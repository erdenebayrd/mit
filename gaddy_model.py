"""Gaddy recognition model (dgaddy/silent_speech), self-contained.
Layer names are identical to the public checkpoint (Zenodo 7183877), so the saved state dicts
from Gaddy_compress_v2.ipynb and Gaddy_compress_KD_only.ipynb load directly.
The model definitions are copied unchanged from Gaddy_compress_v2.ipynb (cell 4); only the
loader and inference helpers at the bottom are new.

forward(x_feat, x_raw, session_ids): x_raw is raw EMG (batch, time, 8); returns logits
(batch, time/8, 38). x_feat and session_ids are not used by this model.
"""
import copy, random
import torch
from torch import nn
import torch.nn.functional as F

D_MODEL, N_HEAD, N_LAYERS, FFN, DROPOUT, RPD = 768, 8, 6, 3072, 0.0, 100

class LearnedRelativePositionalEmbedding(nn.Module):
    def __init__(self, max_relative_pos, num_heads, embedding_dim, unmasked=False,
                 heads_share_embeddings=False, add_to_values=False):
        super().__init__()
        self.max_relative_pos = max_relative_pos; self.num_heads = num_heads
        self.embedding_dim = embedding_dim; self.unmasked = unmasked
        self.heads_share_embeddings = heads_share_embeddings; self.add_to_values = add_to_values
        num_embeddings = 2 * max_relative_pos - 1 if unmasked else max_relative_pos
        embedding_size = [num_embeddings, embedding_dim, 1] if heads_share_embeddings \
            else [num_heads, num_embeddings, embedding_dim, 1]
        if add_to_values: embedding_size[-1] = 2
        initial_stddev = embedding_dim**(-0.5)
        self.embeddings = nn.Parameter(torch.zeros(*embedding_size))
        nn.init.normal_(self.embeddings, mean=0.0, std=initial_stddev)
    def forward(self, query, saved_state=None):
        length = query.shape[0]; decoder_step = False
        used = self.get_embeddings_for_query(length)
        vals = used[..., 1] if self.add_to_values else None
        pl = self.calculate_positional_logits(query, used[..., 0])
        pl = self.relative_to_absolute_indexing(pl, decoder_step)
        return (pl, vals)
    def get_embeddings_for_query(self, length):
        pad_length = max(length - self.max_relative_pos, 0)
        start_pos = max(self.max_relative_pos - length, 0)
        if self.unmasked:
            with torch.no_grad():
                padded = nn.functional.pad(self.embeddings, (0,0,0,0,pad_length,pad_length))
            used = padded.narrow(-3, start_pos, 2*length-1)
        else:
            with torch.no_grad():
                padded = nn.functional.pad(self.embeddings, (0,0,0,0,pad_length,0))
            used = padded.narrow(-3, start_pos, length)
        return used
    def calculate_positional_logits(self, query, relative_embeddings):
        if self.heads_share_embeddings:
            pl = torch.einsum("lbd,md->lbm", query, relative_embeddings)
        else:
            query = query.view(query.shape[0], -1, self.num_heads, self.embedding_dim)
            pl = torch.einsum("lbhd,hmd->lbhm", query, relative_embeddings)
            pl = pl.contiguous().view(pl.shape[0], -1, pl.shape[-1])
        length = query.size(0)
        if length > self.max_relative_pos:
            pad_length = length - self.max_relative_pos
            pl[:,:,:pad_length] -= 1e8
            if self.unmasked: pl[:,:,-pad_length:] -= 1e8
        return pl
    def relative_to_absolute_indexing(self, x, decoder_step):
        length, bsz_heads, _ = x.shape
        if decoder_step: return x.contiguous().view(bsz_heads, 1, -1)
        if self.unmasked:
            x = nn.functional.pad(x, (0,1)); x = x.transpose(0,1)
            x = x.contiguous().view(bsz_heads, length*2*length)
            x = nn.functional.pad(x, (0, length-1))
            x = x.view(bsz_heads, length+1, 2*length-1)
            return x[:, :length, length-1:]
        else:
            x = nn.functional.pad(x, (1,0)); x = x.transpose(0,1)
            x = x.contiguous().view(bsz_heads, length+1, length)
            return x[:, 1:, :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, n_head=4, dropout=0.1, relative_positional=True,
                 relative_positional_distance=100):
        super().__init__()
        self.d_model = d_model; self.n_head = n_head
        d_qkv = d_model // n_head
        assert d_qkv * n_head == d_model
        self.d_qkv = d_qkv
        self.w_q = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_k = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_v = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_o = nn.Parameter(torch.Tensor(n_head, d_qkv, d_model))
        nn.init.xavier_normal_(self.w_q); nn.init.xavier_normal_(self.w_k)
        nn.init.xavier_normal_(self.w_v); nn.init.xavier_normal_(self.w_o)
        self.dropout = nn.Dropout(dropout)
        if relative_positional:
            self.relative_positional = LearnedRelativePositionalEmbedding(
                relative_positional_distance, n_head, d_qkv, True)
        else:
            self.relative_positional = None
    def forward(self, x):
        q = torch.einsum('tbf,hfa->bhta', x, self.w_q)
        k = torch.einsum('tbf,hfa->bhta', x, self.w_k)
        v = torch.einsum('tbf,hfa->bhta', x, self.w_v)
        logits = torch.einsum('bhqa,bhka->bhqk', q, k) / (self.d_qkv ** 0.5)
        if self.relative_positional is not None:
            q_pos = q.permute(2,0,1,3)
            l,b,h,d = q_pos.size()
            position_logits, _ = self.relative_positional(q_pos.reshape(l,b*h,d))
            logits = logits + position_logits.view(b,h,l,l)
        probs = F.softmax(logits, dim=-1); probs = self.dropout(probs)
        o = torch.einsum('bhqk,bhka->bhqa', probs, v)
        out = torch.einsum('bhta,haf->tbf', o, self.w_o)
        return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 relative_positional=True, relative_positional_distance=100):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout,
            relative_positional=relative_positional, relative_positional_distance=relative_positional_distance)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2); src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2); src = self.norm2(src)
        return src

class SimpleEncoder(nn.Module):
    """Key-identical, modern-torch-safe replacement for nn.TransformerEncoder."""
    def __init__(self, layer, n, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n)])
    def forward(self, x, mask=None, src_key_padding_mask=None, is_causal=None):
        for l in self.layers: x = l(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)
        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None
    def forward(self, x):
        iv = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        res = self.res_norm(self.residual_path(iv)) if self.residual_path is not None else iv
        return F.relu(x + res)

class Model(nn.Module):
    def __init__(self, num_features, num_outs, num_aux_outs=None,
                 model_size=D_MODEL, num_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ResBlock(8, model_size, 2),
            ResBlock(model_size, model_size, 2),
            ResBlock(model_size, model_size, 2),
        )
        self.w_raw_in = nn.Linear(model_size, model_size)
        layer = TransformerEncoderLayer(d_model=model_size, nhead=N_HEAD, relative_positional=True,
            relative_positional_distance=RPD, dim_feedforward=FFN, dropout=dropout)
        self.transformer = SimpleEncoder(layer, num_layers)
        self.w_out = nn.Linear(model_size, num_outs)
        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(model_size, num_aux_outs)
    def forward(self, x_feat, x_raw, session_ids):
        if self.training:
            r = random.randrange(8)
            if r > 0:
                shifted = torch.zeros_like(x_raw)
                shifted[:, :-r, :] = x_raw[:, r:, :]
                x_raw = shifted
        x_raw = x_raw.transpose(1,2)
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1,2)
        x_raw = self.w_raw_in(x_raw)
        x = x_raw.transpose(0,1)
        x = self.transformer(x)
        x = x.transpose(0,1)
        if self.has_aux_out:
            return self.w_out(x), self.w_aux(x)
        return self.w_out(x)


# ===================== loaders and inference helper (new) =====================
NUM_OUTS = 38   # 37 characters + CTC blank


def _state_dict(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    return sd


def build(model_size=D_MODEL, num_layers=N_LAYERS):
    return Model(0, NUM_OUTS, model_size=model_size, num_layers=num_layers).eval()


def load_fp32(path):
    m = build()
    m.load_state_dict(_state_dict(path), strict=True)
    return m.eval()


def load_fp16(path):
    m = build().half()
    m.load_state_dict(_state_dict(path), strict=True)
    return m.eval()


def load_kd_student(path):
    m = build(model_size=256, num_layers=3)
    m.load_state_dict(_state_dict(path), strict=True)
    return m.eval()


def quantize_int8(fp32_model, engine=None):
    """Dynamic INT8 (linear layers only), made on this device.
    engine: None keeps the default engine (x86 in Colab); use "qnnpack" on the Raspberry Pi."""
    if engine:
        torch.backends.quantized.engine = engine
    return torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(fp32_model).eval(), {nn.Linear}, dtype=torch.qint8).eval()


def load_int8_saved(path, engine=None):
    """The dynamic INT8 state dict saved in Colab, re-packed for the current engine.
    engine: None keeps the default engine (x86 in Colab); use "qnnpack" on the Raspberry Pi."""
    if engine:
        torch.backends.quantized.engine = engine
    m = torch.ao.quantization.quantize_dynamic(build(), {nn.Linear}, dtype=torch.qint8).eval()
    m.load_state_dict(_state_dict(path), strict=True)
    return m


@torch.no_grad()
def logits(model, emg, half=False):
    """emg: numpy array (time, 8) float32 -> logits tensor (1, time/8, 38) float32."""
    x = torch.as_tensor(emg, dtype=torch.float32).unsqueeze(0)
    if half:
        x = x.half()
    dummy = torch.zeros(1, 1, dtype=torch.long)
    return model(x, x, dummy).float()
