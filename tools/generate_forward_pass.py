"""
Generates src/data/forward-pass.json: a real (not decorative) forward pass
through one toy Transformer encoder layer, for Lecture Mode.

Sizes are chosen for projector legibility, not realism:
  seq_len = 4   ("The cat sat down")
  d_model = 16
  heads   = 4   (d_k = 4)
  d_ff    = 32

All weights are fixed (seeded) so the output is reproducible. This is the
single source of truth every Phase 3/4 scene animates against.
"""

import json
import numpy as np

rng = np.random.default_rng(seed=42)

SENTENCE = ["The", "cat", "sat", "down"]
VOCAB = ["<pad>", "<unk>", "The", "cat", "sat", "down", "dog", "chased",
         "the", "quickly", "slowly", "ran", "jumped", "over", "under"]
TOKEN_IDS = [VOCAB.index(w) for w in SENTENCE]

D_MODEL = 16
NUM_HEADS = 4
D_K = D_MODEL // NUM_HEADS
D_FF = 32
SEQ_LEN = len(SENTENCE)


def linear(x, w, b):
    return x @ w + b


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def round_list(arr, ndigits=4):
    return np.round(arr, ndigits).tolist()


# --- Parameters (fixed/seeded, stand-in for "trained" weights) ---
E = rng.normal(0, 0.3, size=(len(VOCAB), D_MODEL))

W_q = rng.normal(0, 0.2, size=(D_MODEL, D_MODEL)); b_q = np.zeros(D_MODEL)
W_k = rng.normal(0, 0.2, size=(D_MODEL, D_MODEL)); b_k = np.zeros(D_MODEL)
W_v = rng.normal(0, 0.2, size=(D_MODEL, D_MODEL)); b_v = np.zeros(D_MODEL)
W_o = rng.normal(0, 0.2, size=(D_MODEL, D_MODEL)); b_o = np.zeros(D_MODEL)

W_ff1 = rng.normal(0, 0.2, size=(D_MODEL, D_FF)); b_ff1 = np.zeros(D_FF)
W_ff2 = rng.normal(0, 0.2, size=(D_FF, D_MODEL)); b_ff2 = np.zeros(D_MODEL)


def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def split_heads(x, num_heads):
    seq_len, d_model = x.shape
    d_k = d_model // num_heads
    return x.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)  # [heads, seq, d_k]


def concat_heads(x):
    heads, seq_len, d_k = x.shape
    return x.transpose(1, 0, 2).reshape(seq_len, heads * d_k)


stages = []


def record(stage_id, label, token_vectors, extra=None):
    entry = {
        "id": stage_id,
        "label": label,
        "tokenVectors": round_list(np.asarray(token_vectors)),
    }
    if extra:
        entry.update(extra)
    stages.append(entry)


# 1. Token ids
record("tokenize", "Tokenization", np.zeros((SEQ_LEN, 1)),
       extra={"tokenIds": TOKEN_IDS, "tokens": SENTENCE})

# 2. Embedding
x_embed = E[TOKEN_IDS]
record("embedding", "Word Embedding", x_embed)

# 3. Positional encoding
pe = positional_encoding(SEQ_LEN, D_MODEL)
x_pe = x_embed + pe
record("positional-enc", "Positional Encoding", x_pe, extra={"pe": round_list(pe)})

# 4. Q, K, V
Q = linear(x_pe, W_q, b_q)
K = linear(x_pe, W_k, b_k)
V = linear(x_pe, W_v, b_v)
record("proj-q", "Linear Projection to Q", Q)
record("proj-k", "Linear Projection to K", K)
record("proj-v", "Linear Projection to V", V)

# 5. Split heads
Qh, Kh, Vh = split_heads(Q, NUM_HEADS), split_heads(K, NUM_HEADS), split_heads(V, NUM_HEADS)

# 6. Attention per head
scores = np.einsum("hqd,hkd->hqk", Qh, Kh) / np.sqrt(D_K)
weights = softmax(scores, axis=-1)  # [heads, seq, seq]
attn_out_heads = np.einsum("hqk,hkd->hqd", weights, Vh)  # [heads, seq, d_k]

record(
    "qk-matmul", "Q × Kᵀ (scaled)", x_pe,  # token vectors unchanged here; scores carried in extra
    extra={"attentionScores": round_list(scores)},
)
record(
    "scale-softmax", "Softmax Attention Weights", x_pe,
    extra={"attention": round_list(weights)},
)

attn_concat = concat_heads(attn_out_heads)
record("weighted-sum", "Weighted Sum (per head, concatenated)", attn_concat)

attn_output = linear(attn_concat, W_o, b_o)
record("output-proj", "Output Projection (Wo)", attn_output)

# 7. Residual + LayerNorm
res1 = x_pe + attn_output
ln1 = layer_norm(res1)
record("residual-1", "Residual Connection", res1)
record("layer-norm-1", "Layer Normalization", ln1)

# 8. FFN
ff_hidden = np.maximum(0, linear(ln1, W_ff1, b_ff1))
ff_out = linear(ff_hidden, W_ff2, b_ff2)
record("ffn", "Feed Forward Network", ff_out)

res2 = ln1 + ff_out
ln2 = layer_norm(res2)
record("residual-2", "Residual Connection", res2)
record("layer-norm-2", "Layer Normalization", ln2)
record("encoder-output", "Encoder Output", ln2)

output = {
    "meta": {
        "sentence": SENTENCE,
        "tokenIds": TOKEN_IDS,
        "dModel": D_MODEL,
        "numHeads": NUM_HEADS,
        "dK": D_K,
        "dFF": D_FF,
        "seqLen": SEQ_LEN,
        "vocab": VOCAB,
    },
    # A handful of real rows from E — the 4 sentence tokens plus a few
    # decoys — for the "lookup table" visual in the Embedding scene. Real
    # vectors, not the full vocab (would be unreadable on a projector).
    "embeddingSample": [
        {
            "word": w,
            "tokenId": VOCAB.index(w),
            "vector": round_list(E[VOCAB.index(w)]),
            "used": w in SENTENCE,
        }
        for w in (SENTENCE + ["dog", "chased", "quickly", "over"])
    ],
    # Real learned-projection weights, for proj-q/proj-k/proj-v's
    # "weight-matrix" sub-step (Self-Attention). Previously these existed
    # only as in-memory arrays here and never reached the JSON, so that
    # sub-step had no real numbers to render. Added, not regenerated from
    # scratch — same seed, same E/W_o/W_ff1/W_ff2, so every other stage's
    # values are unchanged.
    "weights": {
        "Wq": round_list(W_q), "bq": round_list(b_q),
        "Wk": round_list(W_k), "bk": round_list(b_k),
        "Wv": round_list(W_v), "bv": round_list(b_v),
        "Wo": round_list(W_o), "bo": round_list(b_o),
        "W_ff1": round_list(W_ff1), "b_ff1": round_list(b_ff1),
        "W_ff2": round_list(W_ff2), "b_ff2": round_list(b_ff2),
    },
    "stages": stages,
}

with open("../src/data/forward-pass.json", "w") as f:
    json.dump(output, f, indent=2)

with open("../public/data/forward-pass.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Wrote forward-pass.json (copies in src/data/ and public/data/) with {len(stages)} stages for sentence: {' '.join(SENTENCE)}")
