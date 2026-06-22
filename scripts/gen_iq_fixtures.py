import gguf, numpy as np, json, os
from gguf import quants, GGMLQuantizationType as Q

MODEL='/home/vasis/.cache/dnd-llm/gemma4-12b-iq3/gemma-4-12b-it-UD-IQ3_XXS.gguf'
OUT='/home/vasis/projects_hobby/vllm-rust/crates/core/src/quantization/gguf/iq/testdata'
os.makedirs(OUT, exist_ok=True)

# tensor per type + ggml type id
SPEC = {
  'iq2_xs':  ('blk.16.attn_k.weight', Q.IQ2_XS,  17, 74),
  'iq2_s':   ('blk.0.attn_k.weight',  Q.IQ2_S,   22, 82),
  'iq3_xxs': ('blk.0.attn_v.weight',  Q.IQ3_XXS, 18, 98),
  'iq3_s':   ('blk.0.attn_output.weight', Q.IQ3_S, 21, 110),
  'iq4_xs':  ('blk.1.ffn_down.weight', Q.IQ4_XS, 23, 136),
}
N_BLOCKS = 6
QK_K = 256

r = gguf.GGUFReader(MODEL)
tmap = {t.name: t for t in r.tensors}
for key,(tname, qtype, type_id, type_size) in SPEC.items():
    t = tmap[tname]
    assert t.tensor_type == qtype, (t.tensor_type, qtype)
    raw_rows = t.data  # [rows, row_bytes] uint8
    flat = raw_rows.reshape(-1)  # row-major flat bytes
    nbytes = N_BLOCKS * type_size
    chunk = np.ascontiguousarray(flat[:nbytes])
    # dequantize exactly these blocks
    deq = quants.dequantize(chunk.reshape(N_BLOCKS, type_size), qtype).reshape(-1)
    assert deq.shape[0] == N_BLOCKS*QK_K, deq.shape
    fixture = {
      'type': key,
      'ggml_type_id': type_id,
      'block_size': QK_K,
      'type_size': type_size,
      'n_blocks': N_BLOCKS,
      'raw_hex': chunk.tobytes().hex(),
      'golden_f32': [float(x) for x in deq],
    }
    p = os.path.join(OUT, f'{key}.json')
    json.dump(fixture, open(p,'w'))
    print(f'{key}: tensor={tname} raw={nbytes}B golden={len(deq)} vals  range[{deq.min():.4f},{deq.max():.4f}]  -> {p}')
