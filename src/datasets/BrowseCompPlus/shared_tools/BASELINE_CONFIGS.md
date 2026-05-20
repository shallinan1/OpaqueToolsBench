# Baseline optimized configs (EasyTool / Play2Prompt)

Tool configs produced by the baseline methods, used to generate the EasyTool /
Play2Prompt BrowseComp Plus trajectories (the FAISS / Qwen-0.6B-embedder results in the
paper). Referenced by `config_source` in the shipped trajectory metadata.

```
shared_tools/easytool/fully_opaque_faiss_{no-doc,no-doc_search-all}/gpt5_medium_1024/optimized_config.json
shared_tools/play2prompt/fully_opaque_faiss_{no-doc,no-doc_search-all}/<beam-search-hypers>/optimized_config.json
```

Only the FAISS configs are shipped — the paper's reported BrowseComp results use the
FAISS (Qwen3-Embedding-0.6B) backend, not BM25.
