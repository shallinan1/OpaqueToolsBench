# Baseline optimized configs (EasyTool / Play2Prompt)

Tool configs produced by the baseline methods, used to generate the EasyTool /
Play2Prompt chess trajectories (Table 3). Referenced by `config_source` in the shipped
trajectory metadata, so baseline trajectories re-grade against them directly.

```
shared_tools/easytool/<config>_obfuscated/gpt5_medium_1024/optimized_config.json
shared_tools/play2prompt/<config>_obfuscated/<beam-search-hypers>/optimized_config.json
```

Covers both paper tool settings: `all_specialists_obfuscated`, `elo_tools_obfuscated`.
EasyTool descriptions are already merged into each tool's `description`.
