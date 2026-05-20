# Baseline optimized configs (EasyTool / Play2Prompt)

These are the tool configs **produced by the baseline methods** and used to generate
the EasyTool / Play2Prompt trajectories reported in the paper (Table 2 / `bfcl_full`).
They are the configs referenced by `config_source` in the shipped trajectory metadata,
so the shipped baseline trajectories can be re-graded against them directly.

Layout:

```
tool_configs/easytool/<opacity>/gpt5_medium_1024/optimized_config.json
tool_configs/play2prompt/<opacity>/<beam-search-hypers>/optimized_config.json
```

- **EasyTool:** the opaque config with EasyTool's generated description + usage scenario
  merged into each tool's `description` field (one-shot, two-stage generation).
- **Play2Prompt:** the opaque config with the beam-searched winning description per tool.

Opacity settings covered (the three Table 2 rows × two test categories):
`name[all:increasing_number]_desc[all:blank]_param[all:remove_all]`,
`name[all:increasing_number]_param[all:remove_all]`,
`name[all:increasing_number]_desc[all:blank]_param[all:blank_descriptions]`.
