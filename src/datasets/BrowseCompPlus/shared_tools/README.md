# BrowseCompPlus Shared Tools

Each JSON file under this directory bundles:

- `tools[].tool_definition` — what the model sees (name, description, parameters).
- `tools[].searcher_config` — what actually runs (`searcher_type`: `bm25` or `faiss`; `filter_category` or `filter_domain`).

Naming convention:

```
{transparent | fully_opaque}_{bm25 | faiss}_{no-doc | no-doc_search-all | no-doc_search-all-only}.json
```

- `transparent` vs `fully_opaque` — opacity of names/descriptions.
- `bm25` vs `faiss` — retrieval backend.
- `no-doc` — the simple paper variant (no `get_document` tool, one search tool per category).
- `no-doc_search-all` — adds an extra "search all domains" tool.
- `no-doc_search-all-only` — only the search-all tool, no per-category tools.

Paper evaluations use the `*_no-doc.json` variants. The others are ablations.
