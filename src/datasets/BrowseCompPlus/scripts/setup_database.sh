#!/bin/bash
set -euo pipefail

# Setup BrowseCompPlus retrieval assets:
# 1) Download pre-built BM25/FAISS indexes
# 2) Build local docid->URL mapping used by domain-filtered searchers

bash src/vendor/BrowseComp-Plus/scripts_build_index/download_indexes.sh

python -m src.datasets.BrowseCompPlus.extract_base_urls \
  --output src/datasets/BrowseCompPlus/outputs/base_url_counts.json \
  --id-url-mapping src/datasets/BrowseCompPlus/outputs/id_to_url.json

echo "BrowseCompPlus retrieval assets are ready."
