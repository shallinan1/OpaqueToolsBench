"""
Simple domain-filtered searcher that wraps FAISS searcher and filters by source.
"""

SCALING_FACTOR=500
FAISS_GPU_MAX_K = 2048
FAISS_MAX_BATCH_QUERIES_CAP = 256
import logging
from typing import Any, Dict, List, Optional
import sys
import os
import json
import torch
from tqdm import tqdm

# Add vendor path for imports
sys.path.append('src/vendor/BrowseComp-Plus/searcher')
from searchers.faiss_searcher import FaissSearcher

logger = logging.getLogger(__name__)


class FaissDomainFilteredSearcher(FaissSearcher):
    """
    Extends FAISS searcher to filter results by domain.
    Works as regular FAISS searcher if no filter is specified.
    """
    
    @classmethod
    def parse_args(cls, parser):
        parser.add_argument(
            "--filter-domain",
            default=None,  # Changed to None to indicate no filtering
            help="Domain(s) to filter results. Can be: 1) Single domain (e.g., 'wikipedia.org'), 2) Comma-separated domains (e.g., 'wikipedia.org,nature.com'), 3) Not specified for no filtering.",
        )
        parser.add_argument(
            "--categories-file",
            default=None,
            help="JSON file containing category-to-domains mapping (e.g., categories.json)",
        )
        parser.add_argument(
            "--filter-category",
            default=None,
            help="Category name to filter by (requires --categories-file). E.g., 'academic', 'news', 'wikipedia'",
        )
        parser.add_argument(
            "--show-urls",
            action="store_true",
            help="Include URLs in search results (default: False)",
        )
        super().parse_args(parser)
    
    def __init__(self, args):
        # Store filter configuration for later use
        self.categories_file = getattr(args, 'categories_file', None)
        self.categories: Optional[Dict] = None
        self.show_urls = getattr(args, 'show_urls', True)
        self._insufficient_results_warning_count = 0
        requested_batch_queries = int(getattr(args, "max_batch_search_queries", 32))
        self.max_batch_search_queries = min(
            FAISS_MAX_BATCH_QUERIES_CAP,
            max(1, requested_batch_queries),
        )
        if requested_batch_queries != self.max_batch_search_queries:
            logger.warning(
                "Clamped FAISS max_batch_search_queries from %d to %d",
                requested_batch_queries,
                self.max_batch_search_queries,
            )

        # Load categories if provided
        if self.categories_file:
            with open(self.categories_file, 'r') as f:
                self.categories = json.load(f)
            logger.info(f"Loaded {len(self.categories)} categories")

        super().__init__(args)
        
        logger.info("FAISS searcher initialized with URL mappings for dynamic filtering")
    
    def _process_filter_args(self, filter_category: Optional[str] = None, filter_domain: Optional[str] = None):
        """Process filter arguments and return list of domains to filter by and inverse flag."""
        
        # Check for conflicting arguments
        if filter_category and not self.categories_file:
            raise ValueError("filter_category requires categories_file to be specified")
        
        if filter_domain and filter_category:
            raise ValueError("Cannot specify both filter_domain and filter_category. Use one or the other.")
        
        # Process category-based filtering
        if filter_category and self.categories_file:
            
            # Special handling for 'other' category
            if filter_category == 'other':
                # Collect all domains from all categories
                all_domains = set()
                for category_domains in self.categories.values():
                    all_domains.update(d.lower() for d in category_domains)
                
                return list(all_domains), True  # Return domains and inverse_filter=True
            
            # Normal category handling
            if filter_category not in self.categories:
                available = list(self.categories.keys()) + ['other']
                raise ValueError(f"Category '{filter_category}' not found. Available categories: {', '.join(sorted(available))}")
            
            domains = self.categories[filter_category]
            return [d.lower() for d in domains], False  # Return domains and inverse_filter=False

        # Process direct domain filtering
        elif filter_domain:
            # Split by comma and strip whitespace
            domains = [d.strip().lower() for d in filter_domain.split(',')]
            return domains, False  # Return domains and inverse_filter=False
        
        # No filtering
        return None, False
    
    def _load_dataset(self) -> None:
        """Load dataset with both text and URLs in one pass."""
        logger.info(f'Loading dataset: {self.args.dataset_name}')
        
        try:
            dataset_cache = os.getenv('HF_DATASETS_CACHE')
            cache_dir = dataset_cache if dataset_cache else None

            from datasets import load_dataset
            ds = load_dataset(self.args.dataset_name, split='train', cache_dir=cache_dir)
            
            # Extract both text and URLs in single iteration
            self.docid_to_text = {}
            self.docid_to_url = {}
            for row in ds:
                docid = row['docid']
                self.docid_to_text[docid] = row['text']
                # Always store URLs for dynamic filtering capability
                self.docid_to_url[docid] = row.get('url', '')
                
            logger.info(f'Loaded {len(self.docid_to_text)} passages from dataset')
            logger.info(f'URLs loaded for dynamic domain filtering')
        except Exception as e:
            if "doesn't exist on the Hub or cannot be accessed" in str(e):
                logger.error(f"Dataset '{self.args.dataset_name}' access failed. This is likely an authentication issue.")
                logger.error("Possible solutions:")
                logger.error("1. Ensure you are logged in to Hugging Face:")
                logger.error("   huggingface-cli login")
                logger.error("2. Set environment variable:")
                logger.error("   export HF_TOKEN=your_token_here")
                logger.error("3. Check if the dataset name is correct and you have access")
                logger.error(f"Current environment variables:")
                logger.error(f"   HF_TOKEN: {'Set' if os.getenv('HF_TOKEN') else 'Not set'}")
                logger.error(f"   HUGGINGFACE_HUB_TOKEN: {'Set' if os.getenv('HUGGINGFACE_HUB_TOKEN') else 'Not set'}")
                
                try:
                    from huggingface_hub import HfApi
                    api = HfApi()
                    user_info = api.whoami()
                    logger.error(f"   Hugging Face user: {user_info.get('name', 'Unknown')}")
                except Exception as auth_e:
                    logger.error(f"   Hugging Face authentication check failed: {auth_e}")
            
            raise RuntimeError(f"Failed to load dataset '{self.args.dataset_name}': {e}")
    
    def _check_domain(self, docid: str, filter_domains: Optional[List[str]] = None, inverse_filter: bool = False) -> bool:
        """
        Check if document's URL matches the filter criteria.
        - Normal mode: Returns True if URL contains any filter domain
        - Inverse mode ('other'): Returns True if URL doesn't contain any filter domain
        - No filter: Returns True for all documents
        """
        # If no filter domains, accept all documents
        if not filter_domains:
            return True
            
        url = self.docid_to_url.get(docid, '').lower()
        matches_any = any(domain in url for domain in filter_domains)
        
        # Inverse filter: accept if URL doesn't match any category domain
        if inverse_filter:
            return not matches_any
        # Normal filter: accept if URL matches any filter domain
        else:
            return matches_any

    def _log_insufficient_results(self, fetched: int, filtered: int, requested_k: int) -> None:
        """Rate-limit repeated low-yield filtering warnings."""
        self._insufficient_results_warning_count += 1
        count = self._insufficient_results_warning_count

        if count == 6:
            logger.info(
                "Suppressing frequent low-yield filter warnings; will log only periodic summaries."
            )

        if count <= 5 or count in (10, 25, 50) or count % 100 == 0:
            logger.warning(
                "Filtered %d retrieved results to %d after domain/category filtering (requested k=%d) [occurrence=%d]",
                fetched,
                filtered,
                requested_k,
                count,
            )

    @staticmethod
    def _scaled_fetch_k(k: int) -> int:
        """Scale retrieval depth for filtering, capped by FAISS GPU top-k limit."""
        return min(max(k * SCALING_FACTOR, 100), FAISS_GPU_MAX_K)
    
    def search(self, query: str, k: int = 10, filter_category: Optional[str] = None, filter_domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search and optionally filter results by domain.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_category: Category to filter by (requires categories_file)
            filter_domain: Domain(s) to filter by (comma-separated)
        """
        # Process dynamic filter arguments
        filter_domains, inverse_filter = self._process_filter_args(filter_category, filter_domain)
        
        # If no filter, just use parent search directly
        if not filter_domains:
            return super().search(query, k)
        
        # Get more results to account for filtering
        fetch_k = self._scaled_fetch_k(k)
        
        # Get unfiltered results from parent
        all_results = super().search(query, fetch_k)
        
        # Filter by domain
        filtered = []
        for original_position, result in enumerate(all_results):
            docid = result.get("docid")
            if docid and self._check_domain(docid, filter_domains, inverse_filter):
                # Add URL if show_urls is enabled
                if self.show_urls:
                    result["url"] = self.docid_to_url.get(docid, "")
                result["original_position"] = original_position
                filtered.append(result)
                if len(filtered) >= k:
                    break
        
        if len(filtered) != k:
            self._log_insufficient_results(len(all_results), len(filtered), k)

        
        return filtered
    
    def get_document(self, docid: str, filter_category: Optional[str] = None, filter_domain: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get document if it matches domain filter (or if no filter is set).
        
        Args:
            docid: Document ID to retrieve
            filter_category: Category to filter by (requires categories_file)
            filter_domain: Domain(s) to filter by (comma-separated)
        """
        # Process dynamic filter arguments
        filter_domains, inverse_filter = self._process_filter_args(filter_category, filter_domain)
        
        if not self._check_domain(docid, filter_domains, inverse_filter):
            return None
        
        doc = super().get_document(docid)
        
        # Add URL if show_urls is enabled
        if doc and self.show_urls and docid in self.docid_to_url:
            doc["url"] = self.docid_to_url.get(docid, "")
        
        return doc
    
    @property
    def search_type(self) -> str:
        return "FAISS_dynamic_filtered"
    
    def search_description(self, k: int = 10, filter_category: Optional[str] = None, filter_domain: Optional[str] = None) -> str:
        """Generate description based on current filter parameters."""
        try:
            filter_domains, inverse_filter = self._process_filter_args(filter_category, filter_domain)
            
            if filter_domains:
                if inverse_filter:
                    return f"Search documents NOT from any known category ({len(filter_domains)} excluded domains). Returns top-{k} filtered results."
                elif len(filter_domains) == 1:
                    return f"Search documents from {filter_domains[0]}. Returns top-{k} filtered results."
                else:
                    return f"Search documents from {len(filter_domains)} domains. Returns top-{k} filtered results."
        except Exception:
            pass
        
        return f"Search all documents. Returns top-{k} results."
    
    def batch_search(self, queries: List[str], k: int = 10,
                    filter_categories: List[Optional[str]] = None,
                    filter_domains: List[Optional[str]] = None) -> List[List[dict]]:
        """
        Batch search using native FAISS batch processing and GPU acceleration.
        
        Args:
            queries: List of search queries
            k: Number of results per query  
            filter_categories: List of category filters (one per query, can be None)
            filter_domains: List of domain filters (one per query, can be None)
        
        Returns:
            List of result lists, one per query
        """
        # Default handling
        if filter_categories is None:
            filter_categories = [None] * len(queries)
        if filter_domains is None:
            filter_domains = [None] * len(queries)
        
        fetch_k = self._scaled_fetch_k(k)
        num_queries = len(queries)
        chunk_size = self.max_batch_search_queries
        if num_queries > chunk_size:
            logger.info(
                "FAISS batch search chunking: %d queries in chunks of %d (k=%d, fetch_k=%d)",
                num_queries,
                chunk_size,
                k,
                fetch_k,
            )

        results: List[List[dict]] = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        chunk_starts = range(0, num_queries, chunk_size)
        if num_queries > chunk_size:
            chunk_starts = tqdm(
                chunk_starts,
                total=(num_queries + chunk_size - 1) // chunk_size,
                desc="FAISS batch chunks",
                leave=False,
            )

        for start in chunk_starts:
            end = min(start + chunk_size, num_queries)
            chunk_queries = queries[start:end]
            chunk_filter_categories = filter_categories[start:end]
            chunk_filter_domains = filter_domains[start:end]

            # Batch tokenize each chunk
            all_query_texts = [self.args.task_prefix + q for q in chunk_queries]
            batch_dict = self.tokenizer(
                all_query_texts,
                padding=True,
                truncation=True,
                max_length=self.args.max_length,
                return_tensors="pt",
            )

            # Device handling (copy vendor logic)
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

            # Batch encode per chunk
            with torch.amp.autocast(device):
                with torch.no_grad():
                    all_q_reps = self.model.encode_query(batch_dict)
                    all_q_reps = all_q_reps.cpu().detach().numpy()

            # Batch FAISS search per chunk
            all_scores, all_psg_indices = self.retriever.search(all_q_reps, fetch_k)

            # Process each query's results in chunk
            for i, (filter_cat, filter_dom) in enumerate(zip(chunk_filter_categories, chunk_filter_domains)):
                filter_domains_list, inverse_filter = self._process_filter_args(filter_cat, filter_dom)

                if not filter_domains_list:
                    query_results = []
                    for j in range(min(k, len(all_psg_indices[i]))):
                        score = all_scores[i][j]
                        index = all_psg_indices[i][j]
                        passage_id = self.lookup[index]
                        passage_text = self.docid_to_text.get(passage_id, "Text not found")
                        query_results.append({
                            "docid": passage_id,
                            "score": float(score),
                            "text": passage_text
                        })
                else:
                    filtered = []
                    for original_position, (score, index) in enumerate(zip(all_scores[i], all_psg_indices[i])):
                        passage_id = self.lookup[index]
                        if self._check_domain(passage_id, filter_domains_list, inverse_filter):
                            passage_text = self.docid_to_text.get(passage_id, "Text not found")
                            result = {
                                "docid": passage_id,
                                "score": float(score),
                                "text": passage_text,
                                "original_position": original_position
                            }
                            if self.show_urls:
                                result["url"] = self.docid_to_url.get(passage_id, "")
                            filtered.append(result)
                            if len(filtered) >= k:
                                break
                    query_results = filtered

                results.append(query_results)

        return results
