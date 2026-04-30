"""
Simple domain-filtered searcher that wraps BM25 searcher and filters by source.
Uses external id_to_url.json mapping file for URL lookups.
"""

SCALING_FACTOR=100
import json
import logging
from typing import Any, Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from tqdm import tqdm

# Add vendor path for imports
sys.path.append('src/vendor/BrowseComp-Plus/searcher')
from searchers.bm25_searcher import BM25Searcher

logger = logging.getLogger(__name__)


class BM25DomainFilteredSearcher(BM25Searcher):
    """
    Extends BM25 searcher to filter results by domain.
    Uses external id_to_url.json mapping file for URL lookups.
    Works as regular BM25 searcher if no filter is specified.
    """
    
    @classmethod
    def parse_args(cls, parser):
        # Add domain filter argument
        parser.add_argument(
            "--filter-domain",
            default=None,
            help="Domain(s) to filter results. Can be: 1) Single domain (e.g., 'wikipedia.org'), 2) Comma-separated domains (e.g., 'wikipedia.org,nature.com'), 3) Not specified for no filtering.",
        )
        # Add URL mapping file argument
        parser.add_argument(
            "--url-mapping-file",
            default="src/datasets/BrowseCompPlus/id_to_url.json",
            help="Path to JSON file mapping document IDs to URLs",
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
        self.url_mapping_file = args.url_mapping_file
        self.categories_file = getattr(args, 'categories_file', None)
        self.show_urls = getattr(args, 'show_urls', True)
        self.id_to_url: Dict[str, str] = {}
        self.categories: Optional[Dict] = None
        self._insufficient_results_warning_count = 0
        
        # Always load URL mapping for dynamic filtering capability
        self._load_url_mapping()
        
        # Load categories if provided
        if self.categories_file:
            with open(self.categories_file, 'r') as f:
                self.categories = json.load(f)
            logger.info(f"Loaded {len(self.categories)} categories")
        
        super().__init__(args)
        
        logger.info(f"BM25 searcher initialized with URL mappings for dynamic filtering")
        logger.info(f"Loaded {len(self.id_to_url)} URL mappings from {self.url_mapping_file}")
    
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
    
    def _load_url_mapping(self):
        """Load the id-to-url mapping from JSON file."""
        with open(self.url_mapping_file, 'r') as f:
            self.id_to_url = json.load(f)
        logger.info(f"Loaded {len(self.id_to_url)} URL mappings")

    
    def _get_url(self, docid: str) -> str:
        """Get URL for a document from the preloaded mapping."""
        return self.id_to_url.get(docid, "")
    
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
            
        url = self._get_url(docid).lower()
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
    
    def search(self, query: str, k: int = 10, filter_category: Optional[str] = None, filter_domain: Optional[str] = None) -> list[dict[str, Any]]:
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
        fetch_k = max(k * SCALING_FACTOR, 100)
        
        # Get unfiltered results from parent
        all_results = super().search(query, fetch_k)
        
        # Filter by domain
        filtered = []
        for original_position, result in enumerate(all_results):
            docid = result.get("docid")
            if docid and self._check_domain(docid, filter_domains, inverse_filter):
                # Add URL if show_urls is enabled
                if self.show_urls:
                    result["url"] = self._get_url(docid)
                result["original_position"] = original_position
                filtered.append(result)
                if len(filtered) >= k:
                    break
        
        if len(filtered) != k:
            self._log_insufficient_results(len(all_results), len(filtered), k)

        return filtered
    
    def get_document(self, docid: str, filter_category: Optional[str] = None, filter_domain: Optional[str] = None) -> Optional[dict[str, Any]]:
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
        if doc and self.show_urls:
            doc["url"] = self._get_url(docid)
        
        return doc
    
    @property
    def search_type(self) -> str:
        return "BM25_dynamic_filtered"
    
    def search_description(self, k: int = 10, filter_category: Optional[str] = None, filter_domain: Optional[str] = None) -> str:
        """Generate description based on current filter parameters."""
        try:
            filter_domains, inverse_filter = self._process_filter_args(filter_category, filter_domain)
            
            if filter_domains:
                if inverse_filter:
                    return f"Search documents NOT from any known category ({len(filter_domains)} excluded domains) using BM25. Returns top-{k} filtered results."
                elif len(filter_domains) == 1:
                    return f"Search documents from {filter_domains[0]} using BM25. Returns top-{k} filtered results."
                else:
                    return f"Search documents from {len(filter_domains)} domains using BM25. Returns top-{k} filtered results."
        except Exception:
            pass
        
        return f"Search all documents using BM25. Returns top-{k} results."
    
    def batch_search(self, queries: List[str], k: int = 10,
                    filter_categories: List[Optional[str]] = None,
                    filter_domains: List[Optional[str]] = None) -> List[List[dict]]:
        """
        Batch search using ThreadPoolExecutor on thread-safe LuceneSearcher.
        
        Args:
            queries: List of search queries
            k: Number of results per query  
            filter_categories: List of category filters (one per query, can be None)
            filter_domains: List of domain filters (one per query, can be None)
        
        Returns:
            List of result lists, one per query
        """
        # Simple defaults
        if filter_categories is None:
            filter_categories = [None] * len(queries)
        if filter_domains is None:
            filter_domains = [None] * len(queries)
        
        # Use ThreadPoolExecutor for parallel searches and optionally show progress.
        num_queries = len(queries)
        results: List[Optional[List[dict]]] = [None] * num_queries
        show_progress = num_queries >= 20

        with ThreadPoolExecutor(max_workers=min(num_queries, 8)) as executor:
            future_to_idx = {}
            for i, (query, filter_cat, filter_dom) in enumerate(zip(queries, filter_categories, filter_domains)):
                future = executor.submit(self.search, query, k, filter_cat, filter_dom)
                future_to_idx[future] = i

            iterator = as_completed(future_to_idx)
            if show_progress:
                iterator = tqdm(iterator, total=num_queries, desc="BM25 batch search", leave=False)

            for future in iterator:
                idx = future_to_idx[future]
                results[idx] = future.result()

        return results
