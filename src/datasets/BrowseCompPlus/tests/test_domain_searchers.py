#!/usr/bin/env python3
"""
Test script for both BM25 and FAISS domain-filtered searchers.
Runs comprehensive tests for all searcher configurations.

Usage:
    python3 -m src.datasets.BrowseCompPlus.tests.test_domain_searchers

"""

import logging
from typing import List, Optional

# Import our domain-filtered searchers
from src.datasets.BrowseCompPlus.custom_searcher.faiss_domain_filtered_searcher import FaissDomainFilteredSearcher
from src.datasets.BrowseCompPlus.custom_searcher.bm25_domain_filtered_searcher import BM25DomainFilteredSearcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_faiss_gpu_usage(searcher):
    """Test if FAISS searcher is properly using GPU acceleration."""
    print(f"\n[GPU Usage Test]")
    print("-" * 40)
    
    try:
        import faiss
        
        # Check if FAISS was compiled with GPU support
        gpu_available = faiss.get_num_gpus() > 0
        print(f"GPUs available to FAISS: {faiss.get_num_gpus()}")
        
        if not gpu_available:
            print("⚠️  No GPUs available - FAISS running on CPU")
            return
        
        # Access the underlying FAISS index from the searcher
        if hasattr(searcher, 'index'):
            index = searcher.index
            
            # Check if index is on GPU
            if hasattr(index, 'device') and index.device >= 0:
                print(f"✅ FAISS index is using GPU device: {index.device}")
            elif str(type(index)).find('Gpu') != -1:
                print(f"✅ FAISS index is GPU-accelerated: {type(index).__name__}")
            else:
                print(f"⚠️  FAISS index appears to be CPU-only: {type(index).__name__}")
                
            # Check GPU memory usage if possible
            if hasattr(faiss, 'gpu') and hasattr(faiss.gpu, 'GpuResourcesVector'):
                try:
                    gpu_memory = faiss.gpu.get_mem_info(0)  # GPU 0
                    print(f"GPU memory info: {gpu_memory}")
                except Exception:
                    pass
        else:
            print("⚠️  Cannot access FAISS index from searcher")
            
    except ImportError:
        print("❌ FAISS not available for GPU testing")
    except Exception as e:
        print(f"❌ Error testing GPU usage: {e}")

def test_searcher(searcher, searcher_name: str, test_queries: List[str], top_k: int = 3, 
                  filter_category: Optional[str] = None, filter_domain: Optional[str] = None):
    """Test a searcher with given queries."""
    print(f"\n{'='*80}")
    print(f"Testing {searcher_name} ({searcher.search_type})")
    
    # Process filter args to display filtering status
    filter_domains = None
    inverse_filter = False
    if hasattr(searcher, '_process_filter_args'):
        filter_domains, inverse_filter = searcher._process_filter_args(filter_category, filter_domain)
    
    # Display filtering status
    if filter_domains:
        if inverse_filter:
            print(f"Domain filter: ACTIVE - excluding {len(filter_domains)} category domains ('other' category)")
        elif len(filter_domains) == 1:
            print(f"Domain filter: ACTIVE - filtering to '{filter_domains[0]}'")
        else:
            print(f"Domain filter: ACTIVE - filtering to {len(filter_domains)} domains")
            print(f"  Domains: {', '.join(filter_domains[:5])}{' ...' if len(filter_domains) > 5 else ''}")
    else:
        print(f"Domain filter: INACTIVE - searching all domains")
    
    print(f"{'='*80}")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 60)
        
        try:
            results = searcher.search(query, k=top_k, filter_category=filter_category, filter_domain=filter_domain)
            
            if not results:
                if filter_domains:
                    if len(filter_domains) == 1:
                        print(f"❌ No results found matching domain filter '{filter_domains[0]}'.")
                    else:
                        print(f"❌ No results found matching {len(filter_domains)} domain filters.")
                else:
                    print("❌ No results found.")
            else:
                for j, result in enumerate(results, 1):
                    print(f"\nResult {j}:")
                    print(f"  DocID: {result['docid']}")
                    
                    # URL might not be present when not filtering
                    url = result.get('url', 'N/A')
                    print(f"  URL: {url}")
                    print(f"  Score: {result.get('score', 'N/A'):.4f}")
                    
                    # Show original position if available (only when filtering is active)
                    original_pos = result.get('original_position')
                    if original_pos is not None:
                        print(f"  Original Position: {original_pos} (position in unfiltered results)")
                    
                    # Show text snippet
                    text = result.get('text', '')
                    snippet = text[:150] + "..." if len(text) > 150 else text
                    print(f"  Text: {snippet}")
                    
                    # Only verify domain filtering if filter is active
                    if filter_domains:
                        if url != 'N/A':
                            url_lower = url.lower()
                            matching_domain = None
                            for domain in filter_domains:
                                if domain in url_lower:
                                    matching_domain = domain
                                    break
                            
                            # Check for inverse filter (for 'other' category)
                            if inverse_filter:
                                # For 'other' category: NOT matching any domain is correct
                                if not matching_domain:
                                    print(f"  ✅ 'Other' category verified: URL doesn't match any excluded domains")
                                else:
                                    print(f"  ⚠️  'Other' category issue: URL contains excluded domain '{matching_domain}'")
                            else:
                                # Normal filtering: matching a domain is correct
                                if matching_domain:
                                    print(f"  ✅ Domain filter verified: '{matching_domain}' found in URL")
                                else:
                                    print(f"  ⚠️  Domain filter issue: URL '{url}' doesn't match any filter domains")
                        else:
                            print(f"  ⚠️  Cannot verify domain filter: URL not available")
                    else:
                        # When no filter is active, show domain diversity
                        if url != 'N/A':
                            print(f"  ℹ️  Source domain: {url}")
        
        except Exception as e:
            print(f"❌ Error searching for '{query}': {e}")
            import traceback
            traceback.print_exc()
    
    # Test get_document with first result if available
    if 'results' in locals() and results:
        print(f"\n[Document Retrieval Test]")
        print("-" * 40)
        first_docid = results[0]['docid']
        try:
            doc = searcher.get_document(first_docid, filter_category=filter_category, filter_domain=filter_domain)
            if doc:
                print(f"✅ Successfully retrieved document: {first_docid}")
                
                url = doc.get('url', 'N/A')
                print(f"   URL: {url}")
                print(f"   Text length: {len(doc.get('text', ''))}")
                
                # Verify filtering for retrieved document
                if filter_domains and url != 'N/A':
                    url_lower = url.lower()
                    matches_any = any(domain in url_lower for domain in filter_domains)
                    
                    if inverse_filter:
                        # For 'other' category: NOT matching any domain is correct
                        if not matches_any:
                            print(f"   ✅ Document passes 'other' category filter (not in any excluded domain)")
                        else:
                            print(f"   ⚠️  Document fails 'other' category filter (contains excluded domain)")
                    else:
                        # Normal filtering: matching a domain is correct
                        if matches_any:
                            print(f"   ✅ Document passes domain filter")
                        else:
                            print(f"   ⚠️  Document doesn't match domain filter")
            else:
                print(f"❌ Could not retrieve document: {first_docid}")
        except Exception as e:
            print(f"❌ Error retrieving document {first_docid}: {e}")



class MockArgs:
    """Mock args for searcher initialization"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def run_all_tests():
    """Run comprehensive tests for all searcher configurations."""
    test_queries = [
        "What is machine learning?", 
        "Python programming language", 
        "Climate change effects"
    ]
    
    print("="*80)
    print("COMPREHENSIVE SEARCHER TESTS")
    print("="*80)
    
    # Test configurations
    test_configs = [
        # BM25 tests
        {
            "name": "BM25 - No filtering",
            "searcher_type": "bm25",
            "args": MockArgs(
                index_path="src/vendor/BrowseComp-Plus/indexes/bm25",
                url_mapping_file="src/datasets/BrowseCompPlus/outputs/id_to_url.json",
                categories_file=None,
                filter_category=None,
                filter_domain=None
            )
        },
        {
            "name": "BM25 - Wikipedia domain filter",
            "searcher_type": "bm25", 
            "args": MockArgs(
                index_path="src/vendor/BrowseComp-Plus/indexes/bm25",
                url_mapping_file="src/datasets/BrowseCompPlus/outputs/id_to_url.json",
                categories_file=None,
                filter_category=None,
                filter_domain="wikipedia.org"
            )
        },
        {
            "name": "BM25 - Academic category filter",
            "searcher_type": "bm25",
            "args": MockArgs(
                index_path="src/vendor/BrowseComp-Plus/indexes/bm25",
                url_mapping_file="src/datasets/BrowseCompPlus/outputs/id_to_url.json",
                categories_file="src/datasets/BrowseCompPlus/category_mappings/simple_categories.json",
                filter_category="academic",
                filter_domain=None
            )
        },
        # FAISS tests
        {
            "name": "FAISS - No filtering",
            "searcher_type": "faiss",
            "args": MockArgs(
                index_path="src/vendor/BrowseComp-Plus/indexes/qwen3-embedding-0.6b/*.pkl",
                model_name="Qwen/Qwen3-Embedding-0.6B",
                normalize=False,
                pooling="eos",
                torch_dtype="float16", 
                dataset_name="Tevatron/browsecomp-plus-corpus",
                task_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
                max_length=8192,
                categories_file=None,
                filter_category=None,
                filter_domain=None
            )
        },
        {
            "name": "FAISS - Wikipedia category filter",
            "searcher_type": "faiss",
            "args": MockArgs(
                index_path="src/vendor/BrowseComp-Plus/indexes/qwen3-embedding-0.6b/*.pkl",
                model_name="Qwen/Qwen3-Embedding-0.6B",
                normalize=False,
                pooling="eos",
                torch_dtype="float16",
                dataset_name="Tevatron/browsecomp-plus-corpus", 
                task_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
                max_length=8192,
                categories_file="src/datasets/BrowseCompPlus/category_mappings/simple_categories.json",
                filter_category="wikipedia", 
                filter_domain=None
            )
        }
    ]
    
    # Run all test configurations
    for i, config in enumerate(test_configs, 1):
        print(f"\n[{i}/{len(test_configs)}] {config['name']}")
        print("="*80)
        
        try:
            # Extract filter settings from args
            filter_category = config["args"].filter_category if hasattr(config["args"], 'filter_category') else None
            filter_domain = config["args"].filter_domain if hasattr(config["args"], 'filter_domain') else None
            
            if config["searcher_type"] == "bm25":
                searcher = BM25DomainFilteredSearcher(config["args"])
                test_searcher(searcher, f"BM25 - {config['name']}", test_queries, 2, 
                            filter_category=filter_category, filter_domain=filter_domain)
            else:  # faiss
                searcher = FaissDomainFilteredSearcher(config["args"])
                test_searcher(searcher, f"FAISS - {config['name']}", test_queries, 2,
                            filter_category=filter_category, filter_domain=filter_domain)
                # Test GPU usage for FAISS
                test_faiss_gpu_usage(searcher)
        except Exception as e:
            print(f"❌ Error testing {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED!")
    print("="*80)

def main():
    """Main entry point - run all tests by default.""" 
    run_all_tests()


if __name__ == "__main__":
    main()