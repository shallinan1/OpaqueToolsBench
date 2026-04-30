"""
Extract base URLs from BrowseComp-Plus corpus and count their occurrences.

Usage:
    # Save dataset with base URLs column
    python3 -m src.datasets.BrowseCompPlus.extract_base_urls \
    --output src/datasets/BrowseCompPlus/outputs/base_url_counts.json \
    --id-url-mapping src/datasets/BrowseCompPlus/outputs/id_to_url.json
"""
import json
import argparse
from collections import Counter
from urllib.parse import urlparse
from datasets import load_dataset
from tqdm import tqdm

def extract_base_url(url):
    """Extract base domain from a URL."""
    parsed = urlparse(url)
    # Get the netloc (network location) which is the base domain
    base_url = parsed.netloc
    
    # Handle cases where URL might not have scheme
    if not base_url and not parsed.scheme:
        # Try adding https:// prefix
        parsed = urlparse(f"https://{url}")
        base_url = parsed.netloc
        
    return base_url

def main(args):
    print("Loading BrowseComp-Plus corpus from Hugging Face...")
    
    # Load the dataset
    ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
    print(f"Dataset loaded with {len(ds)} entries")
    
    # Process dataset and add base_url column
    def add_base_url(example):
        url = example.get('url', '')
        base_url = extract_base_url(url)
        example['base_url'] = base_url
        return example
    
    print("Extracting base URLs...")
    # Apply the transformation
    ds_with_base = ds.map(
        add_base_url,
        desc="Extracting base URLs",
        num_proc=args.num_processes
    )
    
    # Build id-to-url mapping if requested (one-liner)
    id_to_url_mapping = {ex['docid']: ex['url'] for ex in ds_with_base} if args.id_url_mapping else {}
    
    # Count base URLs
    base_url_counter = Counter()
    for example in tqdm(ds_with_base, desc="Counting base URLs"):
        base_url = example.get('base_url')
        base_url_counter[base_url] += 1
    
    # Convert Counter to regular dict and sort by count (descending)
    base_url_dict = dict(base_url_counter.most_common())
    
    # Save the base URL counts to a file
    if args.output:
        print(f"\nSaving base URL counts to {args.output}...")
        with open(args.output, 'w') as f:
            json.dump(base_url_dict, f, indent=2)
        print(f"Base URL counts saved to: {args.output}")
    
    # Save the id-to-url mapping if requested
    if args.id_url_mapping:
        print(f"Saving id-to-url mapping to {args.id_url_mapping}...")
        with open(args.id_url_mapping, 'w') as f:
            json.dump(id_to_url_mapping, f, indent=2)
        print(f"ID-to-URL mapping saved to: {args.id_url_mapping}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"Total unique base URLs: {len(base_url_dict)}")
    print(f"Total documents with URLs: {sum(base_url_counter.values())}")
    
    return ds_with_base, base_url_dict, id_to_url_mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and count base URLs from BrowseComp-Plus corpus")
    parser.add_argument("--output", type=str, help="Output file for base URL counts (JSON format, sorted by count descending)")
    parser.add_argument("--id-url-mapping", type=str, default=None, help="Output file for id-to-url mapping (JSON format)")
    parser.add_argument("--num-processes", type=int, default=4, help="Number of parallel processes for data processing")
    main(parser.parse_args())