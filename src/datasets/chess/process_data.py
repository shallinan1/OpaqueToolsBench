"""
Extract diverse chess positions from Lichess evaluation database.

Processes lichess_db_eval.jsonl to sample chess positions across different 
game phases (opening/middlegame/endgame) and evaluation ranges (black/white 
winning/crushing/mate).

Positions are sampled according to specified proportions to create a balanced 
dataset for training purposes.

Output:
    - chess_positions.json: Sampled positions with metadata

python3 -m src.datasets.chess.process_data \
    --max-lines 1000000 \
    --total-samples 2000 \
    --input-file src/datasets/chess/artifacts/lichess_db_eval.jsonl \
    --output-dir src/datasets/chess/artifacts
"""
import argparse
import json
import os
import pandas as pd
from collections import Counter
import random

# Define target proportions for each evaluation category
EVAL_PROPORTIONS = {
    'equal': 0.4,           # 50% for equal
    'white_better': 0.1,   # 12.5% for white_better
    'black_better': 0.1,   # 12.5% for black_better
    'white_winning': 0.08,   # 6% for white_winning
    'black_winning': 0.08,   # 6% for black_winning
    'white_crushing': 0.06,  # 4% for white_crushing
    'black_crushing': 0.06,  # 4% for black_crushing
    'white_mate': 0.06,     # 2.5% for white_mate
    'black_mate': 0.06,     # 2.5% for black_mate
}

PHASE_PROPORTIONS = {
    'opening': 0.25,
    'middlegame': 0.4,
    'endgame': 0.25,
    'late_endgame': 0.1
}

def get_game_phase_fast(fen):
    """Determine game phase from FEN by counting pieces"""
    pieces = fen.split()[0]
    # Count all pieces except kings
    total = sum(1 for c in pieces if c.isalpha() and c.lower() != 'k')
    
    if total >= 26: return 'opening'
    if total >= 14: return 'middlegame'
    if total >= 8: return 'endgame'
    return 'late_endgame'

def categorize_evaluation(eval_data):
    """Categorize position by evaluation (handles both cp and mate)"""
    pv = eval_data['pvs'][0]
    
    if 'mate' in pv: # Handle mate situations
        mate_in = pv['mate']
        if mate_in > 0:  # Positive = white has mate
            return 'white_mate'
        else:  # Negative = black has mate
            return 'black_mate'
    
    # Handle centipawn evaluations
    cp = pv.get('cp', 0)
    if cp < -500: return 'black_crushing'
    if cp < -200: return 'black_winning'
    if cp < -50: return 'black_better'
    if cp < 50: return 'equal'
    if cp < 200: return 'white_better'
    if cp < 500: return 'white_winning'
    return 'white_crushing'

def extract_diverse_positions(input_file='lichess_db_eval.jsonl',
                            max_lines=1000000,
                            total_samples=10000):
    """
    Extract diverse positions from Lichess evaluation database

    Args:
        input_file: Path to the JSONL file
        max_lines: Maximum number of lines to process (for speed)
        total_samples: Total number of positions to sample

    Returns:
        DataFrame with sampled positions
    """

    # Initialize list to collect all positions
    all_positions = []
    print(f"Processing {input_file}...")
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num >= max_lines:
                break

            if line_num % 100000 == 0:
                print(f"Processed {line_num} positions...")

            try:
                data = json.loads(line)
                fen = data['fen']

                # Get the evaluation with highest depth (most reliable)
                best_eval = max(data['evals'], key=lambda x: x['depth'])

                # Categorize position
                game_phase = get_game_phase_fast(fen)
                eval_category = categorize_evaluation(best_eval)

                # Store position data
                all_positions.append({
                    'fen': fen,
                    'phase': game_phase,
                    'evaluation': eval_category,
                    'cp': best_eval['pvs'][0].get('cp'),
                    'mate': best_eval['pvs'][0].get('mate'),
                    'depth': best_eval['depth']
                })

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                continue

    # Convert to DataFrame
    df = pd.DataFrame(all_positions)
    print(f"Finished processing. Total positions: {len(df)}")

    # Filter for depth (at least 20)
    df = df[df['depth'] >= 20]
    print(f"Filtered for depth >= 20. Total positions: {len(df)}")

    # Print counts of each phase and evaluation
    print("\nBy game phase:")
    phase_counts = df['phase'].value_counts()
    total = len(df)
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count:6d} ({count/total:.3f})")

    print("\nBy evaluation:")
    eval_counts = df['evaluation'].value_counts()
    for eval_cat, count in eval_counts.items():
        print(f"  {eval_cat}: {count:6d} ({count/total:.3f})")

    print(f"\nSampling {total_samples} positions with target proportions...")
    
    sampled_dfs = []

    # Sample for each game phase
    for phase, phase_prop in PHASE_PROPORTIONS.items():
        phase_df = df[df['phase'] == phase]
        phase_samples_target = int(total_samples * phase_prop)
        
        if len(phase_df) == 0:
            print(f"\n{phase}: No positions available")
            continue

        print(f"\n{phase} (target: {phase_samples_target} positions):")
        
        phase_samples = []
        
        # Try to sample from each evaluation category according to EVAL_PROPORTIONS
        for eval_cat, eval_prop in EVAL_PROPORTIONS.items():
            target_count = int(phase_samples_target * eval_prop)
            available_df = phase_df[phase_df['evaluation'] == eval_cat]
            
            if len(available_df) == 0:
                print(f"  {eval_cat}: No positions available (target: {target_count})")
                import sys; sys.exit()
            
            if len(available_df) >= target_count:
                sampled = available_df.sample(n=target_count, replace=False, random_state=0)
                print(f"  {eval_cat}: {len(sampled)} sampled")
            else:
                # Not enough samples - take all available
                sampled = available_df
                print(f"  {eval_cat}: Only {len(sampled)} available (target: {target_count})")
            
            phase_samples.append(sampled)
        
        if phase_samples:
            phase_combined = pd.concat(phase_samples, ignore_index=True)
            sampled_dfs.append(phase_combined)
            print(f"  Total for {phase}: {len(phase_combined)} positions")

    # Combine all samples
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    # Shuffle the final dataset
    sampled_df = sampled_df.sample(frac=1, random_state=0).reset_index(drop=True)


    return sampled_df

def save_positions(df, output_file='chess_positions.json'):
    """Save the sampled positions DataFrame to a JSON file"""
    df.to_json(output_file, orient='records', indent=2)
    print(f"Saved {len(df)} positions to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Extract diverse chess positions from Lichess evaluation database')
    parser.add_argument('--max-lines', type=int, default=100000, 
                       help='Maximum number of lines to process from the JSONL file')
    parser.add_argument('--total-samples', type=int, default=10000, 
                       help='Total number of positions to sample')
    parser.add_argument('--input-file', type=str, 
                       default='src/datasets/chess/artifacts/lichess_db_eval.jsonl', 
                       help='Path to the input JSONL file')
    parser.add_argument('--output-dir', type=str, default='.', 
                       help='Directory to save output files')

    args = parser.parse_args()

    print(f"Configuration:")
    print(f"  Input file: {args.input_file}")
    print(f"  Max lines to process: {args.max_lines:,}")
    print(f"  Total samples: {args.total_samples:,}")
    print(f"  Output directory: {args.output_dir}")
    print()

    # Extract positions
    df = extract_diverse_positions(
        input_file=args.input_file,
        max_lines=args.max_lines,
        total_samples=args.total_samples
    )
    output_path = os.path.join(args.output_dir, 'chess_positions.json')
    save_positions(df, output_path)

    print(f"\nFinal statistics:")
    print(f"Total positions sampled: {len(df)}")

    # Show distribution
    print("\nBy game phase:")
    phase_counts = df['phase'].value_counts()
    for phase, count in phase_counts.items():
        prop = count / len(df)
        target = PHASE_PROPORTIONS.get(phase, 0)
        print(f"  {phase}: {count} ({prop:.1%}) [target: {target:.1%}]")

    print("\nBy evaluation:")
    eval_counts = df['evaluation'].value_counts()
    for eval_cat, count in eval_counts.items():
        prop = count / len(df)
        target = EVAL_PROPORTIONS.get(eval_cat, 0)
        print(f"  {eval_cat}: {count} ({prop:.1%}) [target: {target:.1%}]")

    print(f"\nOutput saved to: {output_path}")
    return df

if __name__ == "__main__":
    main()