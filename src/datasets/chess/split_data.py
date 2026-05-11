"""
Simple script to split chess positions into train/test sets.

Usage:
    python3 -m src.datasets.chess.split_data
"""
import json
import random
import os

def main():
    # Load data
    input_file = "src/datasets/chess/artifacts/chess_positions.json"

    print(f"Loading data from {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Total positions: {len(data)}")

    # Add test_id to each position
    for i, position in enumerate(data):
        position['test_id'] = f"chess_{i:05d}"

    # Shuffle data with fixed seed for reproducibility
    random.seed(0)
    random.shuffle(data)

    # Split 10% train, 90% test
    split_idx = int(len(data) * 0.1)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    print(f"Train set: {len(train_data)} positions (10%)")
    print(f"Test set: {len(test_data)} positions (90%)")

    # Save as JSONL files
    train_file = "src/datasets/chess/data/train.jsonl"
    test_file = "src/datasets/chess/data/test.jsonl"

    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_file), exist_ok=True)

    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved train set to {train_file}")

    with open(test_file, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved test set to {test_file}")

if __name__ == "__main__":
    main()