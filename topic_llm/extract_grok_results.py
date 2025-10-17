#!/usr/bin/env python
"""Extract Grok results from pickle file"""
import pickle
from pathlib import Path

# Load Grok results
results_file = Path(__file__).parent.parent / 'data' / 'grok_evaluation_results.pkl'
with open(results_file, 'rb') as f:
    results = pickle.load(f)

print("=== GROK EVALUATION RESULTS ===\n")
for dataset_name, data in results.items():
    print(f"\n{dataset_name}:")
    print(f"  Coherence:            {data['scores']['coherence']:.3f}")
    print(f"  Distinctiveness:      {data['scores']['distinctiveness']:.3f}")
    print(f"  Diversity:            {data['scores']['diversity']:.3f}")
    print(f"  Semantic Integration: {data['scores']['semantic_integration']:.3f}")
    print(f"  Overall Score:        {data['scores']['overall_score']:.3f}")

