import pickle
import pprint

# Load Anthropic results
with open('data/anthropic_evaluation_results.pkl', 'rb') as f:
    anthropic = pickle.load(f)

# Load OpenAI results
with open('data/openai_evaluation_results.pkl', 'rb') as f:
    openai = pickle.load(f)

# Load Grok results
with open('data/grok_evaluation_results.pkl', 'rb') as f:
    grok = pickle.load(f)

print("=" * 80)
print("ANTHROPIC RESULTS")
print("=" * 80)
pprint.pprint(anthropic)

print("\n" + "=" * 80)
print("OPENAI RESULTS")
print("=" * 80)
pprint.pprint(openai)

print("\n" + "=" * 80)
print("GROK RESULTS")
print("=" * 80)
pprint.pprint(grok)

