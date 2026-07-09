"""Test Meta-Cognitive Router."""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from soma_mythos_ehra.arc3.local_coder import ARCDomainLLM, ARCCoderConfig
from soma_mythos_ehra.arc3.meta_router import MetaCognitiveRouter

# Build model
config = ARCCoderConfig(vocab_size=8192, d_model=512, n_layer=8, n_head=8, max_seq_len=512, dropout=0.0)
lrlm = ARCDomainLLM(config).cuda()

# Build router
router = MetaCognitiveRouter(lrlm, d_model=512, vocab_size=8192).cuda()

# Test with different prompts
test_prompts = [
    "the emergence of conscious loops inside neural networks",
    "theorem add_zero (n : Nat) : n + 0 = n",
    "let n be an element of natural numbers",
    "write an essay about the meaning of life",
    "forall x : Nat, x + 0 = x",
    "explain how gravity works in simple terms",
]

print("Testing Meta-Cognitive Router:")
print("=" * 60)

for prompt in test_prompts:
    tokens = torch.tensor([[ord(c) % 8192 for c in prompt[:512]]], dtype=torch.long).cuda()
    analysis = router.analyze_prompt(tokens)
    
    print(f"Prompt: {prompt[:50]}...")
    print(f"  Score: {analysis['routing_score']:.3f} | "
          f"Decision: {analysis['decision']} | "
          f"Confidence: {analysis['confidence']:.3f}")
    print()

print("Meta-Cognitive Router: OK")
