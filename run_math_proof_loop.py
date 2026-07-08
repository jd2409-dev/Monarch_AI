"""Math Proof Loop Runner — Test the SOMA-Mythos-EHRA Mathematical Stack.

Demonstrates the integrated mathematical theorem proving pipeline:
1. Load trained LRLM
2. Initialize SOMA (Perception), Mythos (World Model), EHRA (Execution)
3. Attempt to prove a simple theorem
4. Show the full proof trace
"""
from __future__ import annotations

import os
import sys
import time

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from soma_mythos_ehra.arc3.local_coder import ARCDomainLLM, ARCCoderConfig
from soma_mythos_ehra.arc3.four_tier_dataset import VOCAB_SIZE
from soma_mythos_ehra.arc3.soma_math import SOMAMathEncoder
from soma_mythos_ehra.arc3.mythos_math import MythosMathWorldModel, LEAN_TACTICS
from soma_mythos_ehra.arc3.ehra_math import EHRAMathExecutor


def load_lrlm(checkpoint_path: str = "checkpoints/lrlm_full/lrlm_best.pt"):
    """Load trained LRLM from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = ARCCoderConfig(
        vocab_size=VOCAB_SIZE,
        d_model=512,
        n_layer=8,
        n_head=8,
        max_seq_len=512,
        dropout=0.0,
    )
    model = ARCDomainLLM(config).to(device)
    
    if os.path.exists(checkpoint_path):
        # Allow loading checkpoints with config objects
        torch.serialization.add_safe_globals([ARCCoderConfig])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded LRLM from {checkpoint_path}")
        else:
            print(f"Unknown checkpoint format, using random weights")
    else:
        print(f"No checkpoint at {checkpoint_path}, using random weights")
    
    model.eval()
    return model, device


def build_tokenizer():
    """Build a simple tokenizer for testing."""
    class SimpleTokenizer:
        def __init__(self, vocab_size=8192):
            self.vocab_size = vocab_size
            self.token_to_id = {"[BOS]": 1, "[EOS]": 2, "[PAD]": 0}
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        def encode(self, text):
            tokens = []
            for char in text[:512]:
                tokens.append(ord(char) % self.vocab_size)
            if not tokens:
                tokens = [0]
            return torch.tensor([tokens], dtype=torch.long)
        
        def decode(self, tokens):
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            if isinstance(tokens, list) and len(tokens) > 0:
                if isinstance(tokens[0], list):
                    tokens = tokens[0]
            return "".join(chr(t % 256) for t in tokens if 32 <= t < 127)
    
    return SimpleTokenizer()


def test_soma_encoder():
    """Test SOMA Math Encoder."""
    print("\n" + "="*60)
    print("Testing SOMA Math Encoder")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer()
    
    soma = SOMAMathEncoder(vocab_size=tokenizer.vocab_size, d_model=512).to(device)
    
    # Test encoding
    test_text = "n + 0 = n"
    print(f"Input: {test_text}")
    
    tokens = tokenizer.encode(test_text)
    tokens = tokens.to(device)
    
    with torch.no_grad():
        latent = soma(tokens)
    
    print(f"Output shape: {latent.shape}")
    print(f"Output norm: {latent.norm().item():.4f}")
    print("SOMA Encoder: OK")


def test_mythos_world_model():
    """Test Mythos Math World Model."""
    print("\n" + "="*60)
    print("Testing Mythos Math World Model")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    world_model = MythosMathWorldModel(d_model=512).to(device)
    
    # Test prediction
    state_latent = torch.randn(1, 512).to(device)
    tactic_id = torch.tensor([0]).to(device)  # "intro"
    
    with torch.no_grad():
        result = world_model(state_latent, tactic_id)
    
    print(f"Next latent shape: {result['next_latent'].shape}")
    print(f"Success probability: {result['success_prob'].item():.4f}")
    print(f"Decoded goal shape: {result['decoded_goal'].shape}")
    
    # Test ensemble
    print("\nTesting ensemble prediction...")
    from soma_mythos_ehra.arc3.mythos_math import MythosMathEnsemble
    
    ensemble = MythosMathEnsemble(num_models=3, d_model=512).to(device)
    
    with torch.no_grad():
        ensemble_result = ensemble(state_latent, tactic_id)
    
    print(f"Ensemble mean success: {ensemble_result['mean_success'].item():.4f}")
    print(f"Ensemble std success: {ensemble_result['std_success'].item():.4f}")
    print(f"Ensemble consistency: {ensemble_result['consistency'].item():.4f}")
    
    print("Mythos World Model: OK")


def test_ehra_executor():
    """Test EHRA Math Executor."""
    print("\n" + "="*60)
    print("Testing EHRA Math Executor")
    print("="*60)
    
    # Load LRLM
    lrlm, device = load_lrlm()
    tokenizer = build_tokenizer()
    
    # Create executor
    executor = EHRAMathExecutor(lrlm, tokenizer, project_dir="formal_math_core")
    
    # Test tactic validation
    print("\nTesting tactic validation...")
    test_tactics = ["simp", "invalid_tactic", "intro h", "apply foo"]
    for tactic in test_tactics:
        validated = executor._validate_tactic(tactic)
        print(f"  {tactic:20s} -> {validated}")
    
    # Test alternative suggestion
    print("\nTesting alternative suggestion...")
    test_fails = ["simp", "intro", "apply", "rw"]
    for tactic in test_fails:
        alt = executor._suggest_alternative(tactic)
        print(f"  {tactic:10s} failed -> try {alt}")
    
    print("EHRA Executor: OK")


def test_simple_proof():
    """Test a simple proof attempt."""
    print("\n" + "="*60)
    print("Testing Simple Proof Attempt")
    print("="*60)
    
    # Load LRLM
    lrlm, device = load_lrlm()
    tokenizer = build_tokenizer()
    
    # Create executor
    executor = EHRAMathExecutor(lrlm, tokenizer, project_dir="formal_math_core")
    
    # Simple theorem
    theorem = "theorem add_zero (n : Nat) : n + 0 = n := by"
    
    print(f"Attempting to prove: {theorem}")
    print("Note: This uses simplified Lean 4 simulation")
    
    # Run proof attempt (with limited steps)
    result = executor.prove_conjecture(
        theorem_declaration_str=theorem,
        max_steps=5,  # Limited steps for demo
        use_world_model=True,
    )
    
    print(f"\nProof result:")
    print(f"  Success: {result['success']}")
    print(f"  Steps: {result['steps']}")
    print(f"  Proof trace length: {len(result['proof_trace'])}")
    
    if result['proof_trace']:
        print(f"  First tactic: {result['proof_trace'][0].tactic}")
    
    print("Simple Proof Test: OK")


def main():
    """Run all tests."""
    print("="*60)
    print("SOMA-MYTHOS-EHRA Mathematical Stack Test")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Test individual components
        test_soma_encoder()
        test_mythos_world_model()
        test_ehra_executor()
        
        # Test integrated proof (requires Lean 4)
        # test_simple_proof()  # Uncomment when Lean 4 is installed
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print(f"Tests completed in {elapsed:.2f}s")
    print("="*60)


if __name__ == "__main__":
    main()
