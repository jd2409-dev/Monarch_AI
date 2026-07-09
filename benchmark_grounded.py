"""Benchmark with grounded LRLM + calibrated world model."""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from soma_mythos_ehra.arc3.local_coder import ARCDomainLLM, ARCCoderConfig
from soma_mythos_ehra.arc3.active_world_model import HypothesisEnsemble
from soma_mythos_ehra.arc3.lrlm_agent import LRLMAgent, LRLMAgentConfig
from soma_mythos_ehra.arc3.four_tier_dataset import VOCAB_SIZE


def main():
    print("=" * 60)
    print("BENCHMARK: Grounded LRLM + Calibrated World Model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create agent with default config
    agent_config = LRLMAgentConfig(
        max_steps=100,
        max_episodes=2,
        verification_threshold=0.7,  # Strict threshold
        confidence_threshold=0.15,
        ensemble_size=5,
        verbose=True,
    )
    
    agent = LRLMAgent(agent_config)
    
    # Override LRLM with correct config (d_model=512, n_layer=8)
    model_config = ARCCoderConfig(
        vocab_size=VOCAB_SIZE,
        d_model=512,
        n_layer=8,
        n_head=8,
        max_seq_len=512,
        dropout=0.0,
    )
    agent.lrlm = ARCDomainLLM(model_config).to(device)
    
    # Load grounded LRLM
    grounded_path = "checkpoints/lrlm_full/grounded_model.pt"
    if os.path.exists(grounded_path):
        torch.serialization.add_safe_globals([ARCCoderConfig])
        checkpoint = torch.load(grounded_path, map_location=device, weights_only=True)
        agent.lrlm.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded grounded LRLM: {grounded_path}")
        print(f"  Action accuracy: {checkpoint.get('action_accuracy', 'unknown')}")
    else:
        print(f"Grounded model not found at {grounded_path}")
    
    # Load calibrated world model
    calibrated_path = "checkpoints/calibrated_world_model.pt"
    if os.path.exists(calibrated_path):
        state = torch.load(calibrated_path, map_location=device, weights_only=True)
        for i, m in enumerate(agent.world_model.models):
            if f"model_{i}" in state.get("ensemble", {}):
                m.load_state_dict(state["ensemble"][f"model_{i}"])
        print(f"Loaded calibrated world model: {calibrated_path}")
    else:
        print(f"Calibrated model not found at {calibrated_path}")
    
    # Run benchmark
    print("\nRunning benchmark...")
    agent.play_all(max_games=3, episodes=2)


if __name__ == "__main__":
    main()
