"""LRLM Terminal — interactive command shell for the SOMA-Mythos-EHRA architecture.

Launch with:
    python lrlm_terminal.py                    # Standalone mode
    python lrlm_terminal.py --agent agent.pt    # Connected to running agent
"""
from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="SOMA-Mythos-EHRA LRLM Interactive Shell")
    parser.add_argument("--agent", type=str, help="Path to saved agent checkpoint")
    parser.add_argument("--model", type=str, default="checkpoints/lrlm.pt", help="Path to LRLM checkpoint")
    args = parser.parse_args()

    print()
    print("=" * 62)
    print("  SOMA-MYTHOS-EHRA LRLM COGNITIVE CORE INTERACTIVE SHELL")
    print("=" * 62)

    # Initialize components
    from soma_mythos_ehra.arc3.lrlm_core import ARCLRLM, LRLMConfig
    from soma_mythos_ehra.arc3.architecture_chat import SOMA_Mythos_EHRA_ChatHead
    from soma_mythos_ehra.arc3.game_tokenizer import GameTrajectoryTokenizer

    config = LRLMConfig()
    lrlm = ARCLRLM(config)
    print(f"  LRLM: {lrlm.count_parameters():,} parameters initialized")

    # Try loading saved weights
    if os.path.exists(args.model):
        try:
            lrlm = ARCLRLM.load(args.model)
            print(f"  LRLM: Loaded checkpoint from {args.model}")
        except Exception as e:
            print(f"  LRLM: Could not load checkpoint: {e}")
            print(f"  LRLM: Using randomly initialized weights")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    lrlm = lrlm.to(device)
    print(f"  Device: {device}")

    # Try connecting to agent
    world_model = None
    action_model = None
    buffer = None

    if args.agent and os.path.exists(args.agent):
        try:
            from soma_mythos_ehra.arc3.interactive_agent import InteractiveAgent
            agent = InteractiveAgent()
            agent.load(args.agent)
            world_model = agent.ensemble
            buffer = agent.buffer
            print(f"  Agent: Connected ({len(buffer)} transitions)")
        except Exception as e:
            print(f"  Agent: Could not connect: {e}")

    # Build chat controller
    chat = SOMA_Mythos_EHRA_ChatHead(
        lrlm=lrlm,
        world_model=world_model,
        action_model=action_model,
        buffer=buffer,
    )

    print()
    print("  Type 'help' for available commands, 'exit' to quit.")
    print("-" * 62)

    while True:
        try:
            user_input = input("\n[Operator] ──> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Session ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            # Print session summary
            summary = chat.get_session_summary()
            print(f"\n  Session Summary:")
            print(f"    Messages: {summary['total_messages']}")
            print(f"    Queries: {summary['operator_queries']}")
            print(f"    Buffer: {summary['buffer_size']} transitions")
            print(f"    LRLM: {summary['lrlm_params']:,} params")
            print("  Goodbye.")
            break

        response = chat.process_query(user_input)
        print(f"\n[Architecture Brain] ──> {response}")


if __name__ == "__main__":
    main()
