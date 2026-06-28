"""LRLM Terminal — interactive dual-engine command shell.

Launch modes:
    python lrlm_terminal.py                          # Standalone
    python lrlm_terminal.py --agent checkpoints/     # Connected to agent
    python lrlm_terminal.py --system1                # Force System 1
    python lrlm_terminal.py --system2                # Force System 2
"""
from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="SOMA-Mythos-EHRA LRLM Interactive Shell")
    parser.add_argument("--agent", type=str, help="Path to saved agent checkpoint dir")
    parser.add_argument("--model", type=str, default="checkpoints/lrlm.pt")
    parser.add_argument("--system1", action="store_true", help="Force all queries to System 1")
    parser.add_argument("--system2", action="store_true", help="Force all queries to System 2")
    args = parser.parse_args()

    print()
    print("=" * 62)
    print("  SOMA-MYTHOS-EHRA DUAL-ENGINE LRLM COGNITIVE CORE")
    print("=" * 62)

    # Initialize components
    import torch
    from soma_mythos_ehra.arc3.lrlm_core import ARCLRLM, LRLMConfig
    from soma_mythos_ehra.arc3.unified_lrlm import UnifiedLRLM, UnifiedConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Load LRLM
    config = LRLMConfig(vocab_size=128, d_model=256, n_layer=4, n_head=8, max_seq_len=64)
    lrlm = ARCLRLM(config)
    if os.path.exists(args.model):
        try:
            lrlm = ARCLRLM.load(args.model)
            print(f"  LRLM: Loaded {lrlm.count_parameters():,} params")
        except Exception as e:
            print(f"  LRLM: Using fresh weights ({e})")
    lrlm = lrlm.to(device)
    print(f"  LRLM: {lrlm.count_parameters():,} params on {device}")

    # Connect to agent
    world_model = None
    action_model = None
    buffer = None

    if args.agent and os.path.isdir(args.agent):
        agent_path = os.path.join(args.agent, "") if not args.agent.endswith(".pt") else args.agent
        try:
            from soma_mythos_ehra.arc3.interactive_agent import InteractiveAgent
            agent = InteractiveAgent()
            # Try loading from checkpoint directory
            for f in os.listdir(args.agent) if os.path.isdir(args.agent) else []:
                if f.endswith(".pt"):
                    agent.load(os.path.join(args.agent, f))
                    break
            world_model = agent.ensemble
            buffer = agent.buffer
            print(f"  Agent: Connected ({len(buffer)} transitions)")
        except Exception as e:
            print(f"  Agent: Could not connect: {e}")

    # Build unified LRLM
    unified = UnifiedLRLM(
        config=UnifiedConfig(),
        lrlm_model=lrlm,
        world_model=world_model,
        action_model=action_model,
        buffer=buffer,
    )

    force_system = None
    if args.system1:
        force_system = "system1"
    elif args.system2:
        force_system = "system2"

    # Print status
    print()
    print(unified.status())
    print()
    print("  Commands: 'help', 'status', 'stats', 'exit'")
    print("  Queries auto-route between System 1 (creative) and System 2 (symbolic)")
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
            stats = unified.get_routing_stats()
            print(f"\n  Session Summary:")
            print(f"    Total queries: {stats['total']}")
            print(f"    System 1: {stats['system1']}")
            print(f"    System 2: {stats['system2']}")
            print(f"    Avg confidence: {stats.get('avg_confidence', 0):.2f}")
            print("  Goodbye.")
            break

        if user_input.lower() == "status":
            print(f"\n{unified.status()}")
            continue

        if user_input.lower() == "stats":
            stats = unified.get_routing_stats()
            print(f"\n  Routing Stats: {stats}")
            continue

        if user_input.lower() == "help":
            print(
                "\n  Available commands:\n"
                "    status    - System status\n"
                "    stats     - Routing statistics\n"
                "    help      - This message\n"
                "    exit/quit - End session\n"
                "\n  Queries are auto-routed:\n"
                "    System 1: Creative writing, brainstorming, broad reasoning\n"
                "    System 2: Grid operations, game mechanics, architecture queries\n"
                "\n  Use --system1 or --system2 flags to force routing."
            )
            continue

        # Process query through dual-engine router
        response = unified.process_query(
            user_input,
            force_system=force_system,
        )
        print(f"\n[LRLM Brain] ──> {response}")


if __name__ == "__main__":
    main()
