"""Omniscient Intelligence Shell — Meta-Cognitive Dual-Engine Terminal.

Routes user prompts to System 1 (Creative Synthesis) or System 2 (Formal
Verification) using the MetaCognitiveRouter. No keyword matching — pure
entropy-based intent analysis from the LRLM's own hidden states.

Commands:
  - status                    → Show system status
  - exit/quit                 → Exit terminal

Everything else is routed autonomously by the MetaCognitiveRouter.
"""
from __future__ import annotations

import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soma_mythos_ehra.arc3.local_coder import ARCDomainLLM, ARCCoderConfig
from soma_mythos_ehra.arc3.four_tier_dataset import VOCAB_SIZE
from soma_mythos_ehra.arc3.soma_creative import SOMACreativeEngine
from soma_mythos_ehra.arc3.ehra_math import EHRAMathExecutor
from soma_mythos_ehra.arc3.concept_induction import MythosConceptInductor
from soma_mythos_ehra.arc3.meta_router import MetaCognitiveRouter


class OmniscientIntelligenceShell:
    """
    Unified Intelligence Terminal with Meta-Cognitive Routing.
    
    Uses the MetaCognitiveRouter to analyze prompt entropy and hidden states,
    routing to System 1 (Creative) or System 2 (Symbolic) without keywords.
    """

    def __init__(self, checkpoint_path: str = "checkpoints/lrlm_full/lrlm_best.pt"):
        """
        Initialize the Omniscient Intelligence Shell.
        
        Args:
            checkpoint_path: Path to trained LRLM checkpoint
        """
        print("=" * 70)
        print("SOMA-MYTHOS-EHRA OMNISCIENT INTELLIGENCE CORE")
        print("Meta-Cognitive Routing | Zero Keywords | Active Entropy Analysis")
        print("=" * 70)
        print("Initializing systems...")
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {self.device}")
        
        # Build tokenizer
        self.tokenizer = self._build_tokenizer()
        print(f"  Tokenizer: {self.tokenizer.vocab_size} vocab")
        
        # Build LRLM (scratch-built, no local LLMs like Gemma/Qwen3)
        config = ARCCoderConfig(
            vocab_size=VOCAB_SIZE,
            d_model=512,
            n_layer=8,
            n_head=8,
            max_seq_len=512,
            dropout=0.0,  # No dropout for inference
        )
        self.lrlm = ARCDomainLLM(config).to(self.device)
        
        # Load checkpoint if exists
        if os.path.exists(checkpoint_path):
            torch.serialization.add_safe_globals([ARCCoderConfig])
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            if "model_state_dict" in checkpoint:
                self.lrlm.load_state_dict(checkpoint["model_state_dict"])
                print(f"  Loaded checkpoint: {checkpoint_path}")
            else:
                print(f"  Checkpoint format unknown, using random weights")
        else:
            print(f"  No checkpoint found at {checkpoint_path}, using random weights")
        
        self.lrlm.eval()
        
        # Initialize Meta-Cognitive Router (the gatekeeper)
        self.router = MetaCognitiveRouter(self.lrlm, d_model=512, vocab_size=VOCAB_SIZE).to(self.device)
        print(f"  Meta-Cognitive Router: initialized")
        
        # Initialize engines
        self.creative_core = SOMACreativeEngine(self.lrlm, self.tokenizer)
        self.math_core = EHRAMathExecutor(self.lrlm, self.tokenizer)
        self.concept_inductor = MythosConceptInductor(
            vocab_size=VOCAB_SIZE,
            d_model=512,
        ).to(self.device)
        
        # System stats
        self.total_params = sum(p.numel() for p in self.lrlm.parameters())
        self.commands_processed = 0
        self.system1_count = 0
        self.system2_count = 0
        
        print(f"  Parameters: {self.total_params:,}")
        print(f"  Engines: Creative, Math, Concept Induction, Meta-Router")
        print("=" * 70)
        print()

    def _build_tokenizer(self):
        """Build a simple tokenizer wrapper."""
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

    def launch_console(self):
        """Launch the interactive console with meta-cognitive routing."""
        print("Type any prompt. The system will analyze its entropy profile")
        print("and route to the appropriate engine automatically.")
        print("Commands: 'status', 'exit'/'quit'")
        print()
        
        while True:
            try:
                user_input = input("[Operator] ──► ").strip()
                
                if not user_input:
                    continue
                
                # Handle meta-commands directly
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Shutting down intelligence terminal...")
                    break
                
                if user_input.lower() in ["status", "info"]:
                    self._show_status()
                    continue
                
                # Tokenize input for meta-cognitive analysis
                input_ids = self.tokenizer.encode(user_input)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                input_ids = input_ids.to(self.device)
                
                # Meta-Cognitive Analysis
                analysis = self.router.analyze_prompt(input_ids)
                
                routing_score = analysis["routing_score"]
                decision = analysis["decision"]
                confidence = analysis["confidence"]
                entropy_comp = analysis["entropy_component"]
                hidden_comp = analysis["hidden_component"]
                
                # Display meta-cognition analysis
                print(f"\n  [Meta-Cognition] Entropy: {entropy_comp:.3f} | "
                      f"Hidden: {hidden_comp:.3f} | "
                      f"Score: {routing_score:.3f} | "
                      f"Confidence: {confidence:.3f}")
                
                if decision == "SYMBOLIC":
                    print(f"  [Routing] System 2: Formal Logic & Verification Engine")
                    self._handle_math(user_input)
                    self.system2_count += 1
                else:
                    print(f"  [Routing] System 1: Fluid Conceptual Generation Engine")
                    self._handle_creative(user_input)
                    self.system1_count += 1
                
                self.commands_processed += 1
                print()
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"Error: {e}")

    def _handle_creative(self, prompt: str):
        """Handle creative/generative requests."""
        # Generate essay
        essay = self.creative_core.generate_astonishing_essay(
            prompt, target_length=500, temperature=0.8
        )
        print(f"\n  [System 1 Output]")
        print(f"  {essay}")

    def _handle_math(self, prompt: str):
        """Handle formal verification requests."""
        # Clean up prompt for theorem proving
        theorem = prompt
        for prefix in ["theorem", "prove", "lemma", "formal"]:
            theorem = theorem.replace(prefix, "").strip()
        if theorem.startswith(":"):
            theorem = theorem[1:].strip()
        
        # Attempt proof
        result = self.math_core.prove_conjecture(
            theorem, max_steps=30, use_world_model=True
        )
        
        if result["success"]:
            print(f"\n  [System 2] Formally verified in {result['steps']} steps!")
        else:
            print(f"\n  [System 2] Proof search inconclusive after {result['steps']} steps")

    def _show_status(self):
        """Show system status."""
        print(f"\n  {'='*50}")
        print(f"  SOMA-MYTHOS-EHRA System Status")
        print(f"  {'='*50}")
        print(f"  Parameters: {self.total_params:,}")
        print(f"  Device: {self.device}")
        print(f"  Commands Processed: {self.commands_processed}")
        print(f"  System 1 (Creative): {self.system1_count}")
        print(f"  System 2 (Symbolic): {self.system2_count}")
        print(f"  Vocabulary Size: {self.tokenizer.vocab_size}")
        print(f"  Inducted Concepts: {len(self.concept_inductor.inducted_concepts)}")
        print(f"  Router Weights: entropy={torch.sigmoid(self.router.entropy_weight).item():.3f}, "
              f"hidden={torch.sigmoid(self.router.hidden_weight).item():.3f}")
        print(f"  {'='*50}")


def main():
    """Main entry point."""
    shell = OmniscientIntelligenceShell()
    shell.launch_console()


if __name__ == "__main__":
    main()
