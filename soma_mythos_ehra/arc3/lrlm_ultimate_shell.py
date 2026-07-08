"""LRLM Ultimate Intelligence Shell — Unified Dual-Engine Terminal.

Merges System 1 (Creative Synthesis) and System 2 (Formal Verification)
into a single interactive console. Routes user commands to appropriate
engines based on intent detection.

Commands:
  - write/essay/idea <topic>  → System 1: Creative Synthesis
  - prove/theorem <statement> → System 2: Lean 4 Formal Verification
  - concept <topic>           → System 1: Concept Map Generation
  - hypothesis <observation>  → System 1: Scientific Hypothesis
  - status                    → Show system status
  - exit/quit                 → Exit terminal
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


class UltimateIntelligenceShell:
    """
    Unified Intelligence Terminal.
    
    Routes between System 1 (Creative) and System 2 (Math) based on
    user intent. Manages LRLM lifecycle and engine coordination.
    """

    def __init__(self, checkpoint_path: str = "checkpoints/lrlm_full/lrlm_best.pt"):
        """
        Initialize the Ultimate Intelligence Shell.
        
        Args:
            checkpoint_path: Path to trained LRLM checkpoint
        """
        print("=" * 70)
        print("SOMA-MYTHOS-EHRA UNIFIED INTELLIGENCE TERMINAL")
        print("=" * 70)
        print("Initializing systems...")
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {self.device}")
        
        # Build tokenizer (simplified - use existing tokenizer)
        self.tokenizer = self._build_tokenizer()
        print(f"  Tokenizer: {self.tokenizer.vocab_size} vocab")
        
        # Build LRLM
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
            # Allow loading checkpoints with config objects
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
        
        print(f"  Parameters: {self.total_params:,}")
        print(f"  Engines: Creative, Math, Concept Induction")
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
                # Simple character-level encoding for demo
                tokens = []
                for char in text[:512]:  # Truncate to max seq len
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

    def detect_intent(self, user_input: str) -> str:
        """
        Detect user intent from input text.
        
        Returns:
            "creative", "math", "concept", "status", or "exit"
        """
        lower = user_input.lower()
        
        # Exit commands
        if lower in ["exit", "quit", "q"]:
            return "exit"
        
        # Status
        if lower in ["status", "info", "help"]:
            return "status"
        
        # Math/proof commands
        if any(kw in lower for kw in ["theorem", "prove", "lemma", "formal"]):
            return "math"
        
        # Concept map commands
        if any(kw in lower for kw in ["concept", "map", "outline", "structure"]):
            return "concept"
        
        # Hypothesis commands
        if any(kw in lower for kw in ["hypothesis", "theory", "scientific"]):
            return "hypothesis"
        
        # Default to creative
        return "creative"

    def launch_console(self):
        """Launch the interactive console."""
        print("Commands:")
        print("  write/essay/idea <topic>  → Creative Synthesis")
        print("  prove/theorem <statement> → Lean 4 Formal Verification")
        print("  concept <topic>           → Concept Map Generation")
        print("  hypothesis <observation>  → Scientific Hypothesis")
        print("  status                    → Show system status")
        print("  exit/quit                 → Exit terminal")
        print()
        
        while True:
            try:
                user_input = input("[Operator] ──► ").strip()
                
                if not user_input:
                    continue
                
                intent = self.detect_intent(user_input)
                
                if intent == "exit":
                    print("Shutting down intelligence terminal...")
                    break
                
                elif intent == "status":
                    self._show_status()
                
                elif intent == "math":
                    theorem = user_input
                    for prefix in ["theorem", "prove", "lemma", "formal"]:
                        theorem = theorem.replace(prefix, "").strip()
                    if theorem.startswith(":"):
                        theorem = theorem[1:].strip()
                    self._handle_math(theorem)
                
                elif intent == "concept":
                    topic = user_input
                    for prefix in ["concept", "map", "outline", "structure"]:
                        topic = topic.replace(prefix, "").strip()
                    self._handle_concept(topic)
                
                elif intent == "hypothesis":
                    observation = user_input
                    for prefix in ["hypothesis", "theory", "scientific"]:
                        observation = observation.replace(prefix, "").strip()
                    self._handle_hypothesis(observation)
                
                else:  # creative
                    self._handle_creative(user_input)
                
                self.commands_processed += 1
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"Error: {e}")

    def _handle_creative(self, topic: str):
        """Handle creative writing requests."""
        print(f"\n[System 1: Creative Synthesis]")
        print(f"Generating essay on: '{topic}'")
        print("-" * 50)
        
        essay = self.creative_core.generate_astonishing_essay(
            topic, target_length=500, temperature=0.8
        )
        
        print(essay)
        print()

    def _handle_math(self, theorem: str):
        """Handle formal theorem proving."""
        print(f"\n[System 2: Formal Verification]")
        print(f"Attempting to prove: {theorem[:80]}...")
        print("-" * 50)
        
        result = self.math_core.prove_conjecture(
            theorem, max_steps=30, use_world_model=True
        )
        
        if result["success"]:
            print(f"\nFormally verified in {result['steps']} steps!")
        else:
            print(f"\nProof search inconclusive after {result['steps']} steps")
        print()

    def _handle_concept(self, topic: str):
        """Handle concept map generation."""
        print(f"\n[System 1: Concept Mapping]")
        print(f"Generating concept map for: '{topic}'")
        print("-" * 50)
        
        concept_map = self.creative_core.generate_concept_map(topic, num_concepts=5)
        
        print(f"Topic: {concept_map['topic']}")
        print(f"Concepts:")
        for i, concept in enumerate(concept_map['concepts'], 1):
            print(f"  {i}. {concept}")
        print()

    def _handle_hypothesis(self, observation: str):
        """Handle scientific hypothesis generation."""
        print(f"\n[System 1: Scientific Reasoning]")
        print(f"Generating hypothesis from: '{observation}'")
        print("-" * 50)
        
        hypothesis = self.creative_core.generate_hypothesis(observation)
        
        print(f"Observation: {observation}")
        print(f"Hypothesis: {hypothesis}")
        print()

    def _show_status(self):
        """Show system status."""
        print(f"\n{'='*50}")
        print(f"SOMA-MYTHOS-EHRA System Status")
        print(f"{'='*50}")
        print(f"Parameters: {self.total_params:,}")
        print(f"Device: {self.device}")
        print(f"Commands Processed: {self.commands_processed}")
        print(f"Vocabulary Size: {self.tokenizer.vocab_size}")
        print(f"Inducted Concepts: {len(self.concept_inductor.inducted_concepts)}")
        print(f"{'='*50}")
        print()


def main():
    """Main entry point."""
    shell = UltimateIntelligenceShell()
    shell.launch_console()


if __name__ == "__main__":
    main()
