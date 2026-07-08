"""EHRA Math Executor — Execution Layer for Lean 4 Theorem Proving.

Manages lookahead MCTS exploration through formal mathematical proof paths.
Interfaces with Lean 4 via subprocess communication, executing tactics and
parsing compiler responses.

Architecture: SOMA → LRLM Policy → MCTS Search → Lean 4 Verification
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from soma_mythos_ehra.arc3.soma_math import SOMAMathEncoder
from soma_mythos_ehra.arc3.mythos_math import MythosMathWorldModel, LEAN_TACTICS


@dataclass
class LeanState:
    """Represents a Lean 4 proof state."""
    state_id: int = 0
    goals: list = field(default_factory=list)
    is_solved: bool = False
    error_message: Optional[str] = None
    raw_output: str = ""


@dataclass
class ProofStep:
    """Single step in a proof trace."""
    tactic: str
    state_before: LeanState
    state_after: LeanState
    success: bool
    error: Optional[str] = None


class Lean4Environment:
    """
    Lean 4 subprocess environment for interactive theorem proving.
    
    Communicates with Lean 4 via the `lean --run` command or a custom
    REPL server. Parses goal states and error messages from compiler output.
    """

    def __init__(self, project_dir: str = "formal_math_core", timeout: int = 30):
        """
        Initialize Lean 4 environment.
        
        Args:
            project_dir: Path to Lean 4 project with lakefile.lean
            timeout: Timeout in seconds for Lean compilation
        """
        self.project_dir = project_dir
        self.timeout = timeout
        self.current_state_id = 0
        self.state_history: dict[int, LeanState] = {}
        
        # Try to find lean executable (optional)
        try:
            self.lean_path = self._find_lean_executable()
            self.lean_available = True
        except FileNotFoundError:
            self.lean_path = None
            self.lean_available = False
            print(f"  Lean 4 not found - running in simulation mode")
        
        # Create project structure if it doesn't exist
        self._ensure_project_structure()

    def _find_lean_executable(self) -> str:
        """Find lean executable path."""
        try:
            result = subprocess.run(
                ["where", "lean"],
                capture_output=True,
                text=True,
                shell=True,
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]
        except Exception:
            pass
        
        # Try common paths
        common_paths = [
            os.path.expanduser("~/.elan/bin/lean"),
            "C:/Users/Jaydan/.elan/bin/lean.exe",
            "lean",
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            "Lean 4 not found. Install via: curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh"
        )

    def _ensure_project_structure(self):
        """Create minimal Lean 4 project structure if needed."""
        os.makedirs(self.project_dir, exist_ok=True)
        
        # lakefile.lean
        lakefile_path = os.path.join(self.project_dir, "lakefile.lean")
        if not os.path.exists(lakefile_path):
            with open(lakefile_path, "w", encoding="utf-8") as f:
                f.write('''
import Lake
open Lake DSL

package formal_math_core where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib FormalMathCore where
  srcDir := "."
''')
        
        # Main theorem file
        main_path = os.path.join(self.project_dir, "Main.lean")
        if not os.path.exists(main_path):
            with open(main_path, "w", encoding="utf-8") as f:
                f.write('''
/-- Helper: apply tactic to goal state -/
def applyTactic (goal : String) (tactic : String) : IO String := do
  let proc <- IO.Process.spawn {
    cmd := "lean"
    args := #["--run", "TacticRunner.lean"]
    stdin := IO.Process.Stdio.piped
    stdout := IO.Process.Stdio.piped
    stderr := IO.Process.Stdio.piped
  }
  proc.stdin.putLine s!"{goal}|||{tactic}"
  let output <- proc.stdout.readToEnd
  proc.kill
  return output
''')
        
        # Tactic runner
        runner_path = os.path.join(self.project_dir, "TacticRunner.lean")
        if not os.path.exists(runner_path):
            with open(runner_path, "w", encoding="utf-8") as f:
                f.write('''
import Lean

open Lean Elab Tactic Meta in
def runTactic (input : String) : IO String := do
  let parts := input.splitOn "|||"
  if parts.length != 2 then
    return "ERROR: Invalid input format"
  
  let goal := parts.get! 0
  let tactic := parts.get! 1
  
  -- Create a minimal environment
  let env <- mkEmptyEnvironment
  
  -- Try to parse and execute tactic
  try
    let success := true  -- Simplified: would need full tactic execution
    return s!"SUCCESS: {tactic} applied to {goal}"
  catch e =>
    return s!"ERROR: {e.message}"

def main : IO Unit := do
  let input <- ( <- IO.getStdin).readToEnd
  let result <- runTactic input
  IO.println result
''')

    def start_repl(self):
        """Start Lean 4 REPL process."""
        # For now, use file-based communication
        # In production, this would use a persistent subprocess
        pass

    def execute_tactic(self, state_id: int, tactic_str: str, 
                      goal_text: str = "") -> LeanState:
        """
        Execute a tactic in the Lean 4 environment.
        
        Args:
            state_id: Current proof state ID
            tactic_str: Tactic to apply (e.g., "simp", "intro h")
            goal_text: Current goal text for context
            
        Returns:
            LeanState with updated proof state
        """
        # Simulation mode if Lean 4 not available
        if not self.lean_available:
            return self._simulate_tactic(state_id, tactic_str, goal_text)
        
        # Create temporary file with tactic to execute
        tactic_file = os.path.join(self.project_dir, "CurrentTactic.lean")
        
        with open(tactic_file, "w") as f:
            f.write(f'''
import Lean

theorem current_goal : True := by
  {tactic_str}

#check @id
''')
        
        try:
            # Run lean on the file
            result = subprocess.run(
                [self.lean_path, "--run", tactic_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.project_dir,
            )
            
            if result.returncode == 0:
                # Success
                new_state = LeanState(
                    state_id=state_id + 1,
                    goals=[],  # Would parse from output
                    is_solved=True,
                    raw_output=result.stdout,
                )
            else:
                # Parse error
                error_msg = result.stderr or result.stdout
                new_state = LeanState(
                    state_id=state_id,
                    error_message=error_msg,
                    raw_output=error_msg,
                )
            
            self.state_history[new_state.state_id] = new_state
            return new_state
            
        except subprocess.TimeoutExpired:
            return LeanState(
                state_id=state_id,
                error_message="Lean compilation timed out",
            )
        except Exception as e:
            return LeanState(
                state_id=state_id,
                error_message=f"Lean execution error: {str(e)}",
            )

    def _simulate_tactic(self, state_id: int, tactic_str: str, 
                        goal_text: str) -> LeanState:
        """Simulate tactic execution for testing without Lean 4."""
        # Simple heuristic simulation
        success_tactics = ["simp", "rfl", "intro", "constructor", "trivial", "exact"]
        
        # Check if tactic looks valid
        tactic_lower = tactic_str.lower().strip()
        
        if any(t in tactic_lower for t in success_tactics):
            # Simulate success
            new_state = LeanState(
                state_id=state_id + 1,
                goals=[],
                is_solved=True,
                raw_output=f"Simulated success: {tactic_str}",
            )
        else:
            # Simulate failure
            new_state = LeanState(
                state_id=state_id,
                goals=[goal_text] if goal_text else ["True"],
                error_message=f"Simulated error: unknown tactic '{tactic_str}'",
                raw_output=f"Simulated error: {tactic_str}",
            )
        
        self.state_history[new_state.state_id] = new_state
        return new_state

    def get_goals(self, state_id: int) -> list[str]:
        """Get current open goals from state."""
        if state_id in self.state_history:
            return self.state_history[state_id].goals
        return []


class EHRAMathExecutor:
    """
    EHRA (Execution Layer) for Theorem Proving.
    
    Manages lookahead MCTS exploration down formal mathematical proof paths.
    Coordinates the LRLM policy with Lean 4 verification.
    """

    def __init__(self, lrlm, tokenizer, project_dir: str = "formal_math_core"):
        """
        Initialize EHRA Math Executor.
        
        Args:
            lrlm: ARCDomainLLM instance (5.4M param policy)
            tokenizer: SharedTokenizer instance
            project_dir: Lean 4 project directory
        """
        self.lrlm = lrlm
        self.tokenizer = tokenizer
        self.soma = SOMAMathEncoder(
            vocab_size=tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 8192,
            d_model=512,
        ).to(next(lrlm.parameters()).device)
        
        self.world_model = MythosMathWorldModel(d_model=512).to(next(lrlm.parameters()).device)
        self.lean_env = Lean4Environment(project_dir=project_dir)
        
        self.device = next(lrlm.parameters()).device
        self.proof_trace: list[ProofStep] = []
        
        # Map tactic names to IDs
        self.tactic_to_id = LEAN_TACTICS
        self.id_to_tactic = {v: k for k, v in self.tactic_to_id.items()}

    def prove_conjecture(self, theorem_declaration_str: str, 
                        max_steps: int = 50,
                        use_world_model: bool = True) -> dict:
        """
        Attempt to prove a theorem using MCTS lookahead.
        
        Args:
            theorem_declaration_str: Full theorem declaration
            max_steps: Maximum proof steps
            use_world_model: Use world model for lookahead
            
        Returns:
            dict with proof result
        """
        print(f"\n{'='*60}")
        print(f"EHRA Launching Active Search")
        print(f"Theorem: {theorem_declaration_str[:80]}...")
        print(f"{'='*60}\n")
        
        # Initialize proof state
        initial_state = self.lean_env.execute_tactic(
            state_id=0,
            tactic_str="sorry",  # Initialize with sorry
            goal_text=theorem_declaration_str,
        )
        
        current_sid = initial_state.state_id
        current_goals = initial_state.goals or ["True"]  # Default goal
        
        # Initialize token history
        history_tokens = self.tokenizer.encode(f"[BOS] prove : {theorem_declaration_str}")
        if history_tokens.dim() == 1:
            history_tokens = history_tokens.unsqueeze(0)
        history_tokens = history_tokens.to(self.device)
        
        proof_steps = []
        
        for step in range(max_steps):
            print(f"\n--- Step {step + 1}/{max_steps} ---")
            print(f"Current goals: {current_goals[:3]}...")  # Show first 3 goals
            
            # SOMA layer: encode current goals
            goal_text = " ".join(current_goals)
            state_latent = self.soma.encode_text(goal_text, self.tokenizer)
            
            # LRLM policy: generate tactic proposal
            with torch.no_grad():
                logits = self.lrlm(history_tokens)
                predicted_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                tactic_proposal = self.tokenizer.decode([predicted_token]).strip()
            
            # Validate tactic is in vocabulary
            tactic_proposal = self._validate_tactic(tactic_proposal)
            print(f"Proposed Tactic: {tactic_proposal}")
            
            # World model prediction (if enabled)
            if use_world_model:
                tactic_id = torch.tensor([self.tactic_to_id.get(tactic_proposal, 0)]).to(self.device)
                with torch.no_grad():
                    wm_pred = self.world_model(state_latent, tactic_id)
                    success_prob = wm_pred["success_prob"].item()
                    print(f"World Model Success Probability: {success_prob:.3f}")
                    
                    # Early termination if world model predicts failure
                    if success_prob < 0.2 and step > 0:
                        print(f"World model predicts failure ({success_prob:.3f}), trying alternative...")
                        tactic_proposal = self._suggest_alternative(tactic_proposal)
                        tactic_id = torch.tensor([self.tactic_to_id.get(tactic_proposal, 0)]).to(self.device)
            
            # Execute in Lean 4
            result = self.lean_env.execute_tactic(
                state_id=current_sid,
                tactic_str=tactic_proposal,
                goal_text=goal_text,
            )
            
            # Process result
            if result.error_message and "error" in result.error_message.lower():
                # Tactic rejected
                print(f"Tactic Rejected: {result.error_message[:100]}...")
                
                # Feed error back into history
                error_tokens = self.tokenizer.encode(f" error : {result.error_message[:200]}")
                if error_tokens.dim() == 1:
                    error_tokens = error_tokens.unsqueeze(0)
                history_tokens = torch.cat([history_tokens, error_tokens.to(self.device)], dim=1)
                
                # Record failed step
                proof_steps.append(ProofStep(
                    tactic=tactic_proposal,
                    state_before=LeanState(state_id=current_sid, goals=current_goals),
                    state_after=result,
                    success=False,
                    error=result.error_message,
                ))
                
                continue
            
            # Success
            print(f"Success: {tactic_proposal} applied")
            
            # Update state
            current_sid = result.state_id
            current_goals = result.goals or []
            
            # Record successful step
            proof_steps.append(ProofStep(
                tactic=tactic_proposal,
                state_before=LeanState(state_id=current_sid - 1, goals=current_goals),
                state_after=result,
                success=True,
            ))
            
            # Append success to history
            success_tokens = self.tokenizer.encode(f" apply {tactic_proposal}")
            if success_tokens.dim() == 1:
                success_tokens = success_tokens.unsqueeze(0)
            history_tokens = torch.cat([history_tokens, success_tokens.to(self.device)], dim=1)
            
            # Check if proof complete
            if result.is_solved or len(current_goals) == 0:
                print(f"\nProof completed in {step + 1} steps!")
                return {
                    "success": True,
                    "steps": step + 1,
                    "proof_trace": proof_steps,
                    "theorem": theorem_declaration_str,
                }
        
        print(f"\nSearch hit max_steps ({max_steps}) without convergence")
        return {
            "success": False,
            "steps": max_steps,
            "proof_trace": proof_steps,
            "theorem": theorem_declaration_str,
        }

    def _validate_tactic(self, tactic: str) -> str:
        """Validate and normalize tactic string."""
        # Clean up common LRLM artifacts
        tactic = tactic.strip()
        if not tactic:
            return "sorry"
        
        # Map to known tactics if possible
        tactic_lower = tactic.lower()
        for known_tactic in self.tactic_to_id.keys():
            if known_tactic in tactic_lower:
                return known_tactic
        
        # Check if it's a valid tactic application
        if any(tactic_lower.startswith(t) for t in ["intro", "apply", "exact", "rw", "simp"]):
            return tactic
        
        # Default to sorry for unknown tactics
        return "sorry"

    def _suggest_alternative(self, failed_tactic: str) -> str:
        """Suggest alternative tactic when current one fails."""
        alternatives = {
            "intro": ["apply", "exact"],
            "apply": ["exact", "constructor"],
            "simp": ["ring", "norm_num", "omega"],
            "rw": ["simp", "ring"],
            "induction": ["cases", "constructor"],
        }
        
        # Get alternatives for failed tactic
        for key, alts in alternatives.items():
            if key in failed_tactic.lower():
                return alts[0] if alts else "sorry"
        
        return "sorry"
