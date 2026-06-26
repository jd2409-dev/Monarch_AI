from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field

import torch

from soma_mythos_ehra.soma.gpu_simulator import TensorGridSimulator
from soma_mythos_ehra.soma.jepa import JEPAWorldModel
from soma_mythos_ehra.types import DIRECTION_DELTAS, SearchDecision, normalize_actions


@dataclass
class MythosConfig:
    horizon: int = 24
    simulations: int = 256
    exploration: float = 3.5
    cycle_penalty: float = 15.0
    tabu_window: int = 8
    tabu_revisit_penalty: float = 25.0
    seed: int = 2409


@dataclass
class SearchNode:
    state: torch.Tensor
    action_from_parent: int | None = None
    parent: "SearchNode | None" = None
    visits: int = 0
    total_value: float = 0.0
    energy: float = 0.0
    children: dict[int, "SearchNode"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.total_value / max(self.visits, 1)

    def sequence(self) -> tuple[int, ...]:
        node: SearchNode | None = self
        out: list[int] = []
        while node and node.action_from_parent is not None:
            out.append(node.action_from_parent)
            node = node.parent
        return tuple(reversed(out))


def extract_agent_pos(state: torch.Tensor, agent_value: int = 2) -> tuple[int, int]:
    flat = state.detach().cpu().flatten()
    idx = (flat == agent_value).nonzero(as_tuple=False)
    if idx.numel() == 0:
        return (0, 0)
    pos = idx[0].item()
    w = state.shape[-1]
    return (pos // w, pos % w)


class MetacognitiveController:
    def __init__(
        self,
        cycle_penalty: float,
        tabu_window: int = 8,
        tabu_revisit_penalty: float = 25.0,
    ) -> None:
        self.recent_signatures: list[bytes] = []
        self.cycle_penalty = cycle_penalty
        self.tabu_window = tabu_window
        self.tabu_revisit_penalty = tabu_revisit_penalty
        self.recent_positions: deque[tuple[int, int]] = deque(maxlen=tabu_window)
        self.position_visit_counts: dict[tuple[int, int], int] = {}
        self.step_count: int = 0

    def reset(self) -> None:
        self.recent_signatures.clear()
        self.recent_positions.clear()
        self.position_visit_counts.clear()
        self.step_count = 0

    def remember(self, state: torch.Tensor) -> None:
        self.recent_signatures.append(state.detach().cpu().numpy().tobytes())
        self.recent_signatures = self.recent_positions.maxlen and self.recent_signatures[-self.tabu_window * 2:] or self.recent_signatures[-12:]
        pos = extract_agent_pos(state)
        self.recent_positions.append(pos)
        self.position_visit_counts[pos] = self.position_visit_counts.get(pos, 0) + 1
        self.step_count += 1

    def penalty_for(self, state: torch.Tensor, action: int) -> float:
        pos = extract_agent_pos(state)

        # Position tabu: exponential penalty for recently visited positions
        steps_ago = None
        for i, recent_pos in enumerate(self.recent_positions):
            if recent_pos == pos:
                steps_ago = len(self.recent_positions) - i
                break

        position_penalty = 0.0
        if steps_ago is not None:
            # Exponential decay: closer revisit = stronger penalty
            position_penalty = self.tabu_revisit_penalty * (2.0 ** (self.tabu_window - steps_ago))

        # State signature repeat detection
        signature = state.detach().cpu().numpy().tobytes()
        repeats = self.recent_signatures.count(signature)

        # Oscillation detection: moving back to a position seen within 2 steps
        oscillation = action in (1, 2, 3, 4) and steps_ago is not None and steps_ago <= 2

        return position_penalty + self.cycle_penalty * (repeats + int(oscillation))


class MythosSearch:
    """Paced MCTS-style search over SOMA latent projections."""

    def __init__(
        self,
        simulator: TensorGridSimulator,
        world_model: JEPAWorldModel,
        config: MythosConfig | None = None,
    ) -> None:
        self.simulator = simulator
        self.world_model = world_model.to(simulator.device).eval()
        self.config = config or MythosConfig()
        self.meta = MetacognitiveController(
            cycle_penalty=self.config.cycle_penalty,
            tabu_window=self.config.tabu_window,
            tabu_revisit_penalty=self.config.tabu_revisit_penalty,
        )
        self.rng = random.Random(self.config.seed)

    def reset_meta(self) -> None:
        self.meta.reset()

    @torch.no_grad()
    def choose(self, state: torch.Tensor, available_actions: tuple[int, ...] | None = None) -> SearchDecision:
        actions = normalize_actions(available_actions)
        root = SearchNode(self.simulator.to_device(state))
        self.meta.remember(root.state)

        for _ in range(max(1, self.config.simulations)):
            leaf = self._select(root, actions)
            if len(leaf.sequence()) < self.config.horizon:
                self._expand(leaf, actions)
                if leaf.children:
                    leaf = min(leaf.children.values(), key=lambda c: c.energy)
            value = self._evaluate(leaf)
            self._backpropagate(leaf, value)

        best = min(root.children.values(), key=lambda n: (n.energy, -n.visits, n.action_from_parent or 99))
        reason = "lowest latent/physics energy"
        if best.energy > self.config.cycle_penalty:
            reason = "broad fallback after local collision basin"
        elif best.visits > self.config.simulations * 0.3:
            reason = "most-visited escape from cycle"
        return SearchDecision(
            action=int(best.action_from_parent or actions[0]),
            energy=float(best.energy),
            visits=best.visits,
            sequence=best.sequence(),
            reason=reason,
        )

    def _select(self, node: SearchNode, actions: tuple[int, ...]) -> SearchNode:
        while node.children and set(node.children) >= set(actions):
            log_parent = math.log(max(node.visits, 1))
            node = min(
                node.children.values(),
                key=lambda child: child.value
                - self.config.exploration * math.sqrt(log_parent / max(child.visits, 1)),
            )
        return node

    def _expand(self, node: SearchNode, actions: tuple[int, ...]) -> None:
        untried = [a for a in actions if a not in node.children]
        if not untried:
            return
        batch_actions = torch.tensor(untried, device=self.simulator.device)
        batch_states = node.state.repeat(len(untried), 1, 1)
        next_states, physics_energy = self.simulator.step_batch(batch_states, batch_actions)
        latent_energy = self.world_model.energy(batch_states, batch_actions, next_states)
        goal_energy = self.simulator.distance_to_goal_energy(next_states)
        combined = physics_energy + latent_energy + goal_energy
        for idx, action in enumerate(untried):
            candidate = next_states[idx : idx + 1]
            pos = extract_agent_pos(candidate)
            penalty = self.meta.penalty_for(candidate, action)

            # Extra wall-collision repeat: if the agent stays put (blocked by wall)
            # and we've already seen this position, add a heavy structural penalty
            parent_pos = extract_agent_pos(node.state)
            if pos == parent_pos and action in DIRECTION_DELTAS:
                penalty += 8.0

            node.children[action] = SearchNode(
                state=candidate,
                action_from_parent=action,
                parent=node,
                energy=float(combined[idx].item() + penalty),
            )

    def _evaluate(self, node: SearchNode) -> float:
        depth = len(node.sequence())
        # Stronger depth penalty: deeper sequences accumulate more cost
        # to prefer shorter solution paths
        return node.energy + 0.1 * depth

    def _backpropagate(self, node: SearchNode, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent
