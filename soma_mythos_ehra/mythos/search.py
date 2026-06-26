from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch

from soma_mythos_ehra.soma.gpu_simulator import TensorGridSimulator
from soma_mythos_ehra.soma.jepa import JEPAWorldModel
from soma_mythos_ehra.types import SearchDecision, normalize_actions


@dataclass
class MythosConfig:
    horizon: int = 24
    simulations: int = 256
    exploration: float = 1.35
    cycle_penalty: float = 0.8
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


class MetacognitiveController:
    def __init__(self, cycle_penalty: float) -> None:
        self.recent_signatures: list[bytes] = []
        self.cycle_penalty = cycle_penalty

    def remember(self, state: torch.Tensor) -> None:
        self.recent_signatures.append(state.detach().cpu().numpy().tobytes())
        self.recent_signatures = self.recent_signatures[-12:]

    def penalty_for(self, state: torch.Tensor, action: int) -> float:
        signature = state.detach().cpu().numpy().tobytes()
        repeats = self.recent_signatures.count(signature)
        oscillation = action in (1, 2, 3, 4) and repeats > 0
        return self.cycle_penalty * (repeats + int(oscillation))


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
        self.meta = MetacognitiveController(self.config.cycle_penalty)
        self.rng = random.Random(self.config.seed)

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
        if best.energy > 4.0:
            reason = "broad fallback after local collision basin"
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
            penalty = self.meta.penalty_for(next_states[idx : idx + 1], action)
            node.children[action] = SearchNode(
                state=next_states[idx : idx + 1],
                action_from_parent=action,
                parent=node,
                energy=float(combined[idx].item() + penalty),
            )

    def _evaluate(self, node: SearchNode) -> float:
        return node.energy + 0.05 * len(node.sequence())

    def _backpropagate(self, node: SearchNode, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent
