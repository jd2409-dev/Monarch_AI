"""Curriculum Manager — multi-level progression for ARC-AGI-3 games.

Each game has multiple levels of increasing difficulty. The curriculum
manager tracks progress, transfers learned knowledge between levels,
and prioritizes which levels to practice.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class LevelProgress:
    """Progress tracking for a single level."""
    level_idx: int
    attempts: int = 0
    wins: int = 0
    best_actions: int = 999
    avg_actions: float = 0.0
    best_time: float = float("inf")
    mastered: bool = False
    action_history: list[int] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.attempts, 1)

    @property
    def efficiency(self) -> float:
        if not self.action_history:
            return 0.0
        return 1.0 / (1.0 + sum(self.action_history[-5:]) / 5.0)


@dataclass
class GameProgress:
    """Progress tracking for a full game."""
    game_id: str
    levels: list[LevelProgress] = field(default_factory=list)
    total_attempts: int = 0
    total_wins: int = 0
    current_level: int = 0
    human_baselines: list[int] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.total_wins / max(self.total_attempts, 1)

    @property
    def levels_mastered(self) -> int:
        return sum(1 for l in self.levels if l.mastered)

    @property
    def completion_pct(self) -> float:
        if not self.levels:
            return 0.0
        return self.levels_mastered / len(self.levels)

    def get_practice_level(self) -> int:
        """Select the best level to practice next.

        Strategy:
        1. First, work on the current level (not yet mastered)
        2. If current is mastered, try next level
        3. If stuck, revisit hardest mastered level for reinforcement
        """
        # Try current level
        if self.current_level < len(self.levels):
            lvl = self.levels[self.current_level]
            if not lvl.mastered:
                return self.current_level

        # Try next unmastered level
        for i, lvl in enumerate(self.levels):
            if not lvl.mastered:
                return i

        # All mastered — revisit hardest
        if self.levels:
            hardest = max(self.levels, key=lambda l: l.best_actions)
            return hardest.level_idx

        return 0


class CurriculumManager:
    """Manages multi-level curriculum across all games.

    Tracks progress per game per level and provides:
    - Level selection for practice
    - Knowledge transfer between levels
    - Mastery criteria
    - Progress reporting
    """

    def __init__(self, mastery_threshold: int = 3, human_baselines: dict | None = None) -> None:
        self.games: dict[str, GameProgress] = {}
        self.mastery_threshold = mastery_threshold
        self.knowledge_transfer: dict[str, dict] = {}
        self.human_baselines = human_baselines or {}

    def register_game(self, game_id: str, num_levels: int, baselines: list[int] | None = None) -> None:
        """Register a game with its level count."""
        if game_id not in self.games:
            self.games[game_id] = GameProgress(
                game_id=game_id,
                levels=[LevelProgress(level_idx=i) for i in range(num_levels)],
                human_baselines=baselines or [],
            )

    def record_attempt(
        self,
        game_id: str,
        level_idx: int,
        won: bool,
        actions_taken: int,
        elapsed_time: float = 0.0,
    ) -> None:
        """Record an attempt at a level."""
        if game_id not in self.games:
            self.register_game(game_id, level_idx + 1)

        game = self.games[game_id]
        # Extend levels if needed
        while len(game.levels) <= level_idx:
            game.levels.append(LevelProgress(level_idx=len(game.levels)))

        lvl = game.levels[level_idx]
        lvl.attempts += 1
        lvl.action_history.append(actions_taken)
        game.total_attempts += 1

        if won:
            lvl.wins += 1
            game.total_wins += 1
            lvl.best_actions = min(lvl.best_actions, actions_taken)
            lvl.best_time = min(lvl.best_time, elapsed_time)

            # Check mastery
            if lvl.wins >= self.mastery_threshold:
                lvl.mastered = True

            # Update current level
            if level_idx >= game.current_level and lvl.mastered:
                game.current_level = min(level_idx + 1, len(game.levels) - 1)

        # Running average
        lvl.avg_actions = sum(lvl.action_history[-10:]) / len(lvl.action_history[-10:])

    def get_next_level(self, game_id: str) -> int:
        """Get the next level to attempt for a game."""
        if game_id not in self.games:
            return 0
        return self.games[game_id].get_practice_level()

    def should_transfer_knowledge(self, game_id: str) -> bool:
        """Check if we should transfer knowledge from other games."""
        if game_id not in self.games:
            return False
        game = self.games[game_id]
        # Transfer if we have some mastered levels but current is stuck
        if game.levels_mastered > 0 and game.current_level < len(game.levels):
            current = game.levels[game.current_level]
            if current.attempts > 10 and current.win_rate < 0.3:
                return True
        return False

    def get_transfer_source(self, game_id: str) -> str | None:
        """Find the best game to transfer knowledge from."""
        if game_id not in self.games:
            return None

        best_source = None
        best_score = -1

        for gid, gprog in self.games.items():
            if gid == game_id:
                continue
            if gprog.levels_mastered > 0:
                score = gprog.completion_pct * gprog.win_rate
                if score > best_score:
                    best_score = score
                    best_source = gid

        return best_source

    def save_progress(self) -> dict:
        """Serialize progress to dict."""
        return {
            gid: {
                "levels": [
                    {
                        "idx": l.level_idx,
                        "attempts": l.attempts,
                        "wins": l.wins,
                        "best": l.best_actions,
                        "mastered": l.mastered,
                    }
                    for l in g.levels
                ],
                "current": g.current_level,
                "total_attempts": g.total_attempts,
                "total_wins": g.total_wins,
            }
            for gid, g in self.games.items()
        }

    def report(self) -> str:
        """Generate a progress report."""
        lines = ["=== Curriculum Progress ==="]
        for gid, g in self.games.items():
            mastered = g.levels_mastered
            total = len(g.levels)
            lines.append(
                f"  {gid}: {mastered}/{total} mastered, "
                f"win={g.win_rate:.0%}, current_level={g.current_level}"
            )
        return "\n".join(lines)
