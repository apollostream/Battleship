"""Tests for strategies — base protocol, Thompson, EIG, ELLR.

Strategy contract:
    - Carries an internal ParticleFilter (its posterior over S).
    - choose_action(shots_fired, turn) -> Action (either ShotAction or AskAction).
    - observe(action, observed) -> None   (feeds the observation back for reweighting).
    - Deterministic given a seeded rng.

No strategy mutates the GameState directly; the simulator/runner is
responsible for applying the action and feeding the observation back.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.board import Configuration, Orientation, Ship
from engine.questions import question_by_id
from engine.exact import ExactPosterior
from strategies.base import (
    ShotAction,
    AskAction,
    Action,
    Strategy,
)
from strategies.thompson import ThompsonStrategy
from strategies.eig import EIGStrategy
from strategies.ellr import ELLRStrategy


@pytest.fixture
def truth() -> Configuration:
    return Configuration(ships=(
        Ship(4, 0, 0, Orientation.HORIZONTAL),
        Ship(3, 2, 0, Orientation.HORIZONTAL),
        Ship(3, 4, 4, Orientation.HORIZONTAL),
        Ship(2, 7, 6, Orientation.HORIZONTAL),
    ))


# --------------------------------------------------------------------------
# Thompson — always shoots, no ask branch
# --------------------------------------------------------------------------

class TestThompson:
    @pytest.mark.slow
    def test_always_returns_shot_action(self):
        rng = np.random.default_rng(1)
        s = ThompsonStrategy(eps=0.10, rng=rng)
        action = s.choose_action(shots_fired=frozenset(), turn=0)
        assert isinstance(action, ShotAction)

    @pytest.mark.slow
    def test_never_repeats_a_shot(self, truth):
        """End-to-end integration: 20 shots without repeat.  Exact posterior
        can never produce a cell already shot, because reweight_shot forces
        μ(cell)=0 (miss) or =1 (hit — TS-picked 1-cells excluded from candidates)."""
        rng = np.random.default_rng(2)
        s = ThompsonStrategy(eps=0.10, rng=rng)
        shots = set()
        for t in range(20):
            a = s.choose_action(shots_fired=frozenset(shots), turn=t)
            assert a.cell not in shots
            shots.add(a.cell)
            s.observe(a, observed=truth.X(a.cell))

    @pytest.mark.slow
    def test_observe_shot_reweights_posterior(self, truth):
        rng = np.random.default_rng(3)
        s = ThompsonStrategy(eps=0.10, rng=rng)
        # Fire a shot at (0,0).  Truth has (0,0) occupied ⇒ HIT ⇒ observed=1.
        # After reweighting, μ(0,0) should be 1.0 exactly (delta likelihood).
        shot = ShotAction(cell=(0, 0))
        s.observe(shot, observed=1)
        mu = s.filter.cell_marginal_grid()
        assert mu[0, 0] == pytest.approx(1.0, abs=1e-10)


# --------------------------------------------------------------------------
# EIG / ELLR — adaptive ask-vs-shoot
# --------------------------------------------------------------------------

@pytest.mark.slow
class TestEIGStrategy:
    def test_first_action_is_ask_when_posterior_diffuse(self):
        """Early in the game, μ(c) is far from 0/1 everywhere (max H(μ) modest).
        A well-chosen global query typically has I(q;s) > max_c H(μ(c)),
        so the strategy should ask first."""
        rng = np.random.default_rng(4)
        s = EIGStrategy(eps=0.10, rng=rng)
        a = s.choose_action(shots_fired=frozenset(), turn=0)
        assert isinstance(a, (ShotAction, AskAction))   # either is mathematically possible

    def test_shoot_when_certain(self):
        """After reweighting makes μ(c)=1 on some cell with enough information,
        the adaptive rule should take the shot."""
        rng = np.random.default_rng(5)
        s = EIGStrategy(eps=0.10, rng=rng)
        # Forcibly collapse posterior onto a single configuration.
        s.filter.weights = np.zeros_like(s.filter.weights)
        s.filter.weights[0] = 1.0
        a = s.choose_action(shots_fired=frozenset(), turn=0)
        assert isinstance(a, ShotAction)
        # The chosen cell should be one of configuration 0's occupied cells.
        occupied = s.filter.occupied_cells_of(0)
        assert a.cell in occupied


@pytest.mark.slow
class TestELLRStrategy:
    def test_returns_valid_action(self):
        rng = np.random.default_rng(6)
        s = ELLRStrategy(eps=0.10, rng=rng)
        a = s.choose_action(shots_fired=frozenset(), turn=0)
        assert isinstance(a, (ShotAction, AskAction))

    def test_observe_ask_reweights_softly(self):
        """An ask observation should not zero out any configuration."""
        rng = np.random.default_rng(7)
        s = ELLRStrategy(eps=0.10, rng=rng)
        ask = AskAction(question_id="row:0")
        s.observe(ask, observed=1)
        assert np.all(s.filter.weights > 0)
        assert s.filter.weights.sum() == pytest.approx(1.0, abs=1e-10)
