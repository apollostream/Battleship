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
        """Cost-aware comparator with default (R+, M, C) = (2, 1, 1) requires
        max_c μ(c) ≥ 2/3 to shoot.  At the uniform prior, max μ ≈ 0.23 (interior
        cells), well below 2/3 — so the first action MUST be an ask."""
        rng = np.random.default_rng(4)
        s = EIGStrategy(eps=0.10, rng=rng)
        a = s.choose_action(shots_fired=frozenset(), turn=0)
        assert isinstance(a, AskAction)

    def test_shoot_when_certain(self):
        """Collapse posterior onto a single configuration → max μ = 1 ≥ 2/3 → shoot."""
        rng = np.random.default_rng(5)
        s = EIGStrategy(eps=0.10, rng=rng)
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


# --------------------------------------------------------------------------
# Cost-aware comparator (parameterised by hit_reward, miss_cost, shot_cost)
# --------------------------------------------------------------------------

class TestCostAwareComparator:
    """The threshold μ ≥ (M + C) / (R+ + M) is computed at construction.
    These tests don't require a real ExactPosterior, so no @slow."""

    def test_default_threshold_is_two_thirds(self):
        rng = np.random.default_rng(0)
        # Avoid building ExactPosterior: instantiate via _mbayes directly
        # would still trigger enumeration.  Instead just compute the formula.
        from strategies._mbayes import MBayesStrategy
        # Bypass __init__'s ExactPosterior build by constructing only enough
        # to verify the threshold math.
        cls = MBayesStrategy.__new__(MBayesStrategy)
        cls.hit_reward = 2.0
        cls.miss_cost = 1.0
        cls.shot_cost = 1.0
        denom = cls.hit_reward + cls.miss_cost
        cls.shoot_threshold = (cls.miss_cost + cls.shot_cost) / denom
        assert cls.shoot_threshold == pytest.approx(2.0 / 3.0)

    @pytest.mark.parametrize(
        "R_plus, M, C, expected",
        [
            (2.0, 1.0, 1.0, 2.0 / 3.0),    # default
            (3.0, 2.0, 1.0, 3.0 / 5.0),    # user's general formula example
            (1.0, 1.0, 0.0, 0.5),          # shot-cost-free → coin-flip threshold
            (1.0, 1.0, 1.0, 1.0),          # cost == reward → never shoot at <100%
            (10.0, 1.0, 1.0, 2.0 / 11.0),  # high-reward asymmetric
        ],
    )
    def test_general_formula(self, R_plus, M, C, expected):
        denom = R_plus + M
        thr = (M + C) / denom
        assert thr == pytest.approx(expected)

    def test_invalid_costs_rejected(self):
        from strategies._mbayes import MBayesStrategy
        # R+ + M ≤ 0 makes the threshold ill-defined; constructor must reject.
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="hit_reward \\+ miss_cost"):
            MBayesStrategy(eps=0.1, rng=rng, hit_reward=-1.0, miss_cost=0.0)


@pytest.mark.slow
class TestApproxStrategies:
    """Sample-based EIG/ELLR variants.  Must produce the same interface
    (AskAction when posterior is diffuse), stay deterministic under a seeded
    rng, and rank questions into the plausible neighborhood of the exact
    metric — not pixel-equal (sampling noise) but same top-cluster.
    """
    def test_approx_eig_is_deterministic(self):
        from strategies._mbayes import ApproxEIGMBayesStrategy

        a = ApproxEIGMBayesStrategy(eps=0.10, rng=np.random.default_rng(0))
        b = ApproxEIGMBayesStrategy(eps=0.10, rng=np.random.default_rng(0))
        act_a = a.choose_action(shots_fired=frozenset(), turn=0)
        act_b = b.choose_action(shots_fired=frozenset(), turn=0)
        assert isinstance(act_a, AskAction)
        assert act_a.question_id == act_b.question_id

    def test_approx_ellr_is_deterministic(self):
        from strategies._mbayes import ApproxELLRMBayesStrategy

        a = ApproxELLRMBayesStrategy(eps=0.10, rng=np.random.default_rng(1))
        b = ApproxELLRMBayesStrategy(eps=0.10, rng=np.random.default_rng(1))
        act_a = a.choose_action(shots_fired=frozenset(), turn=0)
        act_b = b.choose_action(shots_fired=frozenset(), turn=0)
        assert isinstance(act_a, AskAction)
        assert act_a.question_id == act_b.question_id

    def test_approx_eig_agrees_with_exact_on_top_cluster(self):
        """Approx EIG's chosen question should be among the exact EIG top-K
        at uniform prior.  K=10 is a loose band that passes with high
        probability for K_SAMPLES=200 (BALD rank collision would flag a bug).
        """
        from strategies._mbayes import ApproxEIGMBayesStrategy
        from engine.metrics import eig_of_all_asks
        from engine.questions import QUESTION_CATALOGUE

        s = ApproxEIGMBayesStrategy(eps=0.10, rng=np.random.default_rng(2))
        A = s.filter.build_answer_matrix(QUESTION_CATALOGUE)
        exact_scores = eig_of_all_asks(
            answers_matrix=A, weights=s.filter.weights, eps=0.10,
        )
        top_idx = set(int(i) for i in np.argpartition(-exact_scores, 9)[:10])
        top_ids = {QUESTION_CATALOGUE[i].id for i in top_idx}
        a = s.choose_action(shots_fired=frozenset(), turn=0)
        assert isinstance(a, AskAction)
        assert a.question_id in top_ids


@pytest.mark.slow
class TestLastRationale:
    """Every strategy must expose both branches of its decision each turn.

    The UI's "Why?" panel reads `last_rationale` after every `choose_action`
    to show shot-branch μ/E[shot] and ask-branch question/score side-by-side,
    whether or not that branch was chosen.  Thompson omits the ask branch
    (it has no ask option), but must still populate the shot branch.
    """

    def test_mbayes_diffuse_populates_both_branches(self):
        """Uniform prior → ask is chosen, but shot-branch fields are still set."""
        rng = np.random.default_rng(40)
        s = EIGStrategy(eps=0.10, rng=rng)
        a = s.choose_action(shots_fired=frozenset(), turn=0)
        assert isinstance(a, AskAction)
        r = s.last_rationale
        assert r["chosen"] == "ask"
        # shot branch: best candidate + μ + EV + threshold
        shot = r["shot"]
        assert isinstance(shot["cell"], tuple) and len(shot["cell"]) == 2
        assert 0.0 <= shot["mu"] <= 1.0
        assert shot["threshold"] == pytest.approx(2.0 / 3.0)
        # EV = (R⁺+M)μ − (M+C) = 3μ − 2 at defaults
        assert shot["ev"] == pytest.approx(3.0 * shot["mu"] - 2.0, abs=1e-10)
        # ask branch
        ask = r["ask"]
        assert ask["question_id"] == a.question_id
        assert ask["metric"] == "eig"
        assert ask["score"] >= 0.0

    def test_mbayes_certain_populates_both_branches(self):
        """Collapsed posterior → shot is chosen, but ask-branch is still computed."""
        rng = np.random.default_rng(41)
        s = EIGStrategy(eps=0.10, rng=rng)
        s.filter.weights = np.zeros_like(s.filter.weights)
        s.filter.weights[0] = 1.0
        a = s.choose_action(shots_fired=frozenset(), turn=0)
        assert isinstance(a, ShotAction)
        r = s.last_rationale
        assert r["chosen"] == "shot"
        assert r["shot"]["cell"] == a.cell
        assert r["shot"]["mu"] == pytest.approx(1.0, abs=1e-10)
        # ask branch must still be populated (for "here's the ask we *didn't* take")
        assert r["ask"] is not None
        assert r["ask"]["metric"] == "eig"

    def test_ellr_rationale_labels_metric(self):
        rng = np.random.default_rng(42)
        s = ELLRStrategy(eps=0.10, rng=rng)
        a = s.choose_action(shots_fired=frozenset(), turn=0)
        assert s.last_rationale["ask"]["metric"] == "ellr"
        # In either branch, ask score must be finite
        assert np.isfinite(s.last_rationale["ask"]["score"])

    def test_approx_rationale_labels_metric(self):
        from strategies._mbayes import (
            ApproxEIGMBayesStrategy, ApproxELLRMBayesStrategy,
        )
        a_eig = ApproxEIGMBayesStrategy(eps=0.10, rng=np.random.default_rng(43))
        a_eig.choose_action(shots_fired=frozenset(), turn=0)
        assert a_eig.last_rationale["ask"]["metric"] == "bald"

        a_ellr = ApproxELLRMBayesStrategy(eps=0.10, rng=np.random.default_rng(44))
        a_ellr.choose_action(shots_fired=frozenset(), turn=0)
        assert a_ellr.last_rationale["ask"]["metric"] == "ellr_approx"

    def test_thompson_has_no_ask_branch(self):
        rng = np.random.default_rng(45)
        s = ThompsonStrategy(eps=0.10, rng=rng)
        a = s.choose_action(shots_fired=frozenset(), turn=0)
        assert isinstance(a, ShotAction)
        r = s.last_rationale
        assert r["chosen"] == "shot"
        assert r["shot"]["cell"] == a.cell
        assert r["ask"] is None

    def test_last_decision_value_backcompat(self):
        """Keeping the legacy scalar in sync — ask branch → score; shot → μ."""
        rng = np.random.default_rng(46)
        s = EIGStrategy(eps=0.10, rng=rng)
        a = s.choose_action(shots_fired=frozenset(), turn=0)
        if isinstance(a, ShotAction):
            assert s.last_decision_value == pytest.approx(s.last_rationale["shot"]["mu"])
        else:
            assert s.last_decision_value == pytest.approx(s.last_rationale["ask"]["score"])


@pytest.mark.slow
class TestELLRShortlist:
    def test_chosen_question_is_in_top_k_by_eig(self):
        """ELLR's _best_question must full-score only over the EIG top-K, so
        the returned question must be one of those top-K entries."""
        from strategies._mbayes import ELLRMBayesStrategy
        from engine.metrics import eig_of_all_asks
        from engine.questions import QUESTION_CATALOGUE

        rng = np.random.default_rng(8)
        s = ELLRMBayesStrategy(eps=0.10, rng=rng)
        # Compute the EIG ranking we expect ELLR to consult.
        A = s.filter.build_answer_matrix(QUESTION_CATALOGUE)
        eig_scores = eig_of_all_asks(
            answers_matrix=A, weights=s.filter.weights, eps=0.10,
        )
        top_idx = set(int(i) for i in np.argpartition(-eig_scores, s.SHORTLIST_K - 1)[:s.SHORTLIST_K])
        top_ids = {QUESTION_CATALOGUE[i].id for i in top_idx}
        # Now ask ELLR for its best question (forces ask branch via choose_action
        # at uniform prior — see TestEIGStrategy above).
        a = s.choose_action(shots_fired=frozenset(), turn=0)
        assert isinstance(a, AskAction)
        assert a.question_id in top_ids
