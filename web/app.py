"""FastAPI bridge — wraps simulator.GameSession behind a session-keyed HTTP API.

Endpoints
---------
    POST   /session                    start a new game (body selects strategy,
                                       truth_seed, eps, t_max)
    GET    /session/{sid}/state        board, hits/misses, posterior, terminal
    POST   /session/{sid}/step         advance one turn (for AI strategies);
                                       returns the turn record
    POST   /session/{sid}/action       submit a user action (shot or ask);
                                       advances the turn for UserStrategy
    GET    /session/{sid}/trajectory   full JSON-ready trajectory (schema:
                                       simulator.runner module docstring)
    DELETE /session/{sid}              drop the session
    GET    /strategies                 names the runner knows about

Static
------
    /ui                                serves mockups/admiralty_dashboard.html
    /ui/{path}                         other mockup assets
    /results/{path}                    results/ JSON (for ?traj=/results/... loads)

Sessions live in a process-local dict keyed by UUID — fine for single-user
dev and for the benchmark review viewer; a persistent store would be needed
for multi-process deploys but isn't in scope.

Spin up::

    .venv/bin/uvicorn web.app:app --host 127.0.0.1 --port 8000 --reload

Then open http://127.0.0.1:8000/ui/ for the dashboard (auto-redirects to the
mockup HTML).
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from engine.board import BOARD_SIZE
from engine.questions import question_by_id
from engine.smc import sample_configuration
from simulator.runner import GameSession, _STRATEGY_REGISTRY, run_game
from strategies.base import AskAction, ShotAction
from strategies.user import UserStrategy


_REPO_ROOT = Path(__file__).resolve().parent.parent
_MOCKUPS = _REPO_ROOT / "mockups"
_RESULTS = _REPO_ROOT / "results"


app = FastAPI(title="Battleship — strategy-comparison bridge", version="0.1")


# --------------------------------------------------------------------------
# In-memory session store
# --------------------------------------------------------------------------

_SESSIONS: dict[str, GameSession] = {}


def _get_session(sid: str) -> GameSession:
    try:
        return _SESSIONS[sid]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"session {sid!r} not found")


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------

class CreateSessionBody(BaseModel):
    strategy: str = Field(default="eig_approx")
    seed: int = 0
    t_max: int = 80
    eps: float = 0.10
    truth_seed: int | None = Field(default=None,
        description="if given, draw truth with this seed; otherwise uses `seed`")

    @field_validator("strategy")
    @classmethod
    def _known_strategy(cls, v: str) -> str:
        if v not in _STRATEGY_REGISTRY:
            known = ", ".join(sorted(_STRATEGY_REGISTRY))
            raise ValueError(f"unknown strategy {v!r}; known: {known}")
        return v


class SimulateBody(BaseModel):
    """One-shot simulation: run a named strategy on a fleet drawn from
    ``truth_seed`` and return the full trajectory dict.  Used by the UI
    to populate peer-strategy traces against the same fleet the user
    just played, so the hits-over-turns chart is apples-to-apples."""
    strategy: str = Field(default="eig_approx")
    seed: int = 0
    t_max: int = 80
    eps: float = 0.10
    truth_seed: int = 0

    @field_validator("strategy")
    @classmethod
    def _known_strategy(cls, v: str) -> str:
        if v not in _STRATEGY_REGISTRY:
            known = ", ".join(sorted(_STRATEGY_REGISTRY))
            raise ValueError(f"unknown strategy {v!r}; known: {known}")
        if v == "user":
            raise ValueError("user strategy requires interaction; use /session instead")
        return v


class ShotActionBody(BaseModel):
    kind: Literal["shot"]
    cell: tuple[int, int]

    @field_validator("cell")
    @classmethod
    def _in_bounds(cls, v: tuple[int, int]) -> tuple[int, int]:
        r, c = v
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            raise ValueError(f"cell {v} out of bounds")
        return v


class AskActionBody(BaseModel):
    kind: Literal["ask"]
    question_id: str

    @field_validator("question_id")
    @classmethod
    def _known_qid(cls, v: str) -> str:
        try:
            question_by_id(v)
        except KeyError:
            raise ValueError(f"unknown question_id {v!r}")
        return v


ActionBody = ShotActionBody | AskActionBody


# --------------------------------------------------------------------------
# State snapshots
# --------------------------------------------------------------------------

def _session_state(s: GameSession) -> dict[str, Any]:
    """Summary snapshot — enough to render the board + posterior live.

    Returns: strategy, seed, eps, t_max, turn, terminated, hits, n_shots,
    n_asks, shots_fired (as [[r,c], ...]), last_turn (or None), truth (full —
    used by the mockup to compute hits/misses/ship overlay), and the live
    cell-marginal grid.
    """
    n_shots = sum(1 for t in s.turns if t["action"]["kind"] == "shot")
    n_asks = s.state.turn - n_shots
    return {
        "session_id": None,  # caller fills in
        "strategy": s.strategy_name,
        "seed": s.seed,
        "eps": s.eps,
        "t_max": s.t_max,
        "turn": s.state.turn,
        "terminated": s.terminated,
        "hits": s.state.hits,
        "n_shots": n_shots,
        "n_asks": n_asks,
        "shots_fired": [list(c) for c in sorted(s.state.shots_fired)],
        "turns": list(s.turns),
        "last_turn": s.turns[-1] if s.turns else None,
        "posterior": s.strategy.filter.cell_marginal_grid().tolist(),
        "truth": {
            "ships": [
                {"length": sh.length, "row": sh.row, "col": sh.col,
                 "orientation": sh.orientation.value}
                for sh in s.truth.ships
            ],
        },
        "cum_net_reward": s.cum_net_reward,
    }


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/strategies")
def list_strategies() -> dict[str, Any]:
    return {"strategies": sorted(_STRATEGY_REGISTRY.keys())}


@app.get("/questions")
def list_questions() -> dict[str, Any]:
    """The binary-question catalogue — stable order matches engine.questions.
    UI uses this to populate the ask picker."""
    from engine.questions import QUESTION_CATALOGUE
    return {
        "questions": [
            {"id": q.id, "kind": q.kind, "label": _question_label(q)}
            for q in QUESTION_CATALOGUE
        ],
    }


def _question_label(q) -> str:
    """Human-readable label, mirroring the JS ``qidToLabel`` in the mockup."""
    rows_letters = "ABCDEFGH"
    if q.kind == "cell":
        return f"Cell {rows_letters[q.row]}{q.col + 1}"
    if q.kind == "row":
        return f"Row {rows_letters[q.row]}"
    if q.kind == "col":
        return f"Col {q.col + 1}"
    if q.kind == "quadrant":
        return f"Quad {q.quadrant}"
    if q.kind == "hparity":
        return f"H-par L={q.length}"
    return q.id


@app.post("/session")
def create_session(body: CreateSessionBody) -> dict[str, Any]:
    truth_rng = np.random.default_rng(
        body.truth_seed if body.truth_seed is not None else body.seed,
    )
    truth = sample_configuration(truth_rng)
    s = GameSession(
        strategy_name=body.strategy, truth=truth,
        t_max=body.t_max, eps=body.eps, seed=body.seed,
    )
    sid = uuid.uuid4().hex
    _SESSIONS[sid] = s
    state = _session_state(s)
    state["session_id"] = sid
    return state


@app.get("/session/{sid}/state")
def get_state(sid: str) -> dict[str, Any]:
    s = _get_session(sid)
    state = _session_state(s)
    state["session_id"] = sid
    return state


@app.post("/session/{sid}/step")
def step_session(sid: str) -> dict[str, Any]:
    """Advance one turn for an AI strategy.  Raises 409 if UserStrategy has
    no pending action, 410 if the game has already terminated."""
    s = _get_session(sid)
    if s.terminated:
        raise HTTPException(status_code=410, detail="session terminated")
    if isinstance(s.strategy, UserStrategy) and s.strategy._pending is None:
        raise HTTPException(
            status_code=409,
            detail="UserStrategy has no pending action; "
                   "POST /session/{sid}/action first",
        )
    record = s.step()
    state = _session_state(s)
    state["session_id"] = sid
    state["step_record"] = record
    return state


@app.post("/session/{sid}/action")
def submit_action(sid: str, body: ActionBody) -> dict[str, Any]:
    """Queue a user action and advance one turn.  Only valid for UserStrategy."""
    s = _get_session(sid)
    if not isinstance(s.strategy, UserStrategy):
        raise HTTPException(
            status_code=409,
            detail=f"session strategy is {s.strategy_name!r}, not 'user'; "
                   "use POST /session/{sid}/step instead",
        )
    if s.terminated:
        raise HTTPException(status_code=410, detail="session terminated")
    if isinstance(body, ShotActionBody):
        if tuple(body.cell) in s.state.shots_fired:
            raise HTTPException(
                status_code=409,
                detail=f"cell {list(body.cell)} already shot",
            )
        s.strategy.set_next_action(ShotAction(cell=tuple(body.cell)))
    else:
        s.strategy.set_next_action(AskAction(question_id=body.question_id))
    record = s.step()
    state = _session_state(s)
    state["session_id"] = sid
    state["step_record"] = record
    return state


@app.get("/session/{sid}/trajectory")
def get_trajectory(sid: str) -> dict[str, Any]:
    return _get_session(sid).trajectory()


@app.post("/simulate")
def simulate(body: SimulateBody) -> dict[str, Any]:
    """Run a full game to completion (no session state retained) and return
    the trajectory dict.  The UI fires this in parallel for peer strategies
    after the user's live game ends, using the user's seed as ``truth_seed``
    so all strategies face the same fleet."""
    truth_rng = np.random.default_rng(body.truth_seed)
    truth = sample_configuration(truth_rng)
    return run_game(
        strategy_name=body.strategy, truth=truth,
        t_max=body.t_max, eps=body.eps, seed=body.seed,
    )


@app.delete("/session/{sid}")
def delete_session(sid: str) -> dict[str, str]:
    _SESSIONS.pop(sid, None)
    return {"session_id": sid, "status": "deleted"}


# --------------------------------------------------------------------------
# Static — mockup UI and results/ JSON
# --------------------------------------------------------------------------

@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@app.get("/ui/")
def ui_index() -> FileResponse:
    return FileResponse(_MOCKUPS / "admiralty_dashboard.html")


app.mount("/ui", StaticFiles(directory=_MOCKUPS, html=True), name="mockups")

if _RESULTS.exists():
    app.mount("/results", StaticFiles(directory=_RESULTS), name="results")
