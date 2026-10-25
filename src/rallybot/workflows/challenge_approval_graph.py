"""Separate, human-approved challenge workflow with durable replay protection.

Only the runner is an application entry point. A compiled graph alone does not
authenticate a caller; actor contexts must be supplied by trusted application code.
"""

import csv
import hashlib
import io
import logging
import os
import sqlite3
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast
from uuid import UUID, uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from langsmith import tracing_context

from rallybot.challenges.repository import ChallengeRepository
from rallybot.challenges.service import ChallengeService, ChallengeValidationError
from rallybot.database.connection import database_path
from rallybot.ladder.models import Player
from rallybot.ladder.repository import PlayerRepository
from rallybot.ladder.service import PlayerResolutionError, resolve_player

logger = logging.getLogger(__name__)


class ChallengeApprovalState(TypedDict):
    requester_player_id: int
    opponent_player_id: int
    opponent_display_name: str
    approval_status: str
    challenge_id: int | None
    workflow_status: str
    error_code: str | None


@dataclass(frozen=True)
class ApprovalActor:
    """Identity supplied by a trusted adapter, never by the LLM/resume payload."""

    player_id: int
    principal: str


class ApprovalError(ValueError):
    """Only a fixed category is exposed to the caller."""


def local_principal() -> str:
    """Read the actual OS identity, rather than trusting USERNAME environment data."""
    if os.name == "nt":
        executable = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/whoami.exe"
        result = subprocess.run([str(executable), "/user", "/fo", "csv", "/nh"],
                                capture_output=True, text=True, check=True, timeout=5)
        sid = next(csv.reader(io.StringIO(result.stdout)))[1]
        if not sid.startswith("S-1-"):
            raise ApprovalError("IDENTITY_UNAVAILABLE")
        return sid
    return f"uid:{os.getuid()}"


def _valid_id(value: object) -> bool:
    return type(value) is int and 1 <= value <= 9223372036854775807


def resolve_opponent(players: PlayerRepository, opponent: int | str) -> Player:
    """Exact case-insensitive names must resolve uniquely; IDs remain authoritative."""
    try:
        return resolve_player(players, opponent)
    except PlayerResolutionError as error:
        raise ApprovalError(str(error)) from None


def build_challenge_approval_graph(service: ChallengeService, checkpointer: SqliteSaver) -> CompiledStateGraph:
    """Compile only; the runner below supplies authorization and serialization."""
    def prepare(state: ChallengeApprovalState) -> dict[str, object]:
        logger.info("Challenge approval preparation started")
        try:
            _, opponent = service.prepare_challenge(state["requester_player_id"], state["opponent_player_id"])
            return {"opponent_display_name": opponent.display_name, "workflow_status": "AWAITING_APPROVAL"}
        except Exception:
            logger.warning("Challenge approval failed: ELIGIBILITY_FAILED")
            return {"workflow_status": "ERROR", "error_code": "ELIGIBILITY_FAILED"}

    def await_approval(state: ChallengeApprovalState) -> dict[str, object]:
        # Interrupt is deliberately outside try/except; this node restarts on resume.
        decision = interrupt({
            "requester_player_id": state["requester_player_id"],
            "opponent_player_id": state["opponent_player_id"],
            "opponent_display_name": state["opponent_display_name"],
            "proposed_action": "CREATE_CHALLENGE",
            "question": f"Create a pending challenge to {state['opponent_display_name']}?",
        })
        if type(decision) is not str or decision not in ("approve", "reject"):
            return {"approval_status": "INVALID", "workflow_status": "ERROR", "error_code": "INVALID_DECISION"}
        if decision == "reject":
            logger.info("Challenge approval rejected")
            return {"approval_status": "REJECTED", "workflow_status": "REJECTED"}
        return {"approval_status": "APPROVED", "workflow_status": "APPROVED"}

    def create_challenge(state: ChallengeApprovalState, config: RunnableConfig) -> dict[str, object]:
        try:
            # Revalidates players and atomically enforces pending-pair uniqueness.
            challenge = service.create_approved_challenge(
                state["requester_player_id"], state["opponent_player_id"],
                config["configurable"]["thread_id"],
            )
            return {"challenge_id": challenge.id, "workflow_status": "COMPLETED"}
        except ChallengeValidationError:
            logger.warning("Challenge approval failed: ELIGIBILITY_CHANGED")
            return {"workflow_status": "ERROR", "error_code": "ELIGIBILITY_CHANGED"}
        except Exception:
            logger.warning("Challenge approval failed: CHALLENGE_UNAVAILABLE")
            return {"workflow_status": "ERROR", "error_code": "CHALLENGE_UNAVAILABLE"}

    graph = StateGraph(ChallengeApprovalState)
    graph.add_node("prepare", prepare)
    graph.add_node("await_approval", await_approval)
    graph.add_node("create_challenge", create_challenge)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", lambda state: state["workflow_status"],
                               {"AWAITING_APPROVAL": "await_approval", "ERROR": END})
    graph.add_conditional_edges("await_approval", lambda state: state["approval_status"],
                               {"APPROVED": "create_challenge", "REJECTED": END, "INVALID": END})
    graph.add_edge("create_challenge", END)
    return graph.compile(checkpointer=checkpointer)


class ChallengeApprovalWorkflow:
    """Short-lived connections, per-store process lock, and caller-bound resumes."""

    def __init__(self, database_url: str, checkpoint_file: Path) -> None:
        business = database_path(database_url)
        self.checkpoint_file = checkpoint_file.resolve()
        self.control_file = self.checkpoint_file.with_name(self.checkpoint_file.name + ".control.sqlite3")
        if business is None or any(path == business or
                (path.exists() and business.exists() and path.samefile(business))
                for path in (self.checkpoint_file, self.control_file)):
            raise ApprovalError("SEPARATE_FILE_STORAGE_REQUIRED")
        self.database_key = hashlib.sha256(str(business).encode()).hexdigest()
        self.players = PlayerRepository(database_url)
        self.service = ChallengeService(players=self.players, challenges=ChallengeRepository(database_url))

    @contextmanager
    def _session(self) -> Iterator[tuple[sqlite3.Connection, CompiledStateGraph]]:
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        # A separate control DB lets the saver commit checkpoints while this
        # transaction serializes start/resume operations across threads/processes.
        control = sqlite3.connect(str(self.control_file), timeout=30, isolation_level=None)
        try:
            control.execute("CREATE TABLE IF NOT EXISTS approval_owners "
                            "(workflow_id TEXT PRIMARY KEY, player_id INTEGER NOT NULL, "
                            "principal TEXT NOT NULL, database_key TEXT NOT NULL)")
            control.execute("BEGIN IMMEDIATE")
            connection = sqlite3.connect(str(self.checkpoint_file), timeout=30, check_same_thread=False)
            try:
                saver = SqliteSaver(connection, serde=JsonPlusSerializer(
                    pickle_fallback=False, allowed_json_modules=[], allowed_msgpack_modules=[],
                ))
                with tracing_context(enabled=False):
                    yield control, build_challenge_approval_graph(self.service, saver)
                control.commit()
            finally:
                connection.close()
        except BaseException:
            control.rollback()
            raise
        finally:
            control.close()

    def start(self, actor: ApprovalActor, opponent: int | str) -> tuple[str, dict[str, object]]:
        if not _valid_id(actor.player_id) or not actor.principal:
            raise ApprovalError("UNAUTHORIZED")
        target = resolve_opponent(self.players, opponent)
        if target.id == actor.player_id:
            raise ApprovalError("SELF_CHALLENGE")
        workflow_id = str(uuid4())
        initial: ChallengeApprovalState = {
            "requester_player_id": actor.player_id, "opponent_player_id": target.id,
            "opponent_display_name": target.display_name, "approval_status": "PENDING",
            "challenge_id": None, "workflow_status": "PREPARING", "error_code": None,
        }
        with self._session() as (control, graph):
            control.execute("INSERT INTO approval_owners VALUES (?, ?, ?, ?)",
                            (workflow_id, actor.player_id, actor.principal, self.database_key))
            result = graph.invoke(initial, {"configurable": {"thread_id": workflow_id}})
        logger.info("Challenge approval workflow started")
        return workflow_id, result

    def resume(self, workflow_id: str, decision: object, *, actor: ApprovalActor) -> dict[str, object]:
        return self._resume(workflow_id, decision, actor=actor, principal=None)

    def resume_local(self, workflow_id: str, decision: object) -> dict[str, object]:
        """Local adapter: identity comes from OS, never an --actor-id argument."""
        return self._resume(workflow_id, decision, actor=None, principal=local_principal())

    def _resume(self, workflow_id: str, decision: object, *,
                actor: ApprovalActor | None, principal: str | None) -> dict[str, object]:
        try:
            if str(UUID(workflow_id)) != workflow_id:
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            raise ApprovalError("WORKFLOW_NOT_AVAILABLE") from None
        with self._session() as (control, graph):
            owner = control.execute("SELECT player_id, principal, database_key FROM approval_owners WHERE workflow_id = ?",
                                    (workflow_id,)).fetchone()
            if owner is None:
                raise ApprovalError("WORKFLOW_NOT_AVAILABLE")
            trusted = actor if actor is not None else ApprovalActor(owner[0], principal)
            if (not _valid_id(trusted.player_id) or (trusted.player_id, trusted.principal, self.database_key) != owner):
                logger.warning("Challenge approval denied: UNAUTHORIZED")
                raise ApprovalError("UNAUTHORIZED")
            config = {"configurable": {"thread_id": workflow_id}}
            snapshot = graph.get_state(config)
            if not snapshot.values or snapshot.values.get("requester_player_id") != trusted.player_id:
                raise ApprovalError("WORKFLOW_NOT_AVAILABLE")
            if not snapshot.next:
                # Terminal outcomes are immutable; no second decision is executed.
                return cast(dict[str, object], snapshot.values)
            if snapshot.next not in (("await_approval",), ("create_challenge",)):
                raise ApprovalError("WORKFLOW_NOT_AVAILABLE")
            # If a process died after recording approval, recover its create node.
            safe_decision = decision if type(decision) is str and decision in ("approve", "reject") else "invalid"
            result = graph.invoke(Command(resume=safe_decision) if snapshot.next == ("await_approval",) else None, config)
        logger.info("Challenge approval workflow resumed")
        return result
