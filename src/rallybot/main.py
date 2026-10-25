"""Console entry point for RallyBot."""

import argparse
import json
import logging
import sqlite3
import sys
from collections.abc import Sequence

from rallybot.config import get_settings
from rallybot.database.connection import (
    DatabaseError,
    InvalidDatabaseURL,
    check_database_health,
    initialize_database,
)
from rallybot.logging_config import configure_logging
from rallybot.llm.ollama_client import check_ollama_health, valid_model_name
from rallybot.health import check_health
from rallybot.ladder.service import SeedNotAllowedError, seed_development_ladder


def main(argv: Sequence[str] | None = None) -> int:
    """Run startup or an explicit infrastructure health command."""
    parser = argparse.ArgumentParser(prog="rallybot")
    parser.add_argument("command", nargs="?", choices=(
        "db-init", "db-check", "ollama-check", "health", "discord-start", "ladder-seed", "matchmaking-demo", "matchmaking-ai-demo", "intent-demo", "supervisor-demo", "challenge-approval-demo", "tool-demo",
    ))
    parser.add_argument("request", nargs="?", help="Quoted request for intent-demo, supervisor-demo or tool-demo.")
    parser.add_argument("--no-identity-context", action="store_true",
                        help="Simulate missing authenticated caller context for intent-demo only.")
    parser.add_argument("--player-id", type=int, help="Explicit internal requester ID for supervisor-demo or tool-demo.")
    parser.add_argument("--requester-id", type=int, help="Development-only approval initiator player ID.")
    parser.add_argument("--opponent-id", type=int, help="Development-only approval opponent player ID.")
    parser.add_argument("--workflow-id", help="Existing development approval workflow to resume.")
    parser.add_argument("--decision", choices=("approve", "reject"), help="Explicit development approval decision.")
    parser.add_argument("--checkpoint-file", help="Separate SQLite approval checkpoint file.")
    args = parser.parse_args(argv)
    if args.command in ("intent-demo", "supervisor-demo", "tool-demo") and args.request is None:
        parser.error("this demo requires a quoted natural-language request")
    if args.command not in ("intent-demo", "supervisor-demo", "tool-demo") and args.request is not None:
        parser.error("a request argument is only supported by intent-demo, supervisor-demo or tool-demo")
    if args.command != "intent-demo" and args.no_identity_context:
        parser.error("--no-identity-context is only supported by intent-demo")
    if args.command in ("supervisor-demo", "tool-demo") and args.player_id is None:
        parser.error("this demo requires an explicit --player-id")
    if args.command not in ("supervisor-demo", "tool-demo") and args.player_id is not None:
        parser.error("--player-id is only supported by supervisor-demo or tool-demo")
    approval_options = (args.requester_id, args.opponent_id, args.workflow_id, args.decision, args.checkpoint_file)
    if args.command != "challenge-approval-demo" and any(value is not None for value in approval_options):
        parser.error("approval options are only supported by challenge-approval-demo")
    if args.command == "challenge-approval-demo":
        if args.workflow_id is None:
            if args.requester_id is None or args.opponent_id is None or args.decision is not None:
                parser.error("start requires --requester-id and --opponent-id; resume requires --workflow-id and --decision")
        elif args.decision is None or args.requester_id is not None or args.opponent_id is not None:
            parser.error("resume requires --workflow-id and --decision, without player identity arguments")
    if args.command == "health":
        health = check_health(logging_configurer=configure_logging)
        print(f"Application: {health.app_name}")
        print(f"Overall health: {health.overall_status}")
        for label, component in (
            ("Configuration", health.configuration), ("Database", health.database),
            ("Ollama server", health.ollama_server), ("Configured model", health.ollama_model),
            ("Logging", health.logging),
        ):
            print(f"{label}: {component.status} - {component.message}")
        if health.logging.status == "PASS":
            logger = logging.getLogger(__name__)
            logger.log(logging.INFO if health.exit_code == 0 else logging.WARNING,
                       "Application health check completed: %s", health.overall_status)
        return health.exit_code

    settings = get_settings()
    configure_logging(settings.log_level)
    if args.command is None:
        print("RallyBot initialized successfully!")
        return 0

    logger = logging.getLogger(__name__)
    if args.command == "tool-demo":
        from rallybot.workflows.tool_calling_graph import run_tool_calling

        result = run_tool_calling(args.request, args.player_id, settings=settings)
        print(f"Model selection mode: {result['selection_mode'] or 'UNAVAILABLE'}")
        print("Tools invoked: " + (", ".join(result["invoked_tools"]) or "None"))
        print(f"Status: {result['status']}")
        if result["error_code"]:
            print(f"Error code: {result['error_code']}")
        print("Final response: " + result["final_response"])
        return {"COMPLETED": 0, "LIMIT_REACHED": 2}.get(result["status"], 1)

    if args.command == "challenge-approval-demo":
        from pathlib import Path
        from rallybot.workflows.challenge_approval_graph import (
            ApprovalActor, ApprovalError, ChallengeApprovalWorkflow, local_principal,
        )

        print("Development-only approval demo: local OS identity; no Discord authentication.")
        if settings.app_env != "development":
            print("Challenge approval demo is only available in development mode.", file=sys.stderr)
            return 1
        try:
            workflow = ChallengeApprovalWorkflow(settings.database_url,
                Path(args.checkpoint_file or "./data/challenge_approvals.sqlite3"))
            if args.workflow_id is None:
                workflow_id, result = workflow.start(ApprovalActor(args.requester_id, local_principal()), args.opponent_id)
            else:
                workflow_id = args.workflow_id
                result = workflow.resume_local(workflow_id, args.decision)
            print(f"Workflow ID: {workflow_id}")
            print(f"Status: {result['workflow_status']}")
            print(f"Approval: {result['approval_status']}")
            if result.get("__interrupt__"):
                print("Approval request: " + json.dumps(result["__interrupt__"][0].value, ensure_ascii=True))
            if result["challenge_id"] is not None:
                print(f"Challenge ID: {result['challenge_id']}")
            if result["error_code"]:
                print(f"Error code: {result['error_code']}")
            return 1 if result["workflow_status"] == "ERROR" else 0
        except ApprovalError as error:
            # ApprovalError contains only fixed categories controlled by this module.
            logger.warning("Challenge approval demo failed: %s", str(error))
            print(f"Challenge approval unavailable: {error}.", file=sys.stderr)
            return 1
        except Exception:
            logger.warning("Challenge approval demo failed: APPROVAL_UNAVAILABLE")
            print("Challenge approval is temporarily unavailable. Initialize the database and check local storage.", file=sys.stderr)
            return 1

    if args.command == "supervisor-demo":
        from rallybot.workflows.supervisor_graph import create_supervisor_dependencies, run_supervisor

        try:
            dependencies = create_supervisor_dependencies(settings)
            dependencies.get_player(args.player_id)
        except Exception:
            logger.warning("Supervisor demo requester validation failed: INVALID_REQUESTER")
            print("Supervisor demo requires an existing registered player ID and an initialized database.", file=sys.stderr)
            return 1
        result = run_supervisor(args.request, args.player_id, dependencies=dependencies)
        intent = result["classified_intent"]
        print(f"Intent: {intent.intent.value if intent is not None else 'UNKNOWN'}")
        print(f"Route: {result['route'] or 'None'}")
        print(f"Status: {result['status']}")
        print("Result: " + json.dumps(result["result"], ensure_ascii=True))
        if result["error_code"]:
            print(f"Error code: {result['error_code']}")
        return {"COMPLETED": 0, "CLARIFICATION_NEEDED": 2, "NOT_IMPLEMENTED": 2,
                "UNSUPPORTED": 2}.get(result["status"], 1)

    if args.command == "intent-demo":
        from rallybot.intent.classifier import IntentClassifier
        from rallybot.intent.models import IntentType

        result = IntentClassifier(settings).classify(args.request,
            authenticated_identity_expected=not args.no_identity_context)
        print(f"Intent: {result.intent.value}")
        print(f"Confidence: {result.confidence:.2f}")
        for label, value in (("Player name", result.player_name), ("Opponent name", result.opponent_name),
                             ("Requested date", result.requested_date), ("Requested time", result.requested_time)):
            print(f"{label}: {value if value is not None else 'None'}")
        print(f"Clarification needed: {'yes' if result.clarification_needed else 'no'}")
        return 1 if result.intent == IntentType.UNKNOWN else 0

    if args.command in ("matchmaking-demo", "matchmaking-ai-demo"):
        from rallybot.workflows.matchmaking_graph import run_matchmaking
        from rallybot.ladder.repository import PlayerRepository
        from rallybot.tools.player_tools import create_player_tools
        from rallybot.llm.matchmaking_explainer import MatchmakingExplainer

        try:
            repository = PlayerRepository(settings.database_url)
            players = repository.list_players()
            if not players:
                print("No players found. Run db-init, then /register or ladder-seed in development mode.")
                return 1
            explainer = MatchmakingExplainer(settings) if args.command == "matchmaking-ai-demo" else None
            result = run_matchmaking(players[0].id, tools=create_player_tools(repository), explainer=explainer)
        except Exception:
            logger.error("Matchmaking demo failed")
            print("Matchmaking demo is temporarily unavailable. Run db-init and check DATABASE_URL.", file=sys.stderr)
            return 1
        print(f"Requesting player: {result['player_name']} [ID {result['player_id']}] ({result['player_points']} points)")
        print("Eligible opponents:")
        for opponent in result["available_players"]:
            print(f"- {opponent['name']} [ID {opponent['id']}] ({opponent['points']} points)")
        if not result["available_players"]:
            print("- None")
        recommendation = result['recommended_opponent']
        print(f"Recommended opponent: {recommendation} [ID {result['recommended_opponent_id']}]" if recommendation else "Recommended opponent: None")
        print(f"Workflow status: {result['status']}")
        if args.command == "matchmaking-ai-demo":
            print(f"Explanation: {result.get('explanation') or 'No recommendation to explain.'}")
            print(f"Explanation source: {result.get('explanation_source') or 'NONE'}")
        return 0

    if args.command == "discord-start":
        from rallybot.discord_bot.bot import DiscordBotError, start_discord_bot

        try:
            print("Starting Discord bot. Press Ctrl+C to stop.", flush=True)
            start_discord_bot(settings)
        except DiscordBotError as error:
            logger.error("Discord bot startup failed")
            print(str(error), file=sys.stderr)
            return 1
        return 0

    if args.command == "ollama-check":
        result = check_ollama_health(settings)
        model = result.configured_model if valid_model_name(result.configured_model) else "[invalid model name]"
        print("Ollama server reachable: " + ("yes" if result.server_reachable else "no"))
        print(f"Configured model: {model}")
        print("Configured model available: " + ("yes" if result.model_available else "no"))
        print("Installed models: " + (", ".join(result.available_models) or "none reported"))
        print(result.message)
        if result.server_reachable and result.model_available:
            logger.info("Ollama connectivity check passed")
            return 0
        logger.error("Ollama check failed: %s", result.error_code)
        return 1

    try:
        if args.command == "db-init":
            initialize_database(settings.database_url)
            print("Database initialized successfully.")
            logger.info("Database initialization completed")
        elif args.command == "ladder-seed":
            result = seed_development_ladder(settings)
            print(f"Development ladder seed complete: {result.added} added, {result.skipped} skipped.")
            logger.info("Development ladder seed completed")
        else:
            if not check_database_health(settings.database_url):
                raise DatabaseError("Database health check failed.")
            print("Database health check passed.")
            logger.info("Database health check passed")
    except SeedNotAllowedError:
        logger.error("Ladder seeding rejected outside development mode")
        print("Ladder seeding is only allowed in development mode.", file=sys.stderr)
        return 1
    except (DatabaseError, InvalidDatabaseURL, sqlite3.Error, OSError):
        # Never print raw errors or configured URLs: they may contain secrets.
        logger.error("Database command failed")
        print("Database command failed. Check DATABASE_URL and database access.", file=sys.stderr)
        return 1
    return 0
