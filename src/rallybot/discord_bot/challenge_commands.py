"""Thin Discord presentation for deterministic challenge operations."""

import asyncio
import logging

import discord
from discord import app_commands

from rallybot.challenges.models import Challenge, ChallengeStatus
from rallybot.challenges.service import ChallengeService, ChallengeValidationError


def _execute(actor_id: int, argument: int, status: ChallengeStatus | None) -> Challenge:
    service = ChallengeService()
    if status is None:
        return service.create_challenge(actor_id, argument)
    return service.transition_challenge(actor_id, argument, status)


async def _private_response(interaction: discord.Interaction, message: str) -> None:
    try:
        if interaction.response.is_done():
            await interaction.edit_original_response(content=message, allowed_mentions=discord.AllowedMentions.none())
        else:
            await interaction.response.send_message(
                message, ephemeral=True, allowed_mentions=discord.AllowedMentions.none(),
            )
    except Exception:
        logging.getLogger(__name__).error("Challenge private response delivery failed")


async def _handle(
    interaction: discord.Interaction, argument: int, status: ChallengeStatus | None
) -> None:
    logger = logging.getLogger(__name__)
    try:
        # Keep validation and failure messages private even when SQLite waits.
        await interaction.response.defer(thinking=True, ephemeral=True)
        result = await asyncio.to_thread(_execute, interaction.user.id, argument, status)
    except ChallengeValidationError as error:
        await _private_response(interaction, str(error))
        return
    except Exception:
        logger.error("Challenge operation failed")
        await _private_response(interaction, "Challenges are temporarily unavailable. Please try again later.")
        return

    verb = "created" if status is None else status.value.lower()
    message = f"Challenge #{result.id} {verb}. Status: {result.status.value}."
    try:
        # Complete the ephemeral defer before sending a genuinely new public
        # followup; the deferred response's visibility cannot be changed.
        await interaction.edit_original_response(
            content="Challenge action saved. Posting public confirmation.",
            allowed_mentions=discord.AllowedMentions.none(),
        )
        await interaction.followup.send(message, ephemeral=False, allowed_mentions=discord.AllowedMentions.none())
    except Exception:
        logger.error("Challenge public confirmation delivery failed")
        await _private_response(
            interaction, f"Challenge #{result.id} was saved as {result.status.value}, "
            "but its public confirmation could not be delivered. Do not repeat the action.",
        )


@app_commands.describe(opponent="The registered Discord member you want to challenge.")
async def challenge(interaction: discord.Interaction, opponent: discord.Member) -> None:
    await _handle(interaction, opponent.id, None)


@app_commands.describe(challenge_id="The challenge ID shown in its confirmation.")
async def accept(interaction: discord.Interaction, challenge_id: int) -> None:
    await _handle(interaction, challenge_id, ChallengeStatus.ACCEPTED)


@app_commands.describe(challenge_id="The challenge ID shown in its confirmation.")
async def decline(interaction: discord.Interaction, challenge_id: int) -> None:
    await _handle(interaction, challenge_id, ChallengeStatus.DECLINED)


@app_commands.describe(challenge_id="The challenge ID shown in its confirmation.")
async def cancel_challenge(interaction: discord.Interaction, challenge_id: int) -> None:
    await _handle(interaction, challenge_id, ChallengeStatus.CANCELLED)
