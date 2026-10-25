"""Guild-scoped application command registration and callbacks."""

import logging
import asyncio
from collections.abc import Sequence

import discord
from discord import app_commands

from rallybot.ladder.models import PlayerRanking
from rallybot.ladder.service import get_ladder
from rallybot.ladder.registration import RegistrationStatus, register_player
from rallybot.discord_bot.challenge_commands import accept, cancel_challenge, challenge, decline


async def ping(interaction: discord.Interaction) -> None:
    """Respond to a connectivity probe without inspecting its payload."""
    logging.getLogger(__name__).info("Discord ping invoked")
    await interaction.response.send_message("Pong!")


def build_ladder_embed(rankings: Sequence[PlayerRanking]) -> discord.Embed:
    """Present a bounded, mobile-friendly view of the persisted rankings."""
    embed = discord.Embed(
        title="Tennis Ladder",
        description="Rankings by points, highest first.",
        color=discord.Color.green(),
    )
    for player in rankings[:25]:
        name = f"{player.rank}. {player.player_name}"[:256]
        value = f"{player.points:,} points"
        if len(embed) + len(name) + len(value) > 5800:
            break
        embed.add_field(
            name=name,
            value=value,
            inline=False,
        )
    if len(embed.fields) < len(rankings):
        embed.set_footer(text=f"Showing {len(embed.fields)} of {len(rankings)} players.")
    return embed


async def ladder(interaction: discord.Interaction) -> None:
    """Acknowledge promptly, retrieve rankings off-thread, and respond safely."""
    logger = logging.getLogger(__name__)
    logger.info("Discord ladder invoked")
    try:
        await interaction.response.defer(thinking=True)
        rankings = await asyncio.to_thread(get_ladder)
        if not rankings:
            await interaction.edit_original_response(content="The ladder is empty. No players have been added yet.")
            return
        embed = build_ladder_embed(rankings)
        await interaction.edit_original_response(embed=embed)
    except Exception:
        logger.error("Discord ladder command failed")
        message = "Unable to display the ladder right now. Please try again later."
        try:
            if interaction.response.is_done():
                await interaction.edit_original_response(content=message, embed=None)
            else:
                await interaction.response.send_message(message, ephemeral=True)
        except Exception:
            logger.error("Discord ladder error response failed")


@app_commands.describe(name="Your tennis player display name (1–50 characters).")
async def register(interaction: discord.Interaction, name: str) -> None:
    """Register the invoking Discord identity, keeping every response private."""
    logger = logging.getLogger(__name__)
    logger.info("Discord registration invoked")
    try:
        await interaction.response.defer(thinking=True, ephemeral=True)
        result = await asyncio.to_thread(register_player, interaction.user.id, name)
        if result.status == RegistrationStatus.INVALID_NAME:
            message = "Please provide a valid name (1–50 characters)."
        else:
            if result.player is None:
                raise RuntimeError("Missing registration result")
            if result.status == RegistrationStatus.ALREADY_REGISTERED:
                message = f"You are already registered as {result.player.display_name}."
            else:
                message = (
                    f"Registration successful! Welcome to RallyBot, {result.player.display_name}. "
                    "Starting points: 1,000."
                )
        await interaction.edit_original_response(content=message, allowed_mentions=discord.AllowedMentions.none())
    except Exception:
        logger.error("Discord registration failed")
        message = "Registration is temporarily unavailable. Please try again later."
        try:
            if interaction.response.is_done():
                await interaction.edit_original_response(content=message, allowed_mentions=discord.AllowedMentions.none())
            else:
                await interaction.response.send_message(
                    message, ephemeral=True, allowed_mentions=discord.AllowedMentions.none(),
                )
        except Exception:
            logger.error("Discord registration error response failed")


class RallyBotCommandTree(app_commands.CommandTree):
    """Keep application-command errors out of raw library traceback logs."""

    async def on_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError
    ) -> None:
        logging.getLogger(__name__).error("Discord slash command failed")


def register_commands(tree: app_commands.CommandTree, guild: discord.Object) -> None:
    """Register the supported commands once in the supplied development guild."""
    for name, description, callback in (
        ("ping", "Check RallyBot connectivity.", ping),
        ("ladder", "View the tennis ladder rankings.", ladder),
        ("register", "Register as a tennis player.", register),
        ("challenge", "Challenge a registered tennis player.", challenge),
        ("accept", "Accept a pending challenge addressed to you.", accept),
        ("decline", "Decline a pending challenge addressed to you.", decline),
        ("cancel-challenge", "Cancel a pending challenge you created.", cancel_challenge),
    ):
        if tree.get_command(name, guild=guild) is None:
            tree.add_command(
                app_commands.Command(name=name, description=description, callback=callback),
                guild=guild,
            )
            logging.getLogger(__name__).info("Discord %s command registered", name)
