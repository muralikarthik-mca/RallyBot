"""Secure Discord lifecycle with development-guild slash commands."""

import asyncio
import logging

import aiohttp
import discord
from pydantic import SecretStr

from rallybot.config import Settings
from rallybot.discord_bot.commands import RallyBotCommandTree, register_commands


class DiscordBotError(RuntimeError):
    """A Discord startup failure with a safe, user-facing message."""


class RallyBotClient(discord.Client):
    """A guild-aware client with all privileged and message intents disabled."""

    def __init__(self, guild_id: int) -> None:
        if guild_id <= 0:
            raise DiscordBotError("DISCORD_GUILD_ID must be a positive development server ID.")
        intents = discord.Intents.none()
        intents.guilds = True
        super().__init__(intents=intents)
        self.development_guild = discord.Object(id=guild_id)
        self.tree = RallyBotCommandTree(self, fallback_to_global=False)
        self._commands_synced = False

    async def setup_hook(self) -> None:
        """Register and sync once after login, before opening the gateway."""
        if self._commands_synced:
            return
        register_commands(self.tree, self.development_guild)
        await self.tree.sync(guild=self.development_guild)
        self._commands_synced = True
        logging.getLogger(__name__).info("Discord development guild commands synchronized")

    async def on_ready(self) -> None:
        """Report readiness using only the numeric bot identifier."""
        logger = logging.getLogger(__name__)
        if self.user is not None:
            logger.info("Discord bot ready (bot_id=%d)", self.user.id)
        else:
            logger.info("Discord bot ready")

    async def on_error(self, event_method: str, *args: object, **kwargs: object) -> None:
        """Replace discord.py's raw event exception logging with a safe message."""
        logging.getLogger(__name__).error("Discord event handler failed")


async def _connect(token: SecretStr, guild_id: int) -> None:
    # The context manager closes the HTTP session and gateway on cancellation,
    # normal completion, and startup failure. No application retry loop is used.
    async with RallyBotClient(guild_id) as client:
        await client.start(token.get_secret_value(), reconnect=True)


def start_discord_bot(settings: Settings) -> None:
    """Block until shutdown; expose only sanitized startup failure messages."""
    token = settings.discord_bot_token
    if token is None or not token.get_secret_value().strip():
        raise DiscordBotError("Discord bot token is missing. Set DISCORD_BOT_TOKEN in .env.")
    if settings.discord_guild_id is None:
        raise DiscordBotError("Development guild ID is missing. Set DISCORD_GUILD_ID in .env.")
    try:
        # Avoid Client.run(), which installs discord.py's own logging handler.
        asyncio.run(_connect(token, settings.discord_guild_id))
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Discord bot stopped")
    except discord.LoginFailure:
        raise DiscordBotError("Discord authentication failed. Check or reset the bot token.") from None
    except (discord.DiscordException, aiohttp.ClientError, OSError, TimeoutError):
        raise DiscordBotError("Discord connection failed. Check network access and bot configuration.") from None
    except Exception:
        raise DiscordBotError("Discord startup failed. Check the bot configuration.") from None
