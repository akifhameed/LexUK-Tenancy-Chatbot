"""
Base class for all LexUK agents.

Efficient approach: each subclass declares a
display name and an ANSI colour, and inherits a logger that tags every
message with the agent's name. The combination produces a readable
trace when several agents collaborate.
"""

from __future__ import annotations

import logging


class Agent:
    """Parent class for every agent in the system."""

    # ANSI colour palette - used by subclasses.
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    GREY = "\033[90m"
    RESET = "\033[0m"

    # Subclasses override these two.
    name: str = "Agent"
    colour: str = RESET

    def __init__(self) -> None:
        # One logger per agent class, e.g. "agent.PlannerAgent".
        self.log = logging.getLogger(f"agent.{type(self).__name__}")

    def announce(self, message: str) -> None:
        """Emit an INFO log line tagged with the agent's name and colour."""
        self.log.info(f"{self.colour}[{self.name}] {message}{self.RESET}")