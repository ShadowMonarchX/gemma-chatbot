from __future__ import annotations

import datetime as dt
import re
import sys

from .errors import InjectionError, ValidationError


class MessageValidator:
    """Validates message payloads, sanitizes text, and blocks injection attempts."""

    def __init__(self) -> None:
        """Initialize validation constraints and patterns."""
        self._max_message_chars: int = 4096
        self._max_history_messages: int = 20
        self._direction_overrides_pattern = re.compile(r"[\u202A-\u202E\u2066-\u2069]")
        self._control_pattern = re.compile(r"[\x00-\x1F\x7F]")
        self._injection_patterns: list[str] = [
            "ignore previous instructions",
            "disregard your system prompt",
            "you are now",
            "act as",
            "jailbreak",
            "dan mode",
            "roleplay as",
            "pretend you are",
            "system:",
            "###system",
        ]

    def validate_messages(self, messages: list[dict]) -> list[dict]:
        """Validate and sanitize message history.

        Args:
            messages: Raw message dictionaries.

        Returns:
            list[dict]: Sanitized messages.
        """
        if not 1 <= len(messages) <= self._max_history_messages:
            raise ValidationError(
                message="Invalid message count",
                status_code=422,
                log_detail=f"message_count={len(messages)}",
            )

        validated: list[dict] = []
        for message in messages:
            role = str(message.get("role", "")).strip()
            raw_content = str(message.get("content", ""))
            content = self.sanitize_input(raw_content)

            if role not in {"user", "assistant"}:
                raise ValidationError(
                    message="Invalid message role",
                    status_code=422,
                    log_detail=f"invalid_role={role}",
                )

            if not content or len(content) > self._max_message_chars:
                raise ValidationError(
                    message="Invalid message length",
                    status_code=422,
                    log_detail=f"message_length={len(content)}",
                )

            if self.check_injection(content):
                self._log_injection_attempt(content)
                raise InjectionError(
                    message="Input rejected: disallowed pattern",
                    status_code=400,
                    log_detail="prompt_injection_pattern",
                )

            validated.append({"role": role, "content": content})

        return validated

    def sanitize_input(self, text: str) -> str:
        """Remove null bytes, ASCII control chars, and directional override marks.

        Args:
            text: Raw text.

        Returns:
            str: Sanitized text.
        """
        without_directional = self._direction_overrides_pattern.sub("", text)
        without_controls = self._control_pattern.sub("", without_directional)
        return without_controls.strip()

    def check_injection(self, text: str) -> bool:
        """Return whether text contains a known injection phrase.

        Args:
            text: Sanitized text.

        Returns:
            bool: True if a disallowed phrase is present.
        """
        lowered = text.lower()
        return any(pattern in lowered for pattern in self._injection_patterns)

    def _log_injection_attempt(self, text: str) -> None:
        """Write a sanitized injection log line to stderr.

        Args:
            text: Sanitized text.

        Returns:
            None.
        """
        timestamp = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        preview = text[:240]
        print(
            f"[{timestamp}] injection_attempt_rejected preview={preview!r}",
            file=sys.stderr,
        )


message_validator = MessageValidator()
