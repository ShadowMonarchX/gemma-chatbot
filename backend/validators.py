from __future__ import annotations

import datetime as dt
import re
import sys

from .errors import InjectionError, ValidationError


class MessageValidator:
    """Validates, sanitizes, and guards user messages against prompt injection."""

    def __init__(self) -> None:
        """Initialize sanitizer and prompt injection patterns."""
        self._max_message_chars: int = 4096
        self._max_history_messages: int = 20
        self._direction_overrides = re.compile(r"[\u202A-\u202E\u2066-\u2069]")
        self._control_chars = re.compile(r"[\x00-\x1F\x7F]")
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
        """Validate message structure, sanitize content, and block injections.

        Args:
            messages: Raw message list from validated request models.

        Returns:
            list[dict]: Sanitized message list.
        """
        if not 1 <= len(messages) <= self._max_history_messages:
            raise ValidationError(
                message="Invalid message count",
                status_code=422,
                log_detail=f"message_count={len(messages)}",
            )

        sanitized_messages: list[dict] = []
        for message in messages:
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", ""))
            sanitized_content = self.sanitize_input(content)

            if role not in {"user", "assistant"}:
                raise ValidationError(
                    message="Invalid message role",
                    status_code=422,
                    log_detail=f"invalid_role={role}",
                )

            if not sanitized_content or len(sanitized_content) > self._max_message_chars:
                raise ValidationError(
                    message="Invalid message length",
                    status_code=422,
                    log_detail=f"message_length={len(sanitized_content)}",
                )

            if self.check_injection(sanitized_content):
                self._log_injection_attempt(sanitized_content)
                raise InjectionError(
                    message="Input rejected: disallowed pattern",
                    status_code=400,
                    log_detail="prompt injection pattern matched",
                )

            sanitized_messages.append({"role": role, "content": sanitized_content})

        return sanitized_messages

    def sanitize_input(self, text: str) -> str:
        """Remove null bytes, control chars, and direction overrides.

        Args:
            text: Raw input text.

        Returns:
            str: Sanitized plain text.
        """
        without_overrides = self._direction_overrides.sub("", text)
        without_controls = self._control_chars.sub("", without_overrides)
        return without_controls.strip()

    def check_injection(self, text: str) -> bool:
        """Detect known prompt-injection strings.

        Args:
            text: Sanitized message text.

        Returns:
            bool: True if an injection pattern is present.
        """
        lowered = text.lower()
        return any(pattern in lowered for pattern in self._injection_patterns)

    def _log_injection_attempt(self, text: str) -> None:
        """Log sanitized injection attempts to stderr with timestamp.

        Args:
            text: Sanitized message text.

        Returns:
            None.
        """
        timestamp = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        safe_preview = text[:240]
        print(
            f"[{timestamp}] injection_attempt_rejected preview={safe_preview!r}",
            file=sys.stderr,
        )


message_validator = MessageValidator()
