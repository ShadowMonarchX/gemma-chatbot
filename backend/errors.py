from __future__ import annotations


class AppError(Exception):
    """Base exception for controlled application errors."""

    def __init__(self, message: str, status_code: int, log_detail: str = "") -> None:
        """Create an application error.

        Args:
            message: Safe message that can be returned to API clients.
            status_code: HTTP status code associated with this error.
            log_detail: Internal detail for server-side logs.

        Returns:
            None.
        """
        self.message: str = message
        self.status_code: int = status_code
        self.log_detail: str = log_detail
        super().__init__(message)


class ModelError(AppError):
    """Raised when model loading or token generation fails."""


class ValidationError(AppError):
    """Raised when validated input violates application rules."""


class RateLimitError(AppError):
    """Raised when the request rate exceeds the configured threshold."""


class InjectionError(AppError):
    """Raised when a potential prompt-injection pattern is detected."""
