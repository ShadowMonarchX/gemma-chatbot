from __future__ import annotations


class AppError(Exception):
    """Base exception for controlled application failures."""

    def __init__(self, message: str, status_code: int, log_detail: str = "") -> None:
        """Create an application exception.

        Args:
            message: Safe message for API response bodies.
            status_code: HTTP status code associated with the error.
            log_detail: Internal diagnostics safe for server logs.

        Returns:
            None.
        """
        self.message: str = message
        self.status_code: int = status_code
        self.log_detail: str = log_detail
        super().__init__(message)


class ConfigurationError(AppError):
    """Raised when runtime configuration is invalid."""


class ModelError(AppError):
    """Raised when model loading or generation fails."""


class ValidationError(AppError):
    """Raised when request validation fails business constraints."""


class RateLimitError(AppError):
    """Raised when the request rate exceeds configured limits."""


class InjectionError(AppError):
    """Raised when prompt-injection patterns are detected in user input."""
