"""Exceptions raised by the correctable queue."""


class QueueError(Exception):
    """Base class for queue errors."""


class DuplicateRowError(QueueError):
    """Raised when submitting a row id that is already known. Use :meth:`CleanseQueue.correct` to revise it."""


class StaleRevisionError(QueueError):
    """Raised when an operation references a revision that is no longer the current one."""


class UnknownRowError(QueueError):
    """Raised when reading or correcting a row id that was never submitted."""


class ChainError(QueueError):
    """Wraps the error raised while running a revision through the node chain."""

    def __init__(self, message, *, cause=None):
        super().__init__(message)
        self.cause = cause
