"""
SQLite-backed durable storage for the correctable cleansing/loading queue.

The store knows nothing about Bonobo graphs. Its responsibilities are:

* persisting each raw row under its stable id, one row per current revision;
* accepting corrections at any time (a correction supersedes the previous revision immediately and removes the
  previous revision's outputs so the *current* result set never contains stale data);
* atomically claiming a pending revision and, later, atomically accepting exactly one terminal state per revision,
  guarding every write on the revision still being current.

All state transitions that touch more than one row happen inside a single ``BEGIN IMMEDIATE`` transaction, so a crash
can never leave a revision half written, an output set partially replaced, or a terminal state without its outputs.
"""
import json
import sqlite3
import threading
import time

from bonobo.q.constants import FAILED, PENDING, RUNNING, SUCCEEDED
from bonobo.q.errors import DuplicateRowError, StaleRevisionError, UnknownRowError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rows (
    row_id        TEXT PRIMARY KEY,
    revision      INTEGER NOT NULL,
    payload       TEXT    NOT NULL,
    state         TEXT    NOT NULL,
    error         TEXT,
    created_at    REAL    NOT NULL,
    updated_at    REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS revisions (
    row_id        TEXT    NOT NULL,
    revision      INTEGER NOT NULL,
    state         TEXT    NOT NULL,
    is_current    INTEGER NOT NULL,
    created_at    REAL    NOT NULL,
    updated_at    REAL    NOT NULL,
    error         TEXT,
    PRIMARY KEY (row_id, revision)
);

CREATE TABLE IF NOT EXISTS outputs (
    row_id        TEXT    NOT NULL,
    revision      INTEGER NOT NULL,
    position      INTEGER NOT NULL,
    payload       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_outputs_current ON outputs (row_id, revision, position);
CREATE INDEX IF NOT EXISTS idx_revisions_state ON revisions (state, is_current);
"""


def _dumps(payload):
    """JSON-serialize a row payload, failing fast with a clear error on unsupported types."""
    try:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TypeError("Queue row payloads must be JSON-serializable (got {!r}).".format(type(payload))) from exc


def _loads(blob):
    return json.loads(blob)


class CleanseQueue:
    """
    A durable, correctable queue of raw rows, persisted in SQLite.

    :param database: path to the SQLite file, or ``":memory:"`` for a transient queue (mostly useful in tests).
    :param timeout: SQLite busy timeout, in seconds.
    """

    def __init__(self, database=":memory:", *, timeout=30.0):
        self.database = database
        # A re-entrant lock serializes writers within this process; SQLite itself plus BEGIN IMMEDIATE still
        # guarantees correctness against other processes holding the same file.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            database,
            timeout=timeout,
            detect_types=sqlite3.PARSE_DECLTYPES,
            isolation_level=None,  # manual transaction control
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=FULL")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._recover()

    # ------------------------------------------------------------------ lifecycle
    def close(self):
        with self._lock:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _rollback(self):
        # Never let a rollback failure mask the exception that triggered it.
        try:
            self._conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    def _recover(self):
        """
        On reopen, any revision still marked RUNNING never reached a terminal state (the process died mid-run).

        The *current* revision of a row is reset to PENDING so it gets retried. Stale revisions that were still
        running are settled as failed history: they are no longer current and must never be retried or write
        anything.
        """
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "UPDATE revisions SET state = ?, updated_at = ? WHERE state = ? AND is_current = 1",
                    (PENDING, now, RUNNING),
                )
                # Historical revisions still marked RUNNING can never complete (their worker is gone): settle them as
                # failed, superseded history so every revision ends in a well-defined terminal state.
                self._conn.execute(
                    "UPDATE revisions SET state = ?, error = ?, updated_at = ? "
                    "WHERE state = ? AND is_current = 0",
                    (FAILED, "interrupted by shutdown before completion", now, RUNNING),
                )
                self._conn.execute(
                    "UPDATE rows SET state = ?, error = NULL, updated_at = ? WHERE state = ?",
                    (PENDING, now, RUNNING),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._rollback()
                raise

    # ------------------------------------------------------------------ intake
    def submit(self, row_id, payload):
        """
        Submit a brand new raw row at revision 1, in the PENDING state.

        :raises DuplicateRowError: if ``row_id`` is already known. Correct it with :meth:`correct` instead.
        """
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("row_id must be a non-empty string.")
        blob = _dumps(payload)
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                exists = self._conn.execute(
                    "SELECT 1 FROM rows WHERE row_id = ?", (row_id,)
                ).fetchone()
                if exists is not None:
                    raise DuplicateRowError("Row {!r} already exists; use correct() to revise it.".format(row_id))
                self._conn.execute(
                    "INSERT INTO rows (row_id, revision, payload, state, error, created_at, updated_at) "
                    "VALUES (?, 1, ?, ?, NULL, ?, ?)",
                    (row_id, blob, PENDING, now, now),
                )
                self._conn.execute(
                    "INSERT INTO revisions (row_id, revision, state, is_current, created_at, updated_at, error) "
                    "VALUES (?, 1, ?, 1, ?, ?, NULL)",
                    (row_id, PENDING, now, now),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._rollback()
                raise
        return 1

    def correct(self, row_id, payload):
        """
        Accept a correction for ``row_id`` and return its new revision number.

        A correction is always accepted, even while an older revision is RUNNING: the old revision is marked stale
        immediately and its outputs are removed, so the current result set no longer contains the old revision's data
        before the new revision is processed. Submitting the exact same payload again is idempotent and returns the
        current revision without creating a new one.

        :raises UnknownRowError: if ``row_id`` was never :meth:`submit`-ed.
        """
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("row_id must be a non-empty string.")
        blob = _dumps(payload)
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                current = self._conn.execute(
                    "SELECT revision, payload FROM rows WHERE row_id = ?", (row_id,)
                ).fetchone()
                if current is None:
                    raise UnknownRowError("Unknown row {!r}; submit() it first.".format(row_id))

                # Idempotent re-delivery of the same correction: no new revision, no state change.
                if current["payload"] == blob:
                    self._rollback()
                    return current["revision"]

                new_revision = current["revision"] + 1
                # Old revision stops being current, whatever state it was in. A late worker completion for it will
                # then hit the is_current guard and be rejected.
                self._conn.execute(
                    "UPDATE revisions SET is_current = 0, updated_at = ? WHERE row_id = ? AND is_current = 1",
                    (now, row_id),
                )
                self._conn.execute(
                    "INSERT INTO revisions (row_id, revision, state, is_current, created_at, updated_at, error) "
                    "VALUES (?, ?, ?, 1, ?, ?, NULL)",
                    (row_id, new_revision, PENDING, now, now),
                )
                # The current result set must not retain any data from the old revision.
                self._conn.execute("DELETE FROM outputs WHERE row_id = ?", (row_id,))
                self._conn.execute(
                    "UPDATE rows SET revision = ?, payload = ?, state = ?, error = NULL, updated_at = ? "
                    "WHERE row_id = ?",
                    (new_revision, blob, PENDING, now, row_id),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._rollback()
                raise
        return new_revision

    # ------------------------------------------------------------------ dispatch
    def claim(self):
        """
        Atomically claim the oldest pending *current* revision to process.

        Returns ``(row_id, revision, payload)`` or ``None`` when the queue has no pending work. Stale revisions
        (superseded by a correction) are never claimed.
        """
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT r.row_id, r.revision, r.payload "
                    "FROM rows AS r "
                    "JOIN revisions AS v ON v.row_id = r.row_id AND v.revision = r.revision "
                    "WHERE r.state = ? AND v.state = ? AND v.is_current = 1 "
                    "ORDER BY r.updated_at, r.row_id "
                    "LIMIT 1",
                    (PENDING, PENDING),
                ).fetchone()
                if row is None:
                    self._rollback()
                    return None
                self._conn.execute(
                    "UPDATE rows SET state = ?, updated_at = ? WHERE row_id = ? AND revision = ? AND state = ?",
                    (RUNNING, now, row["row_id"], row["revision"], PENDING),
                )
                self._conn.execute(
                    "UPDATE revisions SET state = ?, updated_at = ? "
                    "WHERE row_id = ? AND revision = ? AND state = ? AND is_current = 1",
                    (RUNNING, now, row["row_id"], row["revision"], PENDING),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._rollback()
                raise
        return row["row_id"], row["revision"], _loads(row["payload"])

    def complete(self, row_id, revision, *, success, outputs=None, error=None):
        """
        Record the terminal state of a claimed revision.

        The write is guarded on ``revision`` still being the current revision of the row and still being RUNNING, so
        a late success or a late failure of an old task can never overwrite a newer revision, and each revision
        accepts exactly one terminal state: re-delivering the same completion is a no-op returning ``False``.

        On success, ``outputs`` (the complete, ordered list of bags emitted by the chain) and the SUCCEEDED state are
        committed together in one transaction. On failure, ``error`` is stored and no output row is written.
        """
        outputs = outputs or ()
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                revision_row = self._conn.execute(
                    "SELECT state, is_current FROM revisions WHERE row_id = ? AND revision = ?",
                    (row_id, revision),
                ).fetchone()

                if revision_row is None:
                    raise StaleRevisionError(
                        "No revision {} for row {!r}.".format(revision, row_id)
                    )

                # Idempotent re-delivery: a revision keeps its first terminal state forever.
                if revision_row["state"] in (SUCCEEDED, FAILED):
                    self._rollback()
                    return False

                if revision_row["state"] != RUNNING:
                    # Recovered-but-not-reclaimed rows are no longer ours to complete.
                    self._rollback()
                    return False

                if not revision_row["is_current"]:
                    # A correction superseded this run. Record nothing but the late terminal state on the (now
                    # historical) revision, never touching the current row or its outputs.
                    self._conn.execute(
                        "UPDATE revisions SET state = ?, error = ?, updated_at = ? "
                        "WHERE row_id = ? AND revision = ?",
                        (SUCCEEDED if success else FAILED, error, now, row_id, revision),
                    )
                    self._conn.execute("COMMIT")
                    return False

                if success:
                    # Replace the whole output set atomically. In normal flow outputs were already removed at claim
                    # time's correction boundary; DELETE here also makes success idempotent-ish and defensive.
                    self._conn.execute(
                        "DELETE FROM outputs WHERE row_id = ? AND revision = ?", (row_id, revision)
                    )
                    self._conn.executemany(
                        "INSERT INTO outputs (row_id, revision, position, payload) VALUES (?, ?, ?, ?)",
                        [
                            (row_id, revision, position, _dumps(bag))
                            for position, bag in enumerate(outputs)
                        ],
                    )
                    self._conn.execute(
                        "UPDATE revisions SET state = ?, error = NULL, updated_at = ? "
                        "WHERE row_id = ? AND revision = ?",
                        (SUCCEEDED, now, row_id, revision),
                    )
                    self._conn.execute(
                        "UPDATE rows SET state = ?, error = NULL, updated_at = ? "
                        "WHERE row_id = ? AND revision = ?",
                        (SUCCEEDED, now, row_id, revision),
                    )
                else:
                    self._conn.execute(
                        "UPDATE revisions SET state = ?, error = ?, updated_at = ? "
                        "WHERE row_id = ? AND revision = ?",
                        (FAILED, error, now, row_id, revision),
                    )
                    self._conn.execute(
                        "UPDATE rows SET state = ?, error = ?, updated_at = ? "
                        "WHERE row_id = ? AND revision = ?",
                        (FAILED, error, now, row_id, revision),
                    )
                self._conn.execute("COMMIT")
            except BaseException:
                self._rollback()
                raise
        return True

    # ------------------------------------------------------------------ read side
    def get_current(self, row_id):
        """Return ``(revision, payload, state)`` for the current revision of ``row_id``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT revision, payload, state FROM rows WHERE row_id = ?", (row_id,)
            ).fetchone()
        if row is None:
            raise UnknownRowError("Unknown row {!r}.".format(row_id))
        return row["revision"], _loads(row["payload"]), row["state"]

    def get_status(self, row_id):
        """Return the state of the current revision of ``row_id``."""
        return self.get_current(row_id)[2]

    def get_results(self, row_id):
        """
        Return the real, persisted outputs of the row's current revision, in chain order.

        Duplicate output rows are preserved. Returns an empty list while pending/running or after a failure: a failed
        revision never leaves partial outputs behind.
        """
        with self._lock:
            current = self._conn.execute(
                "SELECT revision FROM rows WHERE row_id = ?", (row_id,)
            ).fetchone()
            if current is None:
                raise UnknownRowError("Unknown row {!r}.".format(row_id))
            blobs = self._conn.execute(
                "SELECT payload FROM outputs WHERE row_id = ? AND revision = ? ORDER BY position",
                (row_id, current["revision"]),
            ).fetchall()
        return [_loads(row["payload"]) for row in blobs]

    def get_error(self, row_id):
        """Return the failure reason of the current revision of ``row_id``, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT error FROM rows WHERE row_id = ?", (row_id,)
            ).fetchone()
        if row is None:
            raise UnknownRowError("Unknown row {!r}.".format(row_id))
        return row["error"]

    def pending_count(self):
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM rows WHERE state = ?", (PENDING,)
            ).fetchone()[0]

    def __len__(self):
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
