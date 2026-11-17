"""
Lifecycle states shared by the durable queue and its worker.

* :data:`PENDING`    -- revision accepted, waiting to be (or being re-) processed.
* :data:`RUNNING`    -- a worker claimed the revision and is running it through the chain.
* :data:`SUCCEEDED`  -- terminal state: the whole chain completed, all outputs were persisted atomically.
* :data:`FAILED`     -- terminal state for this revision: the chain raised; the raw row and the reason are kept, no
  output row was persisted.

Only ``SUCCEEDED`` and ``FAILED`` are terminal. A failed row can always be corrected, which creates a fresh
``PENDING`` revision.
"""

PENDING = "pending"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"

TERMINAL_STATES = frozenset((SUCCEEDED, FAILED))
