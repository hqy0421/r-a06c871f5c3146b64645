"""
A correctable, durable cleansing/loading queue for Bonobo.

Each raw row is identified by a stable id and may be corrected any number of times. Every correction produces a new
revision, and accepting a new revision immediately supersedes the previous one. Revisions are processed, one at a time,
through an ordinary linear chain of Bonobo nodes, and the results of a revision are only ever persisted atomically
when the whole chain (including generator-style nodes) ran to completion without error.

The public surface is intentionally small:

* :class:`bonobo.q.CleanseQueue` -- the durable queue (SQLite backed), to submit and correct rows, and to read back
  state and results.
* :class:`bonobo.q.QueueWorker` -- drains the queue, running each accepted revision through a linear node chain.

"""
from bonobo.q.constants import FAILED, PENDING, RUNNING, SUCCEEDED
from bonobo.q.store import CleanseQueue
from bonobo.q.worker import QueueWorker

__all__ = ["CleanseQueue", "QueueWorker", "PENDING", "RUNNING", "SUCCEEDED", "FAILED"]
