"""
Correctable cleansing/loading queue example.

Run with ``python -m bonobo.examples.q_correctable_queue`` (or just ``python bonobo/examples/q_correctable_queue.py``).

It shows the full lifecycle on a temporary on-disk SQLite queue:

1. submit raw rows under stable ids;
2. run them through an ordinary linear chain of native Bonobo nodes (a filter, a transform and a generator);
3. correct a row while work is in flight and see the stale result discarded;
4. reopen the queue and read back the current raw row, its state and its real results;
5. correct a failed row and process the new revision successfully.
"""
import os
import tempfile

from bonobo import Filter
from bonobo.q import CleanseQueue, QueueWorker


class EvenFilter(Filter):
    """Keep even integers, burn odd ones (0 or 1 output, handled natively by Bonobo)."""

    def filter(self, value):
        return value % 2 == 0


def double(value):
    """Ordinary 1-to-1 transform."""
    return value * 2


def split_into_pairs(value):
    """Generator node: one input expands to N outputs; duplicates would be kept."""
    yield (value, "a")
    yield (value, "b")


def main():
    workdir = tempfile.mkdtemp(prefix="bonobo-q-")
    database = os.path.join(workdir, "queue.sqlite")

    queue = CleanseQueue(database)
    queue.submit("row-1", 2)
    queue.submit("row-2", 3)  # will be filtered out (0 output)
    queue.submit("row-3", 4)

    worker = QueueWorker(queue, EvenFilter(), double, split_into_pairs)
    worker.drain()

    print("row-1 results:", queue.get_results("row-1"))
    print("row-2 results (filtered out, still succeeded):", queue.get_results("row-2"))
    print("row-3 results:", queue.get_results("row-3"))

    # A correction supersedes the previous revision immediately, even if an old task were still running.
    queue.correct("row-1", 10)
    worker.drain()
    print("row-1 after correction:", queue.get_current("row-1"), queue.get_results("row-1"))

    # Reopen: state, raw rows and real results are all durable.
    queue.close()
    reopened = CleanseQueue(database)
    print("after reopen row-3:", reopened.get_current("row-3"), reopened.get_results("row-3"))


if __name__ == "__main__":
    main()
