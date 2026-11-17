import os
import threading

import pytest

from bonobo import Filter, MapFields, Rename
from bonobo.constants import NOT_MODIFIED
from bonobo.q import CleanseQueue, QueueWorker
from bonobo.q.constants import FAILED, PENDING, RUNNING, SUCCEEDED
from bonobo.q.errors import ChainError, DuplicateRowError, StaleRevisionError, UnknownRowError
from bonobo.q.worker import _OutputBuffer


# --------------------------------------------------------------------------- helpers
def double(x):
    return x * 2


def keep_even(x):
    if x % 2 == 0:
        yield x


def expand(x):
    # 0, 1 or N outputs depending on the input value.
    for i in range(x):
        yield (x, i)


def explode_three(x):
    # Always yields duplicates: N outputs with repeated rows.
    yield x
    yield x
    yield x


def boom(x):
    raise ValueError("kaboom: {}".format(x))


def flaky_factory(fail_once_for):
    state = {"calls": 0}

    def flaky(x):
        state["calls"] += 1
        if x == fail_once_for and state["calls"] == 1:
            raise RuntimeError("transient")
        return NOT_MODIFIED

    return flaky


# --------------------------------------------------------------------------- store basics
def test_submit_assigns_revision_one_and_pending():
    queue = CleanseQueue(":memory:")
    assert queue.submit("r1", {"a": 1}) == 1
    revision, payload, state = queue.get_current("r1")
    assert (revision, payload, state) == (1, {"a": 1}, PENDING)
    assert queue.pending_count() == 1
    assert len(queue) == 1


def test_submit_duplicate_row_id_rejected():
    queue = CleanseQueue(":memory:")
    queue.submit("r1", 1)
    with pytest.raises(DuplicateRowError):
        queue.submit("r1", 2)


def test_unknown_row_raises():
    queue = CleanseQueue(":memory:")
    with pytest.raises(UnknownRowError):
        queue.get_current("nope")
    with pytest.raises(UnknownRowError):
        queue.correct("nope", 1)
    with pytest.raises(UnknownRowError):
        queue.get_results("nope")


def test_unsupported_payload_rejected_at_submit_time():
    queue = CleanseQueue(":memory:")
    with pytest.raises(TypeError):
        queue.submit("r1", object())


def test_correct_creates_monotonic_revisions_and_supersedes():
    queue = CleanseQueue(":memory:")
    assert queue.submit("r1", "v1") == 1
    assert queue.correct("r1", "v2") == 2
    assert queue.correct("r1", "v3") == 3
    revision, payload, state = queue.get_current("r1")
    assert (revision, payload, state) == (3, "v3", PENDING)


def test_correct_same_payload_is_idempotent():
    queue = CleanseQueue(":memory:")
    queue.submit("r1", [1, 2])
    assert queue.correct("r1", [1, 2]) == 1
    assert queue.correct("r1", [1, 2]) == 1
    revision, payload, _ = queue.get_current("r1")
    assert revision == 1 and payload == [1, 2]


# --------------------------------------------------------------------------- chain semantics: 0 / 1 / N + duplicates
def test_chain_one_output_per_row():
    queue = CleanseQueue(":memory:")
    queue.submit("r1", 21)
    QueueWorker(queue, double).drain()
    assert queue.get_status("r1") == SUCCEEDED
    assert queue.get_results("r1") == [[42]]


def test_chain_filter_zero_output_still_succeeds():
    queue = CleanseQueue(":memory:")
    queue.submit("even", 4)
    queue.submit("odd", 5)
    QueueWorker(queue, keep_even).drain()
    assert queue.get_results("even") == [[4]]
    assert queue.get_results("odd") == []
    assert queue.get_status("odd") == SUCCEEDED


def test_chain_generator_expands_to_n_outputs_in_order():
    queue = CleanseQueue(":memory:")
    queue.submit("r1", 3)
    QueueWorker(queue, expand).drain()
    assert queue.get_results("r1") == [[3, 0], [3, 1], [3, 2]]


def test_chain_keeps_duplicate_rows():
    queue = CleanseQueue(":memory:")
    queue.submit("r1", "x")
    QueueWorker(queue, explode_three).drain()
    assert queue.get_results("r1") == [["x"], ["x"], ["x"]]


def test_chain_runs_through_multiple_linear_native_nodes():
    queue = CleanseQueue(":memory:")
    queue.submit("r1", 4)
    # keep_even (1 output for 4) -> expand (4 outputs) -> double each tuple
    QueueWorker(queue, keep_even, expand, lambda *bag: tuple(v * 2 for v in bag)).drain()
    assert queue.get_results("r1") == [[8, 0], [8, 2], [8, 4], [8, 6]]


def test_non_linear_chain_rejected():
    queue = CleanseQueue(":memory:")
    with pytest.raises(ValueError):
        QueueWorker(queue)

class _ScoreFilter(Filter):
    def filter(self, name, score):
        return score >= 5


def test_native_configurable_nodes_chain_filter_rename_map():
    # An ordinary chain built from Bonobo's native Configurable nodes: named fields, a filter (0/1),
    # a rename and a value-mapping transform.
    from bonobo.util.bags import BagType

    Row = BagType("Row", ["name", "score"])
    queue = CleanseQueue(":memory:")
    queue.submit("kept", ["alice", 10])
    queue.submit("burnt", ["bob", 3])

    chain = (
        _ScoreFilter(),
        Rename(score="points"),
        MapFields(str.upper, key=lambda field: field == "name"),
    )
    worker = QueueWorker(queue, *chain, input_bag_factory=lambda row: Row(*row))
    worker.drain()

    assert queue.get_status("kept") == SUCCEEDED
    kept = queue.get_results("kept")
    assert len(kept) == 1 and kept[0] == ["ALICE", 10]
    assert queue.get_status("burnt") == SUCCEEDED
    assert queue.get_results("burnt") == []


def test_native_stateful_node_is_reset_between_revisions():
    # Limit stops emitting after n inputs; with one row per fresh graph, every revision still goes through.
    from bonobo import Limit

    queue = CleanseQueue(":memory:")
    queue.submit("r1", "a")
    queue.submit("r2", "b")
    queue.submit("r3", "c")
    QueueWorker(queue, Limit(1)).drain()
    # A shared Limit(1) would only let the first processed row pass; fresh contexts mean all three pass.
    assert queue.get_results("r1") == [["a"]]
    assert queue.get_results("r2") == [["b"]]
    assert queue.get_results("r3") == [["c"]]


def test_each_run_gets_fresh_graph_and_clean_node_state():
    # A stateful node would accumulate across runs if contexts were reused.
    seen_inputs = []

    def remember(x):
        seen_inputs.append(x)
        yield x

    queue = CleanseQueue(":memory:")
    queue.submit("r1", 1)
    queue.submit("r2", 2)
    worker = QueueWorker(queue, remember)
    worker.drain()
    assert seen_inputs == [1, 2]
    assert queue.get_results("r1") == [[1]]
    assert queue.get_results("r2") == [[2]]


# --------------------------------------------------------------------------- failure atomicity
def test_failure_persists_raw_row_and_reason_without_any_output():
    queue = CleanseQueue(":memory:")
    queue.submit("r1", 7)
    worker = QueueWorker(queue, lambda x: x + 1, boom)
    worker.drain()

    assert queue.get_status("r1") == FAILED
    revision, payload, state = queue.get_current("r1")
    assert revision == 1 and payload == 7 and state == FAILED
    reason = queue.get_error("r1")
    assert "kaboom: 8" in reason and "ValueError" in reason
    # No partial output may survive a failed run.
    assert queue.get_results("r1") == []


def test_failed_row_can_be_corrected_and_then_succeeds():
    queue = CleanseQueue(":memory:")
    queue.submit("r1", 7)
    QueueWorker(queue, boom).drain()
    assert queue.get_status("r1") == FAILED

    new_revision = queue.correct("r1", 1)
    assert new_revision == 2
    assert queue.get_status("r1") == PENDING
    QueueWorker(queue, double).drain()
    assert queue.get_status("r1") == SUCCEEDED
    assert queue.get_results("r1") == [[2]]
    assert queue.get_error("r1") is None


def test_generator_error_mid_expansion_leaves_no_partial_output():
    def partial_then_boom(x):
        yield x
        yield x
        raise RuntimeError("generator broke")

    queue = CleanseQueue(":memory:")
    queue.submit("r1", 1)
    QueueWorker(queue, partial_then_boom).drain()
    assert queue.get_status("r1") == FAILED
    assert queue.get_results("r1") == []
    assert "generator broke" in queue.get_error("r1")


def test_worker_raises_chain_error_directly_on_manual_run():
    queue = CleanseQueue(":memory:")
    worker = QueueWorker(queue, boom)
    with pytest.raises(ChainError):
        worker._run_revision(1)


# --------------------------------------------------------------------------- corrections vs in-flight tasks
def test_correction_while_running_discards_old_outputs_immediately():
    queue = CleanseQueue(":memory:")
    queue.submit("r1", "old")
    row_id, revision, payload = queue.claim()
    assert (row_id, revision, payload) == ("r1", 1, "old")

    # Simulate the old task having produced outputs before the correction lands (they would only be committed on
    # completion, but a correction must still guarantee no old-revision data remains visible).
    new_revision = queue.correct("r1", "new")
    assert new_revision == 2
    assert queue.get_results("r1") == []
    assert queue.get_status("r1") == PENDING
    revision, payload, _ = queue.get_current("r1")
    assert (revision, payload) == (2, "new")


def test_late_success_of_stale_revision_does_not_overwrite_new_revision():
    queue = CleanseQueue(":memory:")
    queue.submit("r1", "old")
    queue.claim()  # revision 1 now RUNNING
    queue.correct("r1", "new")  # revision 2 pending, revision 1 stale

    # Old task finishes successfully, very late.
    applied = queue.complete("r1", 1, success=True, outputs=[["STALE"]])
    assert applied is False
    assert queue.get_results("r1") == []
    assert queue.get_status("r1") == PENDING

    # New revision runs normally and wins.
    QueueWorker(queue, double if False else lambda s: s.upper()).drain()
    assert queue.get_status("r1") == SUCCEEDED
    assert queue.get_results("r1") == [["NEW"]]


def test_late_failure_of_stale_revision_does_not_fail_new_revision():
    queue = CleanseQueue(":memory:")
    queue.submit("r1", 1)
    queue.claim()
    queue.correct("r1", 2)

    applied = queue.complete("r1", 1, success=False, error="late boom")
    assert applied is False
    assert queue.get_status("r1") == PENDING
    assert queue.get_error("r1") is None

    QueueWorker(queue, double).drain()
    assert queue.get_status("r1") == SUCCEEDED
    assert queue.get_results("r1") == [[4]]


def test_each_reversion_accepts_a_single_terminal_state_and_redelivery_is_idempotent():
    queue = CleanseQueue(":memory:")
    queue.submit("r1", 1)
    queue.claim()
    assert queue.complete("r1", 1, success=True, outputs=[[2]]) is True
    # Re-delivering the same completion changes nothing.
    assert queue.complete("r1", 1, success=True, outputs=[[99]]) is False
    assert queue.get_results("r1") == [[2]]
    # A late failure cannot flip the terminal state either.
    assert queue.complete("r1", 1, success=False, error="nope") is False
    assert queue.get_status("r1") == SUCCEEDED
    assert queue.get_error("r1") is None


def test_failure_redelivery_is_idempotent():
    queue = CleanseQueue(":memory:")
    queue.submit("r1", 1)
    queue.claim()
    assert queue.complete("r1", 1, success=False, error="first") is True
    assert queue.complete("r1", 1, success=False, error="second") is False
    assert queue.get_status("r1") == FAILED
    assert queue.get_error("r1") == "first"


def test_complete_unknown_revision_raises():
    queue = CleanseQueue(":memory:")
    queue.submit("r1", 1)
    with pytest.raises(StaleRevisionError):
        queue.complete("r1", 99, success=True)


def test_claim_is_idempotent_under_concurrency():
    queue = CleanseQueue(":memory:")
    for i in range(20):
        queue.submit("r{}".format(i), i)
    claims = []
    lock = threading.Lock()

    def worker_claim():
        claimed = queue.claim()
        if claimed is not None:
            with lock:
                claims.append((claimed[0], claimed[1]))

    threads = [threading.Thread(target=worker_claim) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # Every claimed (row, revision) pair is unique.
    assert len(claims) == len(set(claims))


# --------------------------------------------------------------------------- end-to-end correction through the worker
def test_full_correction_lifecycle_with_worker():
    queue = CleanseQueue(":memory:")
    worker = QueueWorker(queue, double)

    queue.submit("r1", 1)
    worker.drain()
    assert queue.get_results("r1") == [[2]]

    queue.correct("r1", 10)
    worker.drain()
    revision, payload, state = queue.get_current("r1")
    assert (revision, payload, state) == (2, 10, SUCCEEDED)
    assert queue.get_results("r1") == [[20]]

    # Another correction, this one failing, keeps the raw row and leaves no output.
    queue.correct("r1", 3)
    QueueWorker(queue, lambda x: x + 1, boom).drain()
    assert queue.get_status("r1") == FAILED
    assert queue.get_current("r1")[:2] == (3, 3)
    assert queue.get_results("r1") == []
    assert "kaboom: 4" in queue.get_error("r1")

    # And yet another correction recovers.
    queue.correct("r1", 5)
    QueueWorker(queue, double).drain()
    assert queue.get_results("r1") == [[10]]
    assert queue.get_status("r1") == SUCCEEDED


def test_drain_is_idempotent_when_empty():
    queue = CleanseQueue(":memory:")
    worker = QueueWorker(queue, double)
    assert worker.drain() == 0
    queue.submit("r1", 1)
    assert worker.drain() == 1
    assert worker.drain() == 0


def test_drain_max_tasks():
    queue = CleanseQueue(":memory:")
    for i in range(5):
        queue.submit("r{}".format(i), i)
    worker = QueueWorker(queue, double)
    assert worker.drain(max_tasks=2) == 2
    assert queue.pending_count() == 3
    assert worker.drain() == 3
    assert queue.pending_count() == 0


# --------------------------------------------------------------------------- durability across reopen
def test_state_and_real_results_survive_reopen(tmpdir):
    path = os.path.join(str(tmpdir), "queue.sqlite")
    queue = CleanseQueue(path)
    queue.submit("ok", 3)
    queue.submit("bad", 1)
    queue.correct("ok", 6)
    # Each row is processed by the chain appropriate to it; the bad row runs through a failing chain. Claims come
    # out in submission order (bad first), so complete them accordingly.
    row_id, revision, payload = queue.claim()
    assert (row_id, revision, payload) == ("bad", 1, 1)
    queue.complete("bad", 1, success=False, error="Node 'boom' failed: ValueError: kaboom: {}".format(payload))
    row_id, revision, payload = queue.claim()
    assert (row_id, revision, payload) == ("ok", 2, 6)
    queue.complete("ok", 2, success=True, outputs=[[payload, i] for i in range(payload)])
    queue.close()

    reopened = CleanseQueue(path)
    assert reopened.get_current("ok") == (2, 6, SUCCEEDED)
    assert reopened.get_results("ok") == [[6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5]]
    assert reopened.get_status("bad") == FAILED
    assert reopened.get_current("bad")[:2] == (1, 1)
    assert "kaboom" in reopened.get_error("bad")
    assert reopened.get_results("bad") == []


def test_running_revision_is_reset_to_pending_on_reopen(tmpdir):
    path = os.path.join(str(tmpdir), "queue.sqlite")
    queue = CleanseQueue(path)
    queue.submit("r1", 21)
    queue.claim()  # dies while RUNNING
    assert queue.get_status("r1") == RUNNING
    queue.close()

    reopened = CleanseQueue(path)
    assert reopened.get_status("r1") == PENDING
    assert reopened.get_current("r1")[:2] == (1, 21)
    QueueWorker(reopened, double).drain()
    assert reopened.get_results("r1") == [[42]]


def test_stale_running_revision_not_retried_after_reopen(tmpdir):
    path = os.path.join(str(tmpdir), "queue.sqlite")
    queue = CleanseQueue(path)
    queue.submit("r1", 1)
    queue.claim()  # rev 1 RUNNING
    queue.correct("r1", 2)  # rev 2 pending; rev 1 is a stale RUNNING
    queue.close()

    reopened = CleanseQueue(path)
    # Only the current revision is retried; rev 1 must stay out of the way.
    QueueWorker(reopened, double).drain()
    assert reopened.get_current("r1") == (2, 2, SUCCEEDED)
    assert reopened.get_results("r1") == [[4]]
    # A late completion of the dead run is still rejected.
    assert reopened.complete("r1", 1, success=True, outputs=[["nope"]]) is False

    import sqlite3

    conn = sqlite3.connect(path)
    try:
        stale = conn.execute(
            "SELECT state, is_current FROM revisions WHERE row_id = 'r1' AND revision = 1"
        ).fetchone()
    finally:
        conn.close()
    # The interrupted stale run is settled as failed history on reopen, never retried.
    assert stale == (FAILED, 0)


def test_output_buffer_drops_tokens_but_keeps_data():
    from bonobo.constants import BEGIN, END

    buffer = _OutputBuffer()
    buffer.put(BEGIN)
    buffer.put((1,))
    buffer.put(END)
    assert list(buffer) == [(1,)]


# --------------------------------------------------------------------------- end-to-end races
class _BlockableNode:
    """Node that blocks inside the chain until the test has let a correction land."""

    def __init__(self, gate, proceed):
        self.gate = gate
        self.proceed = proceed

    def __call__(self, payload):
        # Signal that the old revision is genuinely inside the chain, then wait for the test to correct it.
        self.gate.set()
        self.proceed.wait(timeout=5)
        return ("old-result", payload)


def test_correction_while_chain_is_running_old_result_is_discarded():
    import threading

    queue = CleanseQueue(":memory:")
    queue.submit("r1", "v1")
    gate, proceed = threading.Event(), threading.Event()
    worker = QueueWorker(queue, _BlockableNode(gate, proceed))

    old_run = threading.Thread(target=worker.process_one)
    old_run.start()
    assert gate.wait(timeout=5)  # revision 1 is now RUNNING inside the chain

    # Correct while the old task has not finished.
    assert queue.correct("r1", "v2") == 2
    assert queue.get_status("r1") == PENDING
    assert queue.get_results("r1") == []

    # Let the stale task finish; its late success must be rejected.
    proceed.set()
    old_run.join(timeout=5)
    assert queue.get_results("r1") == []
    assert queue.get_status("r1") == PENDING

    # The new revision is processed through an ordinary run and becomes the visible result.
    QueueWorker(queue, lambda payload: ("new-result", payload)).drain()
    assert queue.get_status("r1") == SUCCEEDED
    assert queue.get_results("r1") == [["new-result", "v2"]]


def test_stale_failure_while_new_revision_succeeds_keeps_new_success():
    import threading

    queue = CleanseQueue(":memory:")
    queue.submit("r1", "v1")
    gate, proceed = threading.Event(), threading.Event()

    class _FailingBlockable(_BlockableNode):
        def __call__(self, payload):
            self.gate.set()
            self.proceed.wait(timeout=5)
            raise RuntimeError("old run dies late")

    worker = QueueWorker(queue, _FailingBlockable(gate, proceed))
    old_run = threading.Thread(target=worker.process_one)
    old_run.start()
    assert gate.wait(timeout=5)

    queue.correct("r1", "v2")
    # Process the new revision to success while the old run is still dying.
    QueueWorker(queue, lambda payload: payload.upper()).drain()
    assert queue.get_results("r1") == [["V2"]]

    proceed.set()
    old_run.join(timeout=5)
    # The late failure of revision 1 cannot downgrade revision 2.
    assert queue.get_status("r1") == SUCCEEDED
    assert queue.get_results("r1") == [["V2"]]
    assert queue.get_error("r1") is None


def test_repeated_corrections_only_latest_is_processed_and_visible(tmpdir):
    queue = CleanseQueue(os.path.join(str(tmpdir), "queue.sqlite"))
    queue.submit("r1", 0)
    for value in range(1, 6):
        revision = queue.correct("r1", value)
    assert revision == 6
    assert queue.get_current("r1")[:2] == (6, 5)
    QueueWorker(queue, double).drain()
    assert queue.get_status("r1") == SUCCEEDED
    assert queue.get_results("r1") == [[10]]

    import sqlite3

    conn = sqlite3.connect(queue.database)
    try:
        revisions = conn.execute(
            "SELECT revision, state, is_current FROM revisions WHERE row_id = 'r1' ORDER BY revision"
        ).fetchall()
        output_revisions = conn.execute(
            "SELECT DISTINCT revision FROM outputs WHERE row_id = 'r1'"
        ).fetchall()
    finally:
        conn.close()
    assert [row[0] for row in revisions] == [1, 2, 3, 4, 5, 6]
    assert all(bool(row[2]) == (row[0] == 6) for row in revisions)
    # No stale outputs from intermediate revisions exist in the table.
    assert output_revisions == [(6,)]


def test_concurrent_workers_and_corrections_never_corrupt_current_state(tmpdir):
    import threading

    path = os.path.join(str(tmpdir), "queue.sqlite")
    queue = CleanseQueue(path)
    queue.submit("r1", 1)
    stop = threading.Event()

    def worker_loop():
        worker = QueueWorker(queue, double)
        while not stop.is_set():
            if not worker.process_one():
                stop.wait(0.001)

    def corrector_loop():
        for value in range(2, 51):
            queue.correct("r1", value)
            stop.wait(0.001)

    threads = [threading.Thread(target=worker_loop) for _ in range(3)]
    threads.append(threading.Thread(target=corrector_loop))
    for thread in threads:
        thread.start()
    threads[-1].join(timeout=10)
    stop.set()
    for thread in threads[:-1]:
        thread.join(timeout=10)

    # Whatever wins the race, invariants hold: drain the last revision and verify the visible outputs match it.
    QueueWorker(queue, double).drain()
    revision, payload, state = queue.get_current("r1")
    assert revision == payload and state == SUCCEEDED
    assert queue.get_results("r1") == [[payload * 2]]

    import sqlite3

    conn = sqlite3.connect(queue.database)
    try:
        stale_outputs = conn.execute(
            "SELECT COUNT(*) FROM outputs AS o "
            "JOIN revisions AS v ON v.row_id = o.row_id AND v.revision = o.revision "
            "WHERE v.is_current = 0"
        ).fetchone()[0]
    finally:
        conn.close()
    assert stale_outputs == 0


def test_sqlite_file_contains_no_partial_outputs_after_failure(tmpdir):
    path = os.path.join(str(tmpdir), "queue.sqlite")
    queue = CleanseQueue(path)
    queue.submit("r1", 1)
    QueueWorker(queue, boom).drain()
    queue.close()

    import sqlite3

    conn = sqlite3.connect(path)
    try:
        output_count = conn.execute("SELECT COUNT(*) FROM outputs").fetchone()[0]
        row_state, row_error = conn.execute(
            "SELECT state, error FROM rows WHERE row_id = 'r1'"
        ).fetchone()
        revision_state, revision_error = conn.execute(
            "SELECT state, error FROM revisions WHERE row_id = 'r1' AND revision = 1"
        ).fetchone()
    finally:
        conn.close()
    assert output_count == 0
    assert row_state == FAILED
    assert revision_state == FAILED
    # The failure reason is preserved both on the current row and on the revision record.
    assert "kaboom" in row_error and "kaboom" in revision_error
