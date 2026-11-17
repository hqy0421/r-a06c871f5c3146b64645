"""
Worker that drains a :class:`~bonobo.q.store.CleanseQueue` through an ordinary linear chain of Bonobo nodes.

Each claimed revision is treated as an independent, one-row graph execution:

* a fresh :class:`bonobo.Graph` and fresh node execution contexts are built for every run, so stateful nodes
  (counters, windows, context processors...) start clean and one revision can never leak state into another;
* the revision's raw row enters the first node exactly like a standard Bonobo input bag and goes through the
  ordinary, linear native node chain -- filters (0 output), ordinary transforms (1 output) and generators
  (N outputs) are handled by Bonobo itself, and duplicate output rows are kept;
* outputs are only collected in memory. The whole chain -- including generator exhaustion -- must finish without any
  node error before the queue is told the revision succeeded; on failure nothing but the raw row and the reason is
  persisted.
"""
import logging

from bonobo.constants import BEGIN, END
from bonobo.execution.contexts.graph import GraphExecutionContext
from bonobo.execution.contexts.node import NodeExecutionContext
from bonobo.q.errors import ChainError
from bonobo.structs.graphs import Graph
from bonobo.structs.tokens import Token
from bonobo.util import get_name


class _OutputBuffer(list):
    """In-memory sink for the chain's last node. Mirrors bonobo.util.testing.QueueList without the pytest dependency."""

    def put(self, item):
        if not isinstance(item, Token):
            self.append(item)


class _CapturingNodeExecutionContext(NodeExecutionContext):
    """Node context that keeps the exc_info of every error the ordinary loop would only log."""

    def error(self, exc_info, *, level=logging.ERROR):
        self.parent.captured_errors.append((self, exc_info))
        super().error(exc_info, level=level)

    def fatal(self, exc_info, *, level=logging.CRITICAL):
        self.parent.captured_errors.append((self, exc_info))
        super().fatal(exc_info, level=level)


class _CapturingGraphExecutionContext(GraphExecutionContext):
    NodeExecutionContextType = _CapturingNodeExecutionContext

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_errors = []


def _format_error(node_context, exc_info):
    _, exc, _ = exc_info
    return "Node {!r} failed: {}: {}".format(get_name(node_context.wrapped), type(exc).__name__, exc)


class QueueWorker:
    """
    Runs pending revisions of a :class:`~bonobo.q.store.CleanseQueue` through a linear node chain.

    :param queue: the :class:`~bonobo.q.store.CleanseQueue` to drain.
    :param nodes: the ordinary Bonobo callables/configurables forming a single linear chain, in execution order.
    :param services: optional services mapping, forwarded to the graph execution contexts.
    :param input_bag_factory: optional callable mapping a persisted raw row to the input bag handed to the first
        node. Defaults to the identity, i.e. the raw row itself enters the chain exactly as stored. Use it to rebuild
        a named :func:`~bonobo.util.bags.BagType` bag from a JSON payload when field-aware nodes (``Filter``,
        ``Rename``...) are used.
    """

    GraphExecutionContextType = _CapturingGraphExecutionContext

    def __init__(self, queue, *nodes, services=None, input_bag_factory=None):
        if not nodes:
            raise ValueError("QueueWorker needs at least one node forming a linear chain.")
        self.queue = queue
        self.nodes = tuple(nodes)
        self.services = services
        self.input_bag_factory = input_bag_factory
        self._validate_linear_chain()

    def _validate_linear_chain(self):
        # Build once to check the topology is a plain, single-input/single-output chain. Every run rebuilds a fresh
        # graph from the same node objects, so what we validate here is what each run gets.
        graph = self._build_graph()
        first = graph.outputs_of(BEGIN)
        if len(first) != 1:
            raise ValueError("The node chain must be linear: BEGIN must connect to exactly one first node.")
        seen = set()
        index = next(iter(first))
        while True:
            if index in seen:
                raise ValueError("The node chain must be linear and acyclic.")
            seen.add(index)
            outputs = graph.outputs_of(index)
            if not outputs:
                break
            if len(outputs) != 1:
                raise ValueError(
                    "The node chain must be linear: node {!r} fans out to {} nodes.".format(
                        get_name(graph[index]), len(outputs)
                    )
                )
            index = next(iter(outputs))
        if len(seen) != len(graph.nodes):
            raise ValueError("The node chain must be linear: disconnected or joined nodes detected.")

    def _build_graph(self):
        graph = Graph()
        graph.add_chain(*self.nodes)
        return graph

    # ------------------------------------------------------------------ one run
    def _run_revision(self, payload):
        """
        Run one raw row through a fresh linear chain execution.

        Returns the ordered list of output bags (0, 1 or N of them, duplicates kept) or raises :class:`ChainError`
        if any node (including generator teardown) failed.
        """
        graph = self._build_graph()
        buffer = _OutputBuffer()
        if self.input_bag_factory is not None:
            payload = self.input_bag_factory(payload)
        context = self.GraphExecutionContextType(graph, services=self.services)
        started = False
        try:
            try:
                context.start()
                started = True
            except Exception as exc:
                raise ChainError(
                    "Chain failed while starting: {}: {}".format(type(exc).__name__, exc), cause=exc
                ) from exc
            # The last node's outputs are the in-memory sink; nothing is written to SQLite during the run.
            context[-1].outputs = [buffer]
            context.write(BEGIN, payload, END)
            context.loop()
        except ChainError:
            raise
        except Exception as exc:  # defensive: the node loop normally captures node errors
            raise ChainError(
                "Chain failed while processing: {}: {}".format(type(exc).__name__, exc), cause=exc
            ) from exc
        finally:
            if started:
                try:
                    context.stop()
                except Exception as exc:
                    raise ChainError(
                        "Chain failed while stopping: {}: {}".format(type(exc).__name__, exc), cause=exc
                    ) from exc

        if context.captured_errors:
            node_context, exc_info = context.captured_errors[0]
            raise ChainError(_format_error(node_context, exc_info), cause=exc_info[1])
        if any(node.xstatus for node in context.nodes):
            node_context = next(node for node in context.nodes if node.xstatus)
            raise ChainError("Node {!r} ended in a defunct state.".format(get_name(node_context.wrapped)))
        return list(buffer)

    def process_one(self):
        """
        Claim and process one pending revision.

        Returns ``True`` if a revision was processed, ``False`` if there was no pending work. Completion is recorded
        through the queue's guarded :meth:`~bonobo.q.store.CleanseQueue.complete`, which rejects stale revisions, so
        a late result of a superseded run never touches the current revision.
        """
        claimed = self.queue.claim()
        if claimed is None:
            return False
        row_id, revision, payload = claimed
        try:
            outputs = self._run_revision(payload)
        except ChainError as exc:
            self.queue.complete(row_id, revision, success=False, error=str(exc))
            return True
        except Exception as exc:  # pragma: no cover - safety net around unexpected worker bugs
            reason = "{}: {}".format(type(exc).__name__, exc)
            self.queue.complete(row_id, revision, success=False, error=reason)
            return True
        self.queue.complete(row_id, revision, success=True, outputs=outputs)
        return True

    def drain(self, *, max_tasks=None):
        """
        Process pending revisions until the queue has no more pending work (or ``max_tasks`` were processed).

        Repeated calls are idempotent: a queue without pending work is simply untouched.
        """
        processed = 0
        while max_tasks is None or processed < max_tasks:
            if not self.process_one():
                break
            processed += 1
        return processed
