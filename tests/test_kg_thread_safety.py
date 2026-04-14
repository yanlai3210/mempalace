"""TDD: KnowledgeGraph read methods must hold self._lock.

query_relationship(), timeline(), and stats() access the shared
sqlite3.Connection without locking. Concurrent reads+writes can
corrupt data.

Written BEFORE the fix.
"""

import inspect

from mempalace.knowledge_graph import KnowledgeGraph


class TestKGThreadSafety:
    """Every method that touches self._conn must hold self._lock."""

    def test_query_relationship_holds_lock(self):
        src = inspect.getsource(KnowledgeGraph.query_relationship)
        assert "self._lock" in src, (
            "query_relationship() does not acquire self._lock. "
            "Concurrent reads+writes can corrupt sqlite data."
        )

    def test_timeline_holds_lock(self):
        src = inspect.getsource(KnowledgeGraph.timeline)
        assert "self._lock" in src, (
            "timeline() does not acquire self._lock. "
            "Concurrent reads+writes can corrupt sqlite data."
        )

    def test_stats_holds_lock(self):
        src = inspect.getsource(KnowledgeGraph.stats)
        assert "self._lock" in src, (
            "stats() does not acquire self._lock. "
            "Concurrent reads+writes can corrupt sqlite data."
        )

    def test_all_methods_with_conn_hold_lock(self):
        """Every method that references self._conn must also reference self._lock."""
        for name, method in inspect.getmembers(KnowledgeGraph, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue
            src = inspect.getsource(method)
            if "self._conn" in src or "self.conn" in src:
                assert "self._lock" in src, (
                    f"KnowledgeGraph.{name}() accesses the connection "
                    f"without holding self._lock."
                )
