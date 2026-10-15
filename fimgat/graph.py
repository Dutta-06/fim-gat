"""Graph container used throughout FIM-GAT.

Edges are directed in the direction influence flows (src -> dst) and carry an
Independent Cascade activation probability. Undirected inputs are stored as
two directed edges with independent coins, which is the standard IC model on
an undirected graph.
"""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np


@dataclass
class FIMGraph:
    name: str
    n: int
    src: np.ndarray            # int64 (m,)
    dst: np.ndarray            # int64 (m,)
    prob: np.ndarray           # float64 (m,) IC activation probability
    groups: np.ndarray         # int64 (n,) sensitive attribute, values in [0, n_groups)
    group_names: list
    meta: dict = field(default_factory=dict)

    @property
    def m(self):
        return len(self.src)

    @property
    def n_groups(self):
        return len(self.group_names)

    @cached_property
    def group_sizes(self):
        return np.bincount(self.groups, minlength=self.n_groups)

    @cached_property
    def present_groups(self):
        """Mask of groups that actually occur in this graph."""
        return self.group_sizes > 0

    def _csr(self, key, val):
        order = np.argsort(key, kind="stable")
        indptr = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(np.bincount(key, minlength=self.n), out=indptr[1:])
        return indptr, val[order].astype(np.int64), self.prob[order].astype(np.float64)

    @cached_property
    def in_csr(self):
        """For each node v: incoming sources u and p(u->v). Used for reverse-reachable sampling."""
        return self._csr(self.dst, self.src)

    @cached_property
    def out_csr(self):
        """For each node u: outgoing targets v and p(u->v). Used for forward simulation."""
        return self._csr(self.src, self.dst)

    @cached_property
    def out_degree(self):
        return np.bincount(self.src, minlength=self.n)

    @cached_property
    def in_degree(self):
        return np.bincount(self.dst, minlength=self.n)

    def with_groups(self, groups, group_names, suffix=None):
        """Same topology, different sensitive attribute."""
        return FIMGraph(self.name if suffix is None else f"{self.name}:{suffix}", self.n,
                        self.src, self.dst, self.prob, np.asarray(groups, dtype=np.int64),
                        list(group_names), dict(self.meta))

    def summary(self):
        sizes = ", ".join(f"{nm}={s}" for nm, s in zip(self.group_names, self.group_sizes) if s > 0)
        return f"{self.name}: n={self.n} m={self.m} groups[{sizes}]"


def from_edges(name, n, edges, groups, group_names, prob=0.1, undirected=True,
               weighted_cascade=False, meta=None):
    """Build an FIMGraph from an (E, 2) edge array.

    prob:              constant IC probability (ignored if weighted_cascade)
    undirected:        add both directions
    weighted_cascade:  p(u->v) = 1 / in_degree(v)  (Kempe et al. 2003)
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if undirected:
        edges = np.concatenate([edges, edges[:, ::-1]])
    code = np.unique(edges[:, 0] * np.int64(n) + edges[:, 1])      # 1-D dedup: fast at 10^8 edges
    src, dst = code // n, code % n
    if weighted_cascade:
        indeg = np.bincount(dst, minlength=n)
        p = 1.0 / indeg[dst]
    else:
        p = np.full(len(src), float(prob))
    return FIMGraph(name, n, src, dst, p, np.asarray(groups, dtype=np.int64),
                    list(group_names), meta or {})
