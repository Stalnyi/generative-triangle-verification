"""
VERIFICATION of Section 2:
DAG interpretation of the generative triangle (sec:dag_interpretation).

Source file:
    2_dag_interpretation.tex

This script verifies the mathematical content of the DAG interpretation as a
standalone graph-theoretic layer over the already verified generative triangle.
It uses the established definitions of levels L_n, the generation rule F, and
the causal reachability relation as dependencies, then verifies that the graph

    G = (V,E),   V=T,   E={(x,F(x,sigma)): x in T, sigma in S}

is a directed acyclic graph with the stated degree, layering, past-cone, and
future-cone properties.

Verified content
----------------
1. Edge definition:
       x -> y  iff  y=F(x,sigma)
   for a unique sigma in S when x is fixed.

2. Strict layering:
       x in L_n and x -> y  implies  y in L_{n+1}.
   No edge stays on the same level, goes backward, or skips a level.

3. DAG property:
   the level index t is a strict rank function on edges, hence every directed
   path of positive length strictly increases t and no directed cycle exists.

4. Local finiteness:
       outdegree(x)=s
   for every checked x in T, and
       indegree(x)=0 for x in A,
       indegree(x)=1 for x in T\\A.

5. Edge bijection on finite truncations:
       (L_0 union ... union L_N) x S
       -> edge targets L_1 union ... union L_{N+1}
   is collision-free and surjective on the target vertices.  This verifies
   that no two different immediate parents create the same noninitial vertex.

6. Past cone:
   for x in L_n and 0<=j<=n,
       |C^-(x) cap L_j| = 1.
   The past cone is a unique linear ancestor chain.

7. Future cone:
   for x in L_n and d>=0,
       |C^+(x) cap L_{n+d}| = s^d.
   Moreover, descendants at depth d are in bijection with S^d, and each
   descendant has a unique path from x.

8. Complete s-ary future tree:
   every node in the depth-bounded future subtree of x has exactly s children
   inside the next future depth, and nodes at the same depth have disjoint
   child sets.

9. Graph reachability equals causal coordinate reachability in the checked
   finite windows.

10. Coordinate cone compatibility for every edge and finite path:
       0 <= Delta r <= Delta t.
   For one edge Delta t=1, the bound is verified by exact integer gaps.

11. Negative guards:
   - a time-ordered pair need not be an edge;
   - a time-ordered pair need not be causally reachable;
   - an edge with a nonmatching target is rejected;
   - an edge skipping a level is rejected;
   - a reversed edge is rejected;
   - invalid parameters, states, levels, and vertices are rejected.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import log
from typing import Callable, Sequence

from sympy import simplify, symbols


class DomainError(ValueError):
    """Raised when a mathematical domain condition is violated."""


def expect_raises(expected_exception: type[BaseException] | tuple[type[BaseException], ...],
                  fn: Callable[[], object]) -> None:
    try:
        fn()
    except expected_exception:
        return
    raise AssertionError(f"Expected {expected_exception} was not raised")


def require_int(value: object, name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def require_positive_int(value: object, name: str) -> int:
    value = require_int(value, name)
    if value < 1:
        raise DomainError(f"{name} must be a positive integer")
    return value


def require_nonnegative_int(value: object, name: str) -> int:
    value = require_int(value, name)
    if value < 0:
        raise DomainError(f"{name} must be a nonnegative integer")
    return value


def require_at_least(value: object, lower: int, name: str) -> int:
    value = require_int(value, name)
    if value < lower:
        raise DomainError(f"{name} must be at least {lower}")
    return value


@dataclass(frozen=True, slots=True)
class Edge:
    source: int
    target: int
    source_level: int
    state: int


@dataclass(frozen=True, slots=True)
class Model:
    m: int
    k: int
    s: int

    def __post_init__(self) -> None:
        require_positive_int(self.m, "m")
        require_at_least(self.k, 2, "k")
        require_at_least(self.s, 2, "s")

    @property
    def cA(self) -> int:
        return (self.s - 1) * self.m + 1 - self.k

    @property
    def A(self) -> tuple[int, ...]:
        return tuple(range(self.m, self.m + self.k))

    @property
    def S(self) -> tuple[int, ...]:
        return tuple(range(1, self.s + 1))

    def validate_state(self, sigma: object) -> int:
        sigma = require_int(sigma, "sigma")
        if not (1 <= sigma <= self.s):
            raise DomainError("sigma must belong to S={1,...,s}")
        return sigma

    def level_start(self, n: object) -> int:
        n = require_nonnegative_int(n, "n")
        numerator = self.k * (self.s**n - 1)
        denominator = self.s - 1
        assert numerator % denominator == 0
        return self.m + numerator // denominator

    def level_size(self, n: object) -> int:
        n = require_nonnegative_int(n, "n")
        return self.k * self.s**n

    def level_end(self, n: object) -> int:
        n = require_nonnegative_int(n, "n")
        return self.level_start(n) + self.level_size(n) - 1

    def level(self, n: object) -> tuple[int, ...]:
        n = require_nonnegative_int(n, "n")
        return tuple(range(self.level_start(n), self.level_end(n) + 1))

    def contains_level_point(self, x: object, n: object) -> bool:
        if not isinstance(x, int):
            return False
        n = require_nonnegative_int(n, "n")
        return self.level_start(n) <= x <= self.level_end(n)

    def require_level_point(self, x: object, n: object) -> int:
        x = require_positive_int(x, "x")
        if not self.contains_level_point(x, n):
            raise DomainError("x does not belong to the declared level")
        return x

    def position(self, x: object, n: object) -> int:
        x = self.require_level_point(x, n)
        n = require_nonnegative_int(n, "n")
        return x - self.level_start(n)

    def point_from_position(self, u: object, n: object) -> int:
        u = require_nonnegative_int(u, "u")
        n = require_nonnegative_int(n, "n")
        if u >= self.level_size(n):
            raise DomainError("u is outside the positional range of L_n")
        return self.level_start(n) + u

    def F(self, x: object, sigma: object) -> int:
        x = require_positive_int(x, "x")
        sigma = self.validate_state(sigma)
        return self.s * x + sigma - self.cA

    def edge(self, x: object, n: object, sigma: object) -> Edge:
        n = require_nonnegative_int(n, "n")
        x = self.require_level_point(x, n)
        sigma = self.validate_state(sigma)
        y = self.F(x, sigma)
        assert self.contains_level_point(y, n + 1)
        return Edge(source=x, target=y, source_level=n, state=sigma)

    def is_edge(self, x: object, nx: object, y: object, ny: object) -> bool:
        nx = require_nonnegative_int(nx, "nx")
        ny = require_nonnegative_int(ny, "ny")
        if ny != nx + 1:
            return False
        x = self.require_level_point(x, nx)
        y = self.require_level_point(y, ny)
        for sigma in self.S:
            if self.F(x, sigma) == y:
                return True
        return False

    def local_preimage(self, y: object, previous_level: object) -> tuple[int, int]:
        previous_level = require_nonnegative_int(previous_level, "previous_level")
        y = self.require_level_point(y, previous_level + 1)
        child_position = self.position(y, previous_level + 1)
        parent_position = child_position // self.s
        sigma = child_position % self.s + 1
        parent = self.point_from_position(parent_position, previous_level)
        assert self.F(parent, sigma) == y
        return parent, sigma

    def immediate_predecessor(self, y: object, level_of_y: object) -> tuple[int, int]:
        level_of_y = require_nonnegative_int(level_of_y, "level_of_y")
        if level_of_y == 0:
            raise DomainError("level-zero vertices have indegree zero")
        return self.local_preimage(y, level_of_y - 1)

    def finite_vertices(self, max_level: object) -> tuple[int, ...]:
        max_level = require_nonnegative_int(max_level, "max_level")
        vertices: list[int] = []
        for n in range(max_level + 1):
            vertices.extend(self.level(n))
        return tuple(vertices)

    def finite_edges_from_levels(self, max_source_level: object) -> tuple[Edge, ...]:
        max_source_level = require_nonnegative_int(max_source_level, "max_source_level")
        edges: list[Edge] = []
        for n in range(max_source_level + 1):
            for x in self.level(n):
                for sigma in self.S:
                    edges.append(self.edge(x, n, sigma))
        return tuple(edges)

    def locate_level(self, x: int, max_level: int) -> int:
        for n in range(max_level + 1):
            if self.contains_level_point(x, n):
                return n
        raise DomainError("vertex is outside the finite level window")

    def iterate(self, x: object, states: Sequence[int]) -> int:
        current = require_positive_int(x, "x")
        if not isinstance(states, tuple):
            raise TypeError("state sequence must be a tuple")
        for sigma in states:
            current = self.F(current, sigma)
        return current

    def digit_from_states(self, states: Sequence[int]) -> int:
        if not isinstance(states, tuple):
            raise TypeError("state sequence must be a tuple")
        depth = len(states)
        total = 0
        for index, sigma in enumerate(states):
            sigma = self.validate_state(sigma)
            total += (sigma - 1) * self.s ** (depth - index - 1)
        return total

    def states_from_digit(self, D: object, depth: object) -> tuple[int, ...]:
        D = require_nonnegative_int(D, "D")
        depth = require_nonnegative_int(depth, "depth")
        if D >= self.s**depth:
            raise DomainError("D is outside the depth range")
        states: list[int] = []
        remaining = D
        for power in range(depth - 1, -1, -1):
            base_power = self.s**power
            digit = remaining // base_power
            remaining %= base_power
            states.append(digit + 1)
        return tuple(states)

    def causal_leq(self, x: object, nx: object, y: object, ny: object) -> bool:
        nx = require_nonnegative_int(nx, "nx")
        ny = require_nonnegative_int(ny, "ny")
        if nx > ny:
            return False
        x = self.require_level_point(x, nx)
        y = self.require_level_point(y, ny)
        depth = ny - nx
        ux = self.position(x, nx)
        uy = self.position(y, ny)
        lower = self.s**depth * ux
        upper = lower + self.s**depth - 1
        return lower <= uy <= upper

    def future_slice(self, x: object, n: object, depth: object) -> tuple[int, ...]:
        n = require_nonnegative_int(n, "n")
        depth = require_nonnegative_int(depth, "depth")
        x = self.require_level_point(x, n)
        ux = self.position(x, n)
        return tuple(
            self.point_from_position(self.s**depth * ux + D, n + depth)
            for D in range(self.s**depth)
        )

    def ancestor_at_level(self, x: object, n: object, target_level: object) -> int:
        n = require_nonnegative_int(n, "n")
        target_level = require_nonnegative_int(target_level, "target_level")
        if target_level > n:
            raise DomainError("target_level cannot exceed the vertex level")
        current = self.require_level_point(x, n)
        current_level = n
        while current_level > target_level:
            child = current
            parent, sigma = self.immediate_predecessor(child, current_level)
            assert self.F(parent, sigma) == child
            current = parent
            current_level -= 1
        return current

    def ancestor_chain(self, x: object, n: object) -> tuple[int, ...]:
        n = require_nonnegative_int(n, "n")
        return tuple(self.ancestor_at_level(x, n, level) for level in range(n + 1))

    def r_coordinate(self, x: object, n: object) -> float:
        u = self.position(x, n)
        return log(u + 1, self.s)


def verify_symbolic_edge_layering_and_cone_bounds() -> None:
    print("\n=== Symbolic verification of DAG edge formulas and cone bounds ===")

    u, sigma, s = symbols("u sigma s", integer=True, positive=True)

    child_u = s * u + sigma - 1

    # One-edge lower and upper logarithmic cone bounds are equivalent to exact
    # nonnegative integer gaps:
    #   child_u + 1 >= u + 1
    #   child_u + 1 <= s(u + 1)
    lower_gap = simplify((child_u + 1) - (u + 1))
    upper_gap = simplify(s * (u + 1) - (child_u + 1))

    assert simplify(lower_gap - ((s - 1) * u + sigma - 1)) == 0
    assert simplify(upper_gap - (s - sigma)) == 0

    # Composing a depth-a path with a depth-b path gives the standard digit
    # composition formula.
    D1, D2, a, b = symbols("D1 D2 a b", integer=True, positive=True)
    intermediate = s**a * u + D1
    final = s**b * intermediate + D2
    composed = s ** (a + b) * u + (s**b * D1 + D2)
    assert simplify(final - composed) == 0

    print("[OK] One-edge coordinate gaps imply 0 <= Delta r <= 1")
    print("[OK] Multi-edge digit composition is symbolic")


def verify_edge_definition_layering_and_degrees() -> None:
    print("\n=== Verification of edge definition, layering, and local degrees ===")

    checked_edges = 0
    checked_outdegrees = 0
    checked_indegrees = 0
    checked_wrong_level_edges = 0

    for model in (
        Model(1, 2, 2),
        Model(2, 3, 2),
        Model(4, 2, 3),
        Model(5, 4, 3),
    ):
        for n in range(0, 7):
            for x in model.level(n):
                outgoing = tuple(model.edge(x, n, sigma) for sigma in model.S)
                checked_outdegrees += 1
                assert len(outgoing) == model.s
                assert len({edge.target for edge in outgoing}) == model.s
                assert {edge.state for edge in outgoing} == set(model.S)

                for edge in outgoing:
                    checked_edges += 1
                    assert edge.source == x
                    assert edge.source_level == n
                    assert model.contains_level_point(edge.target, n + 1)
                    assert model.is_edge(edge.source, n, edge.target, n + 1)

                    # Same-level, backward, and skip-level interpretations are not edges.
                    assert not model.is_edge(edge.source, n, edge.target, n)
                    assert not model.is_edge(edge.target, n + 1, edge.source, n)
                    if n + 2 <= 9:
                        assert not model.is_edge(edge.source, n, edge.target, n + 2)
                    checked_wrong_level_edges += 3

            for y in model.level(n):
                checked_indegrees += 1
                if n == 0:
                    expect_raises(DomainError, lambda y=y, model=model: model.immediate_predecessor(y, 0))
                else:
                    parent, sigma = model.immediate_predecessor(y, n)
                    assert model.contains_level_point(parent, n - 1)
                    assert sigma in model.S
                    assert model.F(parent, sigma) == y

                    all_parents = [
                        (candidate, candidate_sigma)
                        for candidate in model.level(n - 1)
                        for candidate_sigma in model.S
                        if model.F(candidate, candidate_sigma) == y
                    ]
                    assert all_parents == [(parent, sigma)]

    print(f"[OK] Checked {checked_edges} graph edges")
    print(f"[OK] Checked {checked_outdegrees} vertices with outdegree exactly s")
    print(f"[OK] Checked {checked_indegrees} indegree cases")
    print(f"[OK] Checked {checked_wrong_level_edges} same/backward/skip-level non-edge guards")


def verify_dag_acyclicity_and_finite_edge_bijection() -> None:
    print("\n=== Verification of DAG acyclicity and finite edge bijection ===")

    checked_finite_windows = 0
    checked_edges = 0
    checked_targets = 0
    checked_topological_edges = 0

    for model in (
        Model(1, 2, 2),
        Model(2, 3, 2),
        Model(3, 2, 3),
        Model(5, 3, 4),
    ):
        for max_source_level in range(0, 6):
            checked_finite_windows += 1
            edges = model.finite_edges_from_levels(max_source_level)
            target_vertices = tuple(
                vertex
                for level in range(1, max_source_level + 2)
                for vertex in model.level(level)
            )

            image_to_edge: dict[int, Edge] = {}
            for edge in edges:
                checked_edges += 1
                assert edge.target not in image_to_edge
                image_to_edge[edge.target] = edge

                source_level = edge.source_level
                target_level = model.locate_level(edge.target, max_source_level + 1)
                checked_topological_edges += 1
                assert target_level == source_level + 1
                assert target_level > source_level

            assert set(image_to_edge) == set(target_vertices)
            checked_targets += len(target_vertices)

            # A finite DAG with a strict rank function cannot have a directed
            # cycle.  We additionally verify that every edge points forward in
            # the natural topological ordering by level, then by integer value.
            topological_index = {
                vertex: (level, index)
                for level in range(max_source_level + 2)
                for index, vertex in enumerate(model.level(level))
            }
            for edge in edges:
                assert topological_index[edge.source] < topological_index[edge.target]

    print(f"[OK] Checked {checked_finite_windows} finite DAG windows")
    print(f"[OK] Checked {checked_edges} finite edges without target collisions")
    print(f"[OK] Checked {checked_targets} target vertices covered exactly once")
    print(f"[OK] Checked {checked_topological_edges} strict rank-increasing edges")


def verify_reachability_equals_causal_order_and_path_uniqueness() -> None:
    print("\n=== Verification of path reachability, causal order, and unique paths ===")

    checked_paths = 0
    checked_inverse_paths = 0
    checked_nonreachable = 0

    for model in (
        Model(1, 2, 2),
        Model(2, 3, 2),
        Model(3, 2, 3),
    ):
        for n in range(0, 5):
            sample_positions = sorted({0, model.level_size(n) // 2, model.level_size(n) - 1})
            for u in sample_positions:
                x = model.point_from_position(u, n)

                # Empty path.
                assert model.iterate(x, tuple()) == x
                assert model.causal_leq(x, n, x, n)

                for depth in range(1, 5):
                    image_to_states: dict[int, tuple[int, ...]] = {}

                    for states in product(model.S, repeat=depth):
                        states = tuple(states)
                        y = model.iterate(x, states)
                        checked_paths += 1

                        assert model.contains_level_point(y, n + depth)
                        assert model.causal_leq(x, n, y, n + depth)

                        if y in image_to_states:
                            raise AssertionError("two distinct state paths reached the same descendant")
                        image_to_states[y] = states

                        D = model.digit_from_states(states)
                        assert model.states_from_digit(D, depth) == states
                        assert model.position(y, n + depth) == model.s**depth * model.position(x, n) + D

                    future = model.future_slice(x, n, depth)
                    assert set(image_to_states) == set(future)

                    for y in future:
                        checked_inverse_paths += 1
                        chain = model.ancestor_chain(y, n + depth)
                        assert chain[n] == x
                        recovered_states = image_to_states[y]
                        rebuilt = x
                        for offset, sigma in enumerate(recovered_states, start=1):
                            rebuilt = model.F(rebuilt, sigma)
                            assert rebuilt == chain[n + offset]
                        assert rebuilt == y

                    # Adjacent nonreachable witnesses just outside the descendant interval.
                    ux = model.position(x, n)
                    low = model.s**depth * ux
                    high = low + model.s**depth - 1
                    level_size = model.level_size(n + depth)

                    if low > 0:
                        y_bad = model.point_from_position(low - 1, n + depth)
                        assert not model.causal_leq(x, n, y_bad, n + depth)
                        checked_nonreachable += 1
                    if high + 1 < level_size:
                        y_bad = model.point_from_position(high + 1, n + depth)
                        assert not model.causal_leq(x, n, y_bad, n + depth)
                        checked_nonreachable += 1

    print(f"[OK] Checked {checked_paths} generated paths")
    print(f"[OK] Checked {checked_inverse_paths} unique inverse paths from ancestor chains")
    print(f"[OK] Checked {checked_nonreachable} adjacent nonreachable witnesses")


def verify_future_tree_and_linear_past() -> None:
    print("\n=== Verification of complete future tree and linear past ===")

    checked_future_slices = 0
    checked_future_points = 0
    checked_child_disjointness = 0
    checked_past_slices = 0
    checked_past_chains = 0

    for model in (
        Model(1, 2, 2),
        Model(2, 4, 2),
        Model(5, 3, 3),
        Model(3, 2, 4),
    ):
        for n in range(0, 6):
            sample_positions = sorted({0, model.level_size(n) // 2, model.level_size(n) - 1})
            for u in sample_positions:
                x = model.point_from_position(u, n)

                for depth in range(0, 5):
                    future = model.future_slice(x, n, depth)
                    checked_future_slices += 1
                    checked_future_points += len(future)

                    assert len(future) == model.s**depth
                    assert len(set(future)) == len(future)
                    assert all(model.causal_leq(x, n, y, n + depth) for y in future)

                    if depth < 4:
                        child_sets = []
                        for y in future:
                            children = tuple(edge.target for edge in (model.edge(y, n + depth, sigma) for sigma in model.S))
                            assert len(children) == model.s
                            assert len(set(children)) == model.s
                            child_sets.append(set(children))

                        union_children = set().union(*child_sets) if child_sets else set()
                        next_future = set(model.future_slice(x, n, depth + 1))
                        assert union_children == next_future

                        for left_index, left_set in enumerate(child_sets):
                            for right_set in child_sets[left_index + 1:]:
                                assert left_set.isdisjoint(right_set)
                                checked_child_disjointness += 1

                chain = model.ancestor_chain(x, n)
                checked_past_chains += 1
                assert len(chain) == n + 1
                assert chain[-1] == x
                assert chain[0] in model.A

                for level in range(0, n + 1):
                    ancestor = chain[level]
                    checked_past_slices += 1
                    assert model.contains_level_point(ancestor, level)
                    assert model.causal_leq(ancestor, level, x, n)
                    all_past_candidates = [
                        candidate
                        for candidate in model.level(level)
                        if model.causal_leq(candidate, level, x, n)
                    ]
                    assert all_past_candidates == [ancestor]

    print(f"[OK] Checked {checked_future_slices} future cone slices")
    print(f"[OK] Checked {checked_future_points} future cone vertices")
    print(f"[OK] Checked {checked_child_disjointness} disjoint child-set comparisons")
    print(f"[OK] Checked {checked_past_slices} singleton past slices")
    print(f"[OK] Checked {checked_past_chains} linear past chains")


def verify_coordinate_cone_compatibility_for_edges_and_paths() -> None:
    print("\n=== Verification of coordinate cone compatibility for DAG paths ===")

    checked_edges = 0
    checked_paths = 0

    for model in (
        Model(1, 2, 2),
        Model(2, 3, 2),
        Model(3, 2, 3),
    ):
        for n in range(0, 6):
            for x in model.level(n)[:min(12, len(model.level(n)))]:
                ux = model.position(x, n)
                rx = model.r_coordinate(x, n)

                for sigma in model.S:
                    edge = model.edge(x, n, sigma)
                    uy = model.position(edge.target, n + 1)
                    ry = model.r_coordinate(edge.target, n + 1)
                    checked_edges += 1

                    assert uy == model.s * ux + sigma - 1

                    lower_gap = (model.s - 1) * ux + sigma - 1
                    upper_gap = model.s - sigma
                    assert lower_gap >= 0
                    assert upper_gap >= 0
                    assert ry >= rx - 1e-12
                    assert ry - rx <= 1 + 1e-12

                for depth in range(0, 4):
                    for y in model.future_slice(x, n, depth):
                        checked_paths += 1
                        ry = model.r_coordinate(y, n + depth)
                        assert ry >= rx - 1e-12
                        assert ry - rx <= depth + 1e-12

    print(f"[OK] Checked {checked_edges} one-edge coordinate cone bounds")
    print(f"[OK] Checked {checked_paths} finite-path coordinate cone bounds")


def verify_negative_guards_and_invalid_inputs() -> None:
    print("\n=== Negative guards and invalid-domain tests ===")

    model = Model(1, 2, 2)

    # Time-ordered but not an edge.
    x = model.point_from_position(0, 1)
    y_later = model.point_from_position(0, 3)
    assert 1 < 3
    assert not model.is_edge(x, 1, y_later, 3)

    # Time-ordered but not causally reachable.
    x2 = model.point_from_position(1, 1)
    y_noncausal = model.point_from_position(0, 3)
    assert 1 < 3
    assert not model.causal_leq(x2, 1, y_noncausal, 3)

    # Invalid edge target: choose a next-level vertex outside the descendant range.
    # outgoing child block of the source.  A naive target+1 is not valid here,
    # because it may simply be another legitimate child with a different state.
    source = model.point_from_position(1, 1)
    source_u = model.position(source, 1)
    child_low = model.s * source_u
    child_high = child_low + model.s - 1
    corrupted_position = child_high + 1
    assert corrupted_position < model.level_size(2)
    corrupted_target = model.point_from_position(corrupted_position, 2)
    edge = model.edge(source, 1, 1)
    assert model.contains_level_point(corrupted_target, 2)
    assert not model.is_edge(source, 1, corrupted_target, 2)

    # Reversed edge and same-level edge are rejected.
    assert not model.is_edge(edge.target, edge.source_level + 1, edge.source, edge.source_level)
    assert not model.is_edge(edge.source, edge.source_level, edge.target, edge.source_level)

    expect_raises(DomainError, lambda: Model(0, 2, 2))
    expect_raises(DomainError, lambda: Model(1, 1, 2))
    expect_raises(DomainError, lambda: Model(1, 2, 1))
    expect_raises(TypeError, lambda: Model(1.0, 2, 2))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: Model(1, 2.0, 2))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: Model(1, 2, 2.0))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: model.level(-1))
    expect_raises(TypeError, lambda: model.level(1.5))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: model.require_level_point(1, 1))
    expect_raises(DomainError, lambda: model.point_from_position(-1, 0))
    expect_raises(DomainError, lambda: model.point_from_position(model.level_size(2), 2))

    expect_raises(DomainError, lambda: model.F(1, 0))
    expect_raises(DomainError, lambda: model.F(1, 3))
    expect_raises(TypeError, lambda: model.F(1, 1.0))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: model.edge(1, 1, 1))
    expect_raises(DomainError, lambda: model.edge(model.point_from_position(0, 1), 1, 0))
    expect_raises(DomainError, lambda: model.immediate_predecessor(model.m, 0))
    expect_raises(DomainError, lambda: model.local_preimage(model.m, 0))
    expect_raises(DomainError, lambda: model.states_from_digit(-1, 2))
    expect_raises(DomainError, lambda: model.states_from_digit(4, 2))
    expect_raises(TypeError, lambda: model.iterate(1, [1, 2]))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: model.iterate(1, (1, 3)))
    expect_raises(DomainError, lambda: model.ancestor_at_level(model.m, 0, 1))

    print("[OK] Time-ordered non-edge and noncausal witnesses are rejected")
    print("[OK] Nonmatching, reversed, same-level, and skip-level edge interpretations are rejected")
    print("[OK] Invalid parameters, levels, vertices, states, and path data are rejected")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of DAG interpretation (sec:dag_interpretation) ===")
    verify_symbolic_edge_layering_and_cone_bounds()
    verify_edge_definition_layering_and_degrees()
    verify_dag_acyclicity_and_finite_edge_bijection()
    verify_reachability_equals_causal_order_and_path_uniqueness()
    verify_future_tree_and_linear_past()
    verify_coordinate_cone_compatibility_for_edges_and_paths()
    verify_negative_guards_and_invalid_inputs()
    print("\n=== DAG interpretation verification completed successfully ===")


if __name__ == "__main__":
    main()
