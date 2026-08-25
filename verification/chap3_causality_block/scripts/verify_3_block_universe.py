"""
VERIFICATION of Section 3:
Static block structure of spacetime (sec:block-universe).

Source file:
    3_block_universe.tex

This script verifies the mathematical core of the block-universe section over
the already verified generative triangle, partial order, and DAG
interpretation.  It does not attempt to "verify" philosophical exposition.
Instead, it checks the precise mathematical statements that make the
interpretation legitimate:

    T = disjoint union of all L_n,
    all levels are fixed by (A,S,F),
    finite truncations are nested prefixes of one static object,
    causality is an internal relation on T,
    the level index is a foliation parameter,
    one-step edges impose a universal causal speed limit,
    and no extra dynamical law is needed once (A,S,F) is fixed.


Verified content
----------------
1. Static object:
       T = disjoint_union_{n>=0} L_n.
   In finite windows:
       T_N = disjoint_union_{n=0}^N L_n = {m,...,max L_N}.
   The inclusions T_N subset T_{N+1} are exact nested prefix inclusions.

2. Full determination by fixed data:
       (A,S,F) uniquely determines every L_n.
   Recomputing levels by closed formula, recursive generation, or genealogy
   gives the same finite truncations.

3. Static extension:
   increasing N does not modify any already existing L_0,...,L_N; it only
   reveals the next fixed layer L_{N+1}.

4. Transition operator discipline:
   the level transition is correctly represented as the full set image
       Phi_n(L_n) = F(L_n x S) = L_{n+1}.
   It is not a single-valued map L_n -> L_{n+1} covering the whole next level
   when s>=2, because |L_{n+1}|=s|L_n|>|L_n|.

5. Internal causality:
   x <=_c y is a relation defined inside the already fixed T, not an external
   update rule.  Coordinate reachability, DAG paths, and ancestor decoding
   agree in finite windows.

6. Complete past and future:
   every x in L_n has a unique finite past chain to L_0, and for every d>=0
       |C^+(x) cap L_{n+d}| = s^d.
   Thus all finite future branches of any checked depth are already fixed by
   the same data.

7. Foliation:
   t(x)=n iff x in L_n.  The foliation by levels is unique because the levels
   are disjoint adjacent intervals.

8. Internal observer/path distinction:
   an observer-like path choosing one successor per level is a chain with one
   vertex per level.  It is a proper subset of the full block, not a generator
   that creates T.

9. Universal causal speed limit:
   every edge increases time by exactly one.  A path from level n to level n+d
   has exactly d edges; no edge skips a level and no path has length shorter
   than the level difference.  In logarithmic coordinates the checked paths
   satisfy
       0 <= Delta r <= Delta t.

10. Shift-isomorphic uniqueness:
   shifting A by delta gives an isomorphic static block:
       c(A+delta)=c(A)+(s-1)delta,
       L_n(A+delta)=L_n(A)+delta,
       causality and ancestor signatures are preserved.

11. Negative guards:
   - time order alone is not causality;
   - no single-valued Phi:L_n->L_{n+1} can cover L_{n+1} for s>=2;
   - changing c changes the block and misaligns the declared level structure;
   - adding skip-level edges violates the DAG rank condition;
   - an internal path does not equal the whole block;
   - invalid parameters, levels, states, vertices, and noncausal requests are
     rejected.

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

    def level_start_by_sum(self, n: object) -> int:
        n = require_nonnegative_int(n, "n")
        return self.m + self.k * sum(self.s**j for j in range(n))

    def level_size(self, n: object) -> int:
        n = require_nonnegative_int(n, "n")
        return self.k * self.s**n

    def level_end(self, n: object) -> int:
        n = require_nonnegative_int(n, "n")
        return self.level_start(n) + self.level_size(n) - 1

    def level(self, n: object) -> tuple[int, ...]:
        n = require_nonnegative_int(n, "n")
        return tuple(range(self.level_start(n), self.level_end(n) + 1))

    def finite_block(self, N: object) -> tuple[int, ...]:
        N = require_nonnegative_int(N, "N")
        vertices: list[int] = []
        for n in range(N + 1):
            vertices.extend(self.level(n))
        return tuple(vertices)

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

    def locate_level(self, x: object, max_level: object) -> int:
        x = require_positive_int(x, "x")
        max_level = require_nonnegative_int(max_level, "max_level")
        for n in range(max_level + 1):
            if self.contains_level_point(x, n):
                return n
        raise DomainError("x is outside the finite block")

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

    def F(self, x: object, sigma: object, c: int | None = None) -> int:
        x = require_positive_int(x, "x")
        sigma = self.validate_state(sigma)
        if c is None:
            c = self.cA
        c = require_int(c, "c")
        return self.s * x + sigma - c

    def edge(self, x: object, n: object, sigma: object) -> Edge:
        n = require_nonnegative_int(n, "n")
        x = self.require_level_point(x, n)
        sigma = self.validate_state(sigma)
        y = self.F(x, sigma)
        assert self.contains_level_point(y, n + 1)
        return Edge(source=x, target=y, source_level=n, state=sigma)

    def set_transition(self, n: object) -> tuple[int, ...]:
        n = require_nonnegative_int(n, "n")
        return tuple(sorted(self.F(x, sigma) for x in self.level(n) for sigma in self.S))

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
            raise DomainError("level-zero vertices have no predecessor in T")
        return self.local_preimage(y, level_of_y - 1)

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
            raise DomainError("target_level cannot exceed n")
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

    def ancestor_signature(self, x: object, n: object) -> tuple[int, tuple[int, ...]]:
        current = self.require_level_point(x, n)
        n = require_nonnegative_int(n, "n")
        states: list[int] = []
        level = n
        while level > 0:
            child = current
            parent, sigma = self.immediate_predecessor(child, level)
            assert self.F(parent, sigma) == child
            states.append(sigma)
            current = parent
            level -= 1
        return current, tuple(reversed(states))

    def r_coordinate(self, x: object, n: object) -> float:
        return log(self.position(x, n) + 1, self.s)

    def shifted(self, delta: object) -> "Model":
        delta = require_nonnegative_int(delta, "delta")
        return Model(self.m + delta, self.k, self.s)


def assert_bijection(domain: Sequence[object],
                     codomain: Sequence[object],
                     mapping: Callable[[object], object],
                     context: str) -> dict[object, object]:
    image_to_preimage: dict[object, object] = {}
    for item in domain:
        image = mapping(item)
        if image in image_to_preimage:
            raise AssertionError(
                f"Collision in {context}: {item!r} and {image_to_preimage[image]!r} map to {image!r}"
            )
        image_to_preimage[image] = item
    assert set(image_to_preimage) == set(codomain), context
    assert len(image_to_preimage) == len(codomain), context
    return image_to_preimage


def verify_symbolic_static_formulas() -> None:
    print("\n=== Symbolic verification of static block formulas ===")

    m, k, s, n, delta = symbols("m k s n delta", integer=True, positive=True)

    cA = (s - 1) * m + 1 - k
    mn = m + k * (s**n - 1) / (s - 1)
    mnp1 = m + k * (s ** (n + 1) - 1) / (s - 1)

    assert simplify(mnp1 - (s * mn + 1 - cA)) == 0
    assert simplify(mnp1 - (mn + k * s**n)) == 0
    assert simplify((mn + k * s**n - 1 + 1) - mnp1) == 0

    finite_block_size = sum(k * s**j for j in range(0, 5))
    closed_size_5 = k * (s**5 - 1) / (s - 1)
    assert simplify(finite_block_size - closed_size_5) == 0

    # Shift formulas.
    c_shift = (s - 1) * (m + delta) + 1 - k
    assert simplify(c_shift - (cA + (s - 1) * delta)) == 0

    x, sigma = symbols("x sigma", integer=True, positive=True)
    F_base = s * x + sigma - cA
    F_shift = s * (x + delta) + sigma - c_shift
    assert simplify(F_shift - (F_base + delta)) == 0

    print("[OK] Closed level formula, recurrence, and adjacency are symbolic")
    print("[OK] Finite block cardinality is the geometric prefix sum")
    print("[OK] Shifted block formulas commute with generation")


def verify_static_blocks_and_nested_truncations() -> None:
    print("\n=== Verification of static finite blocks and nested truncations ===")

    checked_blocks = 0
    checked_vertices = 0
    checked_stability = 0

    for model in (
        Model(1, 2, 2),
        Model(2, 3, 2),
        Model(3, 2, 3),
        Model(5, 4, 3),
    ):
        previous_block: tuple[int, ...] | None = None

        for N in range(0, 9):
            block = model.finite_block(N)
            checked_blocks += 1
            checked_vertices += len(block)

            assert block == tuple(range(model.m, model.level_end(N) + 1))
            assert len(block) == sum(model.level_size(n) for n in range(N + 1))
            assert len(set(block)) == len(block)

            for n in range(0, N + 1):
                L = model.level(n)
                assert L == tuple(range(model.level_start(n), model.level_end(n) + 1))
                assert set(L).issubset(block)
                for j in range(n + 1, N + 1):
                    assert set(L).isdisjoint(model.level(j))

            if previous_block is not None:
                assert block[:len(previous_block)] == previous_block
                assert set(previous_block).issubset(block)
                new_part = block[len(previous_block):]
                assert new_part == model.level(N)
                checked_stability += len(previous_block)

            previous_block = block

    print(f"[OK] Checked {checked_blocks} finite static blocks")
    print(f"[OK] Checked {checked_vertices} vertices in finite block windows")
    print(f"[OK] Checked {checked_stability} stable old-vertex memberships under extension")


def verify_full_determination_without_extra_dynamics() -> None:
    print("\n=== Verification that fixed (A,S,F) determines all levels ===")

    checked_levels = 0
    checked_bijections = 0
    checked_genealogies = 0

    for model in (
        Model(1, 2, 2),
        Model(2, 3, 2),
        Model(3, 2, 3),
        Model(4, 3, 4),
    ):
        twin = Model(model.m, model.k, model.s)

        for n in range(0, 7):
            checked_levels += 1

            # Closed formula, geometric sum, recursive image, and an identical
            # copy of the same data must give the same level.
            assert model.level_start(n) == model.level_start_by_sum(n)
            assert model.level(n) == twin.level(n)

            if n == 0:
                assert model.level(0) == model.A
            else:
                assert model.set_transition(n - 1) == model.level(n)

            domain = tuple((root, states) for root in model.A for states in product(model.S, repeat=n))
            codomain = model.level(n)
            assert_bijection(
                domain,
                codomain,
                lambda pair, model=model: model.iterate(pair[0], tuple(pair[1])),
                f"A x S^{n} -> L_{n}",
            )
            checked_bijections += len(codomain)

            for x in codomain[:min(40, len(codomain))]:
                root, states = model.ancestor_signature(x, n)
                assert model.iterate(root, states) == x
                checked_genealogies += 1

    print(f"[OK] Checked {checked_levels} levels by closed/recursive/twin construction")
    print(f"[OK] Checked {checked_bijections} genealogy bijection values")
    print(f"[OK] Checked {checked_genealogies} inverse genealogy signatures")


def verify_transition_operator_is_set_image_not_single_valued_cover() -> None:
    print("\n=== Verification of transition-operator discipline ===")

    checked_set_images = 0
    checked_no_single_valued_cover = 0

    for model in (
        Model(1, 2, 2),
        Model(2, 3, 2),
        Model(3, 2, 3),
    ):
        for n in range(0, 7):
            image = model.set_transition(n)
            checked_set_images += 1

            assert image == model.level(n + 1)
            assert len(image) == model.s * len(model.level(n))
            assert len(set(image)) == len(image)

            # Soundness guard: a genuine single-valued map L_n -> L_{n+1}
            # cannot cover L_{n+1}, because |L_{n+1}|=s|L_n| and s>=2.
            assert len(model.level(n)) < len(model.level(n + 1))
            checked_no_single_valued_cover += 1

            # Any deterministic choice of one child per parent gives exactly
            # |L_n| targets, hence a proper subset of L_{n+1}.
            leftmost_choice = tuple(model.F(x, 1) for x in model.level(n))
            rightmost_choice = tuple(model.F(x, model.s) for x in model.level(n))
            assert len(set(leftmost_choice)) == len(model.level(n))
            assert len(set(rightmost_choice)) == len(model.level(n))
            assert set(leftmost_choice) < set(model.level(n + 1))
            assert set(rightmost_choice) < set(model.level(n + 1))

    print(f"[OK] Checked {checked_set_images} full set-image transitions Phi_n(L_n)=L_(n+1)")
    print(f"[OK] Checked {checked_no_single_valued_cover} impossibility guards for single-valued full-cover transitions")


def verify_internal_causality_and_complete_past_future() -> None:
    print("\n=== Verification of internal causality, complete past, and complete future ===")

    checked_past_chains = 0
    checked_past_slices = 0
    checked_future_slices = 0
    checked_future_vertices = 0
    checked_causal_decodings = 0

    for model in (
        Model(1, 2, 2),
        Model(2, 3, 2),
        Model(3, 2, 3),
        Model(4, 3, 4),
    ):
        for n in range(0, 7):
            sample_positions = sorted({0, model.level_size(n) // 2, model.level_size(n) - 1})
            for u in sample_positions:
                x = model.point_from_position(u, n)

                chain = model.ancestor_chain(x, n)
                checked_past_chains += 1
                assert len(chain) == n + 1
                assert chain[0] in model.A
                assert chain[-1] == x

                for level, ancestor in enumerate(chain):
                    checked_past_slices += 1
                    assert model.contains_level_point(ancestor, level)
                    assert model.causal_leq(ancestor, level, x, n)
                    all_candidates = [
                        y for y in model.level(level)
                        if model.causal_leq(y, level, x, n)
                    ]
                    assert all_candidates == [ancestor]

                for depth in range(0, 6):
                    future = model.future_slice(x, n, depth)
                    checked_future_slices += 1
                    checked_future_vertices += len(future)

                    assert len(future) == model.s**depth
                    assert len(set(future)) == len(future)

                    for y in future:
                        assert model.causal_leq(x, n, y, n + depth)
                        ux = model.position(x, n)
                        uy = model.position(y, n + depth)
                        D = uy - model.s**depth * ux
                        assert 0 <= D <= model.s**depth - 1
                        states = model.states_from_digit(D, depth)
                        assert model.iterate(x, states) == y
                        checked_causal_decodings += 1

    print(f"[OK] Checked {checked_past_chains} unique past chains")
    print(f"[OK] Checked {checked_past_slices} singleton past slices")
    print(f"[OK] Checked {checked_future_slices} future cone slices")
    print(f"[OK] Checked {checked_future_vertices} future vertices")
    print(f"[OK] Checked {checked_causal_decodings} future branch decodings")


def verify_foliation_and_observer_path_distinction() -> None:
    print("\n=== Verification of foliation and observer-path distinction ===")

    checked_time_labels = 0
    checked_observer_paths = 0
    checked_path_is_proper_subset = 0

    for model in (
        Model(1, 2, 2),
        Model(2, 3, 2),
        Model(3, 2, 3),
    ):
        for N in range(0, 8):
            block = model.finite_block(N)
            for x in block[:min(100, len(block))]:
                located = model.locate_level(x, N)
                checked_time_labels += 1
                assert model.contains_level_point(x, located)
                for other in range(0, N + 1):
                    if other != located:
                        assert not model.contains_level_point(x, other)

        # Observer-like paths choose one state at each step.  They are chains,
        # not the full block.
        for root in model.A[:min(3, len(model.A))]:
            for N in range(1, 7):
                for constant_state in (1, model.s):
                    states = tuple([constant_state] * N)
                    path = [root]
                    current = root
                    for level, sigma in enumerate(states, start=1):
                        current = model.F(current, sigma)
                        path.append(current)
                        assert model.contains_level_point(current, level)

                    checked_observer_paths += 1
                    assert len(path) == N + 1
                    for level, vertex in enumerate(path):
                        assert model.contains_level_point(vertex, level)

                    block = model.finite_block(N)
                    assert set(path).issubset(block)
                    assert len(path) < len(block)
                    checked_path_is_proper_subset += 1

    print(f"[OK] Checked {checked_time_labels} unique foliation labels")
    print(f"[OK] Checked {checked_observer_paths} observer-like chains")
    print(f"[OK] Checked {checked_path_is_proper_subset} paths as proper subsets of static blocks")


def verify_universal_causal_speed_limit() -> None:
    print("\n=== Verification of universal causal speed limit ===")

    checked_edges = 0
    checked_paths = 0
    checked_no_shorter_paths = 0
    checked_no_skip_edges = 0

    for model in (
        Model(1, 2, 2),
        Model(2, 3, 2),
        Model(3, 2, 3),
    ):
        for n in range(0, 6):
            for x in model.level(n)[:min(12, len(model.level(n)))]:
                rx = model.r_coordinate(x, n)

                for sigma in model.S:
                    edge = model.edge(x, n, sigma)
                    checked_edges += 1
                    ry = model.r_coordinate(edge.target, n + 1)

                    assert edge.source_level + 1 == n + 1
                    assert ry >= rx - 1e-12
                    assert ry - rx <= 1 + 1e-12

                    # Exact one-edge integer gap proof.
                    ux = model.position(x, n)
                    uy = model.position(edge.target, n + 1)
                    assert uy == model.s * ux + sigma - 1
                    assert (uy + 1) - (ux + 1) == (model.s - 1) * ux + sigma - 1
                    assert model.s * (ux + 1) - (uy + 1) == model.s - sigma

                    # Edge cannot be interpreted as skipping a level.
                    assert not model.contains_level_point(edge.target, n + 2)
                    checked_no_skip_edges += 1

                for depth in range(0, 5):
                    for y in model.future_slice(x, n, depth):
                        checked_paths += 1
                        ry = model.r_coordinate(y, n + depth)
                        assert ry >= rx - 1e-12
                        assert ry - rx <= depth + 1e-12

                        # Every DAG edge increases level by exactly one, so a
                        # path from n to n+depth has exactly depth edges.
                        chain = model.ancestor_chain(y, n + depth)
                        assert chain[n] == x
                        assert len(chain[n:]) - 1 == depth
                        checked_no_shorter_paths += 1

    print(f"[OK] Checked {checked_edges} one-edge speed-limit bounds")
    print(f"[OK] Checked {checked_paths} finite-path speed-limit bounds")
    print(f"[OK] Checked {checked_no_shorter_paths} exact path-length/level-difference identities")
    print(f"[OK] Checked {checked_no_skip_edges} no-skip-edge guards")


def verify_shift_isomorphic_static_blocks() -> None:
    print("\n=== Verification of shift-isomorphic static blocks ===")

    checked_shifted_levels = 0
    checked_shifted_causality = 0
    checked_shifted_signatures = 0
    checked_composition = 0

    for model in (
        Model(1, 2, 2),
        Model(2, 3, 2),
        Model(4, 2, 3),
    ):
        for delta in range(0, 8):
            shifted = model.shifted(delta)
            assert shifted.cA == model.cA + (model.s - 1) * delta

            for n in range(0, 6):
                checked_shifted_levels += 1
                assert shifted.level(n) == tuple(x + delta for x in model.level(n))

                for x in model.level(n)[:min(6, len(model.level(n)))]:
                    sx = x + delta
                    assert model.ancestor_signature(x, n)[1] == shifted.ancestor_signature(sx, n)[1]
                    checked_shifted_signatures += 1

                    for depth in range(0, 4):
                        for y in model.future_slice(x, n, depth):
                            sy = y + delta
                            checked_shifted_causality += 1
                            assert model.causal_leq(x, n, y, n + depth)
                            assert shifted.causal_leq(sx, n, sy, n + depth)

            for d1 in range(0, 4):
                for d2 in range(0, 4):
                    lhs = model.shifted(d1).shifted(d2)
                    rhs = model.shifted(d1 + d2)
                    checked_composition += 1
                    for n in range(0, 4):
                        assert lhs.level(n) == rhs.level(n)
                        assert lhs.cA == rhs.cA

    print(f"[OK] Checked {checked_shifted_levels} shifted levels")
    print(f"[OK] Checked {checked_shifted_signatures} shifted ancestor signatures")
    print(f"[OK] Checked {checked_shifted_causality} shifted causal pairs")
    print(f"[OK] Checked {checked_composition} shift-composition witnesses")


def verify_negative_guards() -> None:
    print("\n=== Negative guards and invalid-domain tests ===")

    model = Model(1, 2, 2)

    # Time order alone is not causality.
    x = model.point_from_position(1, 1)
    y_noncausal = model.point_from_position(0, 3)
    assert 1 < 3
    assert not model.causal_leq(x, 1, y_noncausal, 3)

    # Wrong c produces a different, misaligned block.
    wrong_c = model.cA + 1
    wrong_L1 = tuple(sorted(model.F(x0, sigma, c=wrong_c) for x0 in model.A for sigma in model.S))
    assert wrong_L1 != model.level(1)
    assert len(wrong_L1) == len(set(wrong_L1)) == model.k * model.s

    # A single observer path is not the block.
    path = [model.A[0]]
    current = model.A[0]
    for sigma in (1, 1, 1, 1):
        current = model.F(current, sigma)
        path.append(current)
    assert set(path) < set(model.finite_block(4))

    # Skip-level edge is not an allowed DAG edge.
    source = model.point_from_position(0, 1)
    skip_target = model.future_slice(source, 1, 2)[0]
    assert model.causal_leq(source, 1, skip_target, 3)
    assert not model.contains_level_point(skip_target, 2)

    # Single-valued Phi cannot cover next level.
    chosen_children = tuple(model.F(x0, 1) for x0 in model.level(2))
    assert len(set(chosen_children)) == len(model.level(2))
    assert set(chosen_children) < set(model.level(3))

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
    expect_raises(DomainError, lambda: model.immediate_predecessor(model.m, 0))
    expect_raises(DomainError, lambda: model.states_from_digit(-1, 2))
    expect_raises(DomainError, lambda: model.states_from_digit(4, 2))
    expect_raises(TypeError, lambda: model.iterate(1, [1, 2]))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: model.iterate(1, (1, 3)))
    expect_raises(DomainError, lambda: model.ancestor_at_level(model.m, 0, 1))
    expect_raises(DomainError, lambda: model.shifted(-1))

    print("[OK] Time-order, wrong-c, single-path, skip-edge, and single-valued-transition guards pass")
    print("[OK] Invalid parameters, levels, vertices, states, and path data are rejected")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of static block-universe structure (sec:block-universe) ===")
    verify_symbolic_static_formulas()
    verify_static_blocks_and_nested_truncations()
    verify_full_determination_without_extra_dynamics()
    verify_transition_operator_is_set_image_not_single_valued_cover()
    verify_internal_causality_and_complete_past_future()
    verify_foliation_and_observer_path_distinction()
    verify_universal_causal_speed_limit()
    verify_shift_isomorphic_static_blocks()
    verify_negative_guards()
    print("\n=== Block-universe verification completed successfully ===")


if __name__ == "__main__":
    main()
