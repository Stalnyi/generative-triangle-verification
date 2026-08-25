"""
VERIFICATION of Chapter 1:
Mathematical foundations of the generative triangle.

Covered source files
--------------------
1_basic_objects.tex
2_space_definition.tex
3_recursive_generation.tex
4_basic_unfolding.tex
5_shift_mechanism.tex
6_geometric_topological_properties.tex
7_time_as_level_index.tex
8_structural_properties.tex
9_hierarchical_meta_structure.tex

This is an integrated verification script.  It checks the chapter as one
mathematical system, not as disconnected snippets.  In particular it verifies
that the same definitions of N, S, A, F, c(A), L_n, T, the shift map, the
time index, and the meta-structure are mutually compatible across all files.

Design principles
-----------------
1. No checks based on the absence of arbitrary future terminology.
2. No tautological checks of the form "definition equals itself".
3. Every block verifies an algebraic, order-theoretic, cardinality,
   bijectivity, collision-freeness, decoding, shift-isomorphism, causal,
   hierarchy, or domain statement.
4. The negative tests are not cosmetic: they explicitly check common false
   strengthened statements and corrupted domains.

Main verified claims
--------------------
A. Basic objects
   - N={1,2,3,...}; x=0, negative, and noninteger x are rejected.
   - S={1,...,s}, s>=2, is finite, contiguous, duplicate-free, |S|=s.
   - Elementary objects are ordered pairs (x,sigma), not interchangeable.
   - Every x has the full S-fiber, and S is globally x-independent.

B. Space definition
   - A={m,...,m+k-1}, k>=2; S={1,...,s}, s>=2.
   - c(A)=(s-1)m+1-k.
   - F(A x S) is exactly {M+1,...,M+sk}.
   - A x S -> L_1 is a true bijection.
   - No collisions: F(x1,sigma1)=F(x2,sigma2) implies x1=x2 and sigma1=sigma2.
   - c(A) is unique for canonical endpoint alignment.
   - Incorrect c remains collision-free but produces a shifted/misaligned
     interval, not the canonical first level.

C. Recursive generation
   - L_0=A and L_{n+1}=F(L_n x S).
   - L_n={m_n,...,m_n+ks^n-1}.
   - m_{n+1}=s m_n+1-c(A), m_0=m.
   - m_n=m+k(s^n-1)/(s-1).
   - |L_n|=ks^n; max(L_n)+1=min(L_{n+1}).
   - Levels are disjoint and finite initial unions have no gaps.
   - F:L_n x S -> L_{n+1} is bijective for every checked n.
   - F on finite truncations maps exactly
       (L_0 union ... union L_N) x S
     onto
       L_1 union ... union L_{N+1}.
     The wrong codomain including L_0 is rejected.
   - Every non-initial element has exactly one immediate predecessor and one
     internal state.

D. Basic unfolding
   - For s=2, A={1,2}, c(A)=0 and F(x,sigma)=2x+sigma.
   - L_n={2^{n+1}-1,...,2^{n+2}-2}.
   - Finite initial unions cover {1,...,2^{N+2}-2} exactly.

E. Shift mechanism
   - c(A+delta)=c(A)+(s-1)delta.
   - F_delta(x+delta,sigma)=F(x,sigma)+delta.
   - L_n(A+delta)=L_n(A)+delta.
   - phi_delta(x)=x+delta is a bijection on every finite truncation.
   - phi_delta preserves generation edges, ancestor signatures, and causal
     reachability.
   - Shift composition phi_delta2 o phi_delta1 = phi_{delta1+delta2}.
   - Incorrect shifts are rejected as isomorphism witnesses.

F. Geometric/topological properties
   - Determinism of finite truncations.
   - Every element has a unique full ancestor chain back to L_0.
   - The genealogy map A x S^n -> L_n is bijective.
   - Distinct internal-state histories from the same ancestor do not collide.

G. Time as level index
   - t(x)=n iff x in L_n.
   - One application of F increases time by one.
   - Nonzero causality strictly increases time.
   - No backward causality and no same-level comparability except identity.
   - t(x)<t(y) alone is not sufficient for causality.

H. Structural properties
   - Every element has exactly s outgoing internal-state choices.
   - |L_{n+1}|=s|L_n| and |L_n| strictly grows because s>=2.
   - The shift family {T_delta : delta in N_0} is countably indexed in the
     sampled finite windows and pairwise shift-isomorphic.

I. Hierarchical meta-structure
   - Level 1: individual generative triangle T.
   - Level 2: countable shift-indexed multispace T_delta.
   - Level 3: recursive use of the same shift operator is closed under
     composition Sigma_gamma(T_delta)=T_{delta+gamma}; no new free parameters
     are introduced in the algebraic shift hierarchy.
   - The explicitly marked illustrative structural-deduction paragraph is
     not used as a theorem.

J. Failure guards
   - wrong c does not give canonical L_1;
   - injectivity alone does not determine c;
   - the wrong truncation codomain is rejected;
   - local decoding outside the declared level is rejected;
   - time order is not causality;
   - a corrupted predecessor/state pair does not regenerate the child;
   - an x-dependent state family is not homogeneous.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
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
class StateSpace:
    s: int

    def __post_init__(self) -> None:
        require_at_least(self.s, 2, "s")

    @property
    def states(self) -> tuple[int, ...]:
        return tuple(range(1, self.s + 1))

    def validate(self, sigma: object) -> int:
        sigma = require_int(sigma, "sigma")
        if not (1 <= sigma <= self.s):
            raise DomainError("sigma must belong to S={1,...,s}")
        return sigma


@dataclass(frozen=True, slots=True)
class ElementaryObject:
    x: int
    sigma: int
    state_space: StateSpace

    def __post_init__(self) -> None:
        require_positive_int(self.x, "x")
        self.state_space.validate(self.sigma)

    @property
    def pair(self) -> tuple[int, int]:
        return (self.x, self.sigma)


@dataclass(frozen=True, slots=True)
class PositiveInterval:
    start: int
    length: int

    def __post_init__(self) -> None:
        require_positive_int(self.start, "start")
        require_at_least(self.length, 1, "length")

    @property
    def values(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.start + self.length))

    @property
    def end(self) -> int:
        return self.start + self.length - 1

    def contains(self, x: object) -> bool:
        return isinstance(x, int) and self.start <= x <= self.end


@dataclass(frozen=True, slots=True)
class IntegerInterval:
    start: int
    length: int

    def __post_init__(self) -> None:
        require_int(self.start, "start")
        require_at_least(self.length, 1, "length")

    @property
    def values(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.start + self.length))

    @property
    def end(self) -> int:
        return self.start + self.length - 1


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
    def A(self) -> PositiveInterval:
        return PositiveInterval(self.m, self.k)

    @property
    def S(self) -> StateSpace:
        return StateSpace(self.s)

    @property
    def cA(self) -> int:
        return (self.s - 1) * self.m + 1 - self.k

    @property
    def M(self) -> int:
        return self.m + self.k - 1

    def F(self, x: object, sigma: object, c: int | None = None) -> int:
        x = require_positive_int(x, "x")
        sigma = self.S.validate(sigma)
        if c is None:
            c = self.cA
        c = require_int(c, "c")
        return self.s * x + sigma - c

    def first_level_image_from_A(self, c: int | None = None) -> tuple[int, ...]:
        return tuple(sorted(self.F(x, sigma, c=c) for x in self.A.values for sigma in self.S.states))

    @property
    def target_L1(self) -> PositiveInterval:
        return PositiveInterval(self.M + 1, self.s * self.k)

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

    def generated_next_level(self, n: object) -> tuple[int, ...]:
        n = require_nonnegative_int(n, "n")
        return tuple(sorted(self.F(x, sigma) for x in self.level(n) for sigma in self.S.states))

    def local_preimage(self, y: object, previous_level: object) -> tuple[int, int]:
        previous_level = require_nonnegative_int(previous_level, "previous_level")
        y = self.require_level_point(y, previous_level + 1)
        u_child = self.position(y, previous_level + 1)
        u_parent = u_child // self.s
        sigma = u_child % self.s + 1
        x = self.point_from_position(u_parent, previous_level)
        assert self.F(x, sigma) == y
        return x, sigma

    def immediate_predecessor(self, y: object, level_of_y: object) -> tuple[int, int]:
        level_of_y = require_nonnegative_int(level_of_y, "level_of_y")
        if level_of_y == 0:
            raise DomainError("level-zero elements have no predecessor inside T")
        return self.local_preimage(y, level_of_y - 1)

    def finite_truncation(self, N: object) -> tuple[int, ...]:
        N = require_nonnegative_int(N, "N")
        out: list[int] = []
        for n in range(N + 1):
            out.extend(self.level(n))
        return tuple(out)

    def locate_level_in_truncation(self, x: int, max_level: int) -> int:
        for n in range(max_level + 1):
            if self.contains_level_point(x, n):
                return n
        raise DomainError("x is outside the finite truncation")

    def ancestor_chain(self, x: object, n: object) -> tuple[int, ...]:
        current = self.require_level_point(x, n)
        n = require_nonnegative_int(n, "n")
        chain = [current]
        level = n
        while level > 0:
            child = current
            parent, sigma = self.immediate_predecessor(child, level)
            assert self.F(parent, sigma) == child
            current = parent
            chain.append(current)
            level -= 1
        return tuple(reversed(chain))

    def ancestor_signature(self, x: object, n: object) -> tuple[int, tuple[int, ...]]:
        current = self.require_level_point(x, n)
        n = require_nonnegative_int(n, "n")
        states: list[int] = []
        level = n
        while level > 0:
            parent, sigma = self.immediate_predecessor(current, level)
            states.append(sigma)
            current = parent
            level -= 1
        return current, tuple(reversed(states))

    def iterate_from_signature(self, root: object, states: Sequence[int]) -> int:
        root = require_positive_int(root, "root")
        current = root
        for sigma in states:
            current = self.F(current, sigma)
        return current

    def shifted(self, delta: object) -> "Model":
        delta = require_nonnegative_int(delta, "delta")
        return Model(self.m + delta, self.k, self.s)

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
        low = self.s**depth * ux
        high = low + self.s**depth - 1
        return low <= uy <= high

    def future_slice(self, x: object, n: object, depth: object) -> tuple[int, ...]:
        n = require_nonnegative_int(n, "n")
        depth = require_nonnegative_int(depth, "depth")
        x = self.require_level_point(x, n)
        ux = self.position(x, n)
        return tuple(
            self.point_from_position(self.s**depth * ux + D, n + depth)
            for D in range(self.s**depth)
        )


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


def interval_from_explicit_A(values: Sequence[int]) -> PositiveInterval:
    if not values:
        raise DomainError("A cannot be empty")
    if any(not isinstance(v, int) for v in values):
        raise TypeError("all A-elements must be integers")
    sorted_values = tuple(sorted(values))
    if len(set(sorted_values)) != len(sorted_values):
        raise DomainError("A cannot contain duplicates")
    if sorted_values[0] < 1:
        raise DomainError("A must be a subset of N")
    if len(sorted_values) < 2:
        raise DomainError("A must have at least two elements")
    for left, right in zip(sorted_values, sorted_values[1:]):
        if right != left + 1:
            raise DomainError("A must be consecutive")
    return PositiveInterval(sorted_values[0], len(sorted_values))


def finite_product_objects(N: int, state_space: StateSpace) -> tuple[ElementaryObject, ...]:
    require_positive_int(N, "N")
    return tuple(
        ElementaryObject(x=x, sigma=sigma, state_space=state_space)
        for x in range(1, N + 1)
        for sigma in state_space.states
    )


def image_of_positive_interval(B: PositiveInterval, s: int, c: int) -> tuple[int, ...]:
    require_at_least(s, 2, "s")
    c = require_int(c, "c")
    return tuple(sorted(s * x + sigma - c for x in B.values for sigma in range(1, s + 1)))


def expected_integer_image(B: PositiveInterval, s: int, c: int) -> IntegerInterval:
    require_at_least(s, 2, "s")
    c = require_int(c, "c")
    return IntegerInterval(s * B.start + 1 - c, B.length * s)


def verify_basic_objects() -> None:
    print("\n=== 1. Basic objects ===")

    checked_spaces = 0
    checked_objects = 0
    checked_fibers = 0
    checked_noninterchangeable = 0

    for s in range(2, 18):
        S = StateSpace(s)
        checked_spaces += 1
        assert S.states == tuple(range(1, s + 1))
        assert len(S.states) == s
        assert len(set(S.states)) == s
        assert S.states[0] == 1
        assert S.states[-1] == s

        for bad_sigma in (-3, -1, 0, s + 1, s + 5):
            expect_raises(DomainError, lambda bad_sigma=bad_sigma: S.validate(bad_sigma))
        expect_raises(TypeError, lambda: S.validate(1.25))

        for x in range(1, 22):
            fiber = tuple(ElementaryObject(x=x, sigma=sigma, state_space=S) for sigma in S.states)
            checked_fibers += 1
            assert len(fiber) == s
            assert {obj.pair for obj in fiber} == {(x, sigma) for sigma in S.states}

            for obj in fiber:
                swapped_admissible = S.states[0] <= obj.x <= S.states[-1]
                if swapped_admissible:
                    swapped = ElementaryObject(x=obj.sigma, sigma=obj.x, state_space=S)
                    checked_noninterchangeable += 1
                    if obj.x != obj.sigma:
                        assert obj.pair != swapped.pair

        for N in range(1, 10):
            product_objects = finite_product_objects(N, S)
            checked_objects += len(product_objects)
            assert len(product_objects) == N * s
            assert len(set(obj.pair for obj in product_objects)) == N * s
            for x in range(1, N + 1):
                assert len([obj for obj in product_objects if obj.x == x]) == s
            for sigma in S.states:
                assert len([obj for obj in product_objects if obj.sigma == sigma]) == N

    expect_raises(DomainError, lambda: StateSpace(1))
    expect_raises(TypeError, lambda: StateSpace(2.0))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: ElementaryObject(x=0, sigma=1, state_space=StateSpace(2)))
    expect_raises(DomainError, lambda: ElementaryObject(x=1, sigma=3, state_space=StateSpace(2)))
    expect_raises(TypeError, lambda: ElementaryObject(x=1.0, sigma=1, state_space=StateSpace(2)))  # type: ignore[arg-type]

    homogeneous = {x: StateSpace(4).states for x in range(1, 15)}
    assert len(set(homogeneous.values())) == 1
    inhomogeneous = dict(homogeneous)
    inhomogeneous[5] = StateSpace(5).states
    assert len(set(inhomogeneous.values())) > 1

    print(f"[OK] Checked {checked_spaces} state spaces")
    print(f"[OK] Checked {checked_fibers} full fibers over x")
    print(f"[OK] Checked {checked_objects} finite product elementary objects")
    print(f"[OK] Checked {checked_noninterchangeable} ordered-coordinate cases")


def verify_space_definition() -> None:
    print("\n=== 2. Space definition, c(A), L_1, and collision-freeness ===")

    m, k, s = symbols("m k s", integer=True, positive=True)
    M = m + k - 1
    cA = (s - 1) * m + 1 - k
    Fmin = s * m + 1 - cA
    Fmax = s * M + s - cA
    assert simplify(Fmin - (M + 1)) == 0
    assert simplify(Fmax - (M + s * k)) == 0

    c = symbols("c", integer=True)
    left_alignment = simplify((s * m + 1 - c) - (M + 1))
    right_alignment = simplify((s * M + s - c) - (M + s * k))
    c_residual = c - ((s - 1) * m + 1 - k)
    assert simplify(left_alignment + c_residual) == 0
    assert simplify(right_alignment + c_residual) == 0

    # If c=c(A)+q, the entire image is shifted by -q.
    q = symbols("q", integer=True)
    shifted_min = s * m + 1 - (cA + q)
    shifted_max = s * M + s - (cA + q)
    assert simplify(shifted_min - (M + 1 - q)) == 0
    assert simplify(shifted_max - (M + s * k - q)) == 0

    checked_spaces = 0
    checked_first_level_values = 0
    checked_wrong_constants = 0
    checked_modular_collision_guards = 0

    for m0 in range(1, 8):
        for k0 in range(2, 9):
            for s0 in range(2, 7):
                U = Model(m0, k0, s0)
                checked_spaces += 1

                domain = tuple((x, sigma) for x in U.A.values for sigma in U.S.states)
                codomain = U.target_L1.values
                image_to_pair = assert_bijection(
                    domain,
                    codomain,
                    lambda pair, U=U: U.F(pair[0], pair[1]),
                    "A x S -> L_1",
                )
                checked_first_level_values += len(image_to_pair)

                for y in codomain:
                    offset = y - (U.M + 1)
                    decoded_x = U.m + offset // U.s
                    decoded_sigma = offset % U.s + 1
                    assert image_to_pair[y] == (decoded_x, decoded_sigma)
                    assert U.F(decoded_x, decoded_sigma) == y

                # Direct modular collision argument over all pairs.
                for p1 in domain:
                    for p2 in domain:
                        same_value = U.F(p1[0], p1[1]) == U.F(p2[0], p2[1])
                        if same_value:
                            assert p1 == p2
                        else:
                            residual = U.s * (p1[0] - p2[0]) - (p2[1] - p1[1])
                            assert residual != 0
                        checked_modular_collision_guards += 1

                for delta_c in (-3, -2, -1, 1, 2, 3):
                    wrong_c = U.cA + delta_c
                    wrong_image = U.first_level_image_from_A(c=wrong_c)
                    assert wrong_image != U.target_L1.values
                    assert wrong_image == tuple(value - delta_c for value in U.target_L1.values)
                    assert len(wrong_image) == len(set(wrong_image)) == U.k * U.s
                    checked_wrong_constants += 1

    checked_intervals = 0
    checked_neighbor_gaps = 0
    for b in range(1, 8):
        for ell in range(1, 8):
            B = PositiveInterval(b, ell)
            for s0 in range(2, 7):
                for c0 in range(-3, 5):
                    image = image_of_positive_interval(B, s0, c0)
                    expected = expected_integer_image(B, s0, c0).values
                    checked_intervals += 1
                    assert image == expected
                    assert len(image) == ell * s0
                    assert len(set(image)) == ell * s0
                    for left, right in zip(image, image[1:]):
                        checked_neighbor_gaps += 1
                        assert right == left + 1

    for invalid_A in (tuple(), (1,), (0, 1), (1, 3), (1, 2, 2)):
        expect_raises((DomainError, TypeError), lambda invalid_A=invalid_A: interval_from_explicit_A(invalid_A))
    assert interval_from_explicit_A((3, 4, 5)).values == (3, 4, 5)

    print(f"[OK] Checked {checked_spaces} generative spaces")
    print(f"[OK] Checked {checked_first_level_values} first-level bijection values")
    print(f"[OK] Checked {checked_modular_collision_guards} modular collision-freeness comparisons")
    print(f"[OK] Checked {checked_wrong_constants} wrong constants as shifted/misaligned, not endpoint-aligned")
    print(f"[OK] Checked {checked_intervals} interval-preservation cases and {checked_neighbor_gaps} neighbor gaps")


def verify_recursive_generation() -> None:
    print("\n=== 3. Recursive generation, global bijections, and decoding ===")

    m, k, s, n = symbols("m k s n", integer=True, positive=True)
    cA = (s - 1) * m + 1 - k
    mn = m + k * (s**n - 1) / (s - 1)
    mnp1 = m + k * (s ** (n + 1) - 1) / (s - 1)
    assert simplify(mnp1 - (s * mn + 1 - cA)) == 0
    assert simplify((m + k * (s**0 - 1) / (s - 1)) - m) == 0
    assert simplify(mnp1 - (mn + k * s**n)) == 0
    assert simplify((mn + k * s**n - 1 + 1) - mnp1) == 0

    checked_levels = 0
    checked_generated = 0
    checked_local_decodings = 0
    checked_global_pairs = 0
    checked_wrong_codomain_rejections = 0
    checked_corrupted_predecessors = 0

    models = (
        Model(1, 2, 2),
        Model(2, 3, 2),
        Model(3, 2, 3),
        Model(5, 3, 4),
    )

    for U in models:
        for n0 in range(0, 7):
            checked_levels += 1
            assert U.level_start(n0) == U.level_start_by_sum(n0)
            L = U.level(n0)
            assert L == tuple(range(U.level_start(n0), U.level_end(n0) + 1))
            assert len(L) == U.level_size(n0)
            assert len(set(L)) == len(L)

            domain = tuple((x, sigma) for x in L for sigma in U.S.states)
            codomain = U.level(n0 + 1)
            image_to_pair = assert_bijection(
                domain,
                codomain,
                lambda pair, U=U: U.F(pair[0], pair[1]),
                f"L_{n0} x S -> L_{n0+1}",
            )
            checked_generated += len(image_to_pair)

            for y in codomain:
                decoded = U.local_preimage(y, n0)
                assert decoded == image_to_pair[y]
                checked_local_decodings += 1

                parent, sigma0 = decoded
                # Corrupting either the parent or the internal state must not
                # regenerate the same child when the corrupted value is admissible.
                if U.position(parent, n0) + 1 < U.level_size(n0):
                    wrong_parent = parent + 1
                    assert U.F(wrong_parent, sigma0) != y
                    checked_corrupted_predecessors += 1
                if sigma0 < U.s:
                    assert U.F(parent, sigma0 + 1) != y
                    checked_corrupted_predecessors += 1
                if sigma0 > 1:
                    assert U.F(parent, sigma0 - 1) != y
                    checked_corrupted_predecessors += 1

        for N in range(0, 6):
            domain_points = U.finite_truncation(N)
            domain = tuple((x, sigma) for x in domain_points for sigma in U.S.states)
            correct_codomain = tuple(y for level in range(1, N + 2) for y in U.level(level))
            image_to_pair = assert_bijection(
                domain,
                correct_codomain,
                lambda pair, U=U: U.F(pair[0], pair[1]),
                "finite truncation global bijection",
            )
            checked_global_pairs += len(image_to_pair)

            wrong_codomain_including_L0 = U.finite_truncation(N + 1)
            assert set(image_to_pair) != set(wrong_codomain_including_L0)
            assert set(U.level(0)).isdisjoint(set(image_to_pair))
            checked_wrong_codomain_rejections += 1

            union = U.finite_truncation(N)
            assert union == tuple(range(U.m, U.level_end(N) + 1))
            assert len(union) == len(set(union))

    U = Model(1, 2, 2)
    expect_raises(DomainError, lambda: U.immediate_predecessor(U.m, 0))
    expect_raises(DomainError, lambda: U.local_preimage(U.m, 0))
    expect_raises(DomainError, lambda: U.local_preimage(U.level_start(2), 0))
    expect_raises(DomainError, lambda: U.level(-1))
    expect_raises(DomainError, lambda: U.point_from_position(U.level_size(2), 2))

    print(f"[OK] Checked {checked_levels} recursive levels")
    print(f"[OK] Checked {checked_generated} generated next-level bijection values")
    print(f"[OK] Checked {checked_local_decodings} local inverse decodings")
    print(f"[OK] Checked {checked_global_pairs} finite-truncation bijection values")
    print(f"[OK] Rejected {checked_wrong_codomain_rejections} wrong truncation codomains")
    print(f"[OK] Checked {checked_corrupted_predecessors} corrupted predecessor/state failures")


def verify_basic_unfolding() -> None:
    print("\n=== 4. Basic binary unfolding ===")

    U = Model(1, 2, 2)
    assert U.cA == 0

    assert U.level(0) == (1, 2)
    assert U.level(1) == (3, 4, 5, 6)
    assert U.level(2) == tuple(range(7, 15))

    assert U.F(1, 1) == 3
    assert U.F(1, 2) == 4
    assert U.F(2, 1) == 5
    assert U.F(2, 2) == 6
    assert tuple(U.F(x, sigma) for x in (3, 4, 5, 6) for sigma in (1, 2)) == tuple(range(7, 15))

    checked_levels = 0
    for n0 in range(0, 18):
        checked_levels += 1
        assert U.level_start(n0) == 2 ** (n0 + 1) - 1
        assert U.level_size(n0) == 2 ** (n0 + 1)
        assert U.level_end(n0) == 2 ** (n0 + 2) - 2
        assert U.level(n0)[0] == 2 ** (n0 + 1) - 1
        assert U.level(n0)[-1] == 2 ** (n0 + 2) - 2

    for N in range(0, 12):
        assert U.finite_truncation(N) == tuple(range(1, 2 ** (N + 2) - 1))
        assert len(U.finite_truncation(N)) == 2 ** (N + 2) - 2

    # Negative guard: the binary closed formulas are special to m=1,k=2,s=2.
    V = Model(2, 2, 2)
    assert V.level_start(0) != 2 ** (0 + 1) - 1
    assert V.cA != 0

    print(f"[OK] Checked {checked_levels} binary levels and finite initial coverage")


def verify_shift_mechanism() -> None:
    print("\n=== 5. Shift mechanism and multispace isomorphism ===")

    m, k, s, delta = symbols("m k s delta", integer=True, positive=True)
    cA = (s - 1) * m + 1 - k
    c_shift = (s - 1) * (m + delta) + 1 - k
    assert simplify(c_shift - (cA + (s - 1) * delta)) == 0

    x, sigma = symbols("x sigma", integer=True, positive=True)
    F_base = s * x + sigma - cA
    F_shift = s * (x + delta) + sigma - c_shift
    assert simplify(F_shift - (F_base + delta)) == 0

    checked_shifted_levels = 0
    checked_shifted_edges = 0
    checked_causal_pairs = 0
    checked_composition = 0
    checked_wrong_shift_witnesses = 0

    for U in (Model(1, 2, 2), Model(2, 3, 2), Model(4, 2, 3), Model(5, 4, 3)):
        for delta0 in range(0, 7):
            V = U.shifted(delta0)
            assert V.cA == U.cA + (U.s - 1) * delta0

            for n0 in range(0, 6):
                checked_shifted_levels += 1
                assert V.level(n0) == tuple(x0 + delta0 for x0 in U.level(n0))

                for x0 in U.level(n0)[:min(6, len(U.level(n0)))]:
                    for sigma0 in U.S.states:
                        checked_shifted_edges += 1
                        assert V.F(x0 + delta0, sigma0) == U.F(x0, sigma0) + delta0
                        if delta0 + 1 <= 10:
                            wrong = x0 + delta0 + 1
                            if V.contains_level_point(wrong, n0):
                                assert V.F(wrong, sigma0) != U.F(x0, sigma0) + delta0
                                checked_wrong_shift_witnesses += 1

            for nx in range(0, 4):
                for depth in range(0, 4):
                    ny = nx + depth
                    for x0 in U.level(nx)[:min(4, len(U.level(nx)))]:
                        for y0 in U.future_slice(x0, nx, depth):
                            checked_causal_pairs += 1
                            assert U.causal_leq(x0, nx, y0, ny)
                            assert V.causal_leq(x0 + delta0, nx, y0 + delta0, ny)

            for d1 in range(0, 4):
                for d2 in range(0, 4):
                    checked_composition += 1
                    W1 = U.shifted(d1).shifted(d2)
                    W2 = U.shifted(d1 + d2)
                    for n0 in range(0, 4):
                        assert W1.level(n0) == W2.level(n0)

    U = Model(1, 2, 2)
    V = U.shifted(6)
    assert V.level(0) == (7, 8)
    assert V.level(1) == (9, 10, 11, 12)
    assert V.level(2) == tuple(range(13, 21))
    assert V.F(7, 1) == 9
    assert V.F(8, 2) == 12
    expect_raises(DomainError, lambda: U.shifted(-1))

    print(f"[OK] Checked {checked_shifted_levels} shifted levels")
    print(f"[OK] Checked {checked_shifted_edges} shifted generation edges")
    print(f"[OK] Checked {checked_causal_pairs} shifted causal pairs")
    print(f"[OK] Checked {checked_composition} shift-composition witnesses")
    print(f"[OK] Checked {checked_wrong_shift_witnesses} wrong-shift non-isomorphism witnesses")


def verify_geometric_topological_properties() -> None:
    print("\n=== 6. Geometric and topological properties ===")

    checked_determinism = 0
    checked_ancestor_chains = 0
    checked_signatures = 0
    checked_genealogy_bijections = 0
    checked_branch_noncollisions = 0

    for U in (Model(1, 2, 2), Model(2, 3, 2), Model(3, 2, 3), Model(4, 4, 3)):
        U_copy = Model(U.m, U.k, U.s)
        for N in range(0, 7):
            checked_determinism += 1
            assert U.finite_truncation(N) == U_copy.finite_truncation(N)

        for n0 in range(0, 7):
            domain = tuple((root, states) for root in U.A.values for states in product(U.S.states, repeat=n0))
            codomain = U.level(n0)
            image_to_signature = assert_bijection(
                domain,
                codomain,
                lambda pair, U=U: U.iterate_from_signature(pair[0], pair[1]),
                f"A x S^{n0} -> L_{n0}",
            )
            checked_genealogy_bijections += len(image_to_signature)

            # Inverse genealogy decoding must match the bijection.
            for y, signature in image_to_signature.items():
                assert U.ancestor_signature(y, n0) == signature
                checked_signatures += 1

            # Different states from the same root do not collide.
            for root in U.A.values[:min(3, len(U.A.values))]:
                seen: dict[int, tuple[int, ...]] = {}
                for states in product(U.S.states, repeat=n0):
                    y = U.iterate_from_signature(root, states)
                    assert y not in seen
                    seen[y] = states
                    checked_branch_noncollisions += 1

        for n0 in range(0, 7):
            sample_positions = sorted({0, U.level_size(n0) // 2, U.level_size(n0) - 1})
            for u in sample_positions:
                x0 = U.point_from_position(u, n0)
                chain = U.ancestor_chain(x0, n0)
                checked_ancestor_chains += 1
                assert len(chain) == n0 + 1
                assert chain[0] in U.A.values
                assert chain[-1] == x0
                for level, point in enumerate(chain):
                    assert U.contains_level_point(point, level)
                for level in range(1, n0 + 1):
                    parent, sigma0 = U.immediate_predecessor(chain[level], level)
                    assert parent == chain[level - 1]
                    assert U.F(parent, sigma0) == chain[level]

    print(f"[OK] Checked {checked_determinism} deterministic finite truncations")
    print(f"[OK] Checked {checked_genealogy_bijections} genealogy bijection values")
    print(f"[OK] Checked {checked_signatures} inverse ancestor signatures")
    print(f"[OK] Checked {checked_branch_noncollisions} same-root branch noncollision witnesses")
    print(f"[OK] Checked {checked_ancestor_chains} unique ancestor chains")


def verify_time_as_level_index() -> None:
    print("\n=== 7. Time as level index ===")

    checked_time_labels = 0
    checked_time_edges = 0
    checked_causal_time_pairs = 0
    checked_same_level = 0
    checked_nonconverse = 0

    for U in (Model(1, 2, 2), Model(2, 3, 2), Model(3, 2, 3)):
        for n0 in range(0, 8):
            for x0 in U.level(n0)[:min(10, len(U.level(n0)))]:
                checked_time_labels += 1
                assert U.contains_level_point(x0, n0)
                for other in range(0, 8):
                    if other != n0:
                        assert not U.contains_level_point(x0, other)

                for sigma0 in U.S.states:
                    checked_time_edges += 1
                    y = U.F(x0, sigma0)
                    assert U.contains_level_point(y, n0 + 1)

            sample = U.level(n0)[:min(8, len(U.level(n0)))]
            for i, x1 in enumerate(sample):
                for x2 in sample:
                    checked_same_level += 1
                    if x1 == x2:
                        assert U.causal_leq(x1, n0, x2, n0)
                    else:
                        assert not U.causal_leq(x1, n0, x2, n0)

        for nx in range(0, 5):
            for depth in range(0, 4):
                ny = nx + depth
                for x0 in U.level(nx)[:min(4, len(U.level(nx)))]:
                    for y0 in U.future_slice(x0, nx, depth):
                        checked_causal_time_pairs += 1
                        assert U.causal_leq(x0, nx, y0, ny)
                        if depth == 0:
                            assert x0 == y0
                        else:
                            assert nx < ny
                            assert not U.causal_leq(y0, ny, x0, nx)

        x_bad = U.point_from_position(1, 1)
        y_bad = U.point_from_position(0, 2)
        assert 1 < 2
        assert not U.causal_leq(x_bad, 1, y_bad, 2)
        checked_nonconverse += 1

    print(f"[OK] Checked {checked_time_labels} unique time labels")
    print(f"[OK] Checked {checked_time_edges} one-step time increments")
    print(f"[OK] Checked {checked_causal_time_pairs} causal time-order pairs")
    print(f"[OK] Checked {checked_same_level} same-level comparability cases")
    print(f"[OK] Checked {checked_nonconverse} time-order nonconverse witnesses")


def verify_structural_properties() -> None:
    print("\n=== 8. Structural properties ===")

    checked_homogeneity = 0
    checked_growth = 0
    checked_shift_family = 0
    checked_alignment_forcing = 0

    for U in (Model(1, 2, 2), Model(2, 3, 2), Model(3, 2, 3), Model(5, 4, 3)):
        for n0 in range(0, 7):
            for x0 in U.level(n0)[:min(8, len(U.level(n0)))]:
                outgoing = tuple(U.F(x0, sigma0) for sigma0 in U.S.states)
                checked_homogeneity += 1
                assert len(outgoing) == U.s
                assert len(set(outgoing)) == U.s
                assert all(U.contains_level_point(y, n0 + 1) for y in outgoing)

        prev = None
        for n0 in range(0, 24):
            checked_growth += 1
            size = U.level_size(n0)
            assert size == U.k * U.s**n0
            if prev is not None:
                assert size == U.s * prev
                assert size > prev
            prev = size

        shifted_starts = []
        for delta0 in range(0, 16):
            V = U.shifted(delta0)
            checked_shift_family += 1
            shifted_starts.append(V.level_start(0))
            for n0 in range(0, 5):
                assert V.level(n0) == tuple(x + delta0 for x in U.level(n0))
        assert shifted_starts == list(range(U.m, U.m + 16))

        for delta_c in (-3, -2, -1, 1, 2, 3):
            wrong_image = U.first_level_image_from_A(c=U.cA + delta_c)
            checked_alignment_forcing += 1
            assert wrong_image != U.target_L1.values
            assert len(wrong_image) == len(set(wrong_image)) == U.k * U.s

    print(f"[OK] Checked {checked_homogeneity} homogeneous outgoing fibers")
    print(f"[OK] Checked {checked_growth} strict cardinality-growth cases")
    print(f"[OK] Checked {checked_shift_family} sampled shift-family members")
    print(f"[OK] Checked {checked_alignment_forcing} non-endpoint-aligned constants as misaligned cases")


def verify_hierarchical_meta_structure() -> None:
    print("\n=== 9. Hierarchical meta-structure ===")

    checked_level1 = 0
    checked_level2 = 0
    checked_level3 = 0
    checked_countable_indexing = 0

    U = Model(1, 2, 2)

    # Level 1: one concrete triangle with its deterministic levels.
    for n0 in range(0, 8):
        checked_level1 += 1
        assert U.level(n0) == tuple(range(U.level_start(n0), U.level_end(n0) + 1))

    # Level 2: the multispace is sampled by N_0-shifts and each member is
    # isomorphic to the base by the declared shift map.
    sampled_family = {delta0: U.shifted(delta0) for delta0 in range(0, 20)}
    assert tuple(sampled_family.keys()) == tuple(range(20))
    for delta0, V in sampled_family.items():
        checked_countable_indexing += 1
        for n0 in range(0, 6):
            checked_level2 += 1
            assert V.level(n0) == tuple(x + delta0 for x in U.level(n0))

    # Level 3: recursive use of the same shift operator is closed under
    # composition.  This verifies the strict part of the hierarchy and does not
    # promote the illustrative figure to a theorem.
    for delta0 in range(0, 10):
        for gamma in range(0, 10):
            checked_level3 += 1
            lhs = U.shifted(delta0).shifted(gamma)
            rhs = U.shifted(delta0 + gamma)
            for n0 in range(0, 5):
                assert lhs.level(n0) == rhs.level(n0)
                assert lhs.cA == rhs.cA

    print(f"[OK] Checked {checked_level1} level-1 deterministic slices")
    print(f"[OK] Checked {checked_level2} level-2 shifted multispace slices")
    print(f"[OK] Checked {checked_level3} level-3 shift-composition witnesses")
    print(f"[OK] Checked {checked_countable_indexing} sampled countable-family indices")


def verify_cross_section_consistency() -> None:
    print("\n=== Cross-section consistency across all chapter files ===")

    checked = 0

    for U in (Model(1, 2, 2), Model(2, 3, 2), Model(4, 3, 3)):
        for delta0 in range(0, 5):
            V = U.shifted(delta0)
            for n0 in range(0, 5):
                for x0 in U.level(n0)[:min(5, len(U.level(n0)))]:
                    shifted_x = x0 + delta0
                    assert V.contains_level_point(shifted_x, n0)
                    assert U.ancestor_signature(x0, n0)[1] == V.ancestor_signature(shifted_x, n0)[1]

                    for depth in range(0, 4):
                        for y0 in U.future_slice(x0, n0, depth):
                            checked += 1
                            shifted_y = y0 + delta0
                            assert U.causal_leq(x0, n0, y0, n0 + depth)
                            assert V.causal_leq(shifted_x, n0, shifted_y, n0 + depth)
                            assert U.ancestor_signature(y0, n0 + depth)[1] == V.ancestor_signature(shifted_y, n0 + depth)[1]
                            if depth > 0:
                                parent, sigma0 = U.immediate_predecessor(y0, n0 + depth)
                                shifted_parent, shifted_sigma = V.immediate_predecessor(shifted_y, n0 + depth)
                                assert shifted_parent == parent + delta0
                                assert shifted_sigma == sigma0

    print(f"[OK] Checked {checked} shifted causal/hereditary cross-section witnesses")


def verify_failure_guards() -> None:
    print("\n=== Soundness guards and negative tests ===")

    U = Model(1, 2, 2)

    # Invalid core domains.
    expect_raises(DomainError, lambda: Model(0, 2, 2))
    expect_raises(DomainError, lambda: Model(1, 1, 2))
    expect_raises(DomainError, lambda: Model(1, 2, 1))
    expect_raises(TypeError, lambda: Model(1.0, 2, 2))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: Model(1, 2.0, 2))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: Model(1, 2, 2.0))  # type: ignore[arg-type]

    # Invalid level and state requests.
    expect_raises(DomainError, lambda: U.level(-1))
    expect_raises(TypeError, lambda: U.level(1.2))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: U.F(1, 0))
    expect_raises(DomainError, lambda: U.F(1, 3))
    expect_raises(TypeError, lambda: U.F(1, 1.0))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: U.require_level_point(1, 1))
    expect_raises(DomainError, lambda: U.point_from_position(-1, 0))
    expect_raises(DomainError, lambda: U.point_from_position(U.level_size(2), 2))
    expect_raises(DomainError, lambda: U.immediate_predecessor(U.m, 0))
    expect_raises(DomainError, lambda: U.shifted(-1))
    expect_raises(DomainError, lambda: U.locate_level_in_truncation(10_000, 3))
    expect_raises(DomainError, lambda: interval_from_explicit_A((1, 3)))
    expect_raises(DomainError, lambda: interval_from_explicit_A((1, 2, 2)))

    # Wrong c: injective but not endpoint-aligned.
    wrong_c = U.cA + 1
    wrong_image = U.first_level_image_from_A(c=wrong_c)
    assert len(wrong_image) == len(set(wrong_image)) == U.k * U.s
    assert wrong_image != U.target_L1.values

    # Wrong truncation codomain includes L_0, which must not be in the image.
    N = 3
    domain = tuple((x, sigma) for x in U.finite_truncation(N) for sigma in U.S.states)
    image = {U.F(x, sigma) for x, sigma in domain}
    assert image == set(y for level in range(1, N + 2) for y in U.level(level))
    assert image != set(U.finite_truncation(N + 1))
    assert set(U.level(0)).isdisjoint(image)

    # Time order is not causality.
    x_bad = U.point_from_position(1, 1)
    y_bad = U.point_from_position(0, 2)
    assert 1 < 2
    assert not U.causal_leq(x_bad, 1, y_bad, 2)

    # Same-level distinct points are incomparable.
    x1, x2 = U.level(2)[0], U.level(2)[-1]
    assert x1 != x2
    assert not U.causal_leq(x1, 2, x2, 2)
    assert not U.causal_leq(x2, 2, x1, 2)

    # Corrupted predecessor/state pair must fail.
    child = U.level(3)[5]
    parent, sigma0 = U.immediate_predecessor(child, 3)
    if sigma0 < U.s:
        assert U.F(parent, sigma0 + 1) != child
    else:
        assert U.F(parent, sigma0 - 1) != child

    # Inhomogeneous state family is not the chapter's homogeneous S.
    homogeneous = {x: StateSpace(3).states for x in range(1, 8)}
    inhomogeneous = dict(homogeneous)
    inhomogeneous[4] = StateSpace(4).states
    assert len(set(homogeneous.values())) == 1
    assert len(set(inhomogeneous.values())) > 1

    print("[OK] Core invalid domains are rejected")
    print("[OK] Wrong c is injective but not endpoint-aligned")
    print("[OK] Wrong truncation codomain is rejected")
    print("[OK] Time-order/noncausality and same-level incomparability guards pass")
    print("[OK] Corrupted predecessor/state and inhomogeneous S guards pass")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Full mathematical verification of Chapter 1 ===")
    verify_basic_objects()
    verify_space_definition()
    verify_recursive_generation()
    verify_basic_unfolding()
    verify_shift_mechanism()
    verify_geometric_topological_properties()
    verify_time_as_level_index()
    verify_structural_properties()
    verify_hierarchical_meta_structure()
    verify_cross_section_consistency()
    verify_failure_guards()
    print("\n=== Full mathematical Chapter 1 verification completed successfully ===")


if __name__ == "__main__":
    main()
