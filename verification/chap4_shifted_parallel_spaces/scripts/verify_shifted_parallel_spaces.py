"""
VERIFICATION of Chapter 4 core files:

1_shifted_parallel_spaces.tex
2_shift_operator.tex
3_multidim_causality_local_preservation.tex

This script verifies the three files together as one coherent mathematical
module.  It checks the shifted family of generative triangles, the transverse
shift operator between layers, and the tagged global multiposet in which
causality remains localized inside each fiber.

The verification blocks check algebraic, order-theoretic, graph-theoretic, metric,
cardinality, bijectivity, cone-preservation, fiber-localization, and domain
statements.

Verified content
----------------

A. Shifted parallel spaces
   1. A_delta = A + delta = {a+delta : a in A}.
   2. c(A_delta)=c(A)+(s-1)delta.
   3. F_delta(x+delta,sigma)=F_0(x,sigma)+delta.
   4. L_n^(delta)=L_n^(0)+delta, with |L_n^(delta)|=k*s^n.
   5. phi_delta:T_0 -> T_delta, phi_delta(x)=x+delta, is a bijection on
      every finite truncation.
   6. phi_delta preserves levels, generation edges, causal reachability,
      ancestor signatures, future cones, past cones, and cone cardinalities.
   7. d_P(T_delta,T_eta)=|delta-eta| is a metric on the finite sampled
      shift-index set, with nonnegativity, identity, symmetry, and triangle
      inequality.

B. Shift operator Sigma
   8. Sigma_{delta->eta}(x)=x+eta-delta maps T_delta to T_eta.
   9. Sigma_{delta->delta}=id, Sigma_{eta->delta} is the inverse, and
      Sigma_{eta->rho} o Sigma_{delta->eta}=Sigma_{delta->rho}.
  10. Sigma preserves levels and internal level time n.
  11. Sigma commutes with generation:
        F_eta(Sigma_{delta->eta}(x),sigma)
        =
        Sigma_{delta->eta}(F_delta(x,sigma)).
  12. Sigma preserves future and past causal cones:
        Sigma(C^+_delta(x))=C^+_eta(Sigma(x)),
        Sigma(C^-_delta(x))=C^-_eta(Sigma(x)).
  13. Sigma is not inserted into the causal order as a new edge; it is a
      structural isomorphism between already constructed fibers.

C. Tagged global multiposet M
  14. M is the tagged disjoint union:
        M = disjoint_union_delta {delta} x T_delta.
      Elements (delta,x) and (eta,x) are distinct when delta!=eta even if the
      same integer x belongs to both numeric triangles.
  15. The projection pi(delta,x)=delta has fibers
        M_delta={delta} x T_delta.
  16. The global causal relation is fiberwise:
        (delta,x) <=_M (eta,y)
        iff delta=eta and x <=_delta y.
  17. Each fiber is isomorphic to (T_delta, <=_delta).
  18. There is no cross-fiber causality.
  19. Global future and past cones are localized:
        C^+_M(delta,x)={delta} x C^+_delta(x),
        C^-_M(delta,x)={delta} x C^-_delta(x).
  20. The tagged shift operator
        Sigmahat_{delta->eta}(delta,x)=(eta,Sigma_{delta->eta}(x))
      is a poset isomorphism between fibers and maps localized cones to
      localized cones.
  21. (M, <=_M) is a poset and has no nonzero causal cycles.  Every finite
      causal chain in M stays inside one fiber.
  22. Local causality is preserved: adding other fibers does not add or remove
      any causal relation inside a fixed fiber.

D. Negative guards
  23. Cross-layer Sigma is not a causal edge.
  24. Same numeric value in different tags does not imply identity or
      causality.
  25. The raw operation (delta,x)->(eta,x) is not the shift isomorphism in
      general; it may change the internal level and is never made causal.
  26. Time/order inside one fiber does not imply cross-fiber causality.
  27. Wrong target layer, invalid delta/eta, invalid states, invalid vertices,
      invalid levels, noncausal cone requests, and corrupted generation
      witnesses are rejected.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
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
class TaggedVertex:
    delta: int
    x: int


@dataclass(frozen=True, slots=True)
class BaseFamily:
    """The shift-indexed family determined by base parameters (m,k,s)."""

    m: int
    k: int
    s: int

    def __post_init__(self) -> None:
        require_positive_int(self.m, "m")
        require_at_least(self.k, 2, "k")
        require_at_least(self.s, 2, "s")

    @property
    def S(self) -> tuple[int, ...]:
        return tuple(range(1, self.s + 1))

    @property
    def base_c(self) -> int:
        return (self.s - 1) * self.m + 1 - self.k

    def validate_state(self, sigma: object) -> int:
        sigma = require_int(sigma, "sigma")
        if not (1 <= sigma <= self.s):
            raise DomainError("sigma must belong to S={1,...,s}")
        return sigma

    def validate_delta(self, delta: object, name: str = "delta") -> int:
        return require_nonnegative_int(delta, name)

    def A(self, delta: object = 0) -> tuple[int, ...]:
        delta = self.validate_delta(delta)
        return tuple(range(self.m + delta, self.m + delta + self.k))

    def c(self, delta: object = 0) -> int:
        delta = self.validate_delta(delta)
        return (self.s - 1) * (self.m + delta) + 1 - self.k

    def level_start(self, n: object, delta: object = 0) -> int:
        n = require_nonnegative_int(n, "n")
        delta = self.validate_delta(delta)
        numerator = self.k * (self.s**n - 1)
        denominator = self.s - 1
        assert numerator % denominator == 0
        return self.m + delta + numerator // denominator

    def level_size(self, n: object) -> int:
        n = require_nonnegative_int(n, "n")
        return self.k * self.s**n

    def level_end(self, n: object, delta: object = 0) -> int:
        n = require_nonnegative_int(n, "n")
        return self.level_start(n, delta) + self.level_size(n) - 1

    def level(self, n: object, delta: object = 0) -> tuple[int, ...]:
        n = require_nonnegative_int(n, "n")
        return tuple(range(self.level_start(n, delta), self.level_end(n, delta) + 1))

    def finite_block(self, N: object, delta: object = 0) -> tuple[int, ...]:
        N = require_nonnegative_int(N, "N")
        vertices: list[int] = []
        for n in range(N + 1):
            vertices.extend(self.level(n, delta))
        return tuple(vertices)

    def contains_level_point(self, x: object, n: object, delta: object = 0) -> bool:
        if not isinstance(x, int):
            return False
        n = require_nonnegative_int(n, "n")
        delta = self.validate_delta(delta)
        return self.level_start(n, delta) <= x <= self.level_end(n, delta)

    def require_level_point(self, x: object, n: object, delta: object = 0) -> int:
        x = require_positive_int(x, "x")
        if not self.contains_level_point(x, n, delta):
            raise DomainError("x does not belong to the declared shifted level")
        return x

    def locate_level(self, x: object, max_level: object, delta: object = 0) -> int:
        x = require_positive_int(x, "x")
        max_level = require_nonnegative_int(max_level, "max_level")
        for n in range(max_level + 1):
            if self.contains_level_point(x, n, delta):
                return n
        raise DomainError("x is outside the finite shifted block")

    def position(self, x: object, n: object, delta: object = 0) -> int:
        x = self.require_level_point(x, n, delta)
        n = require_nonnegative_int(n, "n")
        return x - self.level_start(n, delta)

    def point_from_position(self, u: object, n: object, delta: object = 0) -> int:
        u = require_nonnegative_int(u, "u")
        n = require_nonnegative_int(n, "n")
        delta = self.validate_delta(delta)
        if u >= self.level_size(n):
            raise DomainError("u is outside the positional range of the shifted level")
        return self.level_start(n, delta) + u

    def F(self, x: object, sigma: object, delta: object = 0) -> int:
        x = require_positive_int(x, "x")
        sigma = self.validate_state(sigma)
        delta = self.validate_delta(delta)
        return self.s * x + sigma - self.c(delta)

    def edge_target(self, x: object, n: object, sigma: object, delta: object = 0) -> int:
        n = require_nonnegative_int(n, "n")
        x = self.require_level_point(x, n, delta)
        y = self.F(x, sigma, delta)
        assert self.contains_level_point(y, n + 1, delta)
        return y

    def local_preimage(self, y: object, previous_level: object, delta: object = 0) -> tuple[int, int]:
        previous_level = require_nonnegative_int(previous_level, "previous_level")
        y = self.require_level_point(y, previous_level + 1, delta)
        child_position = self.position(y, previous_level + 1, delta)
        parent_position = child_position // self.s
        sigma = child_position % self.s + 1
        parent = self.point_from_position(parent_position, previous_level, delta)
        assert self.F(parent, sigma, delta) == y
        return parent, sigma

    def immediate_predecessor(self, y: object, level_of_y: object, delta: object = 0) -> tuple[int, int]:
        level_of_y = require_nonnegative_int(level_of_y, "level_of_y")
        if level_of_y == 0:
            raise DomainError("level-zero vertices have no predecessor in the shifted triangle")
        return self.local_preimage(y, level_of_y - 1, delta)

    def iterate(self, x: object, states: Sequence[int], delta: object = 0) -> int:
        current = require_positive_int(x, "x")
        if not isinstance(states, tuple):
            raise TypeError("state sequence must be a tuple")
        for sigma in states:
            current = self.F(current, sigma, delta)
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

    def causal_leq(self, x: object, nx: object, y: object, ny: object, delta: object = 0) -> bool:
        nx = require_nonnegative_int(nx, "nx")
        ny = require_nonnegative_int(ny, "ny")
        delta = self.validate_delta(delta)
        if nx > ny:
            return False
        x = self.require_level_point(x, nx, delta)
        y = self.require_level_point(y, ny, delta)
        depth = ny - nx
        ux = self.position(x, nx, delta)
        uy = self.position(y, ny, delta)
        lower = self.s**depth * ux
        upper = lower + self.s**depth - 1
        return lower <= uy <= upper

    def future_slice(self, x: object, n: object, depth: object, delta: object = 0) -> tuple[int, ...]:
        n = require_nonnegative_int(n, "n")
        depth = require_nonnegative_int(depth, "depth")
        x = self.require_level_point(x, n, delta)
        ux = self.position(x, n, delta)
        return tuple(
            self.point_from_position(self.s**depth * ux + D, n + depth, delta)
            for D in range(self.s**depth)
        )

    def past_slice(self, x: object, n: object, target_level: object, delta: object = 0) -> tuple[int, ...]:
        return (self.ancestor_at_level(x, n, target_level, delta),)

    def ancestor_at_level(self, x: object, n: object, target_level: object, delta: object = 0) -> int:
        n = require_nonnegative_int(n, "n")
        target_level = require_nonnegative_int(target_level, "target_level")
        delta = self.validate_delta(delta)
        if target_level > n:
            raise DomainError("target_level cannot exceed n")
        current = self.require_level_point(x, n, delta)
        current_level = n
        while current_level > target_level:
            child = current
            parent, sigma = self.immediate_predecessor(child, current_level, delta)
            assert self.F(parent, sigma, delta) == child
            current = parent
            current_level -= 1
        return current

    def ancestor_signature(self, x: object, n: object, delta: object = 0) -> tuple[int, tuple[int, ...]]:
        current = self.require_level_point(x, n, delta)
        n = require_nonnegative_int(n, "n")
        delta = self.validate_delta(delta)
        states: list[int] = []
        level = n
        while level > 0:
            child = current
            parent, sigma = self.immediate_predecessor(child, level, delta)
            assert self.F(parent, sigma, delta) == child
            states.append(sigma)
            current = parent
            level -= 1
        return current, tuple(reversed(states))

    def sigma(self, x: object, delta: object, eta: object) -> int:
        x = require_positive_int(x, "x")
        delta = self.validate_delta(delta, "delta")
        eta = self.validate_delta(eta, "eta")
        return x + eta - delta

    def sigmahat(self, tagged: TaggedVertex, eta: object) -> TaggedVertex:
        eta = self.validate_delta(eta, "eta")
        return TaggedVertex(delta=eta, x=self.sigma(tagged.x, tagged.delta, eta))

    def global_causal_leq(self,
                          a: TaggedVertex,
                          na: object,
                          b: TaggedVertex,
                          nb: object) -> bool:
        na = require_nonnegative_int(na, "na")
        nb = require_nonnegative_int(nb, "nb")
        self.validate_tagged_vertex(a, na)
        self.validate_tagged_vertex(b, nb)
        if a.delta != b.delta:
            return False
        return self.causal_leq(a.x, na, b.x, nb, a.delta)

    def validate_tagged_vertex(self, tagged: TaggedVertex, n: object) -> None:
        if not isinstance(tagged, TaggedVertex):
            raise TypeError("tagged vertex must be TaggedVertex")
        self.validate_delta(tagged.delta, "tagged.delta")
        self.require_level_point(tagged.x, n, tagged.delta)

    def global_future_slice(self, tagged: TaggedVertex, n: object, depth: object) -> tuple[TaggedVertex, ...]:
        self.validate_tagged_vertex(tagged, n)
        return tuple(
            TaggedVertex(tagged.delta, y)
            for y in self.future_slice(tagged.x, n, depth, tagged.delta)
        )

    def global_past_slice(self, tagged: TaggedVertex, n: object, target_level: object) -> tuple[TaggedVertex, ...]:
        self.validate_tagged_vertex(tagged, n)
        return tuple(
            TaggedVertex(tagged.delta, y)
            for y in self.past_slice(tagged.x, n, target_level, tagged.delta)
        )

    def shift_distance(self, delta: object, eta: object) -> int:
        delta = self.validate_delta(delta, "delta")
        eta = self.validate_delta(eta, "eta")
        return abs(delta - eta)


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


def verify_symbolic_shift_formulas() -> None:
    print("\n=== Symbolic verification of shift formulas ===")

    m, k, s, delta, eta, rho = symbols("m k s delta eta rho", integer=True, nonnegative=True)
    x, sigma = symbols("x sigma", integer=True, positive=True)

    c0 = (s - 1) * m + 1 - k
    c_delta = (s - 1) * (m + delta) + 1 - k
    c_eta = (s - 1) * (m + eta) + 1 - k

    assert simplify(c_delta - (c0 + (s - 1) * delta)) == 0
    assert simplify(c_eta - c_delta - (s - 1) * (eta - delta)) == 0

    F_delta_shifted = s * (x + delta) + sigma - c_delta
    F0 = s * x + sigma - c0
    assert simplify(F_delta_shifted - (F0 + delta)) == 0

    Sigma_delta_eta = x + eta - delta
    Sigma_eta_rho_after = Sigma_delta_eta + rho - eta
    Sigma_delta_rho = x + rho - delta
    assert simplify(Sigma_eta_rho_after - Sigma_delta_rho) == 0

    F_eta_after_sigma = s * (x + eta - delta) + sigma - c_eta
    Sigma_after_Fdelta = (s * x + sigma - c_delta) + eta - delta
    assert simplify(F_eta_after_sigma - Sigma_after_Fdelta) == 0

    print("[OK] c(A_delta)=c(A)+(s-1)delta is symbolic")
    print("[OK] F_delta(x+delta,sigma)=F_0(x,sigma)+delta is symbolic")
    print("[OK] Sigma composition and generation commutation are symbolic")


def verify_shifted_parallel_spaces() -> None:
    print("\n=== Verification of shifted parallel spaces ===")

    checked_constants = 0
    checked_levels = 0
    checked_phi_bijections = 0
    checked_edges = 0
    checked_causal_pairs = 0
    checked_cones = 0
    checked_metric_triples = 0

    families = (
        BaseFamily(1, 2, 2),
        BaseFamily(2, 3, 2),
        BaseFamily(3, 2, 3),
        BaseFamily(5, 4, 3),
    )

    for fam in families:
        for delta in range(0, 8):
            checked_constants += 1
            assert fam.A(delta) == tuple(a + delta for a in fam.A(0))
            assert fam.c(delta) == fam.c(0) + (fam.s - 1) * delta

            for n in range(0, 7):
                checked_levels += 1
                assert fam.level(n, delta) == tuple(x + delta for x in fam.level(n, 0))
                assert len(fam.level(n, delta)) == fam.k * fam.s**n

            for N in range(0, 6):
                domain = fam.finite_block(N, 0)
                codomain = fam.finite_block(N, delta)
                assert_bijection(
                    domain,
                    codomain,
                    lambda x, fam=fam, delta=delta: x + delta,
                    "phi_delta finite-truncation bijection",
                )
                checked_phi_bijections += len(domain)

            for n in range(0, 5):
                for x in fam.level(n, 0)[:min(6, len(fam.level(n, 0)))]:
                    for sigma in fam.S:
                        checked_edges += 1
                        assert fam.F(x + delta, sigma, delta) == fam.F(x, sigma, 0) + delta

                    root0, states0 = fam.ancestor_signature(x, n, 0)
                    rootd, statesd = fam.ancestor_signature(x + delta, n, delta)
                    assert rootd == root0 + delta
                    assert statesd == states0

                    for depth in range(0, 4):
                        future0 = fam.future_slice(x, n, depth, 0)
                        futured = fam.future_slice(x + delta, n, depth, delta)
                        checked_cones += 1
                        assert futured == tuple(y + delta for y in future0)
                        assert len(futured) == fam.s**depth

                        for y in future0:
                            checked_causal_pairs += 1
                            assert fam.causal_leq(x, n, y, n + depth, 0)
                            assert fam.causal_leq(x + delta, n, y + delta, n + depth, delta)

        # Discrete metric on sampled shift-index set.
        for delta in range(0, 8):
            assert fam.shift_distance(delta, delta) == 0
            for eta in range(0, 8):
                assert fam.shift_distance(delta, eta) >= 0
                assert fam.shift_distance(delta, eta) == fam.shift_distance(eta, delta)
                assert (fam.shift_distance(delta, eta) == 0) == (delta == eta)
                for rho in range(0, 8):
                    checked_metric_triples += 1
                    assert fam.shift_distance(delta, rho) <= fam.shift_distance(delta, eta) + fam.shift_distance(eta, rho)

    print(f"[OK] Checked {checked_constants} shifted constants and shifted initial intervals")
    print(f"[OK] Checked {checked_levels} shifted levels and cardinalities")
    print(f"[OK] Checked {checked_phi_bijections} phi_delta finite-bijection values")
    print(f"[OK] Checked {checked_edges} shifted generation-edge identities")
    print(f"[OK] Checked {checked_causal_pairs} shifted causal pairs")
    print(f"[OK] Checked {checked_cones} shifted future-cone slices")
    print(f"[OK] Checked {checked_metric_triples} shift-distance triangle cases")


def verify_shift_operator_sigma() -> None:
    print("\n=== Verification of Sigma shift operator between layers ===")

    checked_level_maps = 0
    checked_sigma_bijections = 0
    checked_id_inverse_composition = 0
    checked_generation_commutation = 0
    checked_cone_preservation = 0
    checked_time_preservation = 0

    families = (
        BaseFamily(1, 2, 2),
        BaseFamily(2, 3, 2),
        BaseFamily(4, 2, 3),
    )

    for fam in families:
        for delta in range(0, 6):
            for eta in range(0, 6):
                for n in range(0, 6):
                    domain = fam.level(n, delta)
                    codomain = fam.level(n, eta)
                    image = tuple(fam.sigma(x, delta, eta) for x in domain)
                    checked_level_maps += 1
                    assert image == codomain

                    assert_bijection(
                        domain,
                        codomain,
                        lambda x, fam=fam, delta=delta, eta=eta: fam.sigma(x, delta, eta),
                        "Sigma level bijection",
                    )
                    checked_sigma_bijections += len(domain)

                    for x in domain[:min(8, len(domain))]:
                        checked_time_preservation += 1
                        assert fam.contains_level_point(fam.sigma(x, delta, eta), n, eta)

                        # identity, inverse, and composition
                        assert fam.sigma(x, delta, delta) == x
                        assert fam.sigma(fam.sigma(x, delta, eta), eta, delta) == x
                        for rho in range(0, 6):
                            assert fam.sigma(fam.sigma(x, delta, eta), eta, rho) == fam.sigma(x, delta, rho)
                            checked_id_inverse_composition += 1

                        for sigma in fam.S:
                            checked_generation_commutation += 1
                            lhs = fam.F(fam.sigma(x, delta, eta), sigma, eta)
                            rhs = fam.sigma(fam.F(x, sigma, delta), delta, eta)
                            assert lhs == rhs

                    for x in domain[:min(5, len(domain))]:
                        for depth in range(0, 4):
                            source_future = fam.future_slice(x, n, depth, delta)
                            target_future = fam.future_slice(fam.sigma(x, delta, eta), n, depth, eta)
                            mapped_future = tuple(fam.sigma(y, delta, eta) for y in source_future)
                            checked_cone_preservation += 1
                            assert mapped_future == target_future

                            # Past cone is singleton at each earlier level.
                            for level in range(0, n + 1):
                                source_past = fam.past_slice(x, n, level, delta)
                                target_past = fam.past_slice(fam.sigma(x, delta, eta), n, level, eta)
                                mapped_past = tuple(fam.sigma(y, delta, eta) for y in source_past)
                                assert mapped_past == target_past

    # Binary example with eta=3.
    fam = BaseFamily(1, 2, 2)
    assert fam.level(0, 3) == (4, 5)
    assert fam.level(1, 3) == (6, 7, 8, 9)
    assert tuple(fam.sigma(x, 0, 3) for x in fam.level(1, 0)) == fam.level(1, 3)

    print(f"[OK] Checked {checked_level_maps} Sigma level-set identities")
    print(f"[OK] Checked {checked_sigma_bijections} Sigma level-bijection values")
    print(f"[OK] Checked {checked_id_inverse_composition} identity/inverse/composition cases")
    print(f"[OK] Checked {checked_generation_commutation} Sigma-generation commutation cases")
    print(f"[OK] Checked {checked_cone_preservation} future/past cone preservation cases")
    print(f"[OK] Checked {checked_time_preservation} level-time preservation cases")


def verify_global_tagged_multiposet() -> None:
    print("\n=== Verification of tagged global multipospace M ===")

    checked_fibers = 0
    checked_projection = 0
    checked_no_cross_causality = 0
    checked_global_poset_reflexivity = 0
    checked_global_transitivity = 0
    checked_global_antisymmetry = 0
    checked_cone_localization = 0
    checked_sigmahat = 0
    checked_chains_stay_in_fiber = 0

    fam = BaseFamily(1, 2, 2)

    # Tagged disjointness: same numeric coordinate in different fibers is a
    # different event when both tagged vertices exist.
    tagged_a = TaggedVertex(0, fam.level(1, 0)[-1])  # numeric 6
    tagged_b = TaggedVertex(3, fam.level(0, 3)[0])   # numeric 4, not same; use search below for actual overlap
    assert tagged_a != tagged_b

    for delta in range(0, 6):
        for N in range(0, 5):
            fiber = tuple(TaggedVertex(delta, x) for x in fam.finite_block(N, delta))
            checked_fibers += 1
            assert len(fiber) == len(fam.finite_block(N, delta))
            assert len(set(fiber)) == len(fiber)
            assert all(tv.delta == delta for tv in fiber)
            for tv in fiber[:min(20, len(fiber))]:
                checked_projection += 1
                assert tv.delta == delta

    # Same numeric value may occur in different fibers but tags keep events distinct.
    found_numeric_overlap = False
    for delta in range(0, 5):
        for eta in range(delta + 1, 7):
            block_delta = set(fam.finite_block(4, delta))
            block_eta = set(fam.finite_block(4, eta))
            overlap = sorted(block_delta.intersection(block_eta))
            if overlap:
                x = overlap[0]
                a = TaggedVertex(delta, x)
                b = TaggedVertex(eta, x)
                assert a != b
                assert not fam.global_causal_leq(a, fam.locate_level(x, 4, delta), b, fam.locate_level(x, 4, eta))
                found_numeric_overlap = True
                break
        if found_numeric_overlap:
            break
    assert found_numeric_overlap

    # Poset and fiber locality.
    for delta in range(0, 4):
        for n in range(0, 5):
            for x in fam.level(n, delta)[:min(7, len(fam.level(n, delta)))]:
                a = TaggedVertex(delta, x)
                checked_global_poset_reflexivity += 1
                assert fam.global_causal_leq(a, n, a, n)

                for eta in range(0, 4):
                    if eta == delta:
                        continue
                    b = TaggedVertex(eta, fam.sigma(x, delta, eta))
                    checked_no_cross_causality += 1
                    assert fam.contains_level_point(b.x, n, eta)
                    assert not fam.global_causal_leq(a, n, b, n)
                    assert not fam.global_causal_leq(b, n, a, n)

                for depth1 in range(0, 3):
                    for y in fam.future_slice(x, n, depth1, delta):
                        b = TaggedVertex(delta, y)
                        for depth2 in range(0, 3):
                            for z in fam.future_slice(y, n + depth1, depth2, delta):
                                c = TaggedVertex(delta, z)
                                checked_global_transitivity += 1
                                assert fam.global_causal_leq(a, n, b, n + depth1)
                                assert fam.global_causal_leq(b, n + depth1, c, n + depth1 + depth2)
                                assert fam.global_causal_leq(a, n, c, n + depth1 + depth2)

                sample_same_level = fam.level(n, delta)[:min(8, len(fam.level(n, delta)))]
                for y in sample_same_level:
                    b = TaggedVertex(delta, y)
                    checked_global_antisymmetry += 1
                    if fam.global_causal_leq(a, n, b, n) and fam.global_causal_leq(b, n, a, n):
                        assert a == b

                for depth in range(0, 4):
                    global_future = fam.global_future_slice(a, n, depth)
                    local_future = fam.future_slice(x, n, depth, delta)
                    checked_cone_localization += 1
                    assert global_future == tuple(TaggedVertex(delta, y) for y in local_future)
                    assert all(tv.delta == delta for tv in global_future)

                    if depth > 0:
                        for tv in global_future:
                            checked_chains_stay_in_fiber += 1
                            assert tv.delta == delta

                for eta in range(0, 4):
                    sigmahat_a = fam.sigmahat(a, eta)
                    checked_sigmahat += 1
                    assert sigmahat_a == TaggedVertex(eta, fam.sigma(x, delta, eta))
                    assert fam.contains_level_point(sigmahat_a.x, n, eta)

                    for depth in range(0, 3):
                        mapped_cone = tuple(fam.sigmahat(tv, eta) for tv in fam.global_future_slice(a, n, depth))
                        target_cone = fam.global_future_slice(sigmahat_a, n, depth)
                        assert mapped_cone == target_cone

    print(f"[OK] Checked {checked_fibers} finite tagged fibers")
    print(f"[OK] Checked {checked_projection} projection values pi(delta,x)=delta")
    print(f"[OK] Checked {checked_no_cross_causality} no-cross-fiber causality cases")
    print(f"[OK] Checked {checked_global_poset_reflexivity} global reflexivity cases")
    print(f"[OK] Checked {checked_global_transitivity} global transitivity cases")
    print(f"[OK] Checked {checked_global_antisymmetry} global antisymmetry cases")
    print(f"[OK] Checked {checked_cone_localization} localized global cone slices")
    print(f"[OK] Checked {checked_sigmahat} Sigmahat fiber-isomorphism cases")
    print(f"[OK] Checked {checked_chains_stay_in_fiber} chain-locality witnesses")


def verify_raw_numeric_identification_guard() -> None:
    print("\n=== Negative guard: raw numeric identification is not Sigma and not causality ===")

    fam = BaseFamily(1, 2, 2)
    checked_level_changes = 0
    checked_not_sigma = 0
    checked_not_causal = 0

    for delta in range(0, 4):
        for eta in range(0, 4):
            if delta == eta:
                continue

            for n in range(0, 5):
                for x in fam.level(n, delta)[:min(10, len(fam.level(n, delta)))]:
                    # Correct Sigma sends x to x+eta-delta and preserves level.
                    sigma_x = fam.sigma(x, delta, eta)
                    assert fam.contains_level_point(sigma_x, n, eta)

                    # Raw numeric identification would keep x unchanged.  If the
                    # same integer x is present in the target finite window, it
                    # generally has no reason to be on the same level.
                    if x in fam.finite_block(7, eta):
                        target_level = fam.locate_level(x, 7, eta)
                        if target_level != n:
                            checked_level_changes += 1
                        if x != sigma_x:
                            checked_not_sigma += 1

                        a = TaggedVertex(delta, x)
                        b = TaggedVertex(eta, x)
                        # Regardless of level, cross-fiber causal relation is false.
                        checked_not_causal += 1
                        assert not fam.global_causal_leq(a, n, b, target_level)
                        assert not fam.global_causal_leq(b, target_level, a, n)

    assert checked_not_sigma > 0
    assert checked_not_causal > 0
    print(f"[OK] Checked {checked_level_changes} raw same-number identifications that change level")
    print(f"[OK] Checked {checked_not_sigma} raw same-number maps distinct from Sigma")
    print(f"[OK] Checked {checked_not_causal} raw same-number cross-fiber noncausality cases")


def verify_negative_domains_and_corruption_guards() -> None:
    print("\n=== Negative domain and corruption guards ===")

    fam = BaseFamily(1, 2, 2)

    expect_raises(DomainError, lambda: BaseFamily(0, 2, 2))
    expect_raises(DomainError, lambda: BaseFamily(1, 1, 2))
    expect_raises(DomainError, lambda: BaseFamily(1, 2, 1))
    expect_raises(TypeError, lambda: BaseFamily(1.0, 2, 2))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: BaseFamily(1, 2.0, 2))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: BaseFamily(1, 2, 2.0))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: fam.A(-1))
    expect_raises(DomainError, lambda: fam.level(-1, 0))
    expect_raises(DomainError, lambda: fam.level(0, -1))
    expect_raises(TypeError, lambda: fam.level(1.5, 0))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: fam.level(1, 0.5))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: fam.require_level_point(1, 1, 0))
    expect_raises(DomainError, lambda: fam.point_from_position(-1, 0, 0))
    expect_raises(DomainError, lambda: fam.point_from_position(fam.level_size(2), 2, 0))

    expect_raises(DomainError, lambda: fam.F(1, 0, 0))
    expect_raises(DomainError, lambda: fam.F(1, 3, 0))
    expect_raises(TypeError, lambda: fam.F(1, 1.0, 0))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: fam.sigma(1, -1, 0))
    expect_raises(DomainError, lambda: fam.sigma(1, 0, -1))
    expect_raises(DomainError, lambda: fam.immediate_predecessor(fam.level(0, 0)[0], 0, 0))
    expect_raises(DomainError, lambda: fam.states_from_digit(-1, 2))
    expect_raises(DomainError, lambda: fam.states_from_digit(4, 2))
    expect_raises(TypeError, lambda: fam.iterate(1, [1, 2], 0))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: fam.iterate(1, (1, 3), 0))
    expect_raises(DomainError, lambda: fam.ancestor_at_level(fam.level(0, 0)[0], 0, 1, 0))

    # Corrupted generation witness in a shifted layer.
    delta = 3
    x = fam.point_from_position(1, 1, delta)
    y = fam.F(x, 1, delta)
    assert fam.contains_level_point(y, 2, delta)
    assert fam.F(x, 2, delta) != y

    # Target layer discipline for Sigma: the shifted numerical value belongs
    # to the target layer with the target tag.  It may accidentally also lie in
    # the old numerical interval because different shifted triangles can
    # overlap as subsets of N; the tagged vertex is nevertheless different.
    eta = 5
    sigma_x = fam.sigma(x, delta, eta)
    assert fam.contains_level_point(sigma_x, 1, eta)
    a = TaggedVertex(delta, x)
    b = TaggedVertex(eta, sigma_x)
    assert a != b

    # Noncausal cross-fiber cone request is rejected by global relation.
    assert not fam.global_causal_leq(a, 1, b, 1)

    expect_raises(TypeError, lambda: fam.validate_tagged_vertex(("not", "tagged"), 0))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: fam.validate_tagged_vertex(TaggedVertex(-1, 1), 0))
    expect_raises(DomainError, lambda: fam.validate_tagged_vertex(TaggedVertex(0, 1000), 0))

    print("[OK] Invalid parameters, shifts, levels, states, vertices, and paths are rejected")
    print("[OK] Corrupted generation, wrong-layer, and cross-fiber causality guards pass")


def main() -> None:
    print("=== Integrated verification of shifted multipospace causality ===")
    verify_symbolic_shift_formulas()
    verify_shifted_parallel_spaces()
    verify_shift_operator_sigma()
    verify_global_tagged_multiposet()
    verify_raw_numeric_identification_guard()
    verify_negative_domains_and_corruption_guards()
    print("\n=== Shifted multipospace integrated verification completed successfully ===")


if __name__ == "__main__":
    main()
