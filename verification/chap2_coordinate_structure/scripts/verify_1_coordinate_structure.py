"""
VERIFICATION of Section 1: Coordinate structure
(file: 1_coordinate_structure.tex).

This script provides a full mathematical verification block for the claims in
1_coordinate_structure.tex.  It checks the time coordinate and the unique
path-generating decomposition

    Psi : disjoint_union_{n >= 0} (A x S^n) -> T,
          (a, (sigma_1, ..., sigma_n)) |-> F^n(a; sigma_1, ..., sigma_n),

without re-proving earlier global bijectivity results as external theorems.
Instead, it verifies the new coordinate-structure layer by exact symbolic
identities, exhaustive finite models, inverse decoding, example checks, and
negative domain guards.

Verified content
----------------
1. The time coordinate is the level index:
       t(x) = n  iff  x in L_n.

2. The levels are consecutive, disjoint integer intervals with
       m_n = m + k(s^n - 1)/(s - 1),
       |L_n| = k s^n,
       max(L_n) + 1 = min(L_{n+1}).

3. For S = {1, ..., s}, the finite sequence sets satisfy
       S^0 = {empty tuple},   |S^n| = s^n,
   and the tagged domains A x S^n are disjoint by depth.

4. The iterated generator
       F^n(a; sigma_1, ..., sigma_n)
   maps A x S^n exactly onto L_n, with no collisions and no omissions.

5. Every x in L_n has a unique root a in A and a unique internal-state
   sequence of length n.  The inverse decoder recovers the same data from x.

6. The empty sequence represents exactly the initial level A and no positive
   depth element.

7. For k > 1, the same internal-state sequence may occur above distinct
   roots; therefore the sequence alone is not the full path data.  The pair
   (a, sequence) is required.

8. Causal direction is compatible with time:
       x ancestor of y  ==>  t(x) < t(y)
   for nontrivial ancestry.  The converse is false and is checked by an exact
   counterexample.

9. The numerical illustration in the file is verified exactly for s = 2,
   A = {1, 2}: the proposed incorrect path to 11 fails, while the stated paths
   for 11 and 14 succeed.

10. Deterministic genealogical uniqueness coexists with exponential branching:
       each root has s^n descendants at depth n,
       |L_n| = |A| s^n.
   No external randomness is introduced by this verification.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from sympy import cancel, symbols


StateSeq = tuple[int, ...]


@dataclass(frozen=True)
class GenerativeModel:
    m: int
    k: int
    s: int

    def __post_init__(self) -> None:
        if not isinstance(self.m, int) or not isinstance(self.k, int) or not isinstance(self.s, int):
            raise TypeError("m, k and s must be integers")
        if self.m < 1:
            raise ValueError("m must be a positive integer")
        if self.k < 2:
            raise ValueError("k must be at least 2")
        if self.s < 2:
            raise ValueError("s must be at least 2")

    @property
    def A(self) -> tuple[int, ...]:
        return tuple(range(self.m, self.m + self.k))

    @property
    def S(self) -> tuple[int, ...]:
        return tuple(range(1, self.s + 1))

    @property
    def c(self) -> int:
        return (self.s - 1) * self.m + 1 - self.k

    def level_start(self, n: int) -> int:
        if not isinstance(n, int) or n < 0:
            raise ValueError("level index must be a nonnegative integer")
        numerator = self.k * (self.s**n - 1)
        denominator = self.s - 1
        assert numerator % denominator == 0
        return self.m + numerator // denominator

    def level_size(self, n: int) -> int:
        if not isinstance(n, int) or n < 0:
            raise ValueError("level index must be a nonnegative integer")
        return self.k * self.s**n

    def level_end(self, n: int) -> int:
        return self.level_start(n) + self.level_size(n) - 1

    def level(self, n: int) -> tuple[int, ...]:
        return tuple(range(self.level_start(n), self.level_end(n) + 1))

    def in_level(self, x: int, n: int) -> bool:
        return self.level_start(n) <= x <= self.level_end(n)

    def F(self, x: int, sigma: int) -> int:
        if sigma not in self.S:
            raise ValueError("internal state is outside S")
        if x < 1:
            raise ValueError("x must be a positive integer")
        return self.s * x + sigma - self.c

    def iterate(self, root: int, seq: StateSeq) -> int:
        if root not in self.A:
            raise ValueError("root is outside A")
        check_state_sequence(seq, self.s)
        x = root
        for sigma in seq:
            x = self.F(x, sigma)
        return x

    def time_of(self, x: int, max_depth: int | None = None) -> int:
        if not isinstance(x, int) or x < 1:
            raise ValueError("x must be a positive integer")
        depth = 0
        while True:
            if self.level_start(depth) <= x <= self.level_end(depth):
                return depth
            depth += 1
            if max_depth is not None and depth > max_depth:
                raise ValueError("x was not found up to the requested depth")
            if max_depth is None and self.level_start(depth) > x:
                raise ValueError("x is not in T for this model")

    def inverse_step(self, y: int) -> tuple[int, int]:
        if not isinstance(y, int) or y < 1:
            raise ValueError("y must be a positive integer")
        z = y + self.c
        residue = z % self.s
        sigma = self.s if residue == 0 else residue
        predecessor = (z - sigma) // self.s
        if predecessor < 1 or sigma not in self.S:
            raise ValueError("y has no valid immediate predecessor in this model")
        if self.F(predecessor, sigma) != y:
            raise AssertionError("inverse-step reconstruction failed")
        return predecessor, sigma

    def decode(self, x: int) -> tuple[int, StateSeq]:
        depth = self.time_of(x)
        current = x
        reversed_states: list[int] = []
        for current_depth in range(depth, 0, -1):
            predecessor, sigma = self.inverse_step(current)
            assert self.in_level(predecessor, current_depth - 1)
            reversed_states.append(sigma)
            current = predecessor
        if current not in self.A:
            raise AssertionError("decoded root is outside A")
        return current, tuple(reversed(reversed_states))

    def is_ancestor(self, x: int, y: int) -> bool:
        tx = self.time_of(x)
        ty = self.time_of(y)
        if tx > ty:
            return False
        root_x, seq_x = self.decode(x)
        root_y, seq_y = self.decode(y)
        return root_x == root_y and seq_y[: len(seq_x)] == seq_x


def check_state_sequence(seq: StateSeq, s: int) -> None:
    if not isinstance(seq, tuple):
        raise TypeError("internal-state sequence must be a tuple")
    if s < 2:
        raise ValueError("s must be at least 2")
    for sigma in seq:
        if not isinstance(sigma, int) or not (1 <= sigma <= s):
            raise ValueError("internal state is outside S")


def state_sequences(s: int, n: int) -> tuple[StateSeq, ...]:
    if not isinstance(s, int) or not isinstance(n, int):
        raise TypeError("s and n must be integers")
    if s < 2:
        raise ValueError("s must be at least 2")
    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return ((),)
    return tuple(product(range(1, s + 1), repeat=n))


def expect_raises(exc_type: type[BaseException], fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__} was not raised")


def verify_symbolic_level_structure() -> None:
    print("\n=== Symbolic verification of the level/time coordinate identities ===")

    m, k, s, n = symbols("m k s n", integer=True, positive=True)
    c = (s - 1) * m + 1 - k
    m_n = m + k * (s**n - 1) / (s - 1)
    m_np1 = m + k * (s ** (n + 1) - 1) / (s - 1)
    size_n = k * s**n
    max_n = m_n + size_n - 1

    assert cancel((max_n + 1) - m_np1) == 0
    assert cancel(m_np1 - (s * m_n + 1 - c)) == 0
    assert cancel((m_np1 - m_n) - size_n) == 0

    print("[OK] Symbolic formulas for m_n, level adjacency and the time-level recursion are exact")


def verify_finite_sequence_sets() -> None:
    print("\n=== Verification of S^n and tagged finite-sequence domains ===")

    for s in range(2, 8):
        assert state_sequences(s, 0) == ((),)
        for n in range(0, 6):
            seqs = state_sequences(s, n)
            assert len(seqs) == s**n
            for seq in seqs:
                assert len(seq) == n
                check_state_sequence(seq, s)

        # Tagged domain disjointness by depth: equal tagged objects must have equal depth.
        tagged = {(n, seq) for n in range(0, 5) for seq in state_sequences(s, n)}
        assert len(tagged) == sum(s**n for n in range(0, 5))

    print("[OK] S^0, |S^n| = s^n, state admissibility and depth-tag disjointness verified")


def verify_psi_bijection_on_finite_levels() -> None:
    print("\n=== Exhaustive verification of Psi : A x S^n -> L_n on finite models ===")

    models = [
        GenerativeModel(m=1, k=2, s=2),
        GenerativeModel(m=2, k=3, s=2),
        GenerativeModel(m=1, k=4, s=3),
        GenerativeModel(m=5, k=2, s=4),
        GenerativeModel(m=3, k=5, s=5),
    ]

    total_checked = 0
    for model in models:
        for n in range(0, 6):
            images: dict[int, tuple[int, StateSeq]] = {}
            domain_count = 0
            for root in model.A:
                for seq in state_sequences(model.s, n):
                    y = model.iterate(root, seq)
                    domain_count += 1
                    assert model.in_level(y, n)
                    if y in images:
                        raise AssertionError(f"Collision in Psi for model={model}, n={n}, y={y}")
                    images[y] = (root, seq)

                    decoded_root, decoded_seq = model.decode(y)
                    assert decoded_root == root
                    assert decoded_seq == seq
                    assert model.time_of(y) == n

            assert domain_count == model.k * model.s**n
            assert set(images) == set(model.level(n))
            assert len(images) == len(model.level(n))
            total_checked += domain_count

    print(f"[OK] Checked {total_checked} exact Psi images with inverse decoding and no collisions")


def verify_empty_sequence_and_root_dependence() -> None:
    print("\n=== Verification of empty sequence and root dependence ===")

    model = GenerativeModel(m=1, k=2, s=2)

    for root in model.A:
        assert model.iterate(root, ()) == root
        assert model.time_of(root) == 0
        decoded_root, decoded_seq = model.decode(root)
        assert decoded_root == root
        assert decoded_seq == ()

    positive_depth_element = model.iterate(model.A[0], (1,))
    assert positive_depth_element not in model.A
    decoded_root, decoded_seq = model.decode(positive_depth_element)
    assert decoded_seq != ()
    assert len(decoded_seq) == 1

    common_seq = (1, 1)
    first = model.iterate(model.A[0], common_seq)
    second = model.iterate(model.A[1], common_seq)
    assert first != second
    assert model.decode(first) == (model.A[0], common_seq)
    assert model.decode(second) == (model.A[1], common_seq)

    print("[OK] Empty sequence belongs exactly to level 0; root data is necessary when k > 1")


def verify_causal_time_direction_and_nonconverse() -> None:
    print("\n=== Verification of causal direction versus time coordinate ===")

    model = GenerativeModel(m=1, k=2, s=2)

    reflexive_checks = 0
    for n in range(0, 5):
        for x in model.level(n):
            assert model.is_ancestor(x, x)
            reflexive_checks += 1

    count = 0
    for n in range(0, 4):
        for x in model.level(n):
            root, seq = model.decode(x)
            for extension_len in range(1, 4):
                for extension in state_sequences(model.s, extension_len):
                    y = model.iterate(root, seq + extension)
                    assert model.is_ancestor(x, y)
                    assert model.time_of(x) < model.time_of(y)
                    count += 1

    x = 1
    y = 11  # y = F^2(2; 1, 1), not a descendant of 1.
    assert model.time_of(x) == 0
    assert model.time_of(y) == 2
    assert model.time_of(x) < model.time_of(y)
    assert not model.is_ancestor(x, y)

    print(f"[OK] Checked {reflexive_checks} reflexive causal-reachability cases")
    print(f"[OK] Checked {count} nontrivial ancestry cases with strict time increase")
    print("[OK] Exact counterexample confirms that t(x) < t(y) does not imply ancestry")


def verify_numerical_example_from_section() -> None:
    print("\n=== Verification of the numerical illustration for s = 2, A = {1, 2} ===")

    model = GenerativeModel(m=1, k=2, s=2)

    incorrect_candidate = model.iterate(1, (2, 1))
    assert incorrect_candidate == 9
    assert incorrect_candidate != 11

    assert model.iterate(2, (1, 1)) == 11
    assert model.decode(11) == (2, (1, 1))

    assert model.iterate(2, (2, 2)) == 14
    assert model.decode(14) == (2, (2, 2))

    assert set(model.level(2)) == set(range(7, 15))
    for x in model.level(2):
        root, seq = model.decode(x)
        assert root in model.A
        assert len(seq) == 2
        assert model.iterate(root, seq) == x

    print("[OK] The incorrect path to 11 fails exactly; the stated paths for 11 and 14 succeed")
    print("[OK] Every element of L_2 has exactly one root and one length-2 internal-state sequence")


def verify_deterministic_branching_counts() -> None:
    print("\n=== Verification of deterministic uniqueness with exponential branching ===")

    for model in [GenerativeModel(1, 2, 2), GenerativeModel(4, 3, 3), GenerativeModel(2, 5, 4)]:
        for n in range(0, 7):
            per_root_counts = []
            for root in model.A:
                descendants = {model.iterate(root, seq) for seq in state_sequences(model.s, n)}
                assert len(descendants) == model.s**n
                for y in descendants:
                    decoded_root, _ = model.decode(y)
                    assert decoded_root == root
                per_root_counts.append(len(descendants))

            assert all(count == model.s**n for count in per_root_counts)
            assert sum(per_root_counts) == model.level_size(n)

    print("[OK] Each root has exactly s^n descendants at depth n and |L_n| = |A| s^n")


def verify_inverse_step_arithmetic() -> None:
    print("\n=== Verification of immediate predecessor and internal-state recovery ===")

    for model in [GenerativeModel(1, 2, 2), GenerativeModel(2, 4, 3), GenerativeModel(5, 3, 6)]:
        for n in range(0, 5):
            for x in model.level(n):
                for sigma in model.S:
                    y = model.F(x, sigma)
                    predecessor, recovered_sigma = model.inverse_step(y)
                    assert predecessor == x
                    assert recovered_sigma == sigma
                    assert model.in_level(y, n + 1)

    print("[OK] Immediate inverse decoding recovers the unique predecessor and internal state")


def verify_negative_domain_guards() -> None:
    print("\n=== Negative domain and corruption tests ===")

    expect_raises(ValueError, GenerativeModel, 0, 2, 2)
    expect_raises(ValueError, GenerativeModel, 1, 1, 2)
    expect_raises(ValueError, GenerativeModel, 1, 2, 1)
    expect_raises(TypeError, GenerativeModel, 1.0, 2, 2)

    model = GenerativeModel(m=1, k=2, s=2)

    expect_raises(ValueError, model.F, 1, 0)
    expect_raises(ValueError, model.F, 1, 3)
    expect_raises(ValueError, model.F, 0, 1)
    expect_raises(ValueError, model.iterate, 3, (1,))
    expect_raises(ValueError, model.iterate, 1, (1, 3))
    expect_raises(TypeError, model.iterate, 1, [1, 2])
    expect_raises(ValueError, model.level_start, -1)
    expect_raises(ValueError, model.time_of, 0)
    expect_raises(ValueError, state_sequences, 1, 2)
    expect_raises(ValueError, state_sequences, 2, -1)

    # A corrupted object from outside T for this model must not decode.
    # For m=1,k=2,s=2 the structure covers all positive integers, so use another model.
    shifted = GenerativeModel(m=5, k=2, s=2)
    expect_raises(ValueError, shifted.time_of, 1)

    print("[OK] Invalid parameters, states, roots, depths, and corrupted elements are rejected")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of coordinate structure (1_coordinate_structure.tex) ===")
    verify_symbolic_level_structure()
    verify_finite_sequence_sets()
    verify_psi_bijection_on_finite_levels()
    verify_empty_sequence_and_root_dependence()
    verify_causal_time_direction_and_nonconverse()
    verify_numerical_example_from_section()
    verify_deterministic_branching_counts()
    verify_inverse_step_arithmetic()
    verify_negative_domain_guards()
    print("\n=== Coordinate structure verification completed successfully ===")


if __name__ == "__main__":
    main()
