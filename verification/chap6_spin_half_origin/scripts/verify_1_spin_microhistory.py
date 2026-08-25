"""
VERIFICATION of Section: Binary microhistory of a generative element
(sec:spin-microhistory).

This script provides a full mathematical verification block for the claims in
1_spin_microhistory.tex.  It verifies only the new content of this file: the
binary internal-state history, the complete genealogical code, the difference
between the internal sequence and the full causal genealogy, the exact counting
of binary histories, and the centered half-integer normalization for s = 2.

Previously established facts such as global bijectivity of the generative map,
level disjointness, and unique path decomposition are treated here as structural
dependencies.  The script uses them through explicit finite-level models and
does not re-prove the earlier general theory.

Verified content
----------------
1. Binary generative setting:
      s = 2, S = {1, 2}, A = {m, ..., m+k-1}, k >= 2.

2. Complete genealogical code:
      Pi(x) = (a, chi(x)) in A x {1,2}^n.

3. Exact finite-level bijection:
      Pi : L_n -> A x {1,2}^n
   is checked by constructing all finite codes and all generated elements.

4. Uniqueness of the binary microhistory:
      every x in L_n has exactly one recovered internal-state sequence.

5. Limitation of chi alone:
      for k > 1, the same internal-state sequence corresponds to k distinct
      elements, one for each initial element a in A.  Hence chi is not the full
      causal genealogy.

6. Fixed-origin internal genealogy:
      after fixing a, the internal-state sequence determines the whole causal
      chain of intermediate elements.

7. Counting:
      number of distinct binary internal-state sequences at depth n is 2^n;
      each initial element produces exactly 2^n distinct generated elements;
      the whole level has k*2^n elements.

8. Centered binary representation:
      epsilon(sigma) = sigma - 3/2.

9. Half-integer spectrum:
      epsilon({1,2}) = {-1/2, +1/2};
      the spectrum is symmetric around zero and has radius 1/2.

10. Uniqueness among translations sigma -> sigma - lambda:
      centering forces lambda = 3/2.

11. Non-uniqueness among general affine normalizations:
      eta(sigma) = alpha*sigma + beta is centered iff beta = -3*alpha/2;
      the centered spectrum is {-alpha/2, +alpha/2}.  Unit-spacing normalization
      fixes alpha = 1 and beta = -3/2.

12. Negative domain tests:
      invalid internal states, invalid initial elements, length mismatches,
      and non-binary settings are rejected.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Callable

from sympy import Eq, Rational, simplify, solve, symbols


BinaryPrefix = tuple[int, ...]


def expect_raises(label: str, expected: type[BaseException] | tuple[type[BaseException], ...], fn: Callable[[], object]) -> None:
    """Require fn() to fail with the expected exception class."""
    try:
        fn()
    except expected:
        print(f"[OK] Rejected invalid case: {label}")
        return
    except Exception as exc:  # pragma: no cover - diagnostic branch
        raise AssertionError(f"{label}: raised unexpected exception {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"{label}: invalid input was accepted")


@dataclass(frozen=True)
class BinaryGenerativeSpace:
    """Finite-level s=2 generative model used to verify the section's claims."""

    m: int
    k: int

    def __post_init__(self) -> None:
        if not isinstance(self.m, int) or self.m < 1:
            raise ValueError("m must be a positive integer")
        if not isinstance(self.k, int) or self.k < 2:
            raise ValueError("k must be an integer >= 2")

    @property
    def A(self) -> tuple[int, ...]:
        return tuple(range(self.m, self.m + self.k))

    @property
    def c_shift(self) -> int:
        # c(A) = (s-1)m + 1 - k, specialized to s = 2.
        return self.m + 1 - self.k

    def F(self, x: int, sigma: int) -> int:
        if sigma not in (1, 2):
            raise ValueError(f"invalid internal state {sigma}; expected 1 or 2")
        return 2 * x + sigma - self.c_shift

    def level_start(self, n: int) -> int:
        if n < 0:
            raise ValueError("level index must be nonnegative")
        return self.m + self.k * (2**n - 1)

    def level(self, n: int) -> tuple[int, ...]:
        start = self.level_start(n)
        return tuple(range(start, start + self.k * 2**n))

    def validate_prefix(self, prefix: BinaryPrefix, n: int | None = None) -> None:
        if not isinstance(prefix, tuple):
            raise TypeError("internal-state prefix must be a tuple")
        if n is not None and len(prefix) != n:
            raise ValueError(f"expected length {n}, got {len(prefix)}")
        for sigma in prefix:
            if sigma not in (1, 2):
                raise ValueError(f"invalid internal state {sigma}; expected 1 or 2")

    def iterate(self, a: int, prefix: BinaryPrefix) -> int:
        if a not in self.A:
            raise ValueError(f"initial element {a} is not in A={self.A}")
        self.validate_prefix(prefix)
        x = a
        for sigma in prefix:
            x = self.F(x, sigma)
        return x

    def causal_chain(self, a: int, prefix: BinaryPrefix) -> tuple[int, ...]:
        if a not in self.A:
            raise ValueError(f"initial element {a} is not in A={self.A}")
        self.validate_prefix(prefix)
        chain = [a]
        x = a
        for sigma in prefix:
            x = self.F(x, sigma)
            chain.append(x)
        return tuple(chain)

    def binary_prefixes(self, n: int) -> tuple[BinaryPrefix, ...]:
        if n < 0:
            raise ValueError("depth must be nonnegative")
        return tuple(tuple(p) for p in product((1, 2), repeat=n))

    def full_code_map(self, n: int) -> dict[int, tuple[int, BinaryPrefix]]:
        mapping: dict[int, tuple[int, BinaryPrefix]] = {}
        for a in self.A:
            for prefix in self.binary_prefixes(n):
                x = self.iterate(a, prefix)
                if x in mapping:
                    raise AssertionError(f"duplicate generated element {x}: {mapping[x]} and {(a, prefix)}")
                mapping[x] = (a, prefix)
        return mapping

    def recover_code(self, x: int, n: int) -> tuple[int, BinaryPrefix]:
        mapping = self.full_code_map(n)
        if x not in mapping:
            raise ValueError(f"element {x} is not in L_{n}")
        return mapping[x]


def centered_binary_value(sigma: int) -> Fraction:
    if sigma not in (1, 2):
        raise ValueError(f"invalid internal state {sigma}; expected 1 or 2")
    return Fraction(2 * sigma - 3, 2)


def affine_centered_values(alpha: Fraction) -> tuple[Fraction, Fraction]:
    if alpha == 0:
        raise ValueError("alpha must be nonzero for a nondegenerate affine normalization")
    beta = -Fraction(3, 2) * alpha
    return alpha + beta, 2 * alpha + beta


def verify_exact_code_bijection_and_counting() -> None:
    print("\n=== Exact finite verification of the complete genealogical code ===")

    tested_instances = [
        BinaryGenerativeSpace(m=1, k=2),
        BinaryGenerativeSpace(m=2, k=3),
        BinaryGenerativeSpace(m=5, k=4),
        BinaryGenerativeSpace(m=9, k=6),
    ]

    for space in tested_instances:
        for n in range(0, 8):
            prefixes = space.binary_prefixes(n)
            mapping = space.full_code_map(n)
            level = set(space.level(n))

            assert len(prefixes) == 2**n
            assert len(mapping) == space.k * 2**n
            assert set(mapping.keys()) == level
            assert len(level) == space.k * 2**n

            # Pi is a bijection: every generated element has one and only one
            # pair (a, internal-state prefix), and every pair generates one
            # element at L_n.
            image_codes = set(mapping.values())
            expected_codes = {(a, p) for a in space.A for p in prefixes}
            assert image_codes == expected_codes

            for x, (a, prefix) in mapping.items():
                recovered_a, recovered_prefix = space.recover_code(x, n)
                assert recovered_a == a
                assert recovered_prefix == prefix
                assert len(recovered_prefix) == n
                assert all(sigma in (1, 2) for sigma in recovered_prefix)

    print("[OK] Pi: L_n -> A x {1,2}^n is bijective across multiple exact finite models")
    print("[OK] Counts |{1,2}^n| = 2^n and |L_n| = k*2^n hold exactly")


def verify_microhistory_limitation_and_fixed_origin_genealogy() -> None:
    print("\n=== Verification of chi versus full causal genealogy ===")

    space = BinaryGenerativeSpace(m=3, k=5)

    for n in range(0, 7):
        prefixes = space.binary_prefixes(n)
        mapping = space.full_code_map(n)

        # chi alone is not injective when k > 1: each internal-state prefix has
        # exactly k preimages, one for each initial element.
        for prefix in prefixes:
            generated = [space.iterate(a, prefix) for a in space.A]
            assert len(generated) == space.k
            assert len(set(generated)) == space.k
            assert all(x in mapping for x in generated)
            assert {mapping[x][0] for x in generated} == set(space.A)
            assert {mapping[x][1] for x in generated} == {prefix}

        # For a fixed initial element, the internal-state prefix determines the
        # entire causal chain.
        for a in space.A:
            chains = {prefix: space.causal_chain(a, prefix) for prefix in prefixes}
            assert len(set(chains.values())) == len(prefixes)
            for prefix, chain in chains.items():
                assert len(chain) == n + 1
                assert chain[0] == a
                assert chain[-1] == space.iterate(a, prefix)
                for i, sigma in enumerate(prefix):
                    assert chain[i + 1] == space.F(chain[i], sigma)

    print("[OK] The internal-state sequence alone omits the initial element when k > 1")
    print("[OK] Once a is fixed, the internal-state sequence determines the full finite causal chain")


def verify_fixed_origin_injectivity_and_images() -> None:
    print("\n=== Verification of fixed-origin generation ===")

    for space in (BinaryGenerativeSpace(m=1, k=2), BinaryGenerativeSpace(m=4, k=3), BinaryGenerativeSpace(m=11, k=5)):
        for n in range(0, 8):
            prefixes = space.binary_prefixes(n)
            for a in space.A:
                image = [space.iterate(a, prefix) for prefix in prefixes]
                assert len(image) == 2**n
                assert len(set(image)) == 2**n
                assert all(x in space.level(n) for x in image)

                # The images for different initial elements are disjoint at
                # fixed depth, so the full level decomposes into k binary cones.
                for b in space.A:
                    if b == a:
                        continue
                    other_image = {space.iterate(b, prefix) for prefix in prefixes}
                    assert set(image).isdisjoint(other_image)

    print("[OK] Each initial element produces exactly 2^n distinct depth-n descendants")
    print("[OK] Fixed-origin descendant sets are disjoint at the same depth")


def verify_symbolic_cardinality_identities() -> None:
    print("\n=== Symbolic verification of cardinality identities ===")

    k, n = symbols("k n", integer=True, positive=True)
    number_of_prefixes = 2**n
    number_of_codes = k * number_of_prefixes
    level_size = k * 2**n

    assert simplify(number_of_codes - level_size) == 0
    assert simplify(number_of_prefixes / (2**n) - 1) == 0

    # Concrete monotonicity and boundary checks: depth zero has one internal
    # prefix per initial element, while each increment doubles the internal
    # prefix count.
    for depth in range(0, 12):
        assert 2**depth >= 1
        if depth > 0:
            assert 2**depth == 2 * 2 ** (depth - 1)

    print("[OK] |A x {1,2}^n| = k*2^n and binary depth count 2^n are symbolically consistent")
    print("[OK] Depth recursion doubles the number of internal-state prefixes exactly")


def verify_centered_binary_spectrum() -> None:
    print("\n=== Verification of centered binary half-integer spectrum ===")

    eps1 = centered_binary_value(1)
    eps2 = centered_binary_value(2)

    assert eps1 == Fraction(-1, 2)
    assert eps2 == Fraction(1, 2)
    assert eps1 + eps2 == 0
    assert abs(eps1) == abs(eps2) == Fraction(1, 2)
    assert eps2 - eps1 == 1

    sigma = symbols("sigma", integer=True)
    eps = sigma - Rational(3, 2)
    assert simplify(eps.subs(sigma, 1) + Rational(1, 2)) == 0
    assert simplify(eps.subs(sigma, 2) - Rational(1, 2)) == 0
    assert simplify(eps.subs(sigma, 1) + eps.subs(sigma, 2)) == 0
    assert simplify(eps.subs(sigma, 2) - eps.subs(sigma, 1) - 1) == 0

    print("[OK] epsilon(1)=-1/2 and epsilon(2)=+1/2 exactly")
    print("[OK] The centered binary spectrum is symmetric and has unit spacing")


def verify_translation_centering_uniqueness() -> None:
    print("\n=== Symbolic verification of uniqueness among translations ===")

    lambda_ = symbols("lambda")
    equation = Eq((1 - lambda_) + (2 - lambda_), 0)
    solutions = solve(equation, lambda_)
    assert solutions == [Rational(3, 2)]

    translated_1 = 1 - solutions[0]
    translated_2 = 2 - solutions[0]
    assert simplify(translated_1 + Rational(1, 2)) == 0
    assert simplify(translated_2 - Rational(1, 2)) == 0

    # Nearby translations must fail centering.
    for bad_lambda in (Fraction(1), Fraction(4, 3), Fraction(5, 3), Fraction(2)):
        centered_sum = (Fraction(1) - bad_lambda) + (Fraction(2) - bad_lambda)
        assert centered_sum != 0

    print("[OK] Centering sigma-lambda forces lambda=3/2 uniquely")
    print("[OK] Non-centered translations fail the zero-sum centering condition")


def verify_affine_centering_family_and_scale_fixing() -> None:
    print("\n=== Symbolic verification of affine centered normalizations ===")

    alpha, beta = symbols("alpha beta")
    centering_equation = Eq((alpha + beta) + (2 * alpha + beta), 0)
    beta_solution = solve(centering_equation, beta)
    assert beta_solution == [-Rational(3, 2) * alpha]

    eta1 = alpha + beta_solution[0]
    eta2 = 2 * alpha + beta_solution[0]
    assert simplify(eta1 + alpha / 2) == 0
    assert simplify(eta2 - alpha / 2) == 0
    assert simplify(eta1 + eta2) == 0
    assert simplify(eta2 - eta1 - alpha) == 0

    # Non-uniqueness without scale fixing.
    for a in (Fraction(1, 2), Fraction(2), Fraction(-3), Fraction(5, 3)):
        v1, v2 = affine_centered_values(a)
        assert v1 == -a / 2
        assert v2 == a / 2
        assert v1 + v2 == 0
        assert v2 - v1 == a

    # Adjacent-state unit spacing equals one iff alpha=1.
    unit_spacing_solution = solve(Eq(eta2 - eta1, 1), alpha)
    assert unit_spacing_solution == [1]
    unit_spacing_beta = beta_solution[0].subs(alpha, unit_spacing_solution[0])
    assert simplify(unit_spacing_beta + Rational(3, 2)) == 0

    print("[OK] General affine centering is eta(sigma)=alpha*(sigma-3/2)")
    print("[OK] Unit-spacing normalization fixes alpha=1, beta=-3/2")


def verify_negative_domain_tests() -> None:
    print("\n=== Negative domain verification ===")

    space = BinaryGenerativeSpace(m=2, k=3)

    expect_raises(
        "invalid internal state in F",
        ValueError,
        lambda: space.F(3, 0),
    )
    expect_raises(
        "invalid internal-state prefix symbol",
        ValueError,
        lambda: space.iterate(space.A[0], (1, 2, 3)),
    )
    expect_raises(
        "invalid initial element",
        ValueError,
        lambda: space.iterate(999, (1, 2)),
    )
    expect_raises(
        "length mismatch for declared level",
        ValueError,
        lambda: space.validate_prefix((1, 2), n=3),
    )
    expect_raises(
        "element outside requested level",
        ValueError,
        lambda: space.recover_code(space.level_start(4), n=3),
    )
    expect_raises(
        "invalid centered binary state",
        ValueError,
        lambda: centered_binary_value(4),
    )
    expect_raises(
        "degenerate affine normalization alpha=0",
        ValueError,
        lambda: affine_centered_values(Fraction(0)),
    )
    expect_raises(
        "invalid k below the model domain",
        ValueError,
        lambda: BinaryGenerativeSpace(m=1, k=1),
    )

    # Non-binary branching is explicitly outside this section.  The binary
    # verification function rejects the resulting internal state 3.
    ternary_like_prefix = (1, 2, 3)
    expect_raises(
        "non-binary internal-state sequence",
        ValueError,
        lambda: space.validate_prefix(ternary_like_prefix),
    )

    print("[OK] Invalid domains are rejected without vacuous acceptance")


def verify_structural_nonphysical_guard() -> None:
    print("\n=== Verification of the structural interpretation guard ===")

    spectrum = {centered_binary_value(1), centered_binary_value(2)}
    assert spectrum == {Fraction(-1, 2), Fraction(1, 2)}

    print("[OK] The half-integer spectrum is verified as an internal structural consequence only")
    print("[OK] No external physical spin structure is assumed by the verification")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of binary microhistory (sec:spin-microhistory) ===")
    verify_exact_code_bijection_and_counting()
    verify_microhistory_limitation_and_fixed_origin_genealogy()
    verify_fixed_origin_injectivity_and_images()
    verify_symbolic_cardinality_identities()
    verify_centered_binary_spectrum()
    verify_translation_centering_uniqueness()
    verify_affine_centering_family_and_scale_fixing()
    verify_negative_domain_tests()
    verify_structural_nonphysical_guard()
    print("\n=== Binary microhistory verification completed successfully ===")


if __name__ == "__main__":
    main()
