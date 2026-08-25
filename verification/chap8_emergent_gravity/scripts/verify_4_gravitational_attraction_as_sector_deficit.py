"""
VERIFICATION of Section:
Effective gravitational attraction as sector deficit of compatible stable extensions
(sec:gravitational-attraction-as-sector-deficit).

Source file:
    4_gravitational_attraction_as_sector_deficit.tex

This script verifies the mathematical content of the section as a finite
sector-measure layer over the previously verified causal cone, sector
projection, full combinatorial measure, and sector coefficient

    G_s = (ln s)/s.

The section does not introduce gravity as a primitive force and does not alter
the fundamental causal speed bound c=1.  It defines effective gravitational
attraction as the existence of a strict deficit of compatible stable
realizations in at least one sector intersecting the localization region of
the source structure P.

Verified content
----------------
1. Full causal layer at depth ell:
       C^+(x) cap L_{n+ell}
   has exactly s^ell finite extensions, represented by digit coordinates
       D = 0,...,s^ell-1.

2. Primary sector split:
       sector sigma contains exactly s^(ell-1) extensions,
       corresponding to first digit sigma.

3. Compatible stable set:
       H_{Q|P}^{(n+ell)} subseteq H_Q^{(n+ell)}.

4. Sector compatible set:
       H_{Q|P,sigma}^{(n+ell)}(x) subseteq H_{Q,sigma}^{(n+ell)}(x).

5. Compatibility does not modify the full causal layer:
   the full cone has s^ell extensions before and after imposing compatibility;
   compatibility only selects a subset of stable realizations.

6. Sector measures:
       mu_{Q|P}^{(ell)}(I_sigma|x)
       =
       |H_{Q|P,sigma}^{(n+ell)}(x)| / s^ell,
       mu_{Q,0}^{(ell)}(I_sigma|x)
       =
       |H_{Q,sigma}^{(n+ell)}(x)| / s^ell.

7. Full unperturbed sector reference:
       |Sigma_ell(x) cap I_sigma| / |Sigma_ell(x)| = 1/s.

8. Sector deficit:
       delta_mu_sigma = mu_{Q|P} - mu_{Q,0} <= 0.

9. Strict-sector-deficit criterion:
       delta_mu_sigma < 0
   iff
       H_{Q|P,sigma}^{(n+ell)}(x) proper subset of H_{Q,sigma}^{(n+ell)}(x).

10. Localization sectors:
       S_loc^(ell)(x;P)
   consists exactly of sectors whose full causal sector slice intersects
       Loc_{n+ell}(P), where Loc is the finite-level localization region
   determined by weak/strong causal localization and causal containment.

11. Deficit profile:
       Phi_sigma = - delta_mu_sigma / (G_s m_comb(P)),
       where G_s=(ln s)/s>0 and m_comb(P)>0.

12. Profile signs:
       delta_mu=0  -> Phi=0,
       delta_mu<0 -> Phi>0,
       and Phi>=0 always under compatibility.

13. Decomposition:
       delta_mu_sigma = -G_s m_comb(P) Phi_sigma.

14. Effective gravitational attraction:
   It holds iff there exists a sector
       sigma in S_loc^(ell)(x;P)
   with
       delta_mu_sigma<0.

15. The theorem:
   if m_comb(P)>0 and there exists a localized sector with a proper compatible
   sector subset, then strict sector deficit, positive Phi, exact G_s
   decomposition, and effective attraction all follow.

16. Negative guards:
   - positive m_comb(P) alone is not effective attraction;
   - strict deficit outside S_loc is not effective attraction;
   - a non-strict compatible subset gives no attraction;
   - a sector deficit cannot be computed when compatibility is not a subset;
   - Phi is rejected when m_comb(P)<=0;
   - Phi is rejected if G_s is replaced by a nonpositive or wrong-sector
     coefficient in a reconstruction test;
   - full causal cone size is not allowed to shrink under compatibility;
   - changing the speed bound or adding forbidden extensions is rejected;
   - conditional normalization by |H_Q,sigma| is not the sector measure used
     here;
   - invalid s, ell, sigma, digit coordinates, localization sets, and stable
     subsets are rejected.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isclose, log
from typing import Callable, Iterable, Mapping

from sympy import log as slog
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


def require_branching(value: object) -> int:
    value = require_int(value, "s")
    if value < 2:
        raise DomainError("s must satisfy s>=2")
    return value


def require_positive_depth(value: object) -> int:
    value = require_int(value, "ell")
    if value < 1:
        raise DomainError("sector depth ell must satisfy ell>=1")
    return value


def require_nonnegative_int(value: object, name: str) -> int:
    value = require_int(value, name)
    if value < 0:
        raise DomainError(f"{name} must be nonnegative")
    return value


def validate_sigma(s: int, sigma: object) -> int:
    s = require_branching(s)
    sigma = require_int(sigma, "sigma")
    if not (1 <= sigma <= s):
        raise DomainError("sigma must belong to {1,...,s}")
    return sigma


def full_layer_digits(s: int, ell: int) -> frozenset[int]:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    return frozenset(range(s**ell))


def validate_digit_set(s: int, ell: int, digits: Iterable[int], name: str) -> frozenset[int]:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    full = s**ell
    if not isinstance(digits, (set, frozenset)):
        raise TypeError(f"{name} must be a set or frozenset")
    out: set[int] = set()
    for digit in digits:
        if not isinstance(digit, int):
            raise TypeError(f"{name} contains a noninteger digit")
        if not (0 <= digit < full):
            raise DomainError(f"{name} contains a digit outside the causal layer")
        out.add(digit)
    return frozenset(out)


def first_digit(s: int, ell: int, D: object) -> int:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    D = require_nonnegative_int(D, "D")
    if D >= s**ell:
        raise DomainError("D is outside the depth-ell causal layer")
    return D // (s ** (ell - 1)) + 1


def sector_digits(s: int, ell: int, sigma: object) -> frozenset[int]:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    sigma = validate_sigma(s, sigma)
    low = (sigma - 1) * s ** (ell - 1)
    high = sigma * s ** (ell - 1)
    return frozenset(range(low, high))


def normalized_rho(s: int, ell: int, D: object) -> Fraction:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    D = require_nonnegative_int(D, "D")
    if D >= s**ell:
        raise DomainError("D is outside the depth-ell causal layer")
    return Fraction(D, s**ell)


def in_primary_sector(s: int, ell: int, D: object, sigma: object) -> bool:
    return first_digit(s, ell, D) == validate_sigma(s, sigma)


def G_sector(s: int) -> float:
    s = require_branching(s)
    return log(s) / s


def G_comb(s: int) -> float:
    s = require_branching(s)
    return log(s)


def validate_positive_mass(m_comb: object) -> float:
    if not isinstance(m_comb, (int, float)):
        raise TypeError("m_comb must be numeric")
    m = float(m_comb)
    if not m > 0:
        raise DomainError("m_comb(P)>0 is required for the deficit profile")
    return m


@dataclass(frozen=True, slots=True)
class SectorLayer:
    s: int
    ell: int
    stable_Q: frozenset[int]
    compatible_QP: frozenset[int]
    localization: frozenset[int]

    def __post_init__(self) -> None:
        require_branching(self.s)
        require_positive_depth(self.ell)

        stable = validate_digit_set(self.s, self.ell, self.stable_Q, "stable_Q")
        compatible = validate_digit_set(self.s, self.ell, self.compatible_QP, "compatible_QP")
        localization = validate_digit_set(self.s, self.ell, self.localization, "localization")

        if stable != self.stable_Q:
            raise AssertionError("stable_Q normalization mismatch")
        if compatible != self.compatible_QP:
            raise AssertionError("compatible_QP normalization mismatch")
        if localization != self.localization:
            raise AssertionError("localization normalization mismatch")

        if not compatible.issubset(stable):
            raise DomainError("compatible_QP must be a subset of stable_Q")

    @property
    def full_layer(self) -> frozenset[int]:
        return full_layer_digits(self.s, self.ell)

    @property
    def full_size(self) -> int:
        return self.s ** self.ell

    def stable_sector(self, sigma: int) -> frozenset[int]:
        sigma = validate_sigma(self.s, sigma)
        return self.stable_Q.intersection(sector_digits(self.s, self.ell, sigma))

    def compatible_sector(self, sigma: int) -> frozenset[int]:
        sigma = validate_sigma(self.s, sigma)
        return self.compatible_QP.intersection(sector_digits(self.s, self.ell, sigma))

    def full_sector(self, sigma: int) -> frozenset[int]:
        return sector_digits(self.s, self.ell, sigma)

    def mu_unperturbed(self, sigma: int) -> Fraction:
        return Fraction(len(self.stable_sector(sigma)), self.full_size)

    def mu_compatible(self, sigma: int) -> Fraction:
        return Fraction(len(self.compatible_sector(sigma)), self.full_size)

    def delta_mu(self, sigma: int) -> Fraction:
        return self.mu_compatible(sigma) - self.mu_unperturbed(sigma)

    def is_strict_deficit(self, sigma: int) -> bool:
        stable = self.stable_sector(sigma)
        compatible = self.compatible_sector(sigma)
        return compatible < stable

    def localized_sectors(self) -> frozenset[int]:
        return frozenset(
            sigma
            for sigma in range(1, self.s + 1)
            if self.full_sector(sigma).intersection(self.localization)
        )

    def profile(self, sigma: int, m_comb: float, coefficient: float | None = None) -> float:
        sigma = validate_sigma(self.s, sigma)
        m = validate_positive_mass(m_comb)
        if coefficient is None:
            coefficient = G_sector(self.s)
        if not isinstance(coefficient, (int, float)):
            raise TypeError("coefficient must be numeric")
        coefficient = float(coefficient)
        if coefficient <= 0:
            raise DomainError("positive sector coefficient is required")
        return -float(self.delta_mu(sigma)) / (coefficient * m)

    def reconstruct_delta_from_profile(self, sigma: int, m_comb: float, coefficient: float | None = None) -> float:
        sigma = validate_sigma(self.s, sigma)
        m = validate_positive_mass(m_comb)
        if coefficient is None:
            coefficient = G_sector(self.s)
        phi = self.profile(sigma, m, coefficient)
        return -float(coefficient) * m * phi

    def effective_attraction(self) -> bool:
        return any(
            sigma in self.localized_sectors() and self.delta_mu(sigma) < 0
            for sigma in range(1, self.s + 1)
        )

    def strict_localized_deficit_witnesses(self) -> tuple[int, ...]:
        return tuple(
            sigma
            for sigma in range(1, self.s + 1)
            if sigma in self.localized_sectors() and self.is_strict_deficit(sigma)
        )


def make_layer_from_sector_counts(s: int,
                                  ell: int,
                                  stable_counts: Mapping[int, int],
                                  compatible_counts: Mapping[int, int],
                                  localized_sigmas: Iterable[int]) -> SectorLayer:
    s = require_branching(s)
    ell = require_positive_depth(ell)

    stable: set[int] = set()
    compatible: set[int] = set()
    localization: set[int] = set()

    for sigma in range(1, s + 1):
        sector = sorted(sector_digits(s, ell, sigma))
        sector_size = len(sector)

        stable_count = stable_counts.get(sigma, 0)
        compatible_count = compatible_counts.get(sigma, 0)

        stable_count = require_int(stable_count, "stable_count")
        compatible_count = require_int(compatible_count, "compatible_count")

        if not (0 <= stable_count <= sector_size):
            raise DomainError("stable_count is outside the sector size")
        if not (0 <= compatible_count <= stable_count):
            raise DomainError("compatible_count must satisfy 0<=compatible<=stable")

        stable.update(sector[:stable_count])
        compatible.update(sector[:compatible_count])

    for sigma in localized_sigmas:
        sigma = validate_sigma(s, sigma)
        # Add one canonical point from that sector to localization.
        localization.add(min(sector_digits(s, ell, sigma)))

    return SectorLayer(
        s=s,
        ell=ell,
        stable_Q=frozenset(stable),
        compatible_QP=frozenset(compatible),
        localization=frozenset(localization),
    )


def assert_full_cone_unchanged(layer: SectorLayer, candidate_full_cone: frozenset[int]) -> None:
    candidate = validate_digit_set(layer.s, layer.ell, candidate_full_cone, "candidate_full_cone")
    if candidate != layer.full_layer:
        raise DomainError("compatibility is not allowed to modify the full causal layer")


def assert_speed_bound_unchanged(s: int, ell: int, D: int, claimed_speed_bound: int = 1) -> None:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    D = require_nonnegative_int(D, "D")
    if D >= s**ell:
        raise DomainError("D is outside the depth-ell causal layer")
    claimed_speed_bound = require_int(claimed_speed_bound, "claimed_speed_bound")
    if claimed_speed_bound != 1:
        raise DomainError("the fundamental causal speed bound must remain c=1")
    # In logarithmic cone coordinates: 0 <= Delta r <= Delta t=ell.
    rho_numer = D + 1
    assert 1 <= rho_numer <= s**ell
    # Exact inequality behind Delta r <= ell:
    # log_s(D+1) <= ell  iff  D+1 <= s^ell.
    assert rho_numer <= s**ell


def conditional_sector_fraction(layer: SectorLayer, sigma: int) -> Fraction:
    """Not the measure used by the section; used only as a negative guard."""
    stable = layer.stable_sector(sigma)
    if len(stable) == 0:
        raise DomainError("conditional fraction inside an empty stable sector is undefined")
    return Fraction(len(layer.compatible_sector(sigma)), len(stable))


def verify_symbolic_measure_identities() -> None:
    print("\n=== Symbolic verification of sector-deficit identities ===")

    s, ell, a, b, m = symbols("s ell a b m", positive=True)
    Gs = slog(s) / s

    mu_compat = a / (s**ell)
    mu_unperturbed = b / (s**ell)
    delta_mu = mu_compat - mu_unperturbed

    assert simplify(delta_mu - (a - b) / (s**ell)) == 0

    Phi = -delta_mu / (Gs * m)
    reconstructed = -Gs * m * Phi
    assert simplify(reconstructed - delta_mu) == 0

    full_sector_fraction = (s ** (ell - 1)) / (s**ell)
    assert simplify(full_sector_fraction - 1 / s) == 0

    print("[OK] delta_mu=(|H_compat|-|H_unperturbed|)/s^ell is symbolic")
    print("[OK] delta_mu=-G_s*m_comb*Phi is symbolic")
    print("[OK] full unperturbed sector fraction is 1/s symbolically")


def verify_full_causal_layer_and_sector_counts() -> None:
    print("\n=== Verification of full causal layer and sector counts ===")

    checked_layers = 0
    checked_sectors = 0
    checked_digits = 0
    checked_speed_bounds = 0

    for s in range(2, 10):
        for ell in range(1, 8):
            full = full_layer_digits(s, ell)
            checked_layers += 1

            assert len(full) == s**ell
            assert full == frozenset(range(s**ell))

            union = set()
            for sigma in range(1, s + 1):
                sector = sector_digits(s, ell, sigma)
                checked_sectors += 1
                checked_digits += len(sector)

                assert len(sector) == s ** (ell - 1)
                assert Fraction(len(sector), len(full)) == Fraction(1, s)
                assert all(first_digit(s, ell, D) == sigma for D in sector)
                assert sector.isdisjoint(union)
                union.update(sector)

                for D in (min(sector), max(sector), (min(sector) + max(sector)) // 2):
                    assert in_primary_sector(s, ell, D, sigma)
                    assert normalized_rho(s, ell, D) >= Fraction(sigma - 1, s)
                    assert normalized_rho(s, ell, D) < Fraction(sigma, s)
                    assert_speed_bound_unchanged(s, ell, D)
                    checked_speed_bounds += 1

            assert union == set(full)

    print(f"[OK] Checked {checked_layers} full causal layers")
    print(f"[OK] Checked {checked_sectors} sector cardinality/fraction cases")
    print(f"[OK] Checked {checked_digits} digit assignments to primary sectors")
    print(f"[OK] Checked {checked_speed_bounds} causal speed-bound digit witnesses")


def verify_compatible_subset_and_sector_measures() -> None:
    print("\n=== Verification of compatible subsets and sector measures ===")

    checked_layers = 0
    checked_sector_measures = 0
    checked_monotonicity = 0
    checked_strict_criteria = 0

    for s in range(2, 9):
        for ell in range(1, 6):
            sector_size = s ** (ell - 1)
            stable_counts = {}
            compatible_counts = {}

            for sigma in range(1, s + 1):
                stable_count = max(0, sector_size - ((sigma - 1) % max(1, sector_size)))
                if sigma % 3 == 0 and stable_count > 0:
                    compatible_count = stable_count - 1
                elif sigma % 3 == 1:
                    compatible_count = stable_count
                else:
                    compatible_count = max(0, stable_count // 2)

                stable_counts[sigma] = stable_count
                compatible_counts[sigma] = compatible_count

            layer = make_layer_from_sector_counts(
                s=s,
                ell=ell,
                stable_counts=stable_counts,
                compatible_counts=compatible_counts,
                localized_sigmas=range(1, s + 1),
            )
            checked_layers += 1

            assert layer.compatible_QP.issubset(layer.stable_Q)
            assert_full_cone_unchanged(layer, layer.full_layer)

            for sigma in range(1, s + 1):
                stable_sector = layer.stable_sector(sigma)
                compatible_sector = layer.compatible_sector(sigma)

                checked_sector_measures += 1
                assert compatible_sector.issubset(stable_sector)
                assert layer.mu_unperturbed(sigma) == Fraction(len(stable_sector), s**ell)
                assert layer.mu_compatible(sigma) == Fraction(len(compatible_sector), s**ell)
                assert layer.delta_mu(sigma) == Fraction(len(compatible_sector) - len(stable_sector), s**ell)

                checked_monotonicity += 1
                assert layer.mu_compatible(sigma) <= layer.mu_unperturbed(sigma)
                assert layer.delta_mu(sigma) <= 0

                checked_strict_criteria += 1
                assert (layer.delta_mu(sigma) < 0) == (compatible_sector < stable_sector)
                assert layer.is_strict_deficit(sigma) == (compatible_sector < stable_sector)

    print(f"[OK] Checked {checked_layers} compatible stable layers")
    print(f"[OK] Checked {checked_sector_measures} sector measure definitions")
    print(f"[OK] Checked {checked_monotonicity} monotonicity/deficit inequalities")
    print(f"[OK] Checked {checked_strict_criteria} strict-deficit equivalences")


def verify_strict_deficit_criterion_exhaustively_on_small_sets() -> None:
    print("\n=== Exhaustive verification of strict-deficit criterion on small sector sets ===")

    checked_pairs = 0
    checked_non_subset_rejections = 0

    for s in range(2, 5):
        for ell in range(1, 4):
            for sigma in range(1, s + 1):
                sector = sorted(sector_digits(s, ell, sigma))
                if len(sector) > 8:
                    sector = sector[:8]

                # Enumerate many stable/compatible pairs within a bounded sector.
                all_masks = range(1 << len(sector))
                for stable_mask in all_masks:
                    stable = {sector[i] for i in range(len(sector)) if stable_mask & (1 << i)}
                    stable_global = frozenset(stable)

                    submask = stable_mask
                    while True:
                        compatible = {sector[i] for i in range(len(sector)) if submask & (1 << i)}
                        layer = SectorLayer(
                            s=s,
                            ell=ell,
                            stable_Q=stable_global,
                            compatible_QP=frozenset(compatible),
                            localization=frozenset({sector[0]}),
                        )
                        checked_pairs += 1
                        assert (layer.delta_mu(sigma) < 0) == (frozenset(compatible) < stable_global)
                        if submask == 0:
                            break
                        submask = (submask - 1) & stable_mask

                # Non-subset compatibility must be rejected.
                if len(sector) >= 2:
                    stable = frozenset({sector[0]})
                    compatible = frozenset({sector[1]})
                    expect_raises(DomainError, lambda s=s, ell=ell, stable=stable, compatible=compatible:
                                  SectorLayer(s=s, ell=ell, stable_Q=stable, compatible_QP=compatible, localization=frozenset()))
                    checked_non_subset_rejections += 1

    print(f"[OK] Checked {checked_pairs} finite stable/compatible subset pairs")
    print(f"[OK] Checked {checked_non_subset_rejections} non-subset compatibility rejections")


def verify_localization_sectors_and_effective_attraction() -> None:
    print("\n=== Verification of localization sectors and effective attraction criterion ===")

    checked_layers = 0
    checked_localized_sets = 0
    checked_attraction_witnesses = 0
    checked_negative_cases = 0

    for s in range(2, 10):
        for ell in range(1, 6):
            sector_size = s ** (ell - 1)

            # Strict deficit in one localized sector.
            strict_sigma = 1
            stable_counts = {sigma: min(sector_size, max(0, sector_size)) for sigma in range(1, s + 1)}
            compatible_counts = dict(stable_counts)
            compatible_counts[strict_sigma] = max(0, stable_counts[strict_sigma] - 1)

            layer = make_layer_from_sector_counts(
                s=s,
                ell=ell,
                stable_counts=stable_counts,
                compatible_counts=compatible_counts,
                localized_sigmas={strict_sigma},
            )
            checked_layers += 1

            assert layer.localized_sectors() == frozenset({strict_sigma})
            checked_localized_sets += 1
            assert layer.delta_mu(strict_sigma) < 0
            assert layer.effective_attraction()
            assert layer.strict_localized_deficit_witnesses() == (strict_sigma,)
            checked_attraction_witnesses += 1

            # Strict deficit outside localization is not effective attraction.
            outside_sigma = s if strict_sigma != s else 1
            stable_counts2 = {sigma: sector_size for sigma in range(1, s + 1)}
            compatible_counts2 = dict(stable_counts2)
            compatible_counts2[outside_sigma] = max(0, sector_size - 1)
            layer_outside = make_layer_from_sector_counts(
                s=s,
                ell=ell,
                stable_counts=stable_counts2,
                compatible_counts=compatible_counts2,
                localized_sigmas={strict_sigma},
            )
            assert layer_outside.delta_mu(outside_sigma) < 0
            assert outside_sigma not in layer_outside.localized_sectors()
            assert not layer_outside.effective_attraction()
            checked_negative_cases += 1

            # No strict deficit even inside localization is not effective attraction.
            layer_equal = make_layer_from_sector_counts(
                s=s,
                ell=ell,
                stable_counts=stable_counts,
                compatible_counts=stable_counts,
                localized_sigmas={strict_sigma},
            )
            assert layer_equal.delta_mu(strict_sigma) == 0
            assert not layer_equal.effective_attraction()
            checked_negative_cases += 1

            # Positive mass alone is not enough: no strict localized deficit.
            m_comb = 1.0
            assert m_comb > 0
            assert not layer_equal.effective_attraction()
            checked_negative_cases += 1

    print(f"[OK] Checked {checked_layers} localized-deficit layers")
    print(f"[OK] Checked {checked_localized_sets} localized sector-set computations")
    print(f"[OK] Checked {checked_attraction_witnesses} effective-attraction witnesses")
    print(f"[OK] Checked {checked_negative_cases} non-attraction negative cases")


def verify_deficit_profile_and_Gs_decomposition() -> None:
    print("\n=== Verification of deficit profile and G_s decomposition ===")

    checked_profiles = 0
    checked_zero_profiles = 0
    checked_positive_profiles = 0
    checked_reconstructions = 0
    checked_wrong_coefficients = 0

    for s in range(2, 12):
        for ell in range(1, 6):
            sector_size = s ** (ell - 1)
            stable_counts = {sigma: sector_size for sigma in range(1, s + 1)}

            for strict_sigma in range(1, s + 1):
                compatible_counts = dict(stable_counts)
                compatible_counts[strict_sigma] = max(0, sector_size - 1)

                layer = make_layer_from_sector_counts(
                    s=s,
                    ell=ell,
                    stable_counts=stable_counts,
                    compatible_counts=compatible_counts,
                    localized_sigmas={strict_sigma},
                )

                for m_comb in (0.25, 1.0, 2.5, log(max(2, s), s) + 1):
                    for sigma in range(1, s + 1):
                        phi = layer.profile(sigma, m_comb)
                        reconstructed = layer.reconstruct_delta_from_profile(sigma, m_comb)
                        checked_profiles += 1
                        checked_reconstructions += 1

                        assert phi >= -1e-15
                        assert isclose(reconstructed, float(layer.delta_mu(sigma)), rel_tol=1e-12, abs_tol=1e-12)

                        if layer.delta_mu(sigma) == 0:
                            checked_zero_profiles += 1
                            assert isclose(phi, 0.0, abs_tol=1e-15)
                        else:
                            checked_positive_profiles += 1
                            assert layer.delta_mu(sigma) < 0
                            assert phi > 0

                        wrong_coefficient = G_comb(s)
                        if not isclose(wrong_coefficient, G_sector(s), rel_tol=1e-15, abs_tol=1e-15):
                            wrong_phi = layer.profile(sigma, m_comb, coefficient=wrong_coefficient)
                            # If wrong coefficient is used to define the profile but the
                            # section's G_s is then used to reconstruct, the identity fails
                            # whenever there is a strict deficit.
                            if layer.delta_mu(sigma) < 0:
                                wrong_reconstruction_with_Gs = -G_sector(s) * m_comb * wrong_phi
                                assert not isclose(wrong_reconstruction_with_Gs, float(layer.delta_mu(sigma)), rel_tol=1e-12, abs_tol=1e-12)
                                checked_wrong_coefficients += 1

                expect_raises(DomainError, lambda layer=layer: layer.profile(strict_sigma, 0.0))
                expect_raises(DomainError, lambda layer=layer: layer.profile(strict_sigma, -1.0))
                expect_raises(DomainError, lambda layer=layer: layer.profile(strict_sigma, 1.0, coefficient=0.0))
                expect_raises(DomainError, lambda layer=layer: layer.profile(strict_sigma, 1.0, coefficient=-1.0))

    print(f"[OK] Checked {checked_profiles} deficit-profile values")
    print(f"[OK] Checked {checked_zero_profiles} zero-profile cases")
    print(f"[OK] Checked {checked_positive_profiles} positive-profile cases")
    print(f"[OK] Checked {checked_reconstructions} exact G_s decompositions")
    print(f"[OK] Checked {checked_wrong_coefficients} wrong-coefficient reconstruction failures")


def verify_causal_cone_is_not_deformed_by_compatibility() -> None:
    print("\n=== Verification that compatibility does not deform the full causal cone ===")

    checked_layers = 0
    checked_subset_cones = 0
    checked_shrink_rejections = 0
    checked_forbidden_extension_rejections = 0

    for s in range(2, 9):
        for ell in range(1, 6):
            full = full_layer_digits(s, ell)
            sector_size = s ** (ell - 1)
            stable_counts = {sigma: sector_size for sigma in range(1, s + 1)}

            # Use several compatible subsets, including empty and proper selections.
            compatible_variants = [
                {sigma: 0 for sigma in range(1, s + 1)},
                {sigma: stable_counts[sigma] for sigma in range(1, s + 1)},
                {sigma: max(0, stable_counts[sigma] - 1) for sigma in range(1, s + 1)},
            ]

            for compatible_counts in compatible_variants:
                layer = make_layer_from_sector_counts(
                    s=s,
                    ell=ell,
                    stable_counts=stable_counts,
                    compatible_counts=compatible_counts,
                    localized_sigmas=range(1, s + 1),
                )
                checked_layers += 1
                assert layer.full_layer == full
                assert_full_cone_unchanged(layer, full)

                # Compatible realizations are a subset of stable realizations,
                # not a replacement for the full causal cone.
                assert layer.compatible_QP.issubset(layer.full_layer)
                checked_subset_cones += 1

                if len(full) > 1:
                    shrunk = frozenset(sorted(full)[:-1])
                    expect_raises(DomainError, lambda layer=layer, shrunk=shrunk: assert_full_cone_unchanged(layer, shrunk))
                    checked_shrink_rejections += 1

                forbidden = frozenset(set(full).union({s**ell}))
                expect_raises(DomainError, lambda layer=layer, forbidden=forbidden: assert_full_cone_unchanged(layer, forbidden))
                checked_forbidden_extension_rejections += 1

            for D in (0, s**ell - 1, (s**ell - 1) // 2):
                assert_speed_bound_unchanged(s, ell, D, claimed_speed_bound=1)
                expect_raises(DomainError, lambda s=s, ell=ell, D=D: assert_speed_bound_unchanged(s, ell, D, claimed_speed_bound=2))

    print(f"[OK] Checked {checked_layers} compatibility selections against unchanged full cones")
    print(f"[OK] Checked {checked_subset_cones} compatible subsets inside full cones")
    print(f"[OK] Rejected {checked_shrink_rejections} attempted cone shrinkages")
    print(f"[OK] Rejected {checked_forbidden_extension_rejections} attempted forbidden extensions")


def verify_sector_measure_normalization_guard() -> None:
    print("\n=== Guard: sector measure is normalized by s^ell, not by stable-sector size ===")

    checked_guards = 0
    checked_undefined_conditionals = 0

    for s in range(2, 8):
        for ell in range(1, 6):
            sector_size = s ** (ell - 1)
            stable_counts = {sigma: max(0, sector_size - (sigma % 2)) for sigma in range(1, s + 1)}
            compatible_counts = {sigma: max(0, stable_counts[sigma] - 1) for sigma in range(1, s + 1)}

            layer = make_layer_from_sector_counts(
                s=s,
                ell=ell,
                stable_counts=stable_counts,
                compatible_counts=compatible_counts,
                localized_sigmas=range(1, s + 1),
            )

            for sigma in range(1, s + 1):
                mu = layer.mu_compatible(sigma)
                assert mu == Fraction(len(layer.compatible_sector(sigma)), s**ell)
                checked_guards += 1

                if len(layer.stable_sector(sigma)) > 0:
                    conditional = conditional_sector_fraction(layer, sigma)
                    # Conditional normalization is generally a different quantity.
                    if len(layer.stable_sector(sigma)) != s**ell:
                        assert conditional != mu or len(layer.compatible_sector(sigma)) == 0
                else:
                    expect_raises(DomainError, lambda layer=layer, sigma=sigma: conditional_sector_fraction(layer, sigma))
                    checked_undefined_conditionals += 1

    print(f"[OK] Checked {checked_guards} sector-measure normalizations by s^ell")
    print(f"[OK] Checked {checked_undefined_conditionals} undefined conditional normalizations")


def verify_effective_deformation_theorem() -> None:
    print("\n=== Verification of effective gravitational deformation theorem ===")

    checked_theorem_witnesses = 0
    checked_non_witnesses = 0

    for s in range(2, 10):
        for ell in range(1, 6):
            sector_size = s ** (ell - 1)

            for sigma_star in range(1, s + 1):
                stable_counts = {sigma: sector_size for sigma in range(1, s + 1)}
                compatible_counts = dict(stable_counts)
                compatible_counts[sigma_star] = max(0, sector_size - 1)

                layer = make_layer_from_sector_counts(
                    s=s,
                    ell=ell,
                    stable_counts=stable_counts,
                    compatible_counts=compatible_counts,
                    localized_sigmas={sigma_star},
                )

                m_comb = 1.0
                assert m_comb > 0
                assert sigma_star in layer.localized_sectors()
                assert layer.compatible_sector(sigma_star) < layer.stable_sector(sigma_star)
                assert layer.delta_mu(sigma_star) < 0
                assert layer.profile(sigma_star, m_comb) > 0
                assert isclose(
                    layer.reconstruct_delta_from_profile(sigma_star, m_comb),
                    float(layer.delta_mu(sigma_star)),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                assert layer.effective_attraction()
                checked_theorem_witnesses += 1

                # Same proper subset, but not localized: theorem hypothesis fails.
                layer_not_localized = make_layer_from_sector_counts(
                    s=s,
                    ell=ell,
                    stable_counts=stable_counts,
                    compatible_counts=compatible_counts,
                    localized_sigmas=set(),
                )
                assert layer_not_localized.delta_mu(sigma_star) < 0
                assert not layer_not_localized.effective_attraction()
                checked_non_witnesses += 1

                # Localized but no proper subset: theorem hypothesis fails.
                layer_not_strict = make_layer_from_sector_counts(
                    s=s,
                    ell=ell,
                    stable_counts=stable_counts,
                    compatible_counts=stable_counts,
                    localized_sigmas={sigma_star},
                )
                assert layer_not_strict.delta_mu(sigma_star) == 0
                assert not layer_not_strict.effective_attraction()
                checked_non_witnesses += 1

    print(f"[OK] Checked {checked_theorem_witnesses} theorem witnesses")
    print(f"[OK] Checked {checked_non_witnesses} failed-hypothesis non-witnesses")


def verify_negative_domain_tests() -> None:
    print("\n=== Negative domain tests ===")

    expect_raises(DomainError, lambda: full_layer_digits(1, 1))
    expect_raises(DomainError, lambda: full_layer_digits(2, 0))
    expect_raises(TypeError, lambda: full_layer_digits(2.0, 1))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: full_layer_digits(2, 1.0))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: validate_sigma(2, 0))
    expect_raises(DomainError, lambda: validate_sigma(2, 3))
    expect_raises(TypeError, lambda: validate_sigma(2, 1.0))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: first_digit(2, 3, -1))
    expect_raises(DomainError, lambda: first_digit(2, 3, 8))
    expect_raises(TypeError, lambda: first_digit(2, 3, 1.5))  # type: ignore[arg-type]

    expect_raises(TypeError, lambda: validate_digit_set(2, 3, [1, 2], "bad"))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: validate_digit_set(2, 3, frozenset({8}), "bad"))
    expect_raises(DomainError, lambda: validate_digit_set(2, 3, frozenset({-1}), "bad"))
    expect_raises(TypeError, lambda: validate_digit_set(2, 3, frozenset({1.5}), "bad"))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: make_layer_from_sector_counts(2, 3, {1: 5}, {1: 0}, {1}))
    expect_raises(DomainError, lambda: make_layer_from_sector_counts(2, 3, {1: 1}, {1: 2}, {1}))
    expect_raises(DomainError, lambda: make_layer_from_sector_counts(2, 3, {1: 1}, {1: 1}, {3}))
    expect_raises(TypeError, lambda: make_layer_from_sector_counts(2, 3, {1: 1.5}, {1: 1}, {1}))  # type: ignore[arg-type]

    layer = make_layer_from_sector_counts(
        s=2,
        ell=3,
        stable_counts={1: 4, 2: 4},
        compatible_counts={1: 3, 2: 4},
        localized_sigmas={1},
    )
    expect_raises(DomainError, lambda: layer.profile(1, 0.0))
    expect_raises(DomainError, lambda: layer.profile(1, -1.0))
    expect_raises(DomainError, lambda: layer.profile(1, 1.0, coefficient=0.0))
    expect_raises(DomainError, lambda: layer.profile(1, 1.0, coefficient=-0.5))
    expect_raises(TypeError, lambda: layer.profile(1, "mass"))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: layer.profile(0, 1.0))
    expect_raises(DomainError, lambda: layer.profile(3, 1.0))

    expect_raises(DomainError, lambda: assert_speed_bound_unchanged(2, 3, 8))
    expect_raises(DomainError, lambda: assert_speed_bound_unchanged(2, 3, 1, claimed_speed_bound=0))

    print("[OK] Invalid branching, depths, sectors, digits, stable sets, profiles, and speed claims are rejected")


def main() -> None:
    print("=== Verification of effective gravitational attraction as sector deficit ===")
    verify_symbolic_measure_identities()
    verify_full_causal_layer_and_sector_counts()
    verify_compatible_subset_and_sector_measures()
    verify_strict_deficit_criterion_exhaustively_on_small_sets()
    verify_localization_sectors_and_effective_attraction()
    verify_deficit_profile_and_Gs_decomposition()
    verify_causal_cone_is_not_deformed_by_compatibility()
    verify_sector_measure_normalization_guard()
    verify_effective_deformation_theorem()
    verify_negative_domain_tests()
    print("\n=== Effective gravitational-attraction sector-deficit verification completed successfully ===")


if __name__ == "__main__":
    main()
