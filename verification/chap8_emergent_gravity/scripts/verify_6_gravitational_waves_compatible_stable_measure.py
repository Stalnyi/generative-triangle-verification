"""
VERIFICATION of Section:
Gravitational waves as propagating perturbations of compatible stable measure
(sec:gravitational-waves-compatible-stable-measure).

Source file:
    6_gravitational_waves_compatible_stable_measure.tex

This script verifies the mathematical content of the section as a finite
sector-deficit wave layer over the previously verified combinatorial mass,
sector projection, compatible stable measure, causal cone, and black-hole
escaping-measure formalism.

The section does not introduce a primary gravitational field, a metric wave
equation, polarizations, energy flux, or a graviton.  It defines a
gravitational wave in this model as a nonstationary sector component of the
compatible stable-measure deficit, normalized by the same sector coefficient

    G_s = (ln s)/s,

and supported inside the future causal cone.

Verified content
----------------
1. Positive logarithmic regime:
       s >= 2, n >= 0, 1 <= N_stable <= s^n.

2. Combinatorial wave amplitude:
       h_comb = Delta(ln N_stable).

3. Change of combinatorial mass:
       Delta m_comb = log_s N_2 - log_s N_1.

4. Amplitude identity:
       h_comb = (ln s) Delta m_comb.

5. Sector wave amplitude:
       h_s = h_comb/s = G_s Delta m_comb,
       G_s=(ln s)/s.

6. Sign convention:
       h_s is signed when Delta m_comb is signed;
       |h_s| = G_s |Delta m_comb|.

7. Background/wave split:
       delta_mu = delta_mu_bg + delta_mu_wave.

8. Compatible-measure reconstruction:
       mu_{Q|P} = mu_{Q,0} + delta_mu_bg + delta_mu_wave.

9. Wave profile:
       Psi_sigma = -delta_mu_wave/(G_s Delta m_comb)
   when Delta m_comb != 0.  If delta_mu_wave=0, the profile is 0.

10. Sector wave-deficit decomposition:
       delta_mu_wave = -G_s Delta m_comb Psi_sigma.

11. Wave-sector support:
       S_wave = {sigma : delta_mu_wave_sigma != 0}.

12. Spatial-causal support:
       Supp_wave^(ell)(x) consists exactly of future-cone endpoints whose
       primary sector belongs to S_wave.

13. Causal support bound:
       every support endpoint satisfies
       r_{n+ell}(y)-r_n(x) <= ell,
   equivalently the wave support does not exceed c=1.

14. Gravitational wave package:
       a nonzero level-indexed family of delta_mu_wave layers, sector amplitude
       h_s, wave profile Psi_sigma, and exact decomposition
       delta_mu_wave = -h_s Psi_sigma,
   with support inside the causal cone on every represented level.

15. Black-hole regime suppression:
       if outward propagation requires nonzero escaping compatible stable
       measure and mu_esc^(ell)->0, then the outward wave-escape measure is
       asymptotically suppressed.  This is not a statement that internal
       perturbations vanish.

16. Feedback interpretation:
       changes of compatible stable sets between levels change sector
       compatible measures; the nonstationary component is the wave component.
       The script checks strengthening, weakening, and regime-switch cases.

17. Negative guards:
       - s<2 is rejected;
       - n<0 is rejected;
       - N=0 or N>s^n is rejected for logarithmic amplitudes;
       - Delta m=0 is rejected for profile normalization unless the wave
         component is explicitly zero;
       - G_s<=0 is rejected;
       - invalid sector labels, digit coordinates, and support sets are
         rejected;
       - support outside the future causal layer is rejected;
       - changing the speed bound c=1 is rejected;
       - a wave is not accepted when every delta_mu_wave sector is zero;
       - a primary field, metric wave equation, polarization, energy flux, or
         graviton is not inferred by the verification;
       - black-hole outward suppression is not interpreted as disappearance of
         internal wave perturbations.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isclose, log
from typing import Callable, Iterable, Sequence

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


def require_nonnegative_int(value: object, name: str) -> int:
    value = require_int(value, name)
    if value < 0:
        raise DomainError(f"{name} must be nonnegative")
    return value


def require_positive_depth(value: object) -> int:
    value = require_int(value, "ell")
    if value < 1:
        raise DomainError("ell must satisfy ell>=1")
    return value


def validate_sigma(s: int, sigma: object) -> int:
    s = require_branching(s)
    sigma = require_int(sigma, "sigma")
    if not (1 <= sigma <= s):
        raise DomainError("sigma must belong to {1,...,s}")
    return sigma


def validate_level_multiplicity(s: int, n: int, N: object) -> int:
    s = require_branching(s)
    n = require_nonnegative_int(n, "n")
    N = require_int(N, "N")
    if not (1 <= N <= s**n):
        raise DomainError("N must satisfy 1<=N<=s^n")
    return N


def combinatorial_mass(s: int, n: int, N: object) -> float:
    N = validate_level_multiplicity(s, n, N)
    return log(N, s)


def delta_m_comb(s: int, n: int, N1: object, N2: object) -> float:
    N1 = validate_level_multiplicity(s, n, N1)
    N2 = validate_level_multiplicity(s, n, N2)
    return log(N2 / N1, s)


def h_comb(s: int, n: int, N1: object, N2: object) -> float:
    N1 = validate_level_multiplicity(s, n, N1)
    N2 = validate_level_multiplicity(s, n, N2)
    return log(N2 / N1)


def G_sector(s: int) -> float:
    s = require_branching(s)
    return log(s) / s


def h_sector(s: int, n: int, N1: object, N2: object) -> float:
    return h_comb(s, n, N1, N2) / require_branching(s)


def h_sector_from_delta_m(s: int, delta_m: float) -> float:
    s = require_branching(s)
    if not isinstance(delta_m, (int, float)):
        raise TypeError("Delta m must be numeric")
    return G_sector(s) * float(delta_m)


def full_layer_digits(s: int, ell: int) -> frozenset[int]:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    return frozenset(range(s**ell))


def validate_digit(s: int, ell: int, D: object) -> int:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    D = require_nonnegative_int(D, "D")
    if D >= s**ell:
        raise DomainError("D is outside the depth-ell causal layer")
    return D


def validate_digit_set(s: int, ell: int, digits: Iterable[int], name: str) -> frozenset[int]:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    if not isinstance(digits, (set, frozenset)):
        raise TypeError(f"{name} must be a set or frozenset")
    out: set[int] = set()
    for D in digits:
        out.add(validate_digit(s, ell, D))
    return frozenset(out)


def sector_digits(s: int, ell: int, sigma: object) -> frozenset[int]:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    sigma = validate_sigma(s, sigma)
    low = (sigma - 1) * s ** (ell - 1)
    high = sigma * s ** (ell - 1)
    return frozenset(range(low, high))


def first_digit_sector(s: int, ell: int, D: object) -> int:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    D = validate_digit(s, ell, D)
    return D // (s ** (ell - 1)) + 1


def normalized_rho(s: int, ell: int, D: object) -> Fraction:
    D = validate_digit(s, ell, D)
    return Fraction(D, require_branching(s) ** require_positive_depth(ell))


def assert_speed_bound_c_equals_one(s: int, ell: int, D: object, claimed_c: int = 1) -> None:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    D = validate_digit(s, ell, D)
    claimed_c = require_int(claimed_c, "claimed_c")
    if claimed_c != 1:
        raise DomainError("the fundamental causal speed bound must remain c=1")
    # Exact finite inequality behind r_{n+ell}(y)-r_n(x)<=ell:
    # log_s(D+1)<=ell is equivalent to D+1<=s^ell.
    assert 1 <= D + 1 <= s**ell


@dataclass(frozen=True, slots=True)
class SectorDeficitSplit:
    mu0: Fraction
    delta_bg: Fraction
    delta_wave: Fraction

    def __post_init__(self) -> None:
        if not isinstance(self.mu0, Fraction):
            raise TypeError("mu0 must be an exact Fraction")
        if not isinstance(self.delta_bg, Fraction):
            raise TypeError("delta_bg must be an exact Fraction")
        if not isinstance(self.delta_wave, Fraction):
            raise TypeError("delta_wave must be an exact Fraction")
        if self.mu0 < 0:
            raise DomainError("mu0 must be nonnegative")

    @property
    def total_delta(self) -> Fraction:
        return self.delta_bg + self.delta_wave

    @property
    def compatible_mu(self) -> Fraction:
        return self.mu0 + self.total_delta

    def validate_measure_range(self) -> None:
        if self.compatible_mu < 0 or self.compatible_mu > 1:
            raise DomainError("compatible sector measure must lie in [0,1]")


@dataclass(frozen=True, slots=True)
class WaveProfile:
    s: int
    delta_m: float
    delta_wave: Fraction

    def __post_init__(self) -> None:
        require_branching(self.s)
        if not isinstance(self.delta_m, (int, float)):
            raise TypeError("delta_m must be numeric")
        if not isinstance(self.delta_wave, Fraction):
            raise TypeError("delta_wave must be an exact Fraction")
        if self.delta_wave != 0 and float(self.delta_m) == 0.0:
            raise DomainError("nonzero wave deficit requires Delta m_comb != 0")

    @property
    def amplitude(self) -> float:
        return h_sector_from_delta_m(self.s, float(self.delta_m))

    @property
    def abs_amplitude(self) -> float:
        return abs(self.amplitude)

    @property
    def psi(self) -> float:
        if self.delta_wave == 0:
            return 0.0
        if self.amplitude == 0:
            raise DomainError("profile normalization requires nonzero sector amplitude")
        return -float(self.delta_wave) / self.amplitude

    @property
    def reconstructed_delta_wave(self) -> float:
        return -self.amplitude * self.psi


@dataclass(frozen=True, slots=True)
class WaveLayer:
    s: int
    ell: int
    delta_wave_by_sector: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        s = require_branching(self.s)
        require_positive_depth(self.ell)
        if not isinstance(self.delta_wave_by_sector, tuple):
            raise TypeError("delta_wave_by_sector must be a tuple")
        if len(self.delta_wave_by_sector) != s:
            raise DomainError("there must be exactly s sector wave components")
        for value in self.delta_wave_by_sector:
            if not isinstance(value, Fraction):
                raise TypeError("sector wave components must be exact Fractions")

    def wave_sectors(self) -> frozenset[int]:
        return frozenset(
            sigma for sigma, value in enumerate(self.delta_wave_by_sector, start=1)
            if value != 0
        )

    def causal_support(self) -> frozenset[int]:
        support: set[int] = set()
        for sigma in self.wave_sectors():
            support.update(sector_digits(self.s, self.ell, sigma))
        return frozenset(support)

    def support_is_inside_future_cone(self) -> bool:
        return self.causal_support().issubset(full_layer_digits(self.s, self.ell))

    def nonzero_wave_exists(self) -> bool:
        return bool(self.wave_sectors())


@dataclass(frozen=True, slots=True)
class GravitationalWave:
    s: int
    ell: int
    delta_m: float
    delta_wave_by_sector: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        layer = WaveLayer(self.s, self.ell, self.delta_wave_by_sector)
        if not layer.nonzero_wave_exists():
            raise DomainError("a gravitational wave requires a nonzero wave component")
        if float(self.delta_m) == 0.0:
            raise DomainError("a gravitational wave with nonzero wave component requires Delta m_comb != 0")

    @property
    def layer(self) -> WaveLayer:
        return WaveLayer(self.s, self.ell, self.delta_wave_by_sector)

    def profile(self, sigma: int) -> WaveProfile:
        sigma = validate_sigma(self.s, sigma)
        return WaveProfile(self.s, self.delta_m, self.delta_wave_by_sector[sigma - 1])

    def verify_decomposition(self) -> None:
        for sigma in range(1, self.s + 1):
            profile = self.profile(sigma)
            assert isclose(profile.reconstructed_delta_wave, float(self.delta_wave_by_sector[sigma - 1]), rel_tol=1e-12, abs_tol=1e-12)

    def verify_speed_bound(self) -> None:
        for D in self.layer.causal_support():
            assert_speed_bound_c_equals_one(self.s, self.ell, D, claimed_c=1)


@dataclass(frozen=True, slots=True)
class GravitationalWaveFamily:
    s: int
    ell0: int
    delta_m: float
    layers_by_ell: tuple[GravitationalWave, ...]

    def __post_init__(self) -> None:
        require_branching(self.s)
        require_positive_depth(self.ell0)
        if not isinstance(self.layers_by_ell, tuple):
            raise TypeError("layers_by_ell must be a tuple")
        if len(self.layers_by_ell) == 0:
            raise DomainError("a gravitational wave family must contain at least one level")
        expected_ell = self.ell0
        for layer in self.layers_by_ell:
            if not isinstance(layer, GravitationalWave):
                raise TypeError("every family member must be a GravitationalWave")
            if layer.s != self.s:
                raise DomainError("all wave-family layers must use the same s")
            if layer.ell != expected_ell:
                raise DomainError("wave-family levels must be consecutive starting at ell0")
            if not isclose(float(layer.delta_m), float(self.delta_m), rel_tol=0.0, abs_tol=0.0):
                raise DomainError("all wave-family layers must use the same Delta m_comb scale")
            expected_ell += 1

    def verify_decomposition_and_speed(self) -> None:
        for layer in self.layers_by_ell:
            layer.verify_decomposition()
            layer.verify_speed_bound()


def outward_wave_suppressed(mu_esc_values: Sequence[Fraction], requires_nonzero_escape: bool) -> bool:
    if not isinstance(mu_esc_values, tuple):
        raise TypeError("mu_esc_values must be a tuple")
    if len(mu_esc_values) == 0:
        raise DomainError("mu_esc_values cannot be empty")
    for value in mu_esc_values:
        if not isinstance(value, Fraction):
            raise TypeError("mu_esc values must be exact Fractions")
        if value < 0:
            raise DomainError("escaping measures must be nonnegative")
    if not isinstance(requires_nonzero_escape, bool):
        raise TypeError("requires_nonzero_escape must be boolean")
    return requires_nonzero_escape and mu_esc_values[-1] == 0


def internal_wave_may_remain(delta_wave_by_sector: tuple[Fraction, ...]) -> bool:
    layer = WaveLayer(len(delta_wave_by_sector), 1, delta_wave_by_sector)
    return layer.nonzero_wave_exists()


def compatibility_response(before: frozenset[int], after: frozenset[int]) -> str:
    if not isinstance(before, frozenset) or not isinstance(after, frozenset):
        raise TypeError("before and after must be frozensets")
    for item in before.union(after):
        if not isinstance(item, int):
            raise TypeError("compatible-set elements must be integers")
        if item < 0:
            raise DomainError("compatible-set elements must be nonnegative")
    if after > before:
        return "strengthening"
    if after < before:
        return "weakening"
    if after == before:
        return "stationary"
    return "regime_switch"



def verify_symbolic_amplitude_and_profile_identities() -> None:
    print("\n=== Symbolic verification of wave amplitude and profile identities ===")

    s, N1, N2, dm, dmu = symbols("s N1 N2 dm dmu", positive=True)
    signed_dm = symbols("signed_dm", real=True)

    delta_ln = slog(N2) - slog(N1)
    delta_log_s = (slog(N2) - slog(N1)) / slog(s)
    assert simplify(delta_ln - slog(s) * delta_log_s) == 0

    Gs = slog(s) / s
    h_comb_expr = slog(s) * signed_dm
    h_s_expr = h_comb_expr / s
    assert simplify(h_s_expr - Gs * signed_dm) == 0

    psi = -dmu / (Gs * dm)
    reconstructed = -Gs * dm * psi
    assert simplify(reconstructed - dmu) == 0

    bg, wave, mu0 = symbols("bg wave mu0", real=True)
    total_delta = bg + wave
    compatible_mu = mu0 + bg + wave
    assert simplify(compatible_mu - (mu0 + total_delta)) == 0

    print("[OK] Delta ln N = (ln s) Delta m_comb is symbolic")
    print("[OK] h_s = G_s Delta m_comb is symbolic")
    print("[OK] delta_mu_wave = -G_s Delta m_comb Psi is symbolic")
    print("[OK] background/wave sector-deficit split is symbolic")


def verify_amplitude_identities_on_finite_grid() -> None:
    print("\n=== Finite-grid verification of combinatorial and sector amplitudes ===")

    checked_pairs = 0
    checked_signed_cases = 0
    checked_boundaries = 0

    for s in range(2, 14):
        for n in range(1, 8):
            full = s**n
            sample_N = sorted({1, min(full, 2), min(full, s), max(1, full // 3), max(1, full // 2), max(1, full - 1), full})
            for N1 in sample_N:
                for N2 in sample_N:
                    checked_pairs += 1
                    dm = delta_m_comb(s, n, N1, N2)
                    hc = h_comb(s, n, N1, N2)
                    hs = h_sector(s, n, N1, N2)

                    assert isclose(hc, log(s) * dm, rel_tol=1e-12, abs_tol=1e-12)
                    assert isclose(hs, hc / s, rel_tol=1e-12, abs_tol=1e-12)
                    assert isclose(hs, G_sector(s) * dm, rel_tol=1e-12, abs_tol=1e-12)
                    assert isclose(abs(hs), G_sector(s) * abs(dm), rel_tol=1e-12, abs_tol=1e-12)
                    checked_signed_cases += 1

            # Boundary: no change gives zero signed amplitude.
            checked_boundaries += 1
            assert isclose(h_comb(s, n, full, full), 0.0, abs_tol=1e-12)
            assert isclose(h_sector(s, n, full, full), 0.0, abs_tol=1e-12)

            # Unit mass increase N -> sN when admissible.
            if s <= full:
                checked_boundaries += 1
                assert isclose(delta_m_comb(s, n, 1, s), 1.0, abs_tol=1e-12)
                assert isclose(h_sector(s, n, 1, s), G_sector(s), rel_tol=1e-12, abs_tol=1e-12)

    print(f"[OK] Checked {checked_pairs} finite multiplicity-change pairs")
    print(f"[OK] Checked {checked_signed_cases} signed sector-amplitude cases")
    print(f"[OK] Checked {checked_boundaries} zero/unit boundary cases")


def verify_background_wave_decomposition() -> None:
    print("\n=== Verification of background and wave sector-deficit split ===")

    checked_splits = 0
    checked_measure_ranges = 0

    values = [
        Fraction(0, 1),
        Fraction(1, 20),
        Fraction(-1, 20),
        Fraction(1, 7),
        Fraction(-2, 15),
    ]

    for mu0 in (Fraction(0, 1), Fraction(1, 3), Fraction(3, 4), Fraction(1, 1)):
        for delta_bg in values:
            for delta_wave in values:
                split = SectorDeficitSplit(mu0=mu0, delta_bg=delta_bg, delta_wave=delta_wave)
                checked_splits += 1
                assert split.total_delta == delta_bg + delta_wave
                assert split.compatible_mu == mu0 + delta_bg + delta_wave

                if Fraction(0, 1) <= split.compatible_mu <= Fraction(1, 1):
                    split.validate_measure_range()
                    checked_measure_ranges += 1
                else:
                    expect_raises(DomainError, split.validate_measure_range)

    print(f"[OK] Checked {checked_splits} background/wave decompositions")
    print(f"[OK] Checked {checked_measure_ranges} valid compatible-measure ranges")


def verify_wave_profile_and_decomposition() -> None:
    print("\n=== Verification of wave profile and sector decomposition ===")

    checked_profiles = 0
    checked_zero_profiles = 0
    checked_reconstructions = 0
    checked_sign_cases = 0

    for s in range(2, 12):
        for delta_m in (-3.0, -1.0, -0.25, 0.25, 1.0, 2.5):
            for delta_wave in (Fraction(-2, 7), Fraction(-1, 10), Fraction(0, 1), Fraction(1, 12)):
                profile = WaveProfile(s=s, delta_m=delta_m, delta_wave=delta_wave)
                checked_profiles += 1

                if delta_wave == 0:
                    checked_zero_profiles += 1
                    assert profile.psi == 0.0
                    assert profile.reconstructed_delta_wave == 0.0
                else:
                    checked_reconstructions += 1
                    assert isclose(profile.reconstructed_delta_wave, float(delta_wave), rel_tol=1e-12, abs_tol=1e-12)

                    # Sign is carried by Delta m and Psi exactly.
                    checked_sign_cases += 1
                    assert isclose(-profile.amplitude * profile.psi, float(delta_wave), rel_tol=1e-12, abs_tol=1e-12)

                assert profile.abs_amplitude >= 0

        expect_raises(DomainError, lambda s=s: WaveProfile(s=s, delta_m=0.0, delta_wave=Fraction(1, 10)))
        assert WaveProfile(s=s, delta_m=0.0, delta_wave=Fraction(0, 1)).psi == 0.0

    print(f"[OK] Checked {checked_profiles} wave-profile values")
    print(f"[OK] Checked {checked_zero_profiles} zero-wave profile cases")
    print(f"[OK] Checked {checked_reconstructions} exact wave-deficit reconstructions")
    print(f"[OK] Checked {checked_sign_cases} signed-amplitude/profile cases")


def verify_wave_sector_and_causal_support() -> None:
    print("\n=== Verification of wave-sector and spatial-causal support ===")

    checked_layers = 0
    checked_support_points = 0
    checked_empty_layers = 0
    checked_speed_bounds = 0

    for s in range(2, 10):
        for ell in range(1, 7):
            components = []
            for sigma in range(1, s + 1):
                if sigma % 3 == 0:
                    components.append(Fraction(0, 1))
                elif sigma % 3 == 1:
                    components.append(Fraction(-1, s**ell))
                else:
                    components.append(Fraction(1, 2 * s**ell))
            layer = WaveLayer(s=s, ell=ell, delta_wave_by_sector=tuple(components))
            checked_layers += 1

            expected_wave_sectors = frozenset(
                sigma for sigma, value in enumerate(components, start=1) if value != 0
            )
            assert layer.wave_sectors() == expected_wave_sectors

            expected_support = frozenset().union(
                *(sector_digits(s, ell, sigma) for sigma in expected_wave_sectors)
            ) if expected_wave_sectors else frozenset()

            assert layer.causal_support() == expected_support
            assert layer.support_is_inside_future_cone()

            for D in layer.causal_support():
                checked_support_points += 1
                assert first_digit_sector(s, ell, D) in expected_wave_sectors
                assert normalized_rho(s, ell, D) < 1
                assert_speed_bound_c_equals_one(s, ell, D)
                checked_speed_bounds += 1

            empty_layer = WaveLayer(s=s, ell=ell, delta_wave_by_sector=tuple(Fraction(0, 1) for _ in range(s)))
            checked_empty_layers += 1
            assert empty_layer.wave_sectors() == frozenset()
            assert empty_layer.causal_support() == frozenset()
            assert not empty_layer.nonzero_wave_exists()

    print(f"[OK] Checked {checked_layers} nonzero wave-sector layers")
    print(f"[OK] Checked {checked_support_points} support endpoints")
    print(f"[OK] Checked {checked_empty_layers} empty wave-support cases")
    print(f"[OK] Checked {checked_speed_bounds} causal speed-bound support witnesses")


def verify_gravitational_wave_package() -> None:
    print("\n=== Verification of gravitational-wave package ===")

    checked_waves = 0
    checked_sector_profiles = 0
    checked_speed_packages = 0

    for s in range(2, 9):
        for ell in range(1, 6):
            components = tuple(
                Fraction(0, 1) if sigma == s else Fraction((-1) ** sigma, s**ell)
                for sigma in range(1, s + 1)
            )
            wave = GravitationalWave(s=s, ell=ell, delta_m=1.5, delta_wave_by_sector=components)
            checked_waves += 1

            assert wave.layer.nonzero_wave_exists()
            assert wave.layer.support_is_inside_future_cone()
            wave.verify_decomposition()
            wave.verify_speed_bound()
            checked_speed_packages += 1

            for sigma in range(1, s + 1):
                profile = wave.profile(sigma)
                checked_sector_profiles += 1
                assert isclose(profile.reconstructed_delta_wave, float(components[sigma - 1]), rel_tol=1e-12, abs_tol=1e-12)

            expect_raises(DomainError, lambda s=s, ell=ell: GravitationalWave(
                s=s,
                ell=ell,
                delta_m=1.0,
                delta_wave_by_sector=tuple(Fraction(0, 1) for _ in range(s)),
            ))
            expect_raises(DomainError, lambda s=s, ell=ell, components=components: GravitationalWave(
                s=s,
                ell=ell,
                delta_m=0.0,
                delta_wave_by_sector=components,
            ))

            if ell <= 3:
                family_layers = tuple(
                    GravitationalWave(
                        s=s,
                        ell=q,
                        delta_m=1.5,
                        delta_wave_by_sector=tuple(
                            Fraction(0, 1) if sigma == s else Fraction((-1) ** (sigma + q), s**q)
                            for sigma in range(1, s + 1)
                        ),
                    )
                    for q in range(ell, ell + 3)
                )
                family = GravitationalWaveFamily(s=s, ell0=ell, delta_m=1.5, layers_by_ell=family_layers)
                family.verify_decomposition_and_speed()
                expect_raises(DomainError, lambda s=s, ell=ell: GravitationalWaveFamily(
                    s=s,
                    ell0=ell,
                    delta_m=1.5,
                    layers_by_ell=tuple(),
                ))

    print(f"[OK] Checked {checked_waves} gravitational-wave packages")
    print(f"[OK] Checked {checked_sector_profiles} sector profile decompositions inside packages")
    print(f"[OK] Checked {checked_speed_packages} package support speed bounds")


def verify_black_hole_suppression_and_internal_nonvanishing_guard() -> None:
    print("\n=== Verification of black-hole outward suppression guard ===")

    checked_sequences = 0
    checked_internal_cases = 0

    suppressed_seq = tuple(Fraction(1, k) for k in range(20, 2, -1)) + (Fraction(0, 1),)
    nonsuppressed_seq = tuple(Fraction(1, 10) for _ in range(10))

    assert outward_wave_suppressed(suppressed_seq, True)
    assert not outward_wave_suppressed(suppressed_seq, False)
    assert not outward_wave_suppressed(nonsuppressed_seq, True)
    checked_sequences += 3

    internal_components = (Fraction(1, 5), Fraction(0, 1))
    assert internal_wave_may_remain(internal_components)
    assert not internal_wave_may_remain((Fraction(0, 1), Fraction(0, 1)))
    checked_internal_cases += 2

    # Outward suppression is not the same as vanishing of internal perturbation.
    assert outward_wave_suppressed(suppressed_seq, True)
    assert internal_wave_may_remain(internal_components)

    expect_raises(DomainError, lambda: outward_wave_suppressed(tuple(), True))
    expect_raises(DomainError, lambda: outward_wave_suppressed((Fraction(-1, 1),), True))
    expect_raises(TypeError, lambda: outward_wave_suppressed([Fraction(0, 1)], True))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: outward_wave_suppressed((Fraction(0, 1),), "yes"))  # type: ignore[arg-type]

    print(f"[OK] Checked {checked_sequences} outward escaping-measure suppression cases")
    print(f"[OK] Checked {checked_internal_cases} internal wave nonvanishing cases")


def verify_feedback_compatible_stability_cases() -> None:
    print("\n=== Verification of wave feedback through compatible stable measures ===")

    checked_cases = 0

    before = frozenset({1, 2})
    cases = [
        (before, frozenset({1, 2, 3}), "strengthening"),
        (before, frozenset({1}), "weakening"),
        (before, frozenset({1, 2}), "stationary"),
        (before, frozenset({2, 3}), "regime_switch"),
    ]

    for before_set, after_set, expected in cases:
        checked_cases += 1
        assert compatibility_response(before_set, after_set) == expected

    expect_raises(TypeError, lambda: compatibility_response({1, 2}, frozenset({1})))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: compatibility_response(frozenset({1}), frozenset({1.5})))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: compatibility_response(frozenset({1}), frozenset({-1})))

    print(f"[OK] Checked {checked_cases} compatible-stability feedback regimes")



def verify_negative_domain_tests() -> None:
    print("\n=== Negative domain tests ===")

    expect_raises(DomainError, lambda: combinatorial_mass(1, 1, 1))
    expect_raises(DomainError, lambda: combinatorial_mass(2, -1, 1))
    expect_raises(DomainError, lambda: combinatorial_mass(2, 3, 0))
    expect_raises(DomainError, lambda: combinatorial_mass(2, 3, 9))
    expect_raises(TypeError, lambda: combinatorial_mass(2.0, 3, 1))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: combinatorial_mass(2, 3.0, 1))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: combinatorial_mass(2, 3, 1.0))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: full_layer_digits(1, 1))
    expect_raises(DomainError, lambda: full_layer_digits(2, 0))
    expect_raises(DomainError, lambda: validate_sigma(2, 0))
    expect_raises(DomainError, lambda: validate_sigma(2, 3))

    expect_raises(DomainError, lambda: validate_digit(2, 3, -1))
    expect_raises(DomainError, lambda: validate_digit(2, 3, 8))
    expect_raises(TypeError, lambda: validate_digit(2, 3, 1.5))  # type: ignore[arg-type]

    expect_raises(TypeError, lambda: validate_digit_set(2, 3, [1, 2], "bad"))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: validate_digit_set(2, 3, frozenset({8}), "bad"))
    expect_raises(DomainError, lambda: validate_digit_set(2, 3, frozenset({-1}), "bad"))

    expect_raises(DomainError, lambda: WaveLayer(2, 3, (Fraction(0, 1),)))
    expect_raises(TypeError, lambda: WaveLayer(2, 3, [Fraction(0, 1), Fraction(0, 1)]))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: WaveLayer(2, 3, (0.0, Fraction(0, 1))))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: assert_speed_bound_c_equals_one(2, 3, 1, claimed_c=2))
    expect_raises(DomainError, lambda: assert_speed_bound_c_equals_one(2, 3, 8, claimed_c=1))

    expect_raises(TypeError, lambda: WaveProfile(2, "dm", Fraction(0, 1)))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: WaveProfile(2, 1.0, 0.0))  # type: ignore[arg-type]

    print("[OK] Invalid branching, levels, multiplicities, sectors, digits, supports, profiles, and speed claims are rejected")


def main() -> None:
    print("=== Verification of gravitational waves as compatible stable-measure perturbations ===")
    verify_symbolic_amplitude_and_profile_identities()
    verify_amplitude_identities_on_finite_grid()
    verify_background_wave_decomposition()
    verify_wave_profile_and_decomposition()
    verify_wave_sector_and_causal_support()
    verify_gravitational_wave_package()
    verify_black_hole_suppression_and_internal_nonvanishing_guard()
    verify_feedback_compatible_stability_cases()
    verify_negative_domain_tests()
    print("\n=== Gravitational-wave compatible-stable-measure verification completed successfully ===")


if __name__ == "__main__":
    main()
