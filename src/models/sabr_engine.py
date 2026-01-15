"""
SABR Stochastic Volatility Model Engine
========================================

Implements the SABR (Stochastic Alpha Beta Rho) model from Hagan et al. (2002):
    dF_t = σ_t F_t^β dW^F_t
    dσ_t = ν σ_t dW^σ_t
    ⟨dW^F, dW^σ⟩ = ρdt

Features:
- Hagan approximation for implied volatility
- Obloj correction for improved accuracy
- Smile interpolation across strikes
- Per-expiry calibration (SABR is a single-expiry model)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
from scipy.optimize import minimize, brentq
from scipy.stats import norm


@dataclass
class SABRParameters:
    """
    SABR model parameters for a single expiry.

    Attributes:
        alpha: Initial volatility level (σ_0)
        beta: CEV exponent (0 = normal, 1 = lognormal)
        rho: Correlation between F and σ
        nu: Volatility of volatility
    """
    alpha: float  # Initial vol
    beta: float   # CEV exponent [0, 1]
    rho: float    # Correlation [-1, 1]
    nu: float     # Vol of vol

    def validate(self) -> Tuple[bool, str]:
        """Validate parameter constraints."""
        if self.alpha <= 0:
            return False, "alpha must be positive"
        if not 0 <= self.beta <= 1:
            return False, "beta must be in [0, 1]"
        if not -1 < self.rho < 1:
            return False, "rho must be in (-1, 1)"
        if self.nu < 0:
            return False, "nu must be non-negative"
        return True, "Valid"

    def to_array(self, include_beta: bool = False) -> np.ndarray:
        """Convert to numpy array for optimization."""
        if include_beta:
            return np.array([self.alpha, self.beta, self.rho, self.nu])
        return np.array([self.alpha, self.rho, self.nu])

    @classmethod
    def from_array(cls, arr: np.ndarray, beta: float = 0.5) -> 'SABRParameters':
        """Create from numpy array."""
        if len(arr) == 4:
            return cls(alpha=arr[0], beta=arr[1], rho=arr[2], nu=arr[3])
        return cls(alpha=arr[0], beta=beta, rho=arr[1], nu=arr[2])

    @classmethod
    def default(cls) -> 'SABRParameters':
        """Default parameters."""
        return cls(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)


class SABREngine:
    """
    SABR model pricing engine.

    Implements the Hagan et al. approximation formulas for
    implied volatility with various corrections.
    """

    def __init__(self, approximation: str = "hagan"):
        """
        Initialize engine.

        Args:
            approximation: 'hagan' for original, 'obloj' for improved
        """
        self.approximation = approximation

    def implied_volatility(
        self,
        F: float,
        K: float,
        T: float,
        params: SABRParameters
    ) -> float:
        """
        Calculate SABR implied volatility using Hagan approximation.

        Args:
            F: Forward price
            K: Strike price
            T: Time to expiry
            params: SABR parameters

        Returns:
            Black implied volatility
        """
        alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu

        # Handle ATM case separately
        if abs(F - K) < 1e-10:
            return self._atm_volatility(F, T, params)

        # Hagan formula components
        FK = F * K
        log_FK = np.log(F / K)

        # z and x(z) functions
        z = (nu / alpha) * (FK ** ((1 - beta) / 2)) * log_FK

        # Avoid numerical issues
        if abs(z) < 1e-10:
            x_z = 1.0
        else:
            sqrt_term = np.sqrt(1 - 2 * rho * z + z**2)
            x_z = np.log((sqrt_term + z - rho) / (1 - rho)) / z

        # Correction factors
        FK_beta = FK ** ((1 - beta) / 2)

        # Denominator term
        denom = (1 +
                (1 - beta)**2 / 24 * log_FK**2 +
                (1 - beta)**4 / 1920 * log_FK**4)

        # Numerator first term
        num1 = alpha / (FK_beta * denom)

        # Correction term
        term1 = (1 - beta)**2 / 24 * alpha**2 / (FK ** (1 - beta))
        term2 = 0.25 * rho * beta * nu * alpha / FK_beta
        term3 = (2 - 3 * rho**2) / 24 * nu**2

        correction = 1 + (term1 + term2 + term3) * T

        # Final implied vol
        sigma = num1 * z / x_z * correction

        return max(sigma, 0.001)

    def _atm_volatility(self, F: float, T: float, params: SABRParameters) -> float:
        """ATM volatility approximation."""
        alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu

        F_beta = F ** (1 - beta)

        term1 = (1 - beta)**2 / 24 * alpha**2 / F_beta**2
        term2 = 0.25 * rho * beta * nu * alpha / F_beta
        term3 = (2 - 3 * rho**2) / 24 * nu**2

        sigma_atm = (alpha / F_beta) * (1 + (term1 + term2 + term3) * T)

        return max(sigma_atm, 0.001)

    def implied_volatility_obloj(
        self,
        F: float,
        K: float,
        T: float,
        params: SABRParameters
    ) -> float:
        """
        Obloj (2008) improved SABR approximation.

        More accurate than Hagan for large moneyness/vol-of-vol.
        """
        alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu

        if abs(F - K) < 1e-10:
            return self._atm_volatility(F, T, params)

        # Midpoint and log-moneyness
        F_mid = np.sqrt(F * K)
        y = np.log(F / K)

        # Approximate integral of CEV variance
        if abs(beta - 1) < 1e-10:
            I = y
        elif abs(beta) < 1e-10:
            I = (F - K) / F_mid
        else:
            I = (F**(1-beta) - K**(1-beta)) / ((1 - beta) * F_mid**(1-beta))

        # z variable
        z = nu / alpha * I

        # chi function
        if abs(z) < 1e-10:
            chi = 1.0
        else:
            chi = z / np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))

        # Correction terms
        gamma1 = beta / F_mid
        gamma2 = beta * (beta - 2) / F_mid**2

        correction = (1 +
            (-gamma1**2/24 + gamma2/24) * alpha**2 * T +
            0.25 * rho * nu * alpha * gamma1 * T +
            (2 - 3*rho**2) / 24 * nu**2 * T
        )

        sigma = alpha / I * chi * correction

        return max(sigma, 0.001)

    def smile(
        self,
        F: float,
        strikes: np.ndarray,
        T: float,
        params: SABRParameters
    ) -> np.ndarray:
        """
        Calculate implied volatility smile.

        Args:
            F: Forward price
            strikes: Array of strikes
            T: Time to expiry
            params: SABR parameters

        Returns:
            Array of implied volatilities
        """
        vol_func = (self.implied_volatility_obloj
                   if self.approximation == "obloj"
                   else self.implied_volatility)

        return np.array([vol_func(F, K, T, params) for K in strikes])

    def calibrate(
        self,
        F: float,
        strikes: np.ndarray,
        T: float,
        market_ivs: np.ndarray,
        beta: float = 0.5,
        weights: Optional[np.ndarray] = None,
        initial_guess: Optional[SABRParameters] = None
    ) -> Tuple[SABRParameters, float]:
        """
        Calibrate SABR parameters to market implied vols.

        Args:
            F: Forward price
            strikes: Array of strikes
            T: Time to expiry
            market_ivs: Market implied volatilities
            beta: Fixed beta parameter (often set to 0.5 or 1)
            weights: Optional weights for each strike
            initial_guess: Optional initial parameters

        Returns:
            Tuple of (calibrated parameters, RMSE)
        """
        if weights is None:
            weights = np.ones(len(strikes))

        weights = weights / weights.sum()

        # Initial guess
        if initial_guess:
            x0 = initial_guess.to_array(include_beta=False)
        else:
            atm_iv = market_ivs[np.argmin(np.abs(strikes - F))]
            x0 = np.array([atm_iv * F**(1-beta), -0.3, 0.4])

        # Bounds: alpha > 0, -0.99 < rho < 0.99, nu >= 0
        bounds = [(0.001, 10.0), (-0.99, 0.99), (0.001, 5.0)]

        def objective(x):
            params = SABRParameters(alpha=x[0], beta=beta, rho=x[1], nu=x[2])
            model_ivs = self.smile(F, strikes, T, params)
            errors = (model_ivs - market_ivs) ** 2
            return np.sum(weights * errors)

        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500}
        )

        params = SABRParameters(
            alpha=result.x[0],
            beta=beta,
            rho=result.x[1],
            nu=result.x[2]
        )

        # Calculate RMSE
        model_ivs = self.smile(F, strikes, T, params)
        rmse = np.sqrt(np.mean((model_ivs - market_ivs) ** 2))

        return params, rmse


@dataclass
class SABRSurface:
    """
    Collection of SABR calibrations across expiries.

    Stores per-expiry SABR parameters for full surface interpolation.
    """
    expiries: np.ndarray
    parameters: Dict[float, SABRParameters]
    forward_curve: Dict[float, float]  # T -> F(T)
    calibration_errors: Dict[float, float]  # T -> RMSE

    def get_params(self, T: float) -> SABRParameters:
        """Get SABR parameters for expiry T (with interpolation)."""
        if T in self.parameters:
            return self.parameters[T]

        # Linear interpolation of parameters
        T_arr = np.array(sorted(self.parameters.keys()))
        idx = np.searchsorted(T_arr, T)

        if idx == 0:
            return self.parameters[T_arr[0]]
        if idx == len(T_arr):
            return self.parameters[T_arr[-1]]

        T_lo, T_hi = T_arr[idx-1], T_arr[idx]
        w = (T - T_lo) / (T_hi - T_lo)

        p_lo = self.parameters[T_lo]
        p_hi = self.parameters[T_hi]

        return SABRParameters(
            alpha=p_lo.alpha * (1-w) + p_hi.alpha * w,
            beta=p_lo.beta,  # Keep beta constant
            rho=p_lo.rho * (1-w) + p_hi.rho * w,
            nu=p_lo.nu * (1-w) + p_hi.nu * w
        )

    def implied_vol(self, K: float, T: float, engine: Optional[SABREngine] = None) -> float:
        """Get implied vol at strike K, expiry T."""
        if engine is None:
            engine = SABREngine()

        params = self.get_params(T)
        F = self._interpolate_forward(T)

        return engine.implied_volatility(F, K, T, params)

    def _interpolate_forward(self, T: float) -> float:
        """Interpolate forward price for expiry T."""
        T_arr = np.array(sorted(self.forward_curve.keys()))
        F_arr = np.array([self.forward_curve[t] for t in T_arr])

        if T <= T_arr[0]:
            return F_arr[0]
        if T >= T_arr[-1]:
            return F_arr[-1]

        return np.interp(T, T_arr, F_arr)


class SABRCalibrator:
    """
    Calibrate SABR surface from market data.
    """

    def __init__(self, beta: float = 0.5, engine: Optional[SABREngine] = None):
        """
        Initialize calibrator.

        Args:
            beta: Fixed beta for all expiries
            engine: SABR pricing engine
        """
        self.beta = beta
        self.engine = engine or SABREngine()

    def calibrate_slice(
        self,
        F: float,
        strikes: np.ndarray,
        T: float,
        market_ivs: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Tuple[SABRParameters, float]:
        """Calibrate single expiry slice."""
        return self.engine.calibrate(F, strikes, T, market_ivs, self.beta, weights)

    def calibrate_surface(
        self,
        spot: float,
        strikes_by_expiry: Dict[float, np.ndarray],
        ivs_by_expiry: Dict[float, np.ndarray],
        r: float = 0.05,
        q: float = 0.01
    ) -> SABRSurface:
        """
        Calibrate full SABR surface.

        Args:
            spot: Current spot price
            strikes_by_expiry: Dict mapping T -> strikes array
            ivs_by_expiry: Dict mapping T -> market IVs
            r: Risk-free rate
            q: Dividend yield

        Returns:
            Calibrated SABRSurface
        """
        expiries = np.array(sorted(strikes_by_expiry.keys()))
        parameters = {}
        forward_curve = {}
        errors = {}

        for T in expiries:
            F = spot * np.exp((r - q) * T)
            forward_curve[T] = F

            strikes = strikes_by_expiry[T]
            ivs = ivs_by_expiry[T]

            params, rmse = self.calibrate_slice(F, strikes, T, ivs)
            parameters[T] = params
            errors[T] = rmse

        return SABRSurface(
            expiries=expiries,
            parameters=parameters,
            forward_curve=forward_curve,
            calibration_errors=errors
        )


def sabr_density(
    F: float,
    strikes: np.ndarray,
    T: float,
    params: SABRParameters,
    engine: Optional[SABREngine] = None
) -> np.ndarray:
    """
    Extract risk-neutral density from SABR smile.

    Uses Breeden-Litzenberger formula:
        p(K) = e^(rT) * d²C/dK²

    Args:
        F: Forward price
        strikes: Array of strikes
        T: Time to expiry
        params: SABR parameters
        engine: SABR engine

    Returns:
        Probability density at each strike
    """
    if engine is None:
        engine = SABREngine()

    # Get call prices
    def call_price(K: float) -> float:
        sigma = engine.implied_volatility(F, K, T, params)
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return F * norm.cdf(d1) - K * norm.cdf(d2)

    # Numerical second derivative
    dK = strikes[1] - strikes[0] if len(strikes) > 1 else 0.01 * F
    density = np.zeros(len(strikes))

    for i, K in enumerate(strikes):
        C_up = call_price(K + dK)
        C_mid = call_price(K)
        C_down = call_price(K - dK)
        density[i] = (C_up - 2 * C_mid + C_down) / (dK ** 2)

    return np.maximum(density, 0)
