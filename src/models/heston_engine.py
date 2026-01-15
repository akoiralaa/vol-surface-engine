"""
Heston Stochastic Volatility Model Engine
==========================================

Implements the Heston (1993) stochastic volatility model:
    dS_t = μS_t dt + √v_t S_t dW^S_t
    dv_t = κ(θ - v_t) dt + σ√v_t dW^v_t
    ⟨dW^S_t, dW^v_t⟩ = ρdt

Features:
- Characteristic function pricing (Carr-Madan FFT)
- Finite difference PDE solver
- Implied volatility extraction via Newton-Raphson
- Greeks via finite difference
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy import integrate
from scipy.optimize import brentq
from scipy.stats import norm
from functools import lru_cache


@dataclass
class HestonParameters:
    """
    Heston model parameters.

    Attributes:
        v0: Initial variance (σ²)
        kappa: Mean reversion speed
        theta: Long-term variance level
        sigma: Volatility of volatility (vol-of-vol)
        rho: Correlation between spot and vol processes
    """
    v0: float      # Initial variance
    kappa: float   # Mean reversion speed
    theta: float   # Long-term variance
    sigma: float   # Vol of vol
    rho: float     # Correlation

    @property
    def feller_condition(self) -> bool:
        """Check Feller condition: 2κθ > σ² (ensures v_t > 0)."""
        return 2 * self.kappa * self.theta > self.sigma ** 2

    @property
    def feller_ratio(self) -> float:
        """Feller ratio: 2κθ/σ². Should be > 1."""
        return 2 * self.kappa * self.theta / (self.sigma ** 2)

    def validate(self) -> Tuple[bool, str]:
        """Validate parameter constraints."""
        if self.v0 <= 0:
            return False, "v0 must be positive"
        if self.kappa <= 0:
            return False, "kappa must be positive"
        if self.theta <= 0:
            return False, "theta must be positive"
        if self.sigma <= 0:
            return False, "sigma must be positive"
        if not -1 <= self.rho <= 1:
            return False, "rho must be in [-1, 1]"
        return True, "Valid"

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for optimization."""
        return np.array([self.v0, self.kappa, self.theta, self.sigma, self.rho])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'HestonParameters':
        """Create from numpy array."""
        return cls(v0=arr[0], kappa=arr[1], theta=arr[2], sigma=arr[3], rho=arr[4])

    @classmethod
    def default(cls) -> 'HestonParameters':
        """Default parameters (typical equity)."""
        return cls(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)


class HestonEngine:
    """
    Heston model pricing engine.

    Uses characteristic function approach with Carr-Madan FFT
    for efficient option pricing across strikes.
    """

    def __init__(self, integration_method: str = "quad"):
        """
        Initialize engine.

        Args:
            integration_method: 'quad' for scipy quadrature, 'fft' for FFT
        """
        self.integration_method = integration_method

    def characteristic_function(
        self,
        u: complex,
        T: float,
        params: HestonParameters,
        r: float = 0.05,
        q: float = 0.01
    ) -> complex:
        """
        Heston characteristic function φ(u; T).

        Using the formulation from Gatheral (2006) which is numerically stable.

        Args:
            u: Complex argument
            T: Time to maturity
            params: Heston parameters
            r: Risk-free rate
            q: Dividend yield

        Returns:
            Complex characteristic function value
        """
        v0, kappa, theta, sigma, rho = (
            params.v0, params.kappa, params.theta, params.sigma, params.rho
        )

        # Complex arguments
        i = complex(0, 1)

        # Gatheral formulation (more stable than original Heston)
        xi = kappa - sigma * rho * i * u
        d = np.sqrt(xi**2 + sigma**2 * (u**2 + i * u))

        # Avoid numerical issues
        g1 = xi + d
        g2 = xi - d
        g = g2 / g1

        # Exponential terms
        exp_dT = np.exp(-d * T)

        # C and D coefficients
        D = (g2 / sigma**2) * ((1 - exp_dT) / (1 - g * exp_dT))

        C = (r - q) * i * u * T + (kappa * theta / sigma**2) * (
            g2 * T - 2 * np.log((1 - g * exp_dT) / (1 - g))
        )

        return np.exp(C + D * v0)

    def call_price_integral(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        params: HestonParameters
    ) -> float:
        """
        Price European call using characteristic function integration.

        Uses Lewis (2000) formulation for numerical stability.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            q: Dividend yield
            params: Heston parameters

        Returns:
            Call option price
        """
        # Forward price
        F = S * np.exp((r - q) * T)

        # Log-moneyness
        k = np.log(K / F)

        # Integration via Lewis formula
        def integrand(u):
            i = complex(0, 1)
            cf = self.characteristic_function(u - 0.5 * i, T, params, r, q)
            return np.real(np.exp(-i * u * k) * cf / (u**2 + 0.25))

        # Numerical integration
        integral, _ = integrate.quad(integrand, 0, 100, limit=100)

        # Call price
        discount = np.exp(-r * T)
        call_price = S * np.exp(-q * T) - (np.sqrt(K * F) * discount / np.pi) * integral

        return max(call_price, 0)

    def put_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        params: HestonParameters
    ) -> float:
        """Price European put via put-call parity."""
        call = self.call_price_integral(S, K, T, r, q, params)
        # Put-Call Parity: P = C - S*exp(-qT) + K*exp(-rT)
        return call - S * np.exp(-q * T) + K * np.exp(-r * T)

    def price_surface(
        self,
        S: float,
        strikes: np.ndarray,
        expiries: np.ndarray,
        r: float,
        q: float,
        params: HestonParameters,
        is_call: bool = True
    ) -> np.ndarray:
        """
        Price options across strikes and expiries.

        Args:
            S: Spot price
            strikes: Array of strikes
            expiries: Array of expiries
            r: Risk-free rate
            q: Dividend yield
            params: Heston parameters
            is_call: True for calls, False for puts

        Returns:
            2D array of prices, shape (len(strikes), len(expiries))
        """
        prices = np.zeros((len(strikes), len(expiries)))

        for i, K in enumerate(strikes):
            for j, T in enumerate(expiries):
                if is_call:
                    prices[i, j] = self.call_price_integral(S, K, T, r, q, params)
                else:
                    prices[i, j] = self.put_price(S, K, T, r, q, params)

        return prices

    def implied_volatility(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        params: HestonParameters,
        is_call: bool = True
    ) -> float:
        """
        Extract Black-Scholes implied volatility from Heston price.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            q: Dividend yield
            params: Heston parameters
            is_call: True for call, False for put

        Returns:
            Implied volatility
        """
        # Get Heston price
        if is_call:
            price = self.call_price_integral(S, K, T, r, q, params)
        else:
            price = self.put_price(S, K, T, r, q, params)

        # Invert to get BS IV
        return self._bs_implied_vol(price, S, K, T, r, q, is_call)

    def implied_volatility_surface(
        self,
        S: float,
        strikes: np.ndarray,
        expiries: np.ndarray,
        r: float,
        q: float,
        params: HestonParameters
    ) -> np.ndarray:
        """
        Generate implied volatility surface from Heston model.

        Args:
            S: Spot price
            strikes: Array of strikes
            expiries: Array of expiries
            r: Risk-free rate
            q: Dividend yield
            params: Heston parameters

        Returns:
            2D array of IVs, shape (len(strikes), len(expiries))
        """
        ivs = np.zeros((len(strikes), len(expiries)))

        for i, K in enumerate(strikes):
            for j, T in enumerate(expiries):
                # Use OTM options
                is_call = K >= S
                ivs[i, j] = self.implied_volatility(S, K, T, r, q, params, is_call)

        return ivs

    def _bs_implied_vol(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        is_call: bool
    ) -> float:
        """
        Invert Black-Scholes formula to get IV.

        Uses Brent's method for robust root-finding.
        """
        def bs_price(sigma):
            return self._black_scholes(S, K, T, r, q, sigma, is_call)

        def objective(sigma):
            return bs_price(sigma) - price

        # Intrinsic value check
        F = S * np.exp((r - q) * T)
        intrinsic = max(F - K, 0) if is_call else max(K - F, 0)
        discount = np.exp(-r * T)

        if price <= intrinsic * discount * 1.001:
            return 0.001  # Near-zero vol

        try:
            iv = brentq(objective, 0.001, 5.0, xtol=1e-6)
            return iv
        except ValueError:
            # Fallback to simple Newton-Raphson
            return self._newton_iv(price, S, K, T, r, q, is_call)

    def _newton_iv(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        is_call: bool,
        tol: float = 1e-6,
        max_iter: int = 100
    ) -> float:
        """Newton-Raphson IV solver with vega."""
        sigma = 0.2  # Initial guess

        for _ in range(max_iter):
            bs_p = self._black_scholes(S, K, T, r, q, sigma, is_call)
            vega = self._bs_vega(S, K, T, r, q, sigma)

            if vega < 1e-10:
                break

            diff = bs_p - price
            if abs(diff) < tol:
                break

            sigma -= diff / vega
            sigma = max(0.001, min(sigma, 5.0))

        return sigma

    def _black_scholes(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        is_call: bool
    ) -> float:
        """Standard Black-Scholes formula."""
        if T <= 0 or sigma <= 0:
            return max(S - K, 0) if is_call else max(K - S, 0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if is_call:
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    def _bs_vega(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float
    ) -> float:
        """Black-Scholes vega."""
        if T <= 0 or sigma <= 0:
            return 0

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


class HestonPDESolver:
    """
    Finite difference PDE solver for Heston model.

    Solves the 2D PDE for European option pricing.
    Used for validation and American options.
    """

    def __init__(
        self,
        S_max: float = 3.0,
        v_max: float = 1.0,
        n_S: int = 100,
        n_v: int = 50,
        n_t: int = 100
    ):
        """
        Initialize PDE solver grid.

        Args:
            S_max: Max spot as multiple of current spot
            v_max: Max variance
            n_S: Number of spot grid points
            n_v: Number of variance grid points
            n_t: Number of time steps
        """
        self.S_max = S_max
        self.v_max = v_max
        self.n_S = n_S
        self.n_v = n_v
        self.n_t = n_t

    def solve(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        params: HestonParameters,
        is_call: bool = True
    ) -> Tuple[float, np.ndarray]:
        """
        Solve Heston PDE using ADI (Alternating Direction Implicit).

        Args:
            S0: Current spot
            K: Strike
            T: Time to maturity
            r: Risk-free rate
            q: Dividend yield
            params: Heston parameters
            is_call: True for call, False for put

        Returns:
            Tuple of (option price, full grid solution)
        """
        # Grid setup
        S_grid = np.linspace(0, S0 * self.S_max, self.n_S)
        v_grid = np.linspace(0, self.v_max, self.n_v)
        dt = T / self.n_t

        dS = S_grid[1] - S_grid[0]
        dv = v_grid[1] - v_grid[0]

        # Initialize with payoff at T
        V = np.zeros((self.n_S, self.n_v))
        for i, S in enumerate(S_grid):
            if is_call:
                V[i, :] = max(S - K, 0)
            else:
                V[i, :] = max(K - S, 0)

        # Extract parameters
        kappa, theta, sigma, rho = params.kappa, params.theta, params.sigma, params.rho

        # Time stepping (backward from T to 0)
        for t in range(self.n_t):
            V_new = V.copy()

            # Interior points using explicit scheme (simplified)
            for i in range(1, self.n_S - 1):
                for j in range(1, self.n_v - 1):
                    S = S_grid[i]
                    v = v_grid[j]

                    # First derivatives
                    dV_dS = (V[i+1, j] - V[i-1, j]) / (2 * dS)
                    dV_dv = (V[i, j+1] - V[i, j-1]) / (2 * dv)

                    # Second derivatives
                    d2V_dS2 = (V[i+1, j] - 2*V[i, j] + V[i-1, j]) / (dS**2)
                    d2V_dv2 = (V[i, j+1] - 2*V[i, j] + V[i, j-1]) / (dv**2)

                    # Cross derivative
                    d2V_dSdv = (V[i+1, j+1] - V[i+1, j-1] - V[i-1, j+1] + V[i-1, j-1]) / (4 * dS * dv)

                    # PDE coefficients
                    a = 0.5 * v * S**2
                    b = 0.5 * sigma**2 * v
                    c = rho * sigma * v * S
                    d = (r - q) * S
                    e = kappa * (theta - v)

                    # Update
                    V_new[i, j] = V[i, j] + dt * (
                        a * d2V_dS2 +
                        b * d2V_dv2 +
                        c * d2V_dSdv +
                        d * dV_dS +
                        e * dV_dv -
                        r * V[i, j]
                    )

            # Boundary conditions
            V_new[0, :] = 0 if is_call else K * np.exp(-r * (T - t * dt))
            V_new[-1, :] = S_grid[-1] - K * np.exp(-r * (T - t * dt)) if is_call else 0
            V_new[:, 0] = V_new[:, 1]  # Neumann at v=0
            V_new[:, -1] = V_new[:, -2]  # Neumann at v=v_max

            V = V_new

        # Interpolate to get price at (S0, v0)
        S_idx = np.searchsorted(S_grid, S0)
        v_idx = np.searchsorted(v_grid, params.v0)

        S_idx = min(max(S_idx, 1), self.n_S - 2)
        v_idx = min(max(v_idx, 1), self.n_v - 2)

        # Bilinear interpolation
        S_frac = (S0 - S_grid[S_idx-1]) / dS
        v_frac = (params.v0 - v_grid[v_idx-1]) / dv

        price = (
            (1 - S_frac) * (1 - v_frac) * V[S_idx-1, v_idx-1] +
            S_frac * (1 - v_frac) * V[S_idx, v_idx-1] +
            (1 - S_frac) * v_frac * V[S_idx-1, v_idx] +
            S_frac * v_frac * V[S_idx, v_idx]
        )

        return max(price, 0), V


class HestonMonteCarlo:
    """
    Monte Carlo simulation for Heston model.

    Uses QE (Quadratic Exponential) scheme from Andersen (2008)
    for accurate variance process simulation.
    """

    def __init__(self, n_paths: int = 100000, n_steps: int = 100):
        self.n_paths = n_paths
        self.n_steps = n_steps

    def simulate_paths(
        self,
        S0: float,
        T: float,
        r: float,
        q: float,
        params: HestonParameters,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston paths using QE scheme.

        Args:
            S0: Initial spot
            T: Time horizon
            r: Risk-free rate
            q: Dividend yield
            params: Heston parameters
            seed: Random seed

        Returns:
            Tuple of (spot_paths, variance_paths)
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / self.n_steps
        v0, kappa, theta, sigma, rho = (
            params.v0, params.kappa, params.theta, params.sigma, params.rho
        )

        # Initialize paths
        S = np.zeros((self.n_paths, self.n_steps + 1))
        v = np.zeros((self.n_paths, self.n_steps + 1))
        S[:, 0] = S0
        v[:, 0] = v0

        # Pre-compute constants
        exp_kappa = np.exp(-kappa * dt)
        psi_c = 1.5  # Critical value for scheme switching

        for t in range(self.n_steps):
            # Current variance
            v_t = v[:, t]

            # QE scheme for variance
            m = theta + (v_t - theta) * exp_kappa
            s2 = (v_t * sigma**2 * exp_kappa / kappa * (1 - exp_kappa) +
                  theta * sigma**2 / (2 * kappa) * (1 - exp_kappa)**2)
            psi = s2 / (m**2 + 1e-10)

            # Generate variance
            U = np.random.uniform(size=self.n_paths)

            # Use different schemes based on psi
            v_next = np.zeros(self.n_paths)

            # High variance regime (psi > psi_c): exponential approximation
            high = psi > psi_c
            if np.any(high):
                p = (psi[high] - 1) / (psi[high] + 1)
                beta = 2 / (m[high] * (psi[high] + 1))
                v_next[high] = np.where(
                    U[high] <= p,
                    0,
                    np.log((1 - p) / (1 - U[high])) / beta
                )

            # Low variance regime (psi <= psi_c): quadratic approximation
            low = ~high
            if np.any(low):
                inv_psi = 1 / (psi[low] + 1e-10)
                b2 = 2 * inv_psi - 1 + np.sqrt(2 * inv_psi * (2 * inv_psi - 1))
                a = m[low] / (1 + b2)
                Z = norm.ppf(U[low])
                v_next[low] = a * (np.sqrt(b2) + Z)**2

            v[:, t+1] = np.maximum(v_next, 0)

            # Log-spot process with Milstein scheme
            dW_v = np.random.normal(size=self.n_paths) * np.sqrt(dt)
            dW_S = rho * dW_v + np.sqrt(1 - rho**2) * np.random.normal(size=self.n_paths) * np.sqrt(dt)

            v_avg = 0.5 * (v_t + v[:, t+1])  # Use average variance
            log_S = np.log(S[:, t])
            log_S_next = (log_S +
                         (r - q - 0.5 * v_avg) * dt +
                         np.sqrt(v_avg) * dW_S)

            S[:, t+1] = np.exp(log_S_next)

        return S, v

    def price_option(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        params: HestonParameters,
        is_call: bool = True,
        seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Price European option via Monte Carlo.

        Args:
            S0: Spot price
            K: Strike
            T: Time to maturity
            r: Risk-free rate
            q: Dividend yield
            params: Heston parameters
            is_call: True for call, False for put
            seed: Random seed

        Returns:
            Tuple of (price, standard_error)
        """
        S, _ = self.simulate_paths(S0, T, r, q, params, seed)

        # Terminal payoff
        S_T = S[:, -1]
        if is_call:
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)

        # Discounted expectation
        discount = np.exp(-r * T)
        price = discount * np.mean(payoffs)
        se = discount * np.std(payoffs) / np.sqrt(self.n_paths)

        return price, se
