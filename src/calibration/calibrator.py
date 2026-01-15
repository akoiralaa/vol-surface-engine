"""
Volatility Surface Calibration Framework
=========================================

Unified calibration framework for Heston and SABR models.
Implements multiple optimization algorithms:
- Nelder-Mead (derivative-free)
- Differential Evolution (global)
- L-BFGS-B (gradient-based with bounds)

Features:
- Joint and sequential calibration
- Regularization and penalty terms
- Confidence intervals via bootstrap
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Union, Optional, Dict, List, Tuple, Callable
from enum import Enum
from scipy.optimize import minimize, differential_evolution
import time

from ..data.polygon_client import OptionChainSnapshot, OptionContract
from ..models.heston_engine import HestonEngine, HestonParameters
from ..models.sabr_engine import SABREngine, SABRParameters, SABRSurface


class ModelType(Enum):
    HESTON = "heston"
    SABR = "sabr"


class OptimizationMethod(Enum):
    NELDER_MEAD = "Nelder-Mead"
    L_BFGS_B = "L-BFGS-B"
    DIFFERENTIAL_EVOLUTION = "DE"
    POWELL = "Powell"


@dataclass
class CalibrationResult:
    """
    Calibration output with diagnostics.
    """
    parameters: Union[HestonParameters, SABRParameters, SABRSurface]
    model_type: ModelType
    rmse: float  # Root mean squared error in vol points
    max_error: float  # Maximum absolute error
    n_iterations: int
    elapsed_time: float
    success: bool
    message: str
    diagnostics: Dict = field(default_factory=dict)

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Calibration Result ({self.model_type.value})",
            "=" * 40,
            f"RMSE: {self.rmse:.4f} ({self.rmse*100:.2f}%)",
            f"Max Error: {self.max_error:.4f} ({self.max_error*100:.2f}%)",
            f"Iterations: {self.n_iterations}",
            f"Time: {self.elapsed_time:.2f}s",
            f"Success: {self.success}",
            "",
            "Parameters:",
        ]

        if isinstance(self.parameters, HestonParameters):
            p = self.parameters
            lines.extend([
                f"  v0 (initial var): {p.v0:.6f} (vol: {np.sqrt(p.v0)*100:.2f}%)",
                f"  kappa (mean rev): {p.kappa:.4f}",
                f"  theta (long var): {p.theta:.6f} (vol: {np.sqrt(p.theta)*100:.2f}%)",
                f"  sigma (vol-of-vol): {p.sigma:.4f}",
                f"  rho (correlation): {p.rho:.4f}",
                f"  Feller condition: {'✓' if p.feller_condition else '✗'}",
            ])
        elif isinstance(self.parameters, SABRParameters):
            p = self.parameters
            lines.extend([
                f"  alpha: {p.alpha:.6f}",
                f"  beta: {p.beta:.4f}",
                f"  rho: {p.rho:.4f}",
                f"  nu: {p.nu:.4f}",
            ])

        return "\n".join(lines)


class HestonCalibrator:
    """
    Calibrator for Heston stochastic volatility model.

    Uses Nelder-Mead optimization to minimize IV fitting error.
    """

    def __init__(
        self,
        method: OptimizationMethod = OptimizationMethod.NELDER_MEAD,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        regularization: float = 0.0
    ):
        """
        Initialize calibrator.

        Args:
            method: Optimization method
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            regularization: L2 penalty on parameters
        """
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.engine = HestonEngine()

    def calibrate(
        self,
        snapshot: OptionChainSnapshot,
        initial_params: Optional[HestonParameters] = None,
        weights: Optional[np.ndarray] = None,
        vega_weight: bool = True
    ) -> CalibrationResult:
        """
        Calibrate Heston model to market data.

        Args:
            snapshot: Option chain snapshot
            initial_params: Starting parameters
            weights: Custom weights per contract
            vega_weight: If True, weight by vega (more weight on ATM)

        Returns:
            CalibrationResult with fitted parameters
        """
        start_time = time.time()

        # Extract market data
        spot = snapshot.spot_price
        r = snapshot.risk_free_rate
        q = snapshot.dividend_yield
        contracts = snapshot.contracts

        # Filter valid contracts
        valid_contracts = [
            c for c in contracts
            if c.implied_volatility and c.implied_volatility > 0.01
        ]

        if len(valid_contracts) < 5:
            return CalibrationResult(
                parameters=HestonParameters.default(),
                model_type=ModelType.HESTON,
                rmse=float('inf'),
                max_error=float('inf'),
                n_iterations=0,
                elapsed_time=0,
                success=False,
                message="Insufficient valid contracts"
            )

        # Market data arrays
        strikes = np.array([c.strike for c in valid_contracts])
        expiries = np.array([c.time_to_expiry() for c in valid_contracts])
        market_ivs = np.array([c.implied_volatility for c in valid_contracts])
        is_call = np.array([c.option_type.value == "call" for c in valid_contracts])

        # Compute weights
        if weights is None:
            weights = np.ones(len(valid_contracts))

            if vega_weight:
                # Higher weight for ATM options
                moneyness = np.log(strikes / spot)
                weights *= np.exp(-0.5 * (moneyness / 0.1) ** 2)

        weights = weights / weights.sum()

        # Initial guess
        if initial_params is None:
            atm_iv = np.mean(market_ivs)
            initial_params = HestonParameters(
                v0=atm_iv**2,
                kappa=2.0,
                theta=atm_iv**2,
                sigma=0.5,
                rho=-0.7
            )

        x0 = initial_params.to_array()

        # Parameter bounds
        bounds = [
            (0.001, 1.0),     # v0
            (0.01, 10.0),     # kappa
            (0.001, 1.0),     # theta
            (0.01, 2.0),      # sigma
            (-0.99, 0.99)     # rho
        ]

        # Objective function
        n_evals = [0]

        def objective(x: np.ndarray) -> float:
            n_evals[0] += 1

            # Unpack parameters
            v0, kappa, theta, sigma, rho = x

            # Enforce constraints
            if v0 <= 0 or theta <= 0 or sigma <= 0 or kappa <= 0:
                return 1e10
            if not -1 < rho < 1:
                return 1e10

            params = HestonParameters(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)

            # Compute model IVs
            try:
                model_ivs = np.array([
                    self.engine.implied_volatility(
                        spot, K, T, r, q, params, is_call=ic
                    )
                    for K, T, ic in zip(strikes, expiries, is_call)
                ])
            except Exception:
                return 1e10

            # Weighted RMSE
            sq_errors = (model_ivs - market_ivs) ** 2
            rmse = np.sqrt(np.sum(weights * sq_errors))

            # Regularization (penalize extreme params)
            reg = self.regularization * (
                (kappa - 2) ** 2 +
                (sigma - 0.5) ** 2 +
                (rho + 0.5) ** 2
            )

            # Feller penalty
            if not params.feller_condition:
                feller_penalty = 0.01 * (sigma**2 - 2 * kappa * theta) ** 2
            else:
                feller_penalty = 0

            return rmse + reg + feller_penalty

        # Run optimization
        if self.method == OptimizationMethod.DIFFERENTIAL_EVOLUTION:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=self.max_iterations,
                tol=self.tolerance,
                seed=42,
                workers=1
            )
        else:
            result = minimize(
                objective,
                x0,
                method=self.method.value,
                bounds=bounds if self.method == OptimizationMethod.L_BFGS_B else None,
                options={
                    'maxiter': self.max_iterations,
                    'xatol': self.tolerance,
                    'fatol': self.tolerance
                }
            )

        elapsed = time.time() - start_time

        # Extract fitted parameters
        fitted_params = HestonParameters.from_array(result.x)

        # Compute final errors
        model_ivs = np.array([
            self.engine.implied_volatility(spot, K, T, r, q, fitted_params, is_call=ic)
            for K, T, ic in zip(strikes, expiries, is_call)
        ])

        iv_errors = model_ivs - market_ivs
        rmse = np.sqrt(np.mean(iv_errors ** 2))
        max_error = np.max(np.abs(iv_errors))

        return CalibrationResult(
            parameters=fitted_params,
            model_type=ModelType.HESTON,
            rmse=rmse,
            max_error=max_error,
            n_iterations=n_evals[0],
            elapsed_time=elapsed,
            success=result.success if hasattr(result, 'success') else True,
            message=result.message if hasattr(result, 'message') else "Completed",
            diagnostics={
                'market_ivs': market_ivs,
                'model_ivs': model_ivs,
                'iv_errors': iv_errors,
                'strikes': strikes,
                'expiries': expiries,
                'weights': weights,
                'contracts': valid_contracts
            }
        )


class SABRSurfaceCalibrator:
    """
    Calibrate SABR surface slice-by-slice.
    """

    def __init__(
        self,
        beta: float = 0.5,
        max_iterations: int = 200
    ):
        """
        Initialize calibrator.

        Args:
            beta: Fixed beta parameter (typically 0.5 for FX, 1 for rates)
            max_iterations: Max iterations per slice
        """
        self.beta = beta
        self.max_iterations = max_iterations
        self.engine = SABREngine()

    def calibrate(
        self,
        snapshot: OptionChainSnapshot
    ) -> CalibrationResult:
        """
        Calibrate SABR surface from option chain.

        Args:
            snapshot: Option chain snapshot

        Returns:
            CalibrationResult with SABRSurface
        """
        start_time = time.time()

        spot = snapshot.spot_price
        r = snapshot.risk_free_rate
        q = snapshot.dividend_yield
        contracts = snapshot.contracts

        # Group by expiry
        expiry_groups: Dict[float, List[OptionContract]] = {}
        for c in contracts:
            if c.implied_volatility and c.implied_volatility > 0.01:
                T = round(c.time_to_expiry(), 4)
                if T not in expiry_groups:
                    expiry_groups[T] = []
                expiry_groups[T].append(c)

        # Calibrate each slice
        parameters = {}
        forward_curve = {}
        errors = {}
        total_n_iter = 0

        all_model_ivs = []
        all_market_ivs = []

        for T in sorted(expiry_groups.keys()):
            contracts_T = expiry_groups[T]
            if len(contracts_T) < 3:
                continue

            F = spot * np.exp((r - q) * T)
            forward_curve[T] = F

            strikes = np.array([c.strike for c in contracts_T])
            market_ivs = np.array([c.implied_volatility for c in contracts_T])

            # Sort by strike
            sort_idx = np.argsort(strikes)
            strikes = strikes[sort_idx]
            market_ivs = market_ivs[sort_idx]

            # Calibrate slice
            params, rmse = self.engine.calibrate(
                F, strikes, T, market_ivs, beta=self.beta
            )
            parameters[T] = params
            errors[T] = rmse
            total_n_iter += self.max_iterations

            # Store for overall metrics
            model_ivs = self.engine.smile(F, strikes, T, params)
            all_model_ivs.extend(model_ivs)
            all_market_ivs.extend(market_ivs)

        elapsed = time.time() - start_time

        # Create surface
        surface = SABRSurface(
            expiries=np.array(sorted(parameters.keys())),
            parameters=parameters,
            forward_curve=forward_curve,
            calibration_errors=errors
        )

        # Overall metrics
        all_model_ivs = np.array(all_model_ivs)
        all_market_ivs = np.array(all_market_ivs)
        iv_errors = all_model_ivs - all_market_ivs
        rmse = np.sqrt(np.mean(iv_errors ** 2))
        max_error = np.max(np.abs(iv_errors))

        return CalibrationResult(
            parameters=surface,
            model_type=ModelType.SABR,
            rmse=rmse,
            max_error=max_error,
            n_iterations=total_n_iter,
            elapsed_time=elapsed,
            success=True,
            message="Calibrated all slices",
            diagnostics={
                'market_ivs': all_market_ivs,
                'model_ivs': all_model_ivs,
                'iv_errors': iv_errors,
                'slice_errors': errors
            }
        )


class JointCalibrator:
    """
    Joint calibration of Heston with SABR corrections.

    Fits Heston for term structure, SABR for short-term smile.
    """

    def __init__(
        self,
        heston_weight: float = 0.7,
        sabr_weight: float = 0.3
    ):
        """
        Initialize joint calibrator.

        Args:
            heston_weight: Weight for Heston in combined objective
            sabr_weight: Weight for SABR corrections
        """
        self.heston_weight = heston_weight
        self.sabr_weight = sabr_weight
        self.heston_calibrator = HestonCalibrator()
        self.sabr_calibrator = SABRSurfaceCalibrator()

    def calibrate(
        self,
        snapshot: OptionChainSnapshot
    ) -> Tuple[CalibrationResult, CalibrationResult]:
        """
        Run joint calibration.

        First fits Heston for overall dynamics, then SABR for residuals.

        Args:
            snapshot: Option chain snapshot

        Returns:
            Tuple of (HestonResult, SABRResult)
        """
        # Step 1: Calibrate Heston
        heston_result = self.heston_calibrator.calibrate(snapshot)

        # Step 2: Compute Heston residuals and fit SABR
        # (For now, just run independent SABR calibration)
        sabr_result = self.sabr_calibrator.calibrate(snapshot)

        return heston_result, sabr_result


def bootstrap_confidence_intervals(
    snapshot: OptionChainSnapshot,
    calibrator: Union[HestonCalibrator, SABRSurfaceCalibrator],
    n_bootstrap: int = 100,
    confidence: float = 0.95
) -> Dict[str, Tuple[float, float]]:
    """
    Compute confidence intervals via bootstrap.

    Args:
        snapshot: Option chain data
        calibrator: Calibrator instance
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Dict mapping parameter names to (lower, upper) CI
    """
    n_contracts = len(snapshot.contracts)
    param_samples = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_contracts, n_contracts, replace=True)
        resampled = OptionChainSnapshot(
            underlying=snapshot.underlying,
            spot_price=snapshot.spot_price,
            timestamp=snapshot.timestamp,
            contracts=[snapshot.contracts[i] for i in indices],
            risk_free_rate=snapshot.risk_free_rate,
            dividend_yield=snapshot.dividend_yield
        )

        try:
            result = calibrator.calibrate(resampled)
            if isinstance(result.parameters, HestonParameters):
                param_samples.append(result.parameters.to_array())
            elif isinstance(result.parameters, SABRParameters):
                param_samples.append(result.parameters.to_array(include_beta=True))
        except Exception:
            continue

    if not param_samples:
        return {}

    param_samples = np.array(param_samples)
    alpha = (1 - confidence) / 2

    # Compute percentile intervals
    lower = np.percentile(param_samples, 100 * alpha, axis=0)
    upper = np.percentile(param_samples, 100 * (1 - alpha), axis=0)

    # Map to parameter names
    if isinstance(calibrator, HestonCalibrator):
        names = ['v0', 'kappa', 'theta', 'sigma', 'rho']
    else:
        names = ['alpha', 'beta', 'rho', 'nu']

    return {name: (lower[i], upper[i]) for i, name in enumerate(names)}
