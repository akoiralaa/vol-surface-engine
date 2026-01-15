"""
Volatility Surface Engine
=========================

A comprehensive toolkit for stochastic volatility surface calibration
and smile arbitrage detection.

Modules:
- data: Polygon.io options data fetching
- models: Heston and SABR stochastic volatility models
- calibration: Surface calibration framework
- analysis: Arbitrage detection systems
- visualization: 3D vol surface plotting
- backtesting: Strategy backtesting engine
"""

__version__ = "0.1.0"
__author__ = "Abhi Ekoirala"

from .data.polygon_client import (
    PolygonOptionsClient,
    OptionChainSnapshot,
    OptionContract,
    OptionType,
    create_synthetic_snapshot
)

from .models.heston_engine import (
    HestonEngine,
    HestonParameters,
    HestonPDESolver,
    HestonMonteCarlo
)

from .models.sabr_engine import (
    SABREngine,
    SABRParameters,
    SABRSurface,
    SABRCalibrator
)

from .calibration.calibrator import (
    HestonCalibrator,
    SABRSurfaceCalibrator,
    CalibrationResult,
    ModelType
)

from .analysis.arbitrage_detector import (
    StaticArbitrageDetector,
    DynamicArbitrageDetector,
    SmileArbitrageScanner,
    ArbitrageViolation,
    ArbitrageType,
    generate_arbitrage_report
)

from .visualization.plotter import (
    create_3d_vol_surface,
    create_vol_surface_from_snapshot,
    plot_market_vs_model_surface,
    plot_calibration_fit,
    plot_smile_by_expiry,
    plot_term_structure,
    plot_arbitrage_violations,
    create_calibration_report
)

from .backtesting.backtest_engine import (
    VolatilityBacktester,
    BacktestResult,
    TransactionCosts,
    generate_synthetic_backtest_data
)
