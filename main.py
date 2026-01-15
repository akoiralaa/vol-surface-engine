#!/usr/bin/env python3
"""
Volatility Surface Engine - Main Demo
======================================

Demonstrates the full pipeline:
1. Fetch/generate option chain data
2. Calibrate Heston and SABR models
3. Detect arbitrage opportunities
4. Visualize vol surfaces
5. Backtest vol arbitrage strategy

Usage:
    python main.py --mode demo    # Run with synthetic data
    python main.py --mode live    # Run with live SPY data (requires API key)
"""

import argparse
import numpy as np
from datetime import date, timedelta

# Local imports
from src.data.polygon_client import (
    PolygonOptionsClient,
    create_synthetic_snapshot
)
from src.models.heston_engine import HestonEngine, HestonParameters
from src.models.sabr_engine import SABREngine, SABRParameters
from src.calibration.calibrator import HestonCalibrator, SABRSurfaceCalibrator
from src.analysis.arbitrage_detector import (
    StaticArbitrageDetector,
    generate_arbitrage_report
)
from src.visualization.plotter import (
    create_vol_surface_from_snapshot,
    plot_smile_by_expiry,
    plot_term_structure,
    plot_calibration_fit,
    create_calibration_report,
    plot_arbitrage_violations
)
from src.backtesting.backtest_engine import (
    VolatilityBacktester,
    generate_synthetic_backtest_data
)


def run_demo_mode():
    """Run demonstration with synthetic data."""
    print("=" * 60)
    print("VOLATILITY SURFACE ENGINE - DEMO MODE")
    print("=" * 60)

    # 1. Generate synthetic option chain
    print("\n[1/6] Generating synthetic option chain...")
    snapshot = create_synthetic_snapshot(
        spot=500.0,
        n_expiries=6,
        n_strikes=15,
        base_vol=0.22,
        skew=-0.15,
        term_slope=0.03
    )
    print(f"     Generated {len(snapshot.contracts)} contracts")
    print(f"     Spot: ${snapshot.spot_price:.2f}")
    print(f"     Expiries: {len(snapshot.expiries)}")

    # 2. Calibrate Heston model
    print("\n[2/6] Calibrating Heston model...")
    heston_calibrator = HestonCalibrator(max_iterations=300)
    heston_result = heston_calibrator.calibrate(snapshot)

    print(f"\n{heston_result.summary()}")

    # 3. Calibrate SABR surface
    print("\n[3/6] Calibrating SABR surface (per-expiry)...")
    sabr_calibrator = SABRSurfaceCalibrator(beta=0.5)
    sabr_result = sabr_calibrator.calibrate(snapshot)

    print(f"     SABR RMSE: {sabr_result.rmse*100:.2f}%")
    print(f"     Calibrated {len(sabr_result.parameters.parameters)} expiry slices")

    # 4. Detect arbitrage violations
    print("\n[4/6] Scanning for arbitrage violations...")
    arb_detector = StaticArbitrageDetector()
    violations = arb_detector.detect_all(snapshot)

    print(f"     Found {len(violations)} potential violations:")
    butterfly = [v for v in violations if v.arb_type.value == "butterfly"]
    calendar = [v for v in violations if v.arb_type.value == "calendar"]
    print(f"       - Butterfly: {len(butterfly)}")
    print(f"       - Calendar: {len(calendar)}")

    if violations:
        print("\n     Top 3 arbitrage opportunities:")
        for v in violations[:3]:
            print(f"       {v.description[:70]}...")

    # 5. Generate visualizations
    print("\n[5/6] Generating visualizations...")

    # Vol surface
    fig_surface = create_vol_surface_from_snapshot(snapshot, "Market IV Surface")
    fig_surface.write_html("output/vol_surface.html")
    print("     Saved: output/vol_surface.html")

    # Smile by expiry
    fig_smile = plot_smile_by_expiry(snapshot, heston_result.parameters, n_expiries=4)
    fig_smile.write_html("output/smile_by_expiry.html")
    print("     Saved: output/smile_by_expiry.html")

    # Term structure
    fig_term = plot_term_structure(snapshot, heston_result.parameters)
    fig_term.write_html("output/term_structure.html")
    print("     Saved: output/term_structure.html")

    # Calibration report
    fig_report = create_calibration_report(snapshot, heston_result)
    fig_report.write_html("output/calibration_report.html")
    print("     Saved: output/calibration_report.html")

    # Arbitrage plot
    if violations:
        fig_arb = plot_arbitrage_violations(violations)
        fig_arb.write_html("output/arbitrage_violations.html")
        print("     Saved: output/arbitrage_violations.html")

    # 6. Run backtest
    print("\n[6/6] Running RV-IV spread backtest...")

    price_data, iv_data = generate_synthetic_backtest_data(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        initial_spot=500,
        base_vol=0.20
    )

    backtester = VolatilityBacktester()
    backtest_result = backtester.run_rv_iv_strategy(
        price_data,
        iv_data,
        rv_lookback=21,
        iv_threshold=0.02,
        hold_days=5
    )

    print(f"\n{backtest_result.summary()}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nGenerated files in output/:")
    print("  - vol_surface.html")
    print("  - smile_by_expiry.html")
    print("  - term_structure.html")
    print("  - calibration_report.html")
    print("  - arbitrage_violations.html")

    return heston_result, sabr_result, violations, backtest_result


def run_live_mode(ticker: str = "SPY"):
    """Run with live market data from Polygon.io."""
    print("=" * 60)
    print(f"VOLATILITY SURFACE ENGINE - LIVE MODE ({ticker})")
    print("=" * 60)

    try:
        # Initialize Polygon client
        print("\n[1/5] Fetching live option chain...")
        client = PolygonOptionsClient()
        snapshot = client.get_option_snapshot(
            ticker,
            min_tte=0.02,
            max_tte=0.25,
            moneyness_range=(0.90, 1.10)
        )

        print(f"     Fetched {len(snapshot.contracts)} contracts")
        print(f"     Spot: ${snapshot.spot_price:.2f}")

        # Calibrate Heston
        print("\n[2/5] Calibrating Heston model...")
        heston_calibrator = HestonCalibrator()
        heston_result = heston_calibrator.calibrate(snapshot)
        print(f"\n{heston_result.summary()}")

        # Calibrate SABR
        print("\n[3/5] Calibrating SABR surface...")
        sabr_calibrator = SABRSurfaceCalibrator()
        sabr_result = sabr_calibrator.calibrate(snapshot)
        print(f"     SABR RMSE: {sabr_result.rmse*100:.2f}%")

        # Detect arbitrage
        print("\n[4/5] Scanning for arbitrage...")
        report = generate_arbitrage_report(snapshot, heston_result.parameters)
        n_static = len(report['static_arbitrage'])
        print(f"     Found {n_static} static arbitrage violations")

        if report['smile_signals']:
            print(f"\n     Top smile mispricing signals:")
            for sig in report['smile_signals'][:5]:
                print(f"       {sig['direction']} K={sig['strike']:.0f}: "
                      f"mkt={sig['market_iv']:.1%} vs model={sig['model_iv']:.1%}")

        # Generate visualizations
        print("\n[5/5] Generating visualizations...")
        fig = create_vol_surface_from_snapshot(snapshot, f"{ticker} Live IV Surface")
        fig.write_html(f"output/{ticker.lower()}_live_surface.html")
        print(f"     Saved: output/{ticker.lower()}_live_surface.html")

        return heston_result, report

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure POLYGON_API_KEY is set in your .env file")
        print("Get a free API key at https://polygon.io")
        return None, None


def interactive_mode():
    """Interactive exploration mode."""
    print("=" * 60)
    print("VOLATILITY SURFACE ENGINE - INTERACTIVE MODE")
    print("=" * 60)

    # Create synthetic data
    snapshot = create_synthetic_snapshot(spot=500.0)

    print("\nCommands:")
    print("  calibrate  - Calibrate Heston model")
    print("  surface    - Show vol surface")
    print("  smile      - Show vol smiles")
    print("  arb        - Scan for arbitrage")
    print("  params     - Show current parameters")
    print("  quit       - Exit")

    heston_params = None

    while True:
        cmd = input("\n> ").strip().lower()

        if cmd == "quit":
            break
        elif cmd == "calibrate":
            calibrator = HestonCalibrator()
            result = calibrator.calibrate(snapshot)
            heston_params = result.parameters
            print(result.summary())
        elif cmd == "surface":
            fig = create_vol_surface_from_snapshot(snapshot)
            fig.show()
        elif cmd == "smile":
            fig = plot_smile_by_expiry(snapshot, heston_params)
            fig.show()
        elif cmd == "arb":
            detector = StaticArbitrageDetector()
            violations = detector.detect_all(snapshot)
            print(f"Found {len(violations)} violations")
            for v in violations[:5]:
                print(f"  {v.description}")
        elif cmd == "params":
            if heston_params:
                print(f"v0={heston_params.v0:.4f}, kappa={heston_params.kappa:.2f}, "
                      f"theta={heston_params.theta:.4f}, sigma={heston_params.sigma:.2f}, "
                      f"rho={heston_params.rho:.2f}")
            else:
                print("No parameters calibrated yet. Run 'calibrate' first.")
        else:
            print(f"Unknown command: {cmd}")


def main():
    parser = argparse.ArgumentParser(
        description="Volatility Surface Calibration & Smile Arbitrage Engine"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "live", "interactive"],
        default="demo",
        help="Run mode: demo (synthetic data), live (Polygon API), interactive"
    )
    parser.add_argument(
        "--ticker",
        default="SPY",
        help="Ticker symbol for live mode (default: SPY)"
    )

    args = parser.parse_args()

    # Create output directory
    import os
    os.makedirs("output", exist_ok=True)

    if args.mode == "demo":
        run_demo_mode()
    elif args.mode == "live":
        run_live_mode(args.ticker)
    elif args.mode == "interactive":
        interactive_mode()


if __name__ == "__main__":
    main()
