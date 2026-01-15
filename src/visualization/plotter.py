"""
Volatility Surface Visualization
================================

Creates interactive 3D visualizations using Plotly:
- Market implied volatility surface
- Calibrated model surface
- Arbitrage gap visualization
- Smile and term structure plots
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, List, Tuple, Any

from ..data.polygon_client import OptionChainSnapshot
from ..models.heston_engine import HestonEngine, HestonParameters
from ..models.sabr_engine import SABREngine, SABRParameters
from ..calibration.calibrator import CalibrationResult


def create_3d_vol_surface(
    strikes: np.ndarray,
    expiries: np.ndarray,
    ivs: np.ndarray,
    spot: float,
    title: str = "Implied Volatility Surface",
    use_moneyness: bool = True,
    colorscale: str = "Viridis"
) -> go.Figure:
    """
    Create interactive 3D volatility surface plot.

    Args:
        strikes: Array of strikes
        expiries: Array of expiries (years)
        ivs: 2D array of IVs, shape (len(strikes), len(expiries))
        spot: Spot price (for moneyness calculation)
        title: Plot title
        use_moneyness: If True, use K/S on x-axis
        colorscale: Plotly colorscale name

    Returns:
        Plotly Figure object
    """
    # Create meshgrid
    if use_moneyness:
        x_vals = strikes / spot
        x_label = "Moneyness (K/S)"
    else:
        x_vals = strikes
        x_label = "Strike"

    X, Y = np.meshgrid(x_vals, expiries)
    Z = ivs.T * 100  # Convert to percentage

    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale=colorscale,
            name="IV Surface",
            showscale=True,
            colorbar=dict(
                title="IV (%)",
                titleside="right"
            )
        )
    ])

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis=dict(title=x_label),
            yaxis=dict(title="Time to Maturity (years)"),
            zaxis=dict(title="Implied Volatility (%)"),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.8)
            )
        ),
        width=900,
        height=700,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig


def create_vol_surface_from_snapshot(
    snapshot: OptionChainSnapshot,
    title: str = "Market IV Surface"
) -> go.Figure:
    """
    Create 3D vol surface directly from option chain snapshot.

    Args:
        snapshot: Option chain data
        title: Plot title

    Returns:
        Plotly Figure
    """
    spot = snapshot.spot_price
    contracts = [c for c in snapshot.contracts if c.implied_volatility]

    if not contracts:
        fig = go.Figure()
        fig.add_annotation(text="No IV data available", x=0.5, y=0.5, showarrow=False)
        return fig

    # Extract unique strikes and expiries
    strikes = np.array(sorted(set(c.strike for c in contracts)))
    expiries = np.array(sorted(set(c.time_to_expiry() for c in contracts)))

    # Build IV grid
    iv_grid = np.full((len(strikes), len(expiries)), np.nan)

    for c in contracts:
        k_idx = np.where(strikes == c.strike)[0]
        t_idx = np.where(np.isclose(expiries, c.time_to_expiry(), rtol=0.01))[0]

        if len(k_idx) > 0 and len(t_idx) > 0:
            iv_grid[k_idx[0], t_idx[0]] = c.implied_volatility

    return create_3d_vol_surface(strikes, expiries, iv_grid, spot, title)


def plot_market_vs_model_surface(
    snapshot: OptionChainSnapshot,
    model_ivs: np.ndarray,
    title: str = "Market vs Calibrated Model"
) -> go.Figure:
    """
    Create side-by-side comparison of market and model surfaces.

    Args:
        snapshot: Option chain data
        model_ivs: Calibrated model IVs
        title: Plot title

    Returns:
        Plotly Figure with two subplots
    """
    # Extract data
    spot = snapshot.spot_price
    contracts = snapshot.contracts

    # Group by strike and expiry
    strikes = np.array(sorted(set(c.strike for c in contracts)))
    expiries = np.array(sorted(set(c.time_to_expiry() for c in contracts)))

    # Build grids
    market_grid = np.full((len(strikes), len(expiries)), np.nan)
    model_grid = np.full((len(strikes), len(expiries)), np.nan)

    for i, contract in enumerate(contracts):
        k_idx = np.where(strikes == contract.strike)[0]
        t_idx = np.where(np.isclose(expiries, contract.time_to_expiry()))[0]

        if len(k_idx) > 0 and len(t_idx) > 0:
            market_iv = contract.implied_volatility if contract.implied_volatility else 0.2
            market_grid[k_idx[0], t_idx[0]] = market_iv * 100
            if i < len(model_ivs):
                model_grid[k_idx[0], t_idx[0]] = model_ivs[i] * 100

    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Market IV Surface", "Calibrated Model Surface"),
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        horizontal_spacing=0.05
    )

    moneyness = strikes / spot
    X, Y = np.meshgrid(moneyness, expiries)

    # Market surface
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=market_grid.T,
            colorscale="Blues",
            name="Market",
            showscale=False
        ),
        row=1, col=1
    )

    # Model surface
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=model_grid.T,
            colorscale="Reds",
            name="Model",
            showscale=True,
            colorbar=dict(title="IV (%)", x=1.02)
        ),
        row=1, col=2
    )

    # Update layout
    scene_config = dict(
        xaxis=dict(title="Moneyness (K/S)"),
        yaxis=dict(title="Time to Maturity"),
        zaxis=dict(title="IV (%)"),
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=scene_config,
        scene2=scene_config,
        width=1200,
        height=600,
        margin=dict(l=50, r=100, t=80, b=50)
    )

    return fig


def plot_calibration_fit(
    snapshot: OptionChainSnapshot,
    result: CalibrationResult,
    expiry_filter: Optional[float] = None
) -> go.Figure:
    """
    Plot market data points vs calibrated model curve.

    Args:
        snapshot: Option chain data
        result: Calibration result
        expiry_filter: If provided, only plot this expiry

    Returns:
        Plotly Figure
    """
    spot = snapshot.spot_price
    contracts = snapshot.contracts

    # Extract diagnostics
    model_ivs = result.diagnostics.get('model_ivs', np.array([]))
    market_ivs = result.diagnostics.get('market_ivs', np.array([]))

    if len(model_ivs) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No IV data in calibration result", x=0.5, y=0.5, showarrow=False)
        return fig

    # Get contracts from diagnostics
    diag_contracts = result.diagnostics.get('contracts', contracts)

    # Filter by expiry if specified
    if expiry_filter:
        mask = np.array([
            np.isclose(c.time_to_expiry(), expiry_filter, rtol=0.01)
            for c in diag_contracts
        ])
    else:
        mask = np.ones(len(diag_contracts), dtype=bool)

    strikes = np.array([c.strike for c in diag_contracts])[mask]
    market_iv = market_ivs[mask] * 100
    model_iv = model_ivs[mask] * 100
    moneyness = strikes / spot

    fig = go.Figure()

    # Market data points
    fig.add_trace(go.Scatter(
        x=moneyness,
        y=market_iv,
        mode='markers',
        name='Market IV',
        marker=dict(size=10, color='blue', symbol='circle')
    ))

    # Model curve
    sort_idx = np.argsort(moneyness)
    fig.add_trace(go.Scatter(
        x=moneyness[sort_idx],
        y=model_iv[sort_idx],
        mode='lines+markers',
        name='Model IV',
        line=dict(color='red', width=2),
        marker=dict(size=6, color='red', symbol='x')
    ))

    # Add error bars
    errors = model_iv - market_iv
    fig.add_trace(go.Bar(
        x=moneyness,
        y=errors,
        name='Error (Model - Market)',
        marker_color='rgba(128, 128, 128, 0.5)',
        yaxis='y2'
    ))

    fig.update_layout(
        title=dict(
            text=f"Calibration Fit (RMSE: {result.rmse*100:.2f}%, Max Error: {result.max_error*100:.2f}%)",
            x=0.5
        ),
        xaxis=dict(title="Moneyness (K/S)"),
        yaxis=dict(title="Implied Volatility (%)", side='left'),
        yaxis2=dict(
            title="Error (%)",
            side='right',
            overlaying='y',
            showgrid=False
        ),
        legend=dict(x=0.02, y=0.98),
        width=900,
        height=500
    )

    return fig


def plot_smile_by_expiry(
    snapshot: OptionChainSnapshot,
    model_params: Optional[HestonParameters] = None,
    n_expiries: int = 4
) -> go.Figure:
    """
    Plot volatility smiles for multiple expiries.

    Args:
        snapshot: Option chain data
        model_params: If provided, overlay Heston model smile
        n_expiries: Number of expiries to show

    Returns:
        Plotly Figure
    """
    spot = snapshot.spot_price
    r = snapshot.risk_free_rate
    q = snapshot.dividend_yield

    # Get unique expiries
    all_expiries = sorted(set(c.time_to_expiry() for c in snapshot.contracts))

    # Select subset of expiries
    if len(all_expiries) > n_expiries:
        indices = np.linspace(0, len(all_expiries)-1, n_expiries, dtype=int)
        selected_expiries = [all_expiries[i] for i in indices]
    else:
        selected_expiries = all_expiries

    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    engine = HestonEngine() if model_params else None

    for idx, expiry in enumerate(selected_expiries):
        # Get contracts for this expiry
        expiry_contracts = [
            c for c in snapshot.contracts
            if np.isclose(c.time_to_expiry(), expiry, rtol=0.01)
            and c.implied_volatility
        ]

        if not expiry_contracts:
            continue

        strikes = np.array([c.strike for c in expiry_contracts])
        market_ivs = np.array([c.implied_volatility * 100 for c in expiry_contracts])
        moneyness = strikes / spot

        # Sort by moneyness
        sort_idx = np.argsort(moneyness)

        # Market smile
        fig.add_trace(go.Scatter(
            x=moneyness[sort_idx],
            y=market_ivs[sort_idx],
            mode='markers',
            name=f'Market T={expiry:.2f}y',
            marker=dict(size=8, color=colors[idx % len(colors)])
        ))

        # Model smile (if parameters provided)
        if engine and model_params:
            model_ivs = np.array([
                engine.implied_volatility(spot, k, expiry, r, q, model_params) * 100
                for k in strikes
            ])

            fig.add_trace(go.Scatter(
                x=moneyness[sort_idx],
                y=model_ivs[sort_idx],
                mode='lines',
                name=f'Model T={expiry:.2f}y',
                line=dict(color=colors[idx % len(colors)], dash='dash')
            ))

    fig.update_layout(
        title=dict(text="Volatility Smiles by Expiry", x=0.5),
        xaxis=dict(title="Moneyness (K/S)"),
        yaxis=dict(title="Implied Volatility (%)"),
        legend=dict(x=1.02, y=1, xanchor='left'),
        width=900,
        height=500,
        margin=dict(r=200)
    )

    return fig


def plot_term_structure(
    snapshot: OptionChainSnapshot,
    model_params: Optional[HestonParameters] = None
) -> go.Figure:
    """
    Plot ATM implied volatility term structure.

    Args:
        snapshot: Option chain data
        model_params: If provided, overlay model term structure

    Returns:
        Plotly Figure
    """
    spot = snapshot.spot_price
    r = snapshot.risk_free_rate
    q = snapshot.dividend_yield

    # Find ATM options for each expiry
    expiries = []
    atm_ivs = []

    for expiry_date in sorted(snapshot.expiries):
        expiry_contracts = [
            c for c in snapshot.contracts
            if c.expiry == expiry_date and c.implied_volatility
        ]

        if not expiry_contracts:
            continue

        # Find closest to ATM
        strikes = np.array([c.strike for c in expiry_contracts])
        atm_idx = np.argmin(np.abs(strikes - spot))

        T = expiry_contracts[atm_idx].time_to_expiry()
        iv = expiry_contracts[atm_idx].implied_volatility

        if iv and T > 0:
            expiries.append(T)
            atm_ivs.append(iv * 100)

    fig = go.Figure()

    # Market term structure
    fig.add_trace(go.Scatter(
        x=expiries,
        y=atm_ivs,
        mode='markers+lines',
        name='Market ATM IV',
        marker=dict(size=10, color='blue'),
        line=dict(width=2)
    ))

    # Model term structure (if parameters provided)
    if model_params:
        engine = HestonEngine()
        model_ivs = [
            engine.implied_volatility(spot, spot, T, r, q, model_params) * 100
            for T in expiries
        ]

        fig.add_trace(go.Scatter(
            x=expiries,
            y=model_ivs,
            mode='lines',
            name='Heston Model',
            line=dict(color='red', dash='dash', width=2)
        ))

    fig.update_layout(
        title=dict(text="ATM Volatility Term Structure", x=0.5),
        xaxis=dict(title="Time to Maturity (years)"),
        yaxis=dict(title="ATM Implied Volatility (%)"),
        width=800,
        height=500
    )

    return fig


def plot_arbitrage_violations(
    violations: List[Any],
    title: str = "Detected Arbitrage Violations"
) -> go.Figure:
    """
    Visualize detected arbitrage violations.

    Args:
        violations: List of ArbitrageViolation objects
        title: Plot title

    Returns:
        Plotly Figure
    """
    if not violations:
        fig = go.Figure()
        fig.add_annotation(
            text="No arbitrage violations detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title=title, width=600, height=400)
        return fig

    # Categorize violations
    butterfly = [v for v in violations if v.arb_type.value == "butterfly"]
    calendar = [v for v in violations if v.arb_type.value == "calendar"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Butterfly Violations ({len(butterfly)})",
            f"Calendar Violations ({len(calendar)})"
        )
    )

    # Butterfly violations
    if butterfly:
        strikes = [v.strike2 for v in butterfly]
        expiries = [v.expiry1 for v in butterfly]
        magnitudes = [v.violation_magnitude * 100 for v in butterfly]

        fig.add_trace(go.Scatter(
            x=strikes,
            y=magnitudes,
            mode='markers',
            name='Butterfly',
            marker=dict(size=10, color='red', symbol='diamond'),
            text=[f"T={t:.2f}" for t in expiries],
            hovertemplate="K=%{x:.0f}<br>Violation=%{y:.3f}%<br>%{text}"
        ), row=1, col=1)

    # Calendar violations
    if calendar:
        expiries1 = [v.expiry1 for v in calendar]
        magnitudes = [v.violation_magnitude * 100 for v in calendar]
        strikes = [v.strike1 for v in calendar]

        fig.add_trace(go.Scatter(
            x=expiries1,
            y=magnitudes,
            mode='markers',
            name='Calendar',
            marker=dict(size=10, color='orange', symbol='triangle-up'),
            text=[f"K={k:.0f}" for k in strikes],
            hovertemplate="T=%{x:.3f}<br>Violation=%{y:.4f}<br>%{text}"
        ), row=1, col=2)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        width=1000,
        height=400
    )
    fig.update_xaxes(title_text="Strike", row=1, col=1)
    fig.update_xaxes(title_text="Expiry (years)", row=1, col=2)
    fig.update_yaxes(title_text="Violation Magnitude", row=1, col=1)
    fig.update_yaxes(title_text="Variance Decrease", row=1, col=2)

    return fig


def create_calibration_report(
    snapshot: OptionChainSnapshot,
    result: CalibrationResult
) -> go.Figure:
    """
    Create comprehensive calibration report with multiple plots.

    Args:
        snapshot: Option chain data
        result: Calibration result

    Returns:
        Plotly Figure with multiple subplots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Market vs Model IV",
            "Error Distribution",
            "Smile Fit (Nearest Expiry)",
            "Parameter Summary"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "table"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # Extract data
    model_ivs = result.diagnostics.get('model_ivs', np.array([]))
    market_ivs = result.diagnostics.get('market_ivs', np.array([]))
    iv_errors = result.diagnostics.get('iv_errors', np.array([]))
    contracts = result.diagnostics.get('contracts', snapshot.contracts)

    if len(model_ivs) == 0:
        return fig

    model_ivs = model_ivs * 100
    market_ivs = market_ivs * 100
    iv_errors = iv_errors * 100

    moneyness = np.array([c.strike / snapshot.spot_price for c in contracts])

    # Plot 1: Market vs Model scatter
    fig.add_trace(go.Scatter(
        x=market_ivs,
        y=model_ivs,
        mode='markers',
        name='Market vs Model',
        marker=dict(size=6, color='blue', opacity=0.6)
    ), row=1, col=1)

    # 45-degree line
    iv_min = min(market_ivs.min(), model_ivs.min())
    iv_max = max(market_ivs.max(), model_ivs.max())
    fig.add_trace(go.Scatter(
        x=[iv_min, iv_max], y=[iv_min, iv_max],
        mode='lines',
        name='Perfect Fit',
        line=dict(color='red', dash='dash')
    ), row=1, col=1)

    # Plot 2: Error histogram
    fig.add_trace(go.Histogram(
        x=iv_errors,
        nbinsx=30,
        name='IV Errors',
        marker_color='green'
    ), row=1, col=2)

    # Plot 3: Smile for first expiry
    expiries = np.array([c.time_to_expiry() for c in contracts])
    first_expiry = np.min(expiries)
    mask = np.abs(expiries - first_expiry) < 0.01

    exp_moneyness = moneyness[mask]
    sort_idx = np.argsort(exp_moneyness)

    fig.add_trace(go.Scatter(
        x=exp_moneyness[sort_idx],
        y=market_ivs[mask][sort_idx],
        mode='markers',
        name='Market',
        marker=dict(size=8, color='blue')
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=exp_moneyness[sort_idx],
        y=model_ivs[mask][sort_idx],
        mode='lines',
        name='Model',
        line=dict(color='red', width=2)
    ), row=2, col=1)

    # Plot 4: Parameter table
    params = result.parameters
    if isinstance(params, HestonParameters):
        param_names = ['v0', 'kappa', 'theta', 'sigma', 'rho', 'Feller']
        param_values = [
            f"{params.v0:.6f}",
            f"{params.kappa:.4f}",
            f"{params.theta:.6f}",
            f"{params.sigma:.4f}",
            f"{params.rho:.4f}",
            "Yes" if params.feller_condition else "No"
        ]
    elif isinstance(params, SABRParameters):
        param_names = ['alpha', 'beta', 'rho', 'nu']
        param_values = [
            f"{params.alpha:.6f}",
            f"{params.beta:.4f}",
            f"{params.rho:.4f}",
            f"{params.nu:.4f}"
        ]
    else:
        param_names = ['RMSE', 'Max Error', 'Time']
        param_values = [f"{result.rmse:.4f}", f"{result.max_error:.4f}", f"{result.elapsed_time:.1f}s"]

    fig.add_trace(go.Table(
        header=dict(values=['Parameter', 'Value'], fill_color='lightgray'),
        cells=dict(values=[param_names, param_values])
    ), row=2, col=2)

    fig.update_layout(
        title=dict(
            text=f"Calibration Report | RMSE: {result.rmse*100:.2f}% | Time: {result.elapsed_time:.1f}s",
            x=0.5
        ),
        width=1200,
        height=800,
        showlegend=False
    )

    fig.update_xaxes(title_text="Market IV (%)", row=1, col=1)
    fig.update_yaxes(title_text="Model IV (%)", row=1, col=1)
    fig.update_xaxes(title_text="Error (%)", row=1, col=2)
    fig.update_xaxes(title_text="Moneyness (K/S)", row=2, col=1)
    fig.update_yaxes(title_text="IV (%)", row=2, col=1)

    return fig


def plot_rv_iv_comparison(
    dates: np.ndarray,
    realized_vol: np.ndarray,
    implied_vol: np.ndarray,
    title: str = "Realized vs Implied Volatility"
) -> go.Figure:
    """
    Plot time series of realized vs implied volatility.

    Args:
        dates: Array of dates
        realized_vol: Realized volatility time series
        implied_vol: Implied volatility time series
        title: Plot title

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=realized_vol * 100,
        mode='lines',
        name='Realized Vol',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=implied_vol * 100,
        mode='lines',
        name='Implied Vol',
        line=dict(color='red', width=2)
    ))

    # Shade the spread
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates, dates[::-1]]),
        y=np.concatenate([realized_vol * 100, (implied_vol * 100)[::-1]]),
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='RV-IV Spread',
        showlegend=True
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Volatility (%)"),
        width=1000,
        height=500,
        legend=dict(x=0.02, y=0.98)
    )

    return fig


def plot_pnl_attribution(
    dates: np.ndarray,
    delta_pnl: np.ndarray,
    gamma_pnl: np.ndarray,
    vega_pnl: np.ndarray,
    theta_pnl: np.ndarray,
    title: str = "PnL Attribution"
) -> go.Figure:
    """
    Plot stacked PnL attribution by Greek.

    Args:
        dates: Array of dates
        delta_pnl: Delta PnL contribution
        gamma_pnl: Gamma PnL contribution
        vega_pnl: Vega PnL contribution
        theta_pnl: Theta PnL contribution
        title: Plot title

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=dates, y=delta_pnl,
        name='Delta', marker_color='blue'
    ))
    fig.add_trace(go.Bar(
        x=dates, y=gamma_pnl,
        name='Gamma', marker_color='green'
    ))
    fig.add_trace(go.Bar(
        x=dates, y=vega_pnl,
        name='Vega', marker_color='red'
    ))
    fig.add_trace(go.Bar(
        x=dates, y=theta_pnl,
        name='Theta', marker_color='orange'
    ))

    fig.update_layout(
        barmode='relative',
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Date"),
        yaxis=dict(title="PnL ($)"),
        width=1000,
        height=500
    )

    return fig
