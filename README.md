# Stochastic Volatility Surface Calibration & Smile Arbitrage

A comprehensive Python toolkit for calibrating stochastic volatility models to option market data and detecting smile arbitrage opportunities.

**Built for quantitative finance applications:** Heston model calibration, SABR smile dynamics, no-arbitrage violation detection, and volatility trading strategy backtesting.

---

## Features

### Stochastic Volatility Models

| Model | Implementation |
|-------|----------------|
| **Heston (1993)** | Characteristic function pricing (Gatheral formulation), Carr-Madan FFT, finite difference PDE solver, QE Monte Carlo simulation |
| **SABR** | Hagan et al. (2002) approximation, Obloj (2008) correction, per-expiry slice calibration, risk-neutral density extraction |

### Calibration Framework

- **Optimization algorithms:** Nelder-Mead (derivative-free), L-BFGS-B (bounded), Differential Evolution (global)
- **Weighting schemes:** Vega-weighted (ATM emphasis), liquidity-weighted, custom weights
- **Regularization:** Feller condition penalty, parameter bounds, L2 regularization
- **Diagnostics:** RMSE, max error, parameter confidence intervals via bootstrap

### Arbitrage Detection

| Type | Detection Method |
|------|------------------|
| **Butterfly** | Convexity violation in strike space (d²C/dK² < 0) |
| **Calendar** | Total variance decreasing with maturity |
| **Call Spread** | Monotonicity violation (-e^(-qT) ≤ dC/dK ≤ 0) |
| **RV-IV Divergence** | Realized vol vs implied vol spread signals |
| **Smile Mispricing** | Model IV vs market IV divergence |

### Backtesting Engine

- RV-IV spread trading strategy (long/short straddles)
- Smile mispricing strategy (delta-hedged positions)
- Transaction cost modeling (commissions, bid-ask, market impact)
- Performance metrics: Sharpe ratio, max drawdown, profit factor, win rate

### Visualization

- Interactive 3D volatility surfaces (Plotly)
- Smile curves by expiry with model overlay
- ATM term structure plots
- Calibration diagnostic reports
- Arbitrage violation heatmaps

---

## Project Structure

```
vol_surface_engine/
├── main.py                      # Entry point (demo/live/interactive modes)
├── requirements.txt             # Python dependencies
├── .env.example                 # API key template
├── README.md                    # This file
│
├── src/
│   ├── __init__.py              # Package exports
│   │
│   ├── data/
│   │   └── polygon_client.py    # Polygon.io API client + synthetic data generator
│   │
│   ├── models/
│   │   ├── heston_engine.py     # Heston stochastic volatility model
│   │   └── sabr_engine.py       # SABR model implementation
│   │
│   ├── calibration/
│   │   └── calibrator.py        # Calibration framework (Heston + SABR)
│   │
│   ├── analysis/
│   │   └── arbitrage_detector.py # Arbitrage detection systems
│   │
│   ├── visualization/
│   │   └── plotter.py           # Plotly visualization functions
│   │
│   └── backtesting/
│       └── backtest_engine.py   # Strategy backtesting framework
│
└── output/                      # Generated HTML visualizations
    ├── vol_surface.html
    ├── smile_by_expiry.html
    ├── term_structure.html
    ├── calibration_report.html
    └── arbitrage_violations.html
```

---

## Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
# Clone or navigate to the project
cd ~/Desktop/vol_surface_engine

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### API Key (Optional - for live data)

To fetch live SPY/QQQ option data from Polygon.io:

1. Get a free API key at [polygon.io](https://polygon.io)
2. Copy `.env.example` to `.env`
3. Add your API key:

```bash
cp .env.example .env
# Edit .env and add: POLYGON_API_KEY=your_key_here
```

---

## Usage

### Demo Mode (Synthetic Data)

Run the full pipeline with synthetic option chain data:

```bash
python main.py --mode demo
```

This will:
1. Generate a synthetic vol surface with realistic skew/term structure
2. Calibrate Heston model parameters
3. Calibrate SABR surface (per-expiry)
4. Scan for arbitrage violations
5. Generate interactive HTML visualizations
6. Run a backtest of the RV-IV spread strategy

Output files are saved to `output/` directory.

### Live Mode (Real Market Data)

Fetch live option data and run analysis:

```bash
python main.py --mode live --ticker SPY
```

Requires `POLYGON_API_KEY` in `.env` file.

### Interactive Mode

Explore the engine interactively:

```bash
python main.py --mode interactive
```

Commands: `calibrate`, `surface`, `smile`, `arb`, `params`, `quit`

---

## Quick Start Example

```python
from src.data.polygon_client import create_synthetic_snapshot
from src.calibration.calibrator import HestonCalibrator
from src.visualization.plotter import create_vol_surface_from_snapshot, plot_smile_by_expiry

# Generate synthetic option chain
snapshot = create_synthetic_snapshot(
    spot=500.0,
    n_expiries=6,
    n_strikes=15,
    base_vol=0.22,
    skew=-0.15
)

# Calibrate Heston model
calibrator = HestonCalibrator(max_iterations=300)
result = calibrator.calibrate(snapshot)

print(result.summary())
# Calibration Result (heston)
# RMSE: 0.0452 (4.52%)
# Parameters:
#   v0: 0.0523 (vol: 22.87%)
#   kappa: 2.14
#   theta: 0.0498
#   sigma: 0.42
#   rho: -0.68

# Visualize
fig = plot_smile_by_expiry(snapshot, result.parameters, n_expiries=4)
fig.show()
```

---

## Model Details

### Heston Model

The Heston (1993) stochastic volatility model:

```
dS_t = μS_t dt + √v_t S_t dW^S_t
dv_t = κ(θ - v_t) dt + σ√v_t dW^v_t
⟨dW^S, dW^v⟩ = ρdt
```

**Parameters:**
- `v0`: Initial variance
- `κ` (kappa): Mean reversion speed
- `θ` (theta): Long-term variance
- `σ` (sigma): Volatility of volatility
- `ρ` (rho): Spot-vol correlation (typically negative for equities)

**Feller condition:** `2κθ > σ²` ensures variance stays positive.

### SABR Model

The SABR model for smile dynamics:

```
dF_t = σ_t F_t^β dW^F_t
dσ_t = ν σ_t dW^σ_t
⟨dW^F, dW^σ⟩ = ρdt
```

**Parameters:**
- `α` (alpha): Initial volatility level
- `β` (beta): CEV exponent (0=normal, 1=lognormal)
- `ρ` (rho): Forward-vol correlation
- `ν` (nu): Volatility of volatility

---

## Arbitrage Detection

### Static Arbitrage (No-Arbitrage Constraints)

| Constraint | Mathematical Form | Violation Signal |
|------------|-------------------|------------------|
| Butterfly | `d²C/dK² ≥ 0` | Convexity violation in call prices |
| Calendar | `σ²(T₂)T₂ ≥ σ²(T₁)T₁` for T₂ > T₁ | Total variance decreasing |
| Call Spread | `-e^(-qT) ≤ dC/dK ≤ 0` | Call prices not monotonic |

### Dynamic Arbitrage (Trading Signals)

- **RV-IV Spread:** When realized vol diverges from implied vol
  - RV > IV → Buy volatility (long straddle)
  - RV < IV → Sell volatility (short straddle)

- **Smile Mispricing:** When model IV ≠ market IV
  - Model IV > Market IV → Buy option
  - Model IV < Market IV → Sell option

---

## Performance

Typical calibration times (M1 MacBook Pro):

| Operation | Time |
|-----------|------|
| Heston calibration (300 iterations) | 60-120s |
| SABR surface (6 expiries) | 2-5s |
| Arbitrage scan | <1s |
| Backtest (1 year daily) | <1s |

---

## Dependencies

```
numpy>=1.24.0      # Numerical computing
scipy>=1.11.0      # Optimization, integration
pandas>=2.0.0      # Data manipulation
plotly>=5.18.0     # Interactive visualization
polygon-api-client>=1.12.0  # Options data API
python-dotenv>=1.0.0        # Environment variables
```

---

## References

1. Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"
2. Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide"
3. Hagan, P. S., et al. (2002). "Managing Smile Risk"
4. Obloj, J. (2008). "Fine-Tune Your Smile: Correction to Hagan et al."
5. Andersen, L. (2008). "Efficient Simulation of the Heston Stochastic Volatility Model"
6. Carr, P. & Madan, D. (1999). "Option Valuation Using the Fast Fourier Transform"

---

## License

MIT License

---

## Author

Abhie Koirala
