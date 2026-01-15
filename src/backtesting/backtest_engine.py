"""
Volatility Arbitrage Backtesting Engine
========================================

Backtest volatility trading strategies:
- Realized vs Implied vol spread trading
- Smile mispricing strategies
- Delta-hedged volatility positions

Features:
- Transaction cost modeling
- Greeks-based hedging simulation
- Performance analytics (Sharpe, drawdown, etc.)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
from datetime import date, datetime, timedelta
from scipy.stats import norm

from ..models.heston_engine import HestonEngine, HestonParameters


class PositionType(Enum):
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"
    BUTTERFLY = "butterfly"
    CALENDAR = "calendar"
    DELTA_HEDGED_CALL = "delta_hedged_call"
    DELTA_HEDGED_PUT = "delta_hedged_put"


@dataclass
class TransactionCosts:
    """Transaction cost model."""
    option_commission: float = 0.65  # Per contract
    stock_commission: float = 0.005  # Per share
    bid_ask_cost: float = 0.02  # Fraction of mid (one-way)
    market_impact: float = 0.001  # Price impact for large orders

    def option_cost(self, n_contracts: int, mid_price: float) -> float:
        """Total cost to trade n option contracts."""
        return (
            n_contracts * self.option_commission +
            n_contracts * 100 * mid_price * self.bid_ask_cost
        )

    def stock_cost(self, n_shares: int, price: float) -> float:
        """Total cost to trade n shares."""
        return (
            abs(n_shares) * self.stock_commission +
            abs(n_shares) * price * self.bid_ask_cost * 0.1  # Tighter spread on stock
        )


@dataclass
class Position:
    """Single position in backtest."""
    position_type: PositionType
    entry_date: date
    expiry_date: date
    strike: float
    strike2: Optional[float] = None  # For strangles, spreads
    quantity: int = 1
    entry_price: float = 0.0
    entry_iv: float = 0.0
    delta_hedge_ratio: float = 0.0
    stock_position: int = 0


@dataclass
class TradeResult:
    """Result of a single trade."""
    entry_date: date
    exit_date: date
    position_type: PositionType
    strike: float
    entry_iv: float
    exit_iv: float
    realized_vol: float
    gross_pnl: float
    transaction_costs: float
    net_pnl: float
    holding_period: int
    max_drawdown: float


@dataclass
class BacktestResult:
    """Complete backtest results."""
    trades: List[TradeResult]
    daily_pnl: pd.Series
    cumulative_pnl: pd.Series
    metrics: Dict[str, float]

    def summary(self) -> str:
        """Generate summary string."""
        m = self.metrics
        lines = [
            "Backtest Results",
            "=" * 50,
            f"Total Trades: {len(self.trades)}",
            f"Win Rate: {m.get('win_rate', 0)*100:.1f}%",
            f"Total PnL: ${m.get('total_pnl', 0):,.2f}",
            f"Sharpe Ratio: {m.get('sharpe_ratio', 0):.2f}",
            f"Max Drawdown: {m.get('max_drawdown', 0)*100:.1f}%",
            f"Avg Trade PnL: ${m.get('avg_trade_pnl', 0):,.2f}",
            f"Profit Factor: {m.get('profit_factor', 0):.2f}",
        ]
        return "\n".join(lines)


class BlackScholesGreeks:
    """Black-Scholes Greeks calculator."""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        return BlackScholesGreeks.d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        if T <= 0:
            return max(S - K, 0)
        d1 = BlackScholesGreeks.d1(S, K, T, r, q, sigma)
        d2 = BlackScholesGreeks.d2(S, K, T, r, q, sigma)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        if T <= 0:
            return max(K - S, 0)
        d1 = BlackScholesGreeks.d1(S, K, T, r, q, sigma)
        d2 = BlackScholesGreeks.d2(S, K, T, r, q, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool) -> float:
        if T <= 0:
            if is_call:
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        d1 = BlackScholesGreeks.d1(S, K, T, r, q, sigma)
        if is_call:
            return np.exp(-q * T) * norm.cdf(d1)
        else:
            return np.exp(-q * T) * (norm.cdf(d1) - 1)

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0
        d1 = BlackScholesGreeks.d1(S, K, T, r, q, sigma)
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        if T <= 0:
            return 0
        d1 = BlackScholesGreeks.d1(S, K, T, r, q, sigma)
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change

    @staticmethod
    def theta(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool) -> float:
        if T <= 0:
            return 0
        d1 = BlackScholesGreeks.d1(S, K, T, r, q, sigma)
        d2 = BlackScholesGreeks.d2(S, K, T, r, q, sigma)

        term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))

        if is_call:
            term2 = q * S * np.exp(-q * T) * norm.cdf(d1)
            term3 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
            term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)

        return (term1 + term2 + term3) / 365  # Daily theta


class VolatilityBacktester:
    """
    Backtest volatility trading strategies.
    """

    def __init__(
        self,
        transaction_costs: Optional[TransactionCosts] = None,
        hedge_frequency: str = "daily",  # 'daily', 'weekly', 'none'
        initial_capital: float = 100000
    ):
        """
        Initialize backtester.

        Args:
            transaction_costs: Cost model
            hedge_frequency: How often to delta hedge
            initial_capital: Starting capital
        """
        self.costs = transaction_costs or TransactionCosts()
        self.hedge_frequency = hedge_frequency
        self.initial_capital = initial_capital
        self.bs = BlackScholesGreeks()

    def run_rv_iv_strategy(
        self,
        price_data: pd.DataFrame,
        iv_data: pd.DataFrame,
        rv_lookback: int = 21,
        iv_threshold: float = 0.02,
        hold_days: int = 5,
        position_size: float = 0.02  # Fraction of capital per trade
    ) -> BacktestResult:
        """
        Backtest realized vol vs implied vol spread strategy.

        Strategy:
        - When RV < IV by threshold: Sell straddle (short vol)
        - When RV > IV by threshold: Buy straddle (long vol)

        Args:
            price_data: DataFrame with 'date', 'open', 'high', 'low', 'close'
            iv_data: DataFrame with 'date', 'atm_iv'
            rv_lookback: Days for RV calculation
            iv_threshold: Min RV-IV gap to enter
            hold_days: Days to hold position
            position_size: Fraction of capital per trade

        Returns:
            BacktestResult
        """
        # Merge data
        data = pd.merge(price_data, iv_data, on='date', how='inner')
        data = data.sort_values('date').reset_index(drop=True)

        # Calculate realized vol
        data['log_return'] = np.log(data['close'] / data['close'].shift(1))
        data['rv'] = data['log_return'].rolling(rv_lookback).std() * np.sqrt(252)

        # Drop NaN
        data = data.dropna().reset_index(drop=True)

        trades = []
        daily_pnls = []
        in_position = False
        position_entry = None
        position_type = None
        entry_price = None
        entry_spot = None

        r = 0.05
        q = 0.01

        for i in range(len(data)):
            row = data.iloc[i]
            current_date = row['date']
            spot = row['close']
            iv = row['atm_iv']
            rv = row['rv']

            # Daily PnL tracking
            daily_pnl = 0

            if in_position:
                # Check if should exit
                days_held = (current_date - position_entry['date']).days

                if days_held >= hold_days:
                    # Exit position
                    T = (30 - days_held) / 365  # Approximate remaining time
                    strike = position_entry['strike']

                    exit_call = self.bs.call_price(spot, strike, T, r, q, iv)
                    exit_put = self.bs.put_price(spot, strike, T, r, q, iv)
                    exit_straddle = exit_call + exit_put

                    if position_type == PositionType.SHORT_STRADDLE:
                        gross_pnl = entry_price - exit_straddle
                    else:
                        gross_pnl = exit_straddle - entry_price

                    n_contracts = int(self.initial_capital * position_size / (entry_price * 100))
                    gross_pnl *= n_contracts * 100

                    # Transaction costs
                    tx_cost = self.costs.option_cost(n_contracts * 2, exit_straddle)

                    net_pnl = gross_pnl - tx_cost

                    trades.append(TradeResult(
                        entry_date=position_entry['date'],
                        exit_date=current_date,
                        position_type=position_type,
                        strike=strike,
                        entry_iv=position_entry['iv'],
                        exit_iv=iv,
                        realized_vol=rv,
                        gross_pnl=gross_pnl,
                        transaction_costs=tx_cost,
                        net_pnl=net_pnl,
                        holding_period=days_held,
                        max_drawdown=0  # Simplified
                    ))

                    daily_pnl = net_pnl
                    in_position = False

            if not in_position:
                # Check entry signal
                spread = rv - iv

                if abs(spread) > iv_threshold:
                    # Enter position
                    strike = spot  # ATM
                    T = 30 / 365  # 30-day option

                    call_price = self.bs.call_price(spot, strike, T, r, q, iv)
                    put_price = self.bs.put_price(spot, strike, T, r, q, iv)
                    straddle_price = call_price + put_price

                    position_entry = {
                        'date': current_date,
                        'spot': spot,
                        'strike': strike,
                        'iv': iv,
                        'rv': rv
                    }
                    entry_price = straddle_price
                    entry_spot = spot

                    if spread < 0:
                        # RV < IV: Sell vol
                        position_type = PositionType.SHORT_STRADDLE
                    else:
                        # RV > IV: Buy vol
                        position_type = PositionType.LONG_STRADDLE

                    in_position = True

                    # Entry transaction cost
                    n_contracts = int(self.initial_capital * position_size / (entry_price * 100))
                    tx_cost = self.costs.option_cost(n_contracts * 2, straddle_price)
                    daily_pnl -= tx_cost

            daily_pnls.append({'date': current_date, 'pnl': daily_pnl})

        # Create result
        daily_pnl_df = pd.DataFrame(daily_pnls)
        daily_pnl_series = daily_pnl_df.set_index('date')['pnl']
        cumulative_pnl = daily_pnl_series.cumsum()

        metrics = self._calculate_metrics(trades, cumulative_pnl)

        return BacktestResult(
            trades=trades,
            daily_pnl=daily_pnl_series,
            cumulative_pnl=cumulative_pnl,
            metrics=metrics
        )

    def run_smile_mispricing_strategy(
        self,
        price_data: pd.DataFrame,
        option_data: pd.DataFrame,
        model_params: HestonParameters,
        mispricing_threshold: float = 0.01,
        hold_days: int = 5
    ) -> BacktestResult:
        """
        Backtest smile mispricing strategy.

        Buy underpriced options (model IV > market IV)
        Sell overpriced options (model IV < market IV)

        Args:
            price_data: Underlying price data
            option_data: Historical option data with market IVs
            model_params: Heston parameters for model IV
            mispricing_threshold: Min mispricing to trade
            hold_days: Holding period

        Returns:
            BacktestResult
        """
        engine = HestonEngine()
        r = 0.05
        q = 0.01

        trades = []
        daily_pnls = []

        dates = sorted(price_data['date'].unique())

        for current_date in dates:
            spot = price_data[price_data['date'] == current_date]['close'].values[0]

            # Get options for this date
            day_options = option_data[option_data['date'] == current_date]

            for _, opt in day_options.iterrows():
                K = opt['strike']
                T = opt['tte']
                market_iv = opt['iv']
                is_call = opt['type'] == 'call'

                # Model IV
                model_iv = engine.implied_volatility(
                    spot, K, T, r, q, model_params, is_call
                )

                mispricing = market_iv - model_iv

                if abs(mispricing) > mispricing_threshold:
                    # Trade signal
                    if mispricing > 0:
                        # Market IV too high, sell
                        direction = -1
                    else:
                        # Market IV too low, buy
                        direction = 1

                    # Simplified PnL estimation
                    vega = self.bs.vega(spot, K, T, r, q, market_iv)
                    expected_pnl = direction * abs(mispricing) * vega * 100

                    trades.append(TradeResult(
                        entry_date=current_date,
                        exit_date=current_date + timedelta(days=hold_days),
                        position_type=PositionType.DELTA_HEDGED_CALL if is_call else PositionType.DELTA_HEDGED_PUT,
                        strike=K,
                        entry_iv=market_iv,
                        exit_iv=model_iv,
                        realized_vol=market_iv,
                        gross_pnl=expected_pnl,
                        transaction_costs=5.0,
                        net_pnl=expected_pnl - 5.0,
                        holding_period=hold_days,
                        max_drawdown=0
                    ))

            daily_pnls.append({
                'date': current_date,
                'pnl': sum(t.net_pnl for t in trades if t.entry_date == current_date)
            })

        daily_pnl_df = pd.DataFrame(daily_pnls)
        if len(daily_pnl_df) > 0:
            daily_pnl_series = daily_pnl_df.set_index('date')['pnl']
            cumulative_pnl = daily_pnl_series.cumsum()
        else:
            daily_pnl_series = pd.Series(dtype=float)
            cumulative_pnl = pd.Series(dtype=float)

        metrics = self._calculate_metrics(trades, cumulative_pnl)

        return BacktestResult(
            trades=trades,
            daily_pnl=daily_pnl_series,
            cumulative_pnl=cumulative_pnl,
            metrics=metrics
        )

    def _calculate_metrics(
        self,
        trades: List[TradeResult],
        cumulative_pnl: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not trades:
            return {
                'total_pnl': 0,
                'n_trades': 0,
                'win_rate': 0,
                'avg_trade_pnl': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0
            }

        pnls = [t.net_pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        # Sharpe ratio (annualized)
        if len(cumulative_pnl) > 1:
            daily_returns = cumulative_pnl.diff().dropna()
            if daily_returns.std() > 0:
                sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Max drawdown
        if len(cumulative_pnl) > 0:
            running_max = cumulative_pnl.cummax()
            drawdown = cumulative_pnl - running_max
            max_dd = drawdown.min() / (self.initial_capital + 1e-10)  # As fraction of capital
        else:
            max_dd = 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            'total_pnl': sum(pnls),
            'n_trades': len(trades),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'avg_trade_pnl': np.mean(pnls),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'profit_factor': profit_factor,
            'avg_holding_period': np.mean([t.holding_period for t in trades]),
            'total_costs': sum(t.transaction_costs for t in trades)
        }


def generate_synthetic_backtest_data(
    start_date: date,
    end_date: date,
    initial_spot: float = 500,
    drift: float = 0.05,
    base_vol: float = 0.20,
    vol_of_vol: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic price and IV data for backtesting.

    Args:
        start_date: Start date
        end_date: End date
        initial_spot: Starting price
        drift: Annualized drift
        base_vol: Base volatility level
        vol_of_vol: Volatility of volatility (for IV variation)

    Returns:
        Tuple of (price_data, iv_data) DataFrames
    """
    n_days = (end_date - start_date).days
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    # Simulate prices with stochastic volatility (simplified Heston)
    dt = 1 / 252
    spot = initial_spot
    vol = base_vol

    prices = []
    ivs = []

    for d in dates:
        # Update vol (mean-reverting)
        vol += 2.0 * (base_vol - vol) * dt + vol_of_vol * np.sqrt(vol * dt) * np.random.normal()
        vol = max(vol, 0.05)

        # Update price
        ret = drift * dt + np.sqrt(vol) * np.sqrt(dt) * np.random.normal()
        spot *= np.exp(ret)

        # Simulate IV (vol + premium)
        iv_premium = 0.02 + 0.01 * np.random.normal()
        iv = vol + iv_premium

        prices.append({
            'date': d,
            'open': spot * (1 + 0.001 * np.random.normal()),
            'high': spot * (1 + abs(0.01 * np.random.normal())),
            'low': spot * (1 - abs(0.01 * np.random.normal())),
            'close': spot
        })

        ivs.append({
            'date': d,
            'atm_iv': iv
        })

    return pd.DataFrame(prices), pd.DataFrame(ivs)
