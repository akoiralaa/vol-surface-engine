"""
Polygon.io Options Data Client
==============================

Fetches live SPY/QQQ option chain data from Polygon.io API.
Handles data normalization, IV extraction, and snapshot creation.
"""

import os
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum

import numpy as np
import pandas as pd
from polygon import RESTClient
from dotenv import load_dotenv

load_dotenv()


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class OptionContract:
    """Single option contract with market data."""
    ticker: str
    underlying: str
    strike: float
    expiry: date
    option_type: OptionType
    bid: float
    ask: float
    mid: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

    def time_to_expiry(self, ref_date: Optional[date] = None) -> float:
        """Calculate time to expiry in years (ACT/365)."""
        if ref_date is None:
            ref_date = date.today()
        days = (self.expiry - ref_date).days
        return max(days / 365.0, 1/365)  # Minimum 1 day

    def moneyness(self, spot: float) -> float:
        """Calculate moneyness K/S."""
        return self.strike / spot

    def log_moneyness(self, spot: float) -> float:
        """Calculate log-moneyness ln(K/S)."""
        return np.log(self.strike / spot)

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid."""
        return self.spread / self.mid if self.mid > 0 else float('inf')


@dataclass
class OptionChainSnapshot:
    """Complete option chain snapshot for a single underlying."""
    underlying: str
    spot_price: float
    timestamp: datetime
    contracts: List[OptionContract]
    risk_free_rate: float = 0.05  # Default 5%
    dividend_yield: float = 0.01  # Default 1%

    @property
    def expiries(self) -> List[date]:
        """Unique expiry dates sorted."""
        return sorted(set(c.expiry for c in self.contracts))

    @property
    def strikes(self) -> np.ndarray:
        """Unique strikes sorted."""
        return np.array(sorted(set(c.strike for c in self.contracts)))

    def filter_by_expiry(self, expiry: date) -> List[OptionContract]:
        """Get contracts for a specific expiry."""
        return [c for c in self.contracts if c.expiry == expiry]

    def filter_by_moneyness(self, min_m: float = 0.8, max_m: float = 1.2) -> List[OptionContract]:
        """Filter contracts by moneyness range."""
        return [
            c for c in self.contracts
            if min_m <= c.moneyness(self.spot_price) <= max_m
        ]

    def filter_by_liquidity(self, min_volume: int = 10, max_spread_pct: float = 0.2) -> List[OptionContract]:
        """Filter for liquid contracts."""
        return [
            c for c in self.contracts
            if c.volume >= min_volume and c.spread_pct <= max_spread_pct
        ]

    def get_atm_contracts(self, tolerance: float = 0.02) -> List[OptionContract]:
        """Get near-ATM contracts."""
        return [
            c for c in self.contracts
            if abs(c.moneyness(self.spot_price) - 1.0) <= tolerance
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        records = []
        for c in self.contracts:
            records.append({
                'ticker': c.ticker,
                'strike': c.strike,
                'expiry': c.expiry,
                'type': c.option_type.value,
                'bid': c.bid,
                'ask': c.ask,
                'mid': c.mid,
                'iv': c.implied_volatility,
                'volume': c.volume,
                'oi': c.open_interest,
                'tte': c.time_to_expiry(),
                'moneyness': c.moneyness(self.spot_price),
                'delta': c.delta,
                'gamma': c.gamma,
                'theta': c.theta,
                'vega': c.vega
            })
        return pd.DataFrame(records)


class PolygonOptionsClient:
    """
    Client for fetching options data from Polygon.io.

    Requires POLYGON_API_KEY environment variable.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in environment")
        self.client = RESTClient(self.api_key)

    def get_spot_price(self, ticker: str) -> float:
        """Fetch current spot price for underlying."""
        try:
            agg = self.client.get_previous_close_agg(ticker)
            if agg and len(agg) > 0:
                return agg[0].close
        except Exception as e:
            print(f"Warning: Could not fetch spot for {ticker}: {e}")

        # Fallback to snapshot
        try:
            snapshot = self.client.get_snapshot_ticker("stocks", ticker)
            return snapshot.day.close if snapshot.day else snapshot.prev_day.close
        except Exception as e:
            raise RuntimeError(f"Failed to get spot price for {ticker}: {e}")

    def get_option_chain(
        self,
        underlying: str,
        expiry_range: Tuple[date, date] = None,
        strike_range: Tuple[float, float] = None,
        option_type: Optional[OptionType] = None,
        limit: int = 250
    ) -> List[OptionContract]:
        """
        Fetch option chain contracts from Polygon.

        Args:
            underlying: Ticker symbol (e.g., 'SPY')
            expiry_range: (min_expiry, max_expiry) tuple
            strike_range: (min_strike, max_strike) tuple
            option_type: Filter by CALL or PUT
            limit: Max contracts per request

        Returns:
            List of OptionContract objects
        """
        contracts = []

        # Build query parameters
        params = {
            "underlying_ticker": underlying,
            "limit": limit,
            "order": "asc",
            "sort": "expiration_date"
        }

        if expiry_range:
            params["expiration_date.gte"] = expiry_range[0].isoformat()
            params["expiration_date.lte"] = expiry_range[1].isoformat()

        if strike_range:
            params["strike_price.gte"] = strike_range[0]
            params["strike_price.lte"] = strike_range[1]

        if option_type:
            params["contract_type"] = option_type.value

        try:
            # Fetch contracts list
            for contract in self.client.list_options_contracts(**params):
                contracts.append(contract)
        except Exception as e:
            print(f"Error fetching options chain: {e}")
            return []

        return contracts

    def get_option_snapshot(
        self,
        underlying: str,
        min_tte: float = 0.01,  # ~4 days
        max_tte: float = 0.5,   # 6 months
        moneyness_range: Tuple[float, float] = (0.85, 1.15),
        use_otm_only: bool = True
    ) -> OptionChainSnapshot:
        """
        Get complete option chain snapshot with market data.

        Args:
            underlying: Ticker symbol
            min_tte: Minimum time to expiry (years)
            max_tte: Maximum time to expiry (years)
            moneyness_range: (min, max) moneyness filter
            use_otm_only: If True, use OTM puts for K<S, OTM calls for K>S

        Returns:
            OptionChainSnapshot with all contract data
        """
        # Get spot price
        spot = self.get_spot_price(underlying)

        # Calculate date range
        today = date.today()
        min_expiry = today + timedelta(days=int(min_tte * 365))
        max_expiry = today + timedelta(days=int(max_tte * 365))

        # Calculate strike range
        min_strike = spot * moneyness_range[0]
        max_strike = spot * moneyness_range[1]

        # Fetch option chain snapshot from Polygon
        contracts = []

        try:
            # Use snapshot endpoint for live data with greeks
            snapshot = self.client.list_snapshot_options_chain(
                underlying,
                params={
                    "strike_price.gte": min_strike,
                    "strike_price.lte": max_strike,
                    "expiration_date.gte": min_expiry.isoformat(),
                    "expiration_date.lte": max_expiry.isoformat()
                }
            )

            for opt in snapshot:
                # Parse contract details
                details = opt.details
                day = opt.day
                greeks = opt.greeks if hasattr(opt, 'greeks') else None

                # Determine option type
                opt_type = OptionType.CALL if details.contract_type == "call" else OptionType.PUT

                # OTM filter
                if use_otm_only:
                    if opt_type == OptionType.CALL and details.strike_price < spot:
                        continue
                    if opt_type == OptionType.PUT and details.strike_price > spot:
                        continue

                # Extract prices
                bid = day.close if hasattr(day, 'close') and day.close else 0
                ask = day.close if hasattr(day, 'close') and day.close else 0

                # Try to get bid/ask from quote if available
                if hasattr(opt, 'last_quote') and opt.last_quote:
                    bid = opt.last_quote.bid or bid
                    ask = opt.last_quote.ask or ask

                mid = (bid + ask) / 2 if bid and ask else day.close if hasattr(day, 'close') else 0

                contract = OptionContract(
                    ticker=details.ticker,
                    underlying=underlying,
                    strike=details.strike_price,
                    expiry=datetime.strptime(details.expiration_date, "%Y-%m-%d").date(),
                    option_type=opt_type,
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    last=day.close if hasattr(day, 'close') and day.close else mid,
                    volume=day.volume if hasattr(day, 'volume') and day.volume else 0,
                    open_interest=opt.open_interest if hasattr(opt, 'open_interest') else 0,
                    implied_volatility=opt.implied_volatility if hasattr(opt, 'implied_volatility') else None,
                    delta=greeks.delta if greeks and hasattr(greeks, 'delta') else None,
                    gamma=greeks.gamma if greeks and hasattr(greeks, 'gamma') else None,
                    theta=greeks.theta if greeks and hasattr(greeks, 'theta') else None,
                    vega=greeks.vega if greeks and hasattr(greeks, 'vega') else None
                )

                contracts.append(contract)

        except Exception as e:
            print(f"Error fetching snapshot: {e}")
            raise

        return OptionChainSnapshot(
            underlying=underlying,
            spot_price=spot,
            timestamp=datetime.now(),
            contracts=contracts
        )

    def get_historical_options(
        self,
        ticker: str,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Fetch historical option data for backtesting.

        Args:
            ticker: Option ticker (e.g., 'O:SPY251219C00600000')
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        try:
            aggs = self.client.list_aggs(
                ticker,
                1,
                "day",
                start_date.isoformat(),
                end_date.isoformat()
            )

            records = []
            for agg in aggs:
                records.append({
                    'date': datetime.fromtimestamp(agg.timestamp / 1000).date(),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume,
                    'vwap': agg.vwap if hasattr(agg, 'vwap') else None
                })

            return pd.DataFrame(records)

        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()


def create_synthetic_snapshot(
    spot: float = 500.0,
    n_expiries: int = 5,
    n_strikes: int = 11,
    base_vol: float = 0.20,
    skew: float = -0.1,
    term_slope: float = 0.02
) -> OptionChainSnapshot:
    """
    Create synthetic option chain for testing without API.

    Generates realistic vol surface with skew and term structure.

    Args:
        spot: Spot price
        n_expiries: Number of expiry dates
        n_strikes: Number of strikes per expiry
        base_vol: ATM volatility level
        skew: Skew parameter (negative = put skew)
        term_slope: Term structure slope

    Returns:
        Synthetic OptionChainSnapshot
    """
    contracts = []
    today = date.today()

    # Generate expiries (1 week to 6 months)
    expiry_days = np.linspace(7, 180, n_expiries).astype(int)

    # Generate strikes (80% to 120% of spot)
    strikes = np.linspace(0.85 * spot, 1.15 * spot, n_strikes)

    for exp_days in expiry_days:
        expiry = today + timedelta(days=int(exp_days))
        T = exp_days / 365.0

        for strike in strikes:
            # Generate IV with skew and term structure
            moneyness = np.log(strike / spot)

            # Parabolic smile with skew
            iv = base_vol + skew * moneyness + 0.5 * abs(moneyness) ** 2

            # Add term structure
            iv += term_slope * np.sqrt(T)

            # Add small noise
            iv *= (1 + np.random.normal(0, 0.01))
            iv = max(iv, 0.05)  # Floor at 5%

            # Determine OTM option type
            if strike < spot:
                opt_type = OptionType.PUT
            else:
                opt_type = OptionType.CALL

            # Generate realistic prices using simple Black-Scholes-ish approximation
            d1 = (np.log(spot / strike) + 0.5 * iv**2 * T) / (iv * np.sqrt(T))
            from scipy.stats import norm

            if opt_type == OptionType.CALL:
                price = spot * norm.cdf(d1) - strike * np.exp(-0.05 * T) * norm.cdf(d1 - iv * np.sqrt(T))
            else:
                price = strike * np.exp(-0.05 * T) * norm.cdf(-d1 + iv * np.sqrt(T)) - spot * norm.cdf(-d1)

            price = max(price, 0.01)
            spread = price * 0.02  # 2% spread

            contract = OptionContract(
                ticker=f"O:{expiry.strftime('%y%m%d')}{'C' if opt_type == OptionType.CALL else 'P'}{int(strike*1000):08d}",
                underlying="SYNTH",
                strike=strike,
                expiry=expiry,
                option_type=opt_type,
                bid=price - spread/2,
                ask=price + spread/2,
                mid=price,
                last=price,
                volume=np.random.randint(100, 10000),
                open_interest=np.random.randint(1000, 100000),
                implied_volatility=iv,
                delta=norm.cdf(d1) if opt_type == OptionType.CALL else norm.cdf(d1) - 1
            )
            contracts.append(contract)

    return OptionChainSnapshot(
        underlying="SYNTH",
        spot_price=spot,
        timestamp=datetime.now(),
        contracts=contracts
    )
