import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from enum import Enum
from scipy.stats import norm
from datetime import date, timedelta

from ..data.polygon_client import OptionChainSnapshot, OptionContract
from ..models.heston_engine import HestonEngine, HestonParameters
from ..models.sabr_engine import SABREngine, SABRParameters


class ArbitrageType(Enum):
    BUTTERFLY = "butterfly"
    CALENDAR = "calendar"
    CALL_SPREAD = "call_spread"
    RV_IV = "rv_iv"     # Realized vs Implied


@dataclass
class ArbitrageViolation: 
    """arbitrage violation detected"""
    arb_type: ArbitrageType
    strike1: float
    strike2: Optional[float]
    strike3: Optional[float]    # For butterfly
    expiry1: float
    expiry2: Optional[float]    # For calendar
    violation_magnitude: float  # How severe (in vol or price terms)
    pnl_estimate: float         # Estimated profit from trade
    confidence: float           # 0-1 score
    contracts: List[str]        # Tickers involved
    description: str


class StaticArbitrageDetector: 
    """no-arbitrage violations in option prices detected"""
    def __init__(
        self,
        butterfly_threshold: float = 0.001,
        calendar_threshold: float = 0.0001,
        min_confidence: float = 0.5
    ):
        self.butterfly_threshold = butterfly_threshold  #butterfly_threshold: min clionation to flag butterfly arb
        self.calendar_threshold = calendar_threshold    #calendar_threshold: min variance decrease for calendar arb
        self.min_confidence = min_confidence            #min_confidence: minimum confidence to report

    def detect_butterfly_arbitrage(
        self,
        snapshot: OptionChainSnapshot
    ) -> List[ArbitrageViolation]:
        """ 
        butterfly spread arb deection (convexity violation) 
        butterfly spread: Long K1, Short 2×K2, Long K3
        no-arbitrage requires: Price(K1) + Price(K3) ≥ 2×Price(K2)
        equivalently in IV: second derivative of total variance w.r.t. log-strike ≥ 0
        """
        violations = []
        spot = snapshot.spot_price

        # group by expiry
        expiry_groups: Dict[float, List[OptionContract]] = {}
        for c in snapshot.contracts:
            T = round(c.time_to_expiry(), 4)
            if T not in expiry_groups:
                expiry_groups[T] = []
            expiry_groups[T].append(c)

        for T, contracts in expiry_groups.items():
            if len(contracts) < 3:
                continue

            # sort by strike
            contracts = sorted(contracts, key=lambda c: c.strike)
            strikes = np.array([c.strike for c in contracts])
            prices = np.array([c.mid for c in contracts])

            # checking each triplet
            for i in range(len(contracts) - 2):
                K1, K2, K3 = strikes[i], strikes[i+1], strikes[i+2]
                P1, P2, P3 = prices[i], prices[i+1], prices[i+2]

                # butterfly value should be non-negative
                # interpolate P2_interp from P1, P3
                w = (K2 - K1) / (K3 - K1)
                P2_interp = P1 * (1 - w) + P3 * w

                butterfly_value = P1 + P3 - 2 * P2
                violation = P2 - P2_interp  # convexity violation

                if violation > self.butterfly_threshold * spot:
                    # calculating potential PnL
                    pnl = violation  # PPU if we sell the butterfly

                    # confidence based on liquidity
                    liquidity_score = min(1.0, sum(c.volume for c in contracts[i:i+3]) / 1000)
                    spread_score = 1.0 - min(1.0, sum(c.spread_pct for c in contracts[i:i+3]) / 3)
                    confidence = 0.5 * liquidity_score + 0.5 * spread_score

                    if confidence >= self.min_confidence:
                        violations.append(ArbitrageViolation(
                            arb_type=ArbitrageType.BUTTERFLY,
                            strike1=K1,
                            strike2=K2,
                            strike3=K3,
                            expiry1=T,
                            expiry2=None,
                            violation_magnitude=violation / spot,
                            pnl_estimate=pnl,
                            confidence=confidence,
                            contracts=[contracts[i].ticker, contracts[i+1].ticker, contracts[i+2].ticker],
                            description=f"Butterfly violation at K={K2:.0f}, T={T:.3f}: "
                                       f"center overpriced by {violation:.4f}"
                        ))

        return violations

    def detect_calendar_arbitrage(
        self,
        snapshot: OptionChainSnapshot
    ) -> List[ArbitrageViolation]:
        """ 
        detecting  calendar spread arbitrage (term structure violations)
        no-arbitrage: Total variance σ²T must be increasing in T
        """
        violations = []
        spot = snapshot.spot_price

        # Group by strike
        strike_groups: Dict[float, List[OptionContract]] = {}
        for c in snapshot.contracts:
            if c.implied_volatility and c.implied_volatility > 0.01:
                K = round(c.strike, 2)
                if K not in strike_groups:
                    strike_groups[K] = []
                strike_groups[K].append(c)

        for K, contracts in strike_groups.items():
            if len(contracts) < 2:
                continue

            # Sort by expiry
            contracts = sorted(contracts, key=lambda c: c.time_to_expiry())

            for i in range(len(contracts) - 1):
                c1, c2 = contracts[i], contracts[i+1]
                T1, T2 = c1.time_to_expiry(), c2.time_to_expiry()
                iv1, iv2 = c1.implied_volatility, c2.implied_volatility

                # Total variance
                var1 = iv1**2 * T1
                var2 = iv2**2 * T2

                # Should have var2 > var1
                if var1 > var2 + self.calendar_threshold:
                    variance_decrease = var1 - var2

                    # Calculate potential PnL (approximate)
                    pnl = (np.sqrt(var1) - np.sqrt(var2)) * spot * np.sqrt(T2 - T1)

                    # Confidence
                    liquidity_score = min(1.0, (c1.volume + c2.volume) / 500)
                    confidence = 0.7 * liquidity_score + 0.3 * (1 - min(1.0, abs(K/spot - 1) / 0.1))

                    if confidence >= self.min_confidence:
                        violations.append(ArbitrageViolation(
                            arb_type=ArbitrageType.CALENDAR,
                            strike1=K,
                            strike2=None,
                            strike3=None,
                            expiry1=T1,
                            expiry2=T2,
                            violation_magnitude=variance_decrease,
                            pnl_estimate=pnl,
                            confidence=confidence,
                            contracts=[c1.ticker, c2.ticker],
                            description=f"Calendar arb at K={K:.0f}: var decreases from "
                                       f"T={T1:.3f} to T={T2:.3f} by {variance_decrease:.6f}"
                        ))

        return violations

    def detect_call_spread_arbitrage(
        self,
        snapshot: OptionChainSnapshot
    ) -> List[ArbitrageViolation]:
        """
        Detect call spread constraint violations.

        No-arbitrage: -e^(-qT) ≤ dC/dK ≤ 0
        """
        violations = []
        spot = snapshot.spot_price
        r = snapshot.risk_free_rate
        q = snapshot.dividend_yield

        # Group by expiry
        expiry_groups: Dict[float, List[OptionContract]] = {}
        for c in snapshot.contracts:
            if c.option_type.value == "call":
                T = round(c.time_to_expiry(), 4)
                if T not in expiry_groups:
                    expiry_groups[T] = []
                expiry_groups[T].append(c)

        for T, contracts in expiry_groups.items():
            if len(contracts) < 2:
                continue

            discount = np.exp(-q * T)
            contracts = sorted(contracts, key=lambda c: c.strike)

            for i in range(len(contracts) - 1):
                c1, c2 = contracts[i], contracts[i+1]
                K1, K2 = c1.strike, c2.strike
                P1, P2 = c1.mid, c2.mid

                dC_dK = (P2 - P1) / (K2 - K1)

                # Check constraints
                if dC_dK > 0.01:  # Should be negative
                    violations.append(ArbitrageViolation(
                        arb_type=ArbitrageType.CALL_SPREAD,
                        strike1=K1,
                        strike2=K2,
                        strike3=None,
                        expiry1=T,
                        expiry2=None,
                        violation_magnitude=dC_dK,
                        pnl_estimate=dC_dK * (K2 - K1),
                        confidence=0.8,
                        contracts=[c1.ticker, c2.ticker],
                        description=f"Call spread violation: dC/dK = {dC_dK:.4f} > 0"
                    ))
                elif dC_dK < -discount - 0.01:  # Should be > -e^(-qT)
                    violations.append(ArbitrageViolation(
                        arb_type=ArbitrageType.CALL_SPREAD,
                        strike1=K1,
                        strike2=K2,
                        strike3=None,
                        expiry1=T,
                        expiry2=None,
                        violation_magnitude=abs(dC_dK + discount),
                        pnl_estimate=abs(dC_dK + discount) * (K2 - K1),
                        confidence=0.6,
                        contracts=[c1.ticker, c2.ticker],
                        description=f"Call spread lower bound violation: "
                                   f"dC/dK = {dC_dK:.4f} < -{discount:.4f}"
                    ))

        return violations

    def detect_all(
        self,
        snapshot: OptionChainSnapshot
    ) -> List[ArbitrageViolation]:
        """Detect all types of static arbitrage."""
        violations = []
        violations.extend(self.detect_butterfly_arbitrage(snapshot))
        violations.extend(self.detect_calendar_arbitrage(snapshot))
        violations.extend(self.detect_call_spread_arbitrage(snapshot))
        return sorted(violations, key=lambda v: -v.pnl_estimate)


@dataclass
class RealizedVolEstimate:
    """Realized volatility estimate with confidence."""
    value: float
    std_error: float
    window_days: int
    method: str


class RealizedVolCalculator:
    """
    Calculate realized volatility from historical prices.
    """

    @staticmethod
    def close_to_close(prices: np.ndarray, annualization: float = 252) -> RealizedVolEstimate:
        """
        Standard close-to-close realized vol.

        Args:
            prices: Array of closing prices
            annualization: Trading days per year

        Returns:
            RealizedVolEstimate
        """
        log_returns = np.diff(np.log(prices))
        rv = np.std(log_returns) * np.sqrt(annualization)
        se = rv / np.sqrt(2 * len(log_returns))

        return RealizedVolEstimate(
            value=rv,
            std_error=se,
            window_days=len(prices),
            method="close-to-close"
        )

    @staticmethod
    def parkinson(high: np.ndarray, low: np.ndarray, annualization: float = 252) -> RealizedVolEstimate:
        """
        Parkinson (1980) high-low estimator.

        More efficient than close-to-close.
        """
        log_hl = np.log(high / low)
        rv = np.sqrt(np.mean(log_hl**2) / (4 * np.log(2))) * np.sqrt(annualization)
        se = rv / np.sqrt(2 * len(high))

        return RealizedVolEstimate(
            value=rv,
            std_error=se,
            window_days=len(high),
            method="parkinson"
        )

    @staticmethod
    def garman_klass(
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        annualization: float = 252
    ) -> RealizedVolEstimate:
        """
        Garman-Klass (1980) OHLC estimator.

        Most efficient with OHLC data.
        """
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)

        term1 = 0.5 * log_hl**2
        term2 = (2 * np.log(2) - 1) * log_co**2

        rv = np.sqrt(np.mean(term1 - term2) * annualization)
        se = rv / np.sqrt(2 * len(high))

        return RealizedVolEstimate(
            value=rv,
            std_error=se,
            window_days=len(high),
            method="garman-klass"
        )


class DynamicArbitrageDetector:
    """
    Detect realized vs implied volatility divergences.

    Core strategy: When RV << IV, sell vol (short straddle/strangle)
                   When RV >> IV, buy vol (long straddle/strangle)
    """

    def __init__(
        self,
        rv_iv_threshold: float = 0.02,  # 2 vol points
        lookback_days: int = 21,
        forecast_horizon: int = 5
    ):
        """
        Initialize detector.

        Args:
            rv_iv_threshold: Min RV-IV difference to signal
            lookback_days: Days for RV calculation
            forecast_horizon: Days to hold trade
        """
        self.rv_iv_threshold = rv_iv_threshold
        self.lookback_days = lookback_days
        self.forecast_horizon = forecast_horizon
        self.rv_calculator = RealizedVolCalculator()

    def detect_rv_iv_divergence(
        self,
        snapshot: OptionChainSnapshot,
        historical_prices: np.ndarray
    ) -> List[ArbitrageViolation]:
        """
        Detect realized vs implied vol divergence.

        Args:
            snapshot: Current option chain
            historical_prices: Recent daily closing prices

        Returns:
            List of RV-IV divergence signals
        """
        violations = []

        # Calculate realized vol
        rv_estimate = self.rv_calculator.close_to_close(
            historical_prices[-self.lookback_days:]
        )
        rv = rv_estimate.value

        # Get ATM implied vol for near-term expiry
        spot = snapshot.spot_price
        atm_contracts = snapshot.get_atm_contracts(tolerance=0.03)

        if not atm_contracts:
            return violations

        # Find nearest expiry ATM
        min_expiry_contracts = [
            c for c in atm_contracts
            if c.time_to_expiry() > 0.02  # At least 1 week
        ]

        if not min_expiry_contracts:
            return violations

        nearest = min(min_expiry_contracts, key=lambda c: c.time_to_expiry())
        iv = nearest.implied_volatility

        if iv is None:
            return violations

        # Calculate divergence
        divergence = rv - iv

        if abs(divergence) > self.rv_iv_threshold:
            # Estimate PnL from variance swap approximation
            T = nearest.time_to_expiry()
            notional = spot * 0.01  # 1% of spot
            # Approx vega PnL
            vega_approx = spot * np.sqrt(T) * 0.4 * norm.pdf(0)  # ATM vega
            pnl_estimate = abs(divergence) * vega_approx

            # Confidence based on RV estimation error
            z_score = abs(divergence) / rv_estimate.std_error
            confidence = min(0.95, 0.5 + 0.1 * z_score)

            direction = "RV > IV (buy vol)" if divergence > 0 else "RV < IV (sell vol)"

            violations.append(ArbitrageViolation(
                arb_type=ArbitrageType.RV_IV,
                strike1=nearest.strike,
                strike2=None,
                strike3=None,
                expiry1=T,
                expiry2=None,
                violation_magnitude=abs(divergence),
                pnl_estimate=pnl_estimate,
                confidence=confidence,
                contracts=[nearest.ticker],
                description=f"{direction}: RV={rv:.2%}, IV={iv:.2%}, "
                           f"gap={divergence:.2%}"
            ))

        return violations


class SmileArbitrageScanner:
    """
    Scan for smile mispricing opportunities.

    Compares market IV to model (Heston/SABR) IV to find
    over/underpriced options relative to fitted surface.
    """

    def __init__(
        self,
        model_type: str = "heston",
        mispricing_threshold: float = 0.01  # 1 vol point
    ):
        """
        Initialize scanner.

        Args:
            model_type: 'heston' or 'sabr'
            mispricing_threshold: Min model-market IV gap
        """
        self.model_type = model_type
        self.mispricing_threshold = mispricing_threshold

    def scan(
        self,
        snapshot: OptionChainSnapshot,
        model_params: HestonParameters,
        calibration_rmse: float
    ) -> List[Dict]:
        """
        Scan for smile mispricings relative to calibrated model.

        Args:
            snapshot: Option chain data
            model_params: Calibrated model parameters
            calibration_rmse: Calibration RMSE for confidence

        Returns:
            List of mispricing signals
        """
        signals = []
        engine = HestonEngine()

        spot = snapshot.spot_price
        r = snapshot.risk_free_rate
        q = snapshot.dividend_yield

        for contract in snapshot.contracts:
            if not contract.implied_volatility:
                continue

            K = contract.strike
            T = contract.time_to_expiry()
            market_iv = contract.implied_volatility
            is_call = contract.option_type.value == "call"

            # Get model IV
            model_iv = engine.implied_volatility(
                spot, K, T, r, q, model_params, is_call
            )

            # Mispricing
            mispricing = market_iv - model_iv

            if abs(mispricing) > self.mispricing_threshold:
                # Confidence decreases with calibration error
                base_conf = min(0.9, 0.3 + abs(mispricing) / 0.05)
                confidence = base_conf * (1 - min(1, calibration_rmse / 0.02))

                # Trade direction
                if mispricing > 0:
                    direction = "SELL"
                    trade = f"Sell {contract.option_type.value}"
                else:
                    direction = "BUY"
                    trade = f"Buy {contract.option_type.value}"

                signals.append({
                    'ticker': contract.ticker,
                    'strike': K,
                    'expiry': T,
                    'moneyness': K / spot,
                    'market_iv': market_iv,
                    'model_iv': model_iv,
                    'mispricing': mispricing,
                    'direction': direction,
                    'trade': trade,
                    'confidence': confidence,
                    'bid': contract.bid,
                    'ask': contract.ask,
                    'spread': contract.spread_pct,
                    'volume': contract.volume
                })

        # Sort by absolute mispricing
        signals = sorted(signals, key=lambda x: -abs(x['mispricing']))

        return signals


def generate_arbitrage_report(
    snapshot: OptionChainSnapshot,
    model_params: Optional[HestonParameters] = None,
    historical_prices: Optional[np.ndarray] = None
) -> Dict:
    """
    Generate comprehensive arbitrage analysis report.

    Args:
        snapshot: Option chain data
        model_params: Calibrated Heston parameters (optional)
        historical_prices: Historical prices for RV calculation (optional)

    Returns:
        Report dictionary
    """
    report = {
        'underlying': snapshot.underlying,
        'spot': snapshot.spot_price,
        'timestamp': snapshot.timestamp.isoformat(),
        'n_contracts': len(snapshot.contracts),
        'static_arbitrage': [],
        'dynamic_arbitrage': [],
        'smile_signals': [],
        'summary': {}
    }

    # Static arbitrage
    static_detector = StaticArbitrageDetector()
    static_violations = static_detector.detect_all(snapshot)
    report['static_arbitrage'] = [
        {
            'type': v.arb_type.value,
            'description': v.description,
            'magnitude': v.violation_magnitude,
            'pnl_estimate': v.pnl_estimate,
            'confidence': v.confidence,
            'contracts': v.contracts
        }
        for v in static_violations
    ]

    # Dynamic arbitrage (RV-IV)
    if historical_prices is not None and len(historical_prices) >= 21:
        dynamic_detector = DynamicArbitrageDetector()
        dynamic_violations = dynamic_detector.detect_rv_iv_divergence(
            snapshot, historical_prices
        )
        report['dynamic_arbitrage'] = [
            {
                'type': v.arb_type.value,
                'description': v.description,
                'magnitude': v.violation_magnitude,
                'pnl_estimate': v.pnl_estimate,
                'confidence': v.confidence
            }
            for v in dynamic_violations
        ]

    # Smile signals (if model provided)
    if model_params is not None:
        scanner = SmileArbitrageScanner()
        signals = scanner.scan(snapshot, model_params, calibration_rmse=0.01)
        report['smile_signals'] = signals[:10]  # Top 10

    # Summary stats
    report['summary'] = {
        'n_butterfly_violations': len([v for v in static_violations if v.arb_type == ArbitrageType.BUTTERFLY]),
        'n_calendar_violations': len([v for v in static_violations if v.arb_type == ArbitrageType.CALENDAR]),
        'total_arb_pnl': sum(v.pnl_estimate for v in static_violations),
        'max_confidence_signal': max([v.confidence for v in static_violations], default=0)
    }

    return report
