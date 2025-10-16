import numpy as np
from scipy.stats import norm
from typing import Dict, List, Any

# Import the abstract base class for the trading strategy
from autograder.sdk.strategy_interface import AbstractTradingStrategy

class MyTradingStrategy(AbstractTradingStrategy):
    """
    An aggressive momentum-based strategy that trades frequently by making its
    quotes proportional to market sentiment. It uses tighter spreads and faster
    reactions to capitalize on short-term order flow.
    """

    def __init__(self):
        """Initializes the strategy with more aggressive parameters."""
        super().__init__()
        # Tighter base spread to increase trade frequency
        self.base_spread = 2.5
        # Lower inventory penalty to allow riding trends
        self.inventory_multiplier = 0.05
        # **NEW**: Proportional factor for sentiment. This is the core of the new logic.
        self.sentiment_proportional_factor = 0.6
        # Faster EMA to react more quickly to recent trades
        self.ema_alpha = 0.5
        # Multiplier for adjusting spread based on market volatility
        self.volatility_spread_multiplier = 1.5
        # Tracks the exponential moving average of net order flow
        self.net_order_flow_ema = 0.0

    def on_game_start(self, config: Dict[str, Any]) -> None:
        """Called once at the start of the game."""
        self.team_name = config.get("team_name", "my_team")

    def make_market(
        self,
        marketplace: Any,
        training_rolls: List[int],
        my_trades: List,
        current_rolls: List[int],
        round_info: Any
    ) -> Dict:
        """Main logic loop called to generate quotes."""
        markets = {}
        products = marketplace.get_products()
        all_trades = marketplace.get_trade_history()
        my_trades = marketplace.my_trades
        training_rolls = marketplace.training_rolls
        
        # --- Volatility and Sentiment Analysis ---
        recent_trade_prices = [t.price for t in all_trades[-30:] if t.price]
        market_volatility = np.std(recent_trade_prices) if len(recent_trade_prices) > 1 else 1.0
        dynamic_spread = self.base_spread + market_volatility * self.volatility_spread_multiplier

        recent_trades = all_trades[-20:]
        net_order_flow = sum(t.quantity for t in recent_trades if t.side == 'BUY') - \
                         sum(t.quantity for t in recent_trades if t.side == 'SELL')
        self.net_order_flow_ema = self.ema_alpha * net_order_flow + (1 - self.ema_alpha) * self.net_order_flow_ema

        # **CRITICAL CHANGE**: Sentiment adjustment is now PROPORTIONAL to the EMA of order flow
        # This makes the strategy much more reactive and less reliant on fixed thresholds.
        sentiment_adjustment = self.net_order_flow_ema * self.sentiment_proportional_factor

        for product in products:
            # **BUG FIX**: Accessing product.id instead of product.product_id
            fair_value = self.calculate_fair_value(product.id, training_rolls, marketplace.round_info)
            
            if fair_value is not None:
                adjusted_fair_value = fair_value + sentiment_adjustment
                
                position = my_trades.get_position(product.id)
                inventory_adjustment = position * self.inventory_multiplier
                adjusted_fair_value -= inventory_adjustment

                bid_price = adjusted_fair_value - dynamic_spread / 2
                ask_price = adjusted_fair_value + dynamic_spread / 2
                
                markets[product.id] = (round(bid_price, 1), round(ask_price, 1))

        return markets

    def calculate_fair_value(self, product_id: str, training_rolls: List[int], round_info: Dict[str, Any]) -> float | None:
        """Calculates the theoretical fair value using the Black-Scholes-Merton model for options."""
        try:
            parts = product_id.split(',')
            product_type = parts[1]
            
            expected_roll = np.mean(training_rolls) if training_rolls else 3.5
            total_subrounds = round_info.get("num_sub_rounds", 10)
            total_rolls_per_round = 20000

            if product_type == 'F':
                num_subrounds = int(parts[2])
                return expected_roll * (total_rolls_per_round / total_subrounds * num_subrounds)
            
            elif product_type in ['C', 'P']:
                strike = float(parts[2])
                volatility = np.std(training_rolls) / 10000 if training_rolls and np.std(training_rolls) > 0 else 0.05
                underlying_price = expected_roll * total_rolls_per_round
                current_sub_round = round_info.get("current_sub_round", 1)
                time_to_expiry = max(0.001, (total_subrounds - current_sub_round + 1) / total_subrounds)

                d1 = (np.log(underlying_price / strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
                d2 = d1 - volatility * np.sqrt(time_to_expiry)

                if product_type == 'C':
                    return underlying_price * norm.cdf(d1) - strike * norm.cdf(d2)
                else:
                    return strike * norm.cdf(-d2) - underlying_price * norm.cdf(-d1)
            
            return None
        except (ValueError, IndexError, ZeroDivisionError):
            return None

    def on_round_end(self, result: Dict[str, Any]) -> None:
        """Handles end of round logic."""
        pnl = result.get('your_pnl', 0.0)
        print(f"Round ended. PnL: {pnl:.2f}")
        # Reset EMA for the next round
        self.net_order_flow_ema = 0.0

    def on_game_end(self, summary: Dict[str, Any]) -> None:
        """Handles end of game logic."""
        total_pnl = summary.get('total_pnl', 0.0)
        print(f"Game over. Total PnL: {total_pnl:.2f}")