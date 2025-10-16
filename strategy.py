import numpy as np
from scipy.stats import norm
from typing import Dict, List, Any

# Import the abstract base class for the trading strategy
from autograder.sdk.strategy_interface import AbstractTradingStrategy

class MyTradingStrategy(AbstractTradingStrategy):
    """
    A strategy based on statistical forecasting and Bayesian updating.
    It continuously refines its estimate of the dice roll distribution as new
    data becomes available, and adjusts its trading confidence accordingly.
    It does not rely on market sentiment, only on statistical analysis.
    """

    def __init__(self):
        """Initializes the strategy's parameters."""
        super().__init__()
        # Extremely wide base spread - only trade with huge edge
        self.base_spread = 100.0
        # Minimal inventory management when we have edge
        self.inventory_multiplier = 0.05
        # Total number of dice rolls in a full round
        self.TOTAL_ROLLS = 20000
        # Only trade when we have EXTREME confidence (very late in the round)
        self.min_confidence_to_trade = 0.85
        # When we have massive edge, use extremely tight spreads
        self.aggressive_spread = 0.5

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
        """Called each sub-round to generate quotes based on updated statistical beliefs."""
        markets = {}
        products = marketplace.get_products()
        
        # Combine historical and current data for the most up-to-date statistical sample.
        all_known_rolls = training_rolls + current_rolls
        
        if not all_known_rolls:
            # Not enough data to form a belief, stay out of the market.
            return {}

        # --- Update Statistical Beliefs ---
        updated_mean = np.mean(all_known_rolls)
        updated_std = np.std(all_known_rolls) if len(all_known_rolls) > 1 else 2886.0
        
        # --- Confidence-Based Spread Adjustment ---
        # Our confidence increases as we see more of the actual rolls.
        num_current_rolls = len(current_rolls)
        # Confidence grows from 0 to 1 as the round progresses.
        confidence_level = num_current_rolls / self.TOTAL_ROLLS
        
        # Only trade if we have EXTREME confidence (very late in round)
        if confidence_level < self.min_confidence_to_trade:
            return {}
        
        # When we have massive edge, use extremely tight spreads to capture it aggressively
        if confidence_level > 0.9:
            dynamic_spread = self.aggressive_spread
        else:
            # Still use wide spreads even with high confidence
            dynamic_spread = self.base_spread

        for product in products:
            fair_value = self.calculate_fair_value(
                product.id, 
                updated_mean, 
                updated_std,
                round_info
            )
            
            if fair_value is not None:
                # Use marketplace.my_trades.get_position() directly instead of iterating
                try:
                    position = marketplace.my_trades.get_position(product.id)
                except:
                    position = 0.0
                
                # Only trade if we have a tiny position (avoid any significant losses)
                if abs(position) > 1:
                    continue
                
                # Quadratic inventory penalty to aggressively manage risk
                inventory_adjustment = np.sign(position) * (position**2) * self.inventory_multiplier
                adjusted_fair_value = fair_value - inventory_adjustment

                bid_price = adjusted_fair_value - dynamic_spread / 2
                ask_price = adjusted_fair_value + dynamic_spread / 2
                
                # Ensure bid < ask and both are positive
                if bid_price < ask_price and bid_price > 0:
                    markets[product.id] = (round(bid_price, 1), round(ask_price, 1))

        return markets

    def calculate_fair_value(self, product_id: str, mean: float, std: float, round_info: Any) -> float | None:
        """
        Calculates the theoretical fair value based on the LATEST statistical estimates.
        """
        try:
            parts = product_id.split(',')
            if len(parts) < 2:
                return None
            product_type = parts[1]
            
            # The underlying price is our best forecast of the final sum.
            underlying_price = mean * self.TOTAL_ROLLS

            if product_type == 'F':
                return underlying_price
            
            elif product_type in ['C', 'P']:
                if len(parts) < 3:
                    return None
                strike = float(parts[2])
                # Use std directly per requested spec
                volatility = std
                
                total_subrounds = round_info.get("num_sub_rounds", 10) if hasattr(round_info, 'get') else 10
                current_sub_round = round_info.get("current_sub_round", 1) if hasattr(round_info, 'get') else 1
                time_to_expiry = max(1e-6, (total_subrounds - current_sub_round + 1) / total_subrounds)

                if volatility <= 0 or time_to_expiry <= 0:
                    return None # Cannot price options without volatility or time

                d1 = (np.log(underlying_price / strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
                d2 = d1 - volatility * np.sqrt(time_to_expiry)

                if product_type == 'C':
                    return underlying_price * norm.cdf(d1) - strike * norm.cdf(d2)
                else: # Put
                    return strike * norm.cdf(-d2) - underlying_price * norm.cdf(-d1)
            
            return None
        except (ValueError, IndexError, ZeroDivisionError, OverflowError):
            return None

    def on_round_end(self, result: Dict[str, Any]) -> None:
        """Handles end of round logic."""
        pnl = result.get('your_pnl', 0.0)
        print(f"Round ended. PnL: {pnl:.2f}")

    def on_game_end(self, summary: Dict[str, Any]) -> None:
        """Handles end of game logic."""
        total_pnl = summary.get('total_pnl', 0.0)
        print(f"Game over. Total PnL: {total_pnl:.2f}")
