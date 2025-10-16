import numpy as np
from scipy.stats import norm
from typing import Dict, List, Any

# Import the abstract base class for the trading strategy
from autograder.sdk.strategy_interface import AbstractTradingStrategy

class MyTradingStrategy(AbstractTradingStrategy):
    """
    An advanced trading strategy that incorporates market sentiment, inventory risk,
    and a more sophisticated volatility model. It dynamically adjusts its quotes 
    to adapt to changing market conditions and manage risk.
    """

    def __init__(self):
        """Initializes the strategy with fine-tuned parameters."""
        super().__init__()
        # Starting spread, will be adjusted based on volatility
        self.base_spread = 4.0  
        # Multiplier for adjusting quotes based on inventory risk
        self.inventory_multiplier = 0.07
        # Threshold for detecting significant order flow imbalance
        self.sentiment_threshold = 1.5
        # Factor to adjust the fair value based on market sentiment
        self.sentiment_adjustment_factor = 3.0
        # EMA weight, giving more importance to recent trades
        self.ema_alpha = 0.4
        # Tracks the exponential moving average of net order flow
        self.net_order_flow_ema = 0.0
        # Risk factor to adjust spread based on market volatility
        self.volatility_spread_multiplier = 1.2

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
        """
        Main logic loop called each sub-round to generate quotes.
        """
        markets = {}
        current_sub_round = round_info.get("current_sub_round", 1)
        
        # --- Market Volatility Analysis ---
        # A simple measure of recent price volatility
        try:
            all_trades = marketplace.get_trade_history()
            recent_trade_prices = [t.price for t in all_trades[-30:] if hasattr(t, 'price') and t.price]
            market_volatility = np.std(recent_trade_prices) if len(recent_trade_prices) > 1 else 1.0
        except:
            market_volatility = 1.0
        
        # Dynamically adjust spread based on volatility
        dynamic_spread = self.base_spread + market_volatility * self.volatility_spread_multiplier

        # --- Market Sentiment Analysis ---
        try:
            recent_trades = all_trades[-20:] if 'all_trades' in locals() else []
            net_order_flow = 0.0
            for t in recent_trades:
                if hasattr(t, 'side') and t.side == 'BUY':
                    net_order_flow += getattr(t, 'quantity', 0)
                elif hasattr(t, 'side') and t.side == 'SELL':
                    net_order_flow -= getattr(t, 'quantity', 0)
        except:
            net_order_flow = 0.0
            
        self.net_order_flow_ema = self.ema_alpha * net_order_flow + (1 - self.ema_alpha) * self.net_order_flow_ema

        sentiment_adjustment = 0
        if self.net_order_flow_ema > self.sentiment_threshold:
            sentiment_adjustment = self.sentiment_adjustment_factor
        elif self.net_order_flow_ema < -self.sentiment_threshold:
            sentiment_adjustment = -self.sentiment_adjustment_factor

        # --- Calculate Expected Value from Training Data ---
        expected_roll = np.mean(training_rolls) if training_rolls else 3.5
        total_rolls_per_round = 20000
        total_subrounds = round_info.get("num_sub_rounds", 10)

        # --- Inventory Calculation ---
        inventory = {}
        products = marketplace.get_products()
        for p in products:
            inventory[p.product_id] = 0.0
        
        for trade in my_trades:
            if hasattr(trade, 'buyer_id') and trade.buyer_id == self.team_name:
                inventory[trade.product_id] += getattr(trade, 'quantity', 0)
            elif hasattr(trade, 'seller_id') and trade.seller_id == self.team_name:
                inventory[trade.product_id] -= getattr(trade, 'quantity', 0)

        for product in products:
            product_id = product.product_id
            fair_value = self.calculate_fair_value(product_id, training_rolls, round_info, expected_roll, total_rolls_per_round, total_subrounds)
            
            if fair_value is not None:
                # 1. Start with base fair value
                adjusted_fair_value = fair_value
                
                # 2. Adjust for market sentiment
                adjusted_fair_value += sentiment_adjustment
                
                # 3. Adjust for inventory risk
                position = inventory.get(product_id, 0.0)
                inventory_adjustment = position * self.inventory_multiplier
                adjusted_fair_value -= inventory_adjustment

                # 4. Set final quotes with dynamic spread
                bid_price = adjusted_fair_value - dynamic_spread / 2
                ask_price = adjusted_fair_value + dynamic_spread / 2
                
                # Ensure bid < ask and both are positive
                if bid_price < ask_price and bid_price > 0:
                    markets[product_id] = (round(bid_price, 1), round(ask_price, 1))

        return markets

    def calculate_fair_value(self, product_id: str, training_rolls: List[int], round_info: Dict[str, Any], expected_roll: float, total_rolls_per_round: int, total_subrounds: int) -> float | None:
        """
        Calculates the theoretical fair value using a more robust model.
        """
        try:
            parts = product_id.split(',')
            if len(parts) < 2:
                return None
            product_type = parts[1]

            if product_type == 'F': # Future
                if len(parts) < 3:
                    return None
                num_subrounds = int(parts[2])
                # Fair value is the expected sum of rolls
                return expected_roll * (total_rolls_per_round / total_subrounds * num_subrounds)
            
            elif product_type in ['C', 'P']: # Options
                if len(parts) < 3:
                    return None
                strike = float(parts[2])
                
                # Enhanced volatility: use training data std deviation
                volatility = np.std(training_rolls) / 10000 if training_rolls and np.std(training_rolls) > 0 else 0.05
                
                underlying_price = expected_roll * total_rolls_per_round
                
                # Time to expiry decreases as the round progresses
                current_sub_round = round_info.get("current_sub_round", 1)
                time_to_expiry = (total_subrounds - current_sub_round + 1) / total_subrounds
                
                if time_to_expiry <= 0: 
                    time_to_expiry = 0.001 # Avoid division by zero at expiry

                # Black-Scholes-Merton model for option pricing
                d1 = (np.log(underlying_price / strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
                d2 = d1 - volatility * np.sqrt(time_to_expiry)

                if product_type == 'C': # Call Option
                    price = (underlying_price * norm.cdf(d1) - strike * np.exp(0) * norm.cdf(d2))
                    return price
                else: # Put Option
                    price = (strike * np.exp(0) * norm.cdf(-d2) - underlying_price * norm.cdf(-d1))
                    return price
            
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