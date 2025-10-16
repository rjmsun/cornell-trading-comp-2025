import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple, List, Any
from abc import ABC, abstractmethod

# Import the official AbstractTradingStrategy from the autograder's SDK
# This is the primary fix for the error you encountered.
from autograder.sdk.strategy_interface import AbstractTradingStrategy

# ---------------------------------------------------------------------------- #
#                          STRATEGY IMPLEMENTATION                             #
# ---------------------------------------------------------------------------- #

class MyTradingStrategy(AbstractTradingStrategy):
    """
    A quantitative trading strategy for the Cornell Derivatives Trading Competition.
    
    The strategy is built on three pillars:
    1.  Forecasting: Uses the Central Limit Theorem to model the final sum of
        dice rolls as a Normal distribution, dynamically updating the forecast
        as new rolls are observed.
    2.  Pricing: Calculates the theoretical fair value of futures and options
        using formulas derived for a normally distributed underlying.
    3.  Execution: Employs the Avellaneda-Stoikov market-making model to set
        optimal bid-ask quotes that manage inventory risk and capture spread.
    """

    def on_game_start(self, config: Dict[str, Any]) -> None:
        """
        Called once at the start of the game. Initializes strategy parameters.
        """
        self.team_name = config.get("team_name", "my_team")
        
        # --- Avellaneda-Stoikov Parameters ---
        # Risk aversion parameter. Higher gamma = wider spreads, more conservative.
        self.gamma = 0.1
        # Order book liquidity parameter. Higher kappa = tighter spreads.
        self.kappa = 1.5
        # --- Order-flow Sentiment Parameters ---
        self.ema_alpha = 0.3  # weight for recent trades in EMA
        self.order_flow_ema = 0.0
        self.sentiment_coeff = 0.002  # price shift per unit of EMA

        # Preserve base parameters so we can scale them dynamically each sub-round
        self._gamma_base = self.gamma
        self._kappa_base = self.kappa
        
        # --- Competition Constants ---
        self.TOTAL_ROLLS = 20000
        self.SUBROUNDS_PER_ROUND = 10
        self.ROLLS_PER_SUBROUND = 2000

        # --- Round-specific State ---
        # These will be reset at the start of each round's first subround.
        self.die_mean = 0.0
        self.die_variance = 0.0

    def _calculate_die_stats(self, rolls: List[int]) -> Tuple[float, float]:
        """
        Calculates the sample mean and unbiased sample variance of the die.
        """
        if not rolls:
            return 0.0, 0.0
        
        mean = np.mean(rolls)
        # Use ddof=1 for unbiased sample variance
        variance = np.var(rolls, ddof=1)
        return mean, variance

    def _calculate_call_fair_value(self, mu_sum: float, sigma_sum: float, strike: float) -> float:
        """
        Calculates the fair value of a European call option.
        Formula: (mu - K) * Phi(d) + sigma * phi(d)
        """
        if sigma_sum <= 1e-9:  # Avoid division by zero if variance is negligible
            return max(0.0, mu_sum - strike)
            
        d = (mu_sum - strike) / sigma_sum
        fair_value = (mu_sum - strike) * norm.cdf(d) + sigma_sum * norm.pdf(d)
        return fair_value

    def _calculate_put_fair_value(self, mu_sum: float, sigma_sum: float, strike: float) -> float:
        """
        Calculates the fair value of a European put option.
        Formula: (K - mu) * Phi(-d) + sigma * phi(-d)
        """
        if sigma_sum <= 1e-9: # Avoid division by zero
            return max(0.0, strike - mu_sum)

        d = (mu_sum - strike) / sigma_sum
        fair_value = (strike - mu_sum) * norm.cdf(-d) + sigma_sum * norm.pdf(-d)
        return fair_value

    def make_market(
        self,
        marketplace: Any,
        training_rolls: List[int],
        my_trades: List,
        current_rolls: List[int],
        round_info: Any
    ) -> Dict:
        """
        The core logic for making markets in each subround.
        """
        markets = {}
        current_sub_round = round_info.current_sub_round

        # 1. INITIALIZATION: On the first subround, analyze the training data.
        if current_sub_round == 1:
            self.die_mean, self.die_variance = self._calculate_die_stats(training_rolls)

        # 2. STATE UPDATE: Determine the current state of the round.
        num_obs_rolls = len(current_rolls)
        sum_obs_rolls = sum(current_rolls)
        num_rem_rolls = self.TOTAL_ROLLS - num_obs_rolls

        # 3. FORECAST UPDATE: Update the forecast for the final sum.
        # Mean of the sum of remaining rolls
        mu_rem = num_rem_rolls * self.die_mean
        # Variance of the sum of remaining rolls
        var_rem = num_rem_rolls * self.die_variance
        
        # Updated forecast for the final total sum
        mu_sum = sum_obs_rolls + mu_rem
        sigma_sum = np.sqrt(var_rem)

        # 4. INVENTORY CALCULATION: Calculate current inventory for each product.
        inventory = {}
        products = marketplace.get_products()
        for p in products:
            inventory[p.product_id] = 0.0
        
        for trade in my_trades:
            if trade.buyer_id == self.team_name:
                inventory[trade.product_id] += trade.quantity
            elif trade.seller_id == self.team_name:
                inventory[trade.product_id] -= trade.quantity

        # 5. ITERATE AND QUOTE: Loop through products and generate markets.
        time_remaining = (self.SUBROUNDS_PER_ROUND - current_sub_round + 1) / self.SUBROUNDS_PER_ROUND

        # --- ORDER-FLOW SENTIMENT UPDATE ------------------------------------ #
        try:
            # Prefer a dedicated API if available
            recent_trades = marketplace.get_recent_trades(limit=50)  # type: ignore[attr-defined]
        except AttributeError:
            # Fallback to entire history if helper not provided
            recent_trades = marketplace.get_trade_history()[-50:]  # type: ignore[attr-defined]

        net_flow = 0.0
        for t in recent_trades:
            # We assume trade objects expose side ("BUY"/"SELL") and quantity
            if getattr(t, "side", "") == "BUY":
                net_flow += t.quantity
            elif getattr(t, "side", "") == "SELL":
                net_flow -= t.quantity

        # Exponential moving average of net order flow
        self.order_flow_ema = self.ema_alpha * net_flow + (1 - self.ema_alpha) * self.order_flow_ema

        # Sentiment price shift (capped to avoid extremes)
        sentiment_shift = max(min(self.order_flow_ema * self.sentiment_coeff, 5.0), -5.0)

        # --- DYNAMIC PARAMETER ADJUSTMENTS ---------------------------------- #
        # Reduce risk-aversion as we approach the end of the round (so we quote tighter)
        self.gamma = self._gamma_base * (0.5 + 0.5 * time_remaining)
        # Increase assumed liquidity as we near the end (tighter spreads)
        self.kappa = self._kappa_base * (1.5 - 0.5 * time_remaining)

        for product in products:
            product_id = product.product_id
            parts = product_id.split(',')
            
            # --- Calculate Theoretical Fair Value (s) ---
            fair_value = 0.0
            if len(parts) < 2:
                continue  # malformed id
            product_type = parts[1]
            
            if product_type == 'F': # Future
                fair_value = mu_sum
            elif product_type in ['C', 'P']: # Call or Put Option
                try:
                    if len(parts) < 3:
                        continue
                    strike_price = float(parts[2])
                    if product_type == 'C':
                        fair_value = self._calculate_call_fair_value(mu_sum, sigma_sum, strike_price)
                    else: # 'P'
                        fair_value = self._calculate_put_fair_value(mu_sum, sigma_sum, strike_price)
                except (ValueError, IndexError):
                    continue # Skip malformed product IDs

            # --- Apply Avellaneda-Stoikov Model ---
            q = inventory.get(product_id, 0.0)
            s = fair_value + sentiment_shift  # apply sentiment bias
            # Use variance of the remaining sum as the volatility term
            sigma_sq_T_minus_t = var_rem 
            
            # Calculate Reservation Price (r)
            reservation_price = s - q * self.gamma * sigma_sq_T_minus_t
            
            # Calculate Optimal Spread
            spread_term_1 = self.gamma * sigma_sq_T_minus_t
            spread_term_2 = (2 / self.gamma) * np.log(1 + (self.gamma / self.kappa))
            optimal_spread = spread_term_1 + spread_term_2
            
            # Ensure spread is non-negative
            optimal_spread = max(optimal_spread, 0.01) # Set a minimum spread

            # --- Set Final Bid and Ask Quotes ---
            bid_price = reservation_price - (optimal_spread / 2)
            ask_price = reservation_price + (optimal_spread / 2)
            
            # Sanity check: bid must be less than ask
            if bid_price < ask_price and bid_price > 0:
                markets[product_id] = (round(bid_price, 2), round(ask_price, 2))

        return markets

    def on_round_end(self, result: Dict[str, Any]) -> None:
        """
        Called at the end of each round. Can be used for analysis.
        """
        # Reset round-specific state for the next round
        self.die_mean = 0.0
        self.die_variance = 0.0
        print(f"Round ended. PnL: {result.get('your_pnl')}")

    def on_game_end(self, summary: Dict[str, Any]) -> None:
        """
        Called at the end of the game.
        """
        print(f"Game ended. Final Score: {summary.get('final_score')}")