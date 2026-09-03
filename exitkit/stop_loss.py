"""
Stop-loss exit models for risk management.

This module implements various stop-loss exit strategies including fixed
stop-loss, trailing stop-loss, and adaptive stop-loss based on volatility.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from .types import WindowTensor, SignalOutput
from .marketdata import require, require_entry
from .types import Position
from .base import PnLBasedExitModel


class StopLossExitModel(PnLBasedExitModel):
    """Exit on adverse price movements"""
    
    def __init__(self, stop_loss_pct: float = 0.02, trailing: bool = False,
                 trailing_activation_pct: float = 0.01):
        """
        Initialize stop-loss exit model
        
        Args:
            stop_loss_pct: Stop loss percentage (default 2%)
            trailing: Whether to use trailing stop loss
            trailing_activation_pct: Activation threshold for trailing stop
        """
        super().__init__(
            name="StopLossExit",
            pnl_threshold=-stop_loss_pct,  # Negative for losses
            threshold_type='percentage'
        )
        self.trailing = trailing
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_stops = {}  # Track trailing stops per position
        
        self.parameters.update({
            'trailing': trailing,
            'trailing_activation_pct': trailing_activation_pct
        })
        
        self.logger = logging.getLogger(__name__)
    
    def generate_exit_signals(self, W_t: WindowTensor, 
                             open_positions: List[Position],
                             market_data: Dict) -> List[SignalOutput]:
        """Generate stop-loss exit signals"""
        exit_signals = []
        current_price = require(market_data, 'spot_price')
        
        for position in open_positions:
            position.update_pnl(current_price)
            
            if self.trailing:
                # Update trailing stop
                self._update_trailing_stop(position, current_price)
            
            # Check if stop loss hit
            should_exit, pnl_pct = self.check_pnl_exit(position)
            
            if should_exit and pnl_pct <= -abs(self.pnl_threshold):
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,
                    kappa=abs(pnl_pct) / abs(self.pnl_threshold),
                    confidence=1.0,  # High confidence on stop loss
                    horizon=0.0,
                    position_id=position.position_id,
                    exit_reason='stop_loss',
                    meta={
                        'loss_pct': pnl_pct,
                        'stop_loss_pct': abs(self.pnl_threshold),
                        'entry_price': position.entry_price,
                        'current_price': current_price,
                        'trailing': self.trailing,
                        'holding_hours': position.get_holding_hours()
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated stop-loss exit for position {position.position_id}: "
                                f"loss={pnl_pct:.2%}")
        
        return exit_signals
    
    def _update_trailing_stop(self, position: Position, current_price: float):
        """Update trailing stop for position"""
        pos_id = position.position_id
        pnl_pct = position.get_pnl_pct()
        
        # Initialize trailing stop if not exists
        if pos_id not in self.trailing_stops:
            self.trailing_stops[pos_id] = {
                'highest_pnl': 0.0,
                'trailing_stop': -abs(self.pnl_threshold)
            }
        
        trailing_data = self.trailing_stops[pos_id]
        
        # Update highest P&L if position is profitable
        if pnl_pct > trailing_data['highest_pnl']:
            trailing_data['highest_pnl'] = pnl_pct
            
            # Update trailing stop if we've moved favorably
            if pnl_pct >= self.trailing_activation_pct:
                # Trail the stop loss behind the highest profit
                trailing_data['trailing_stop'] = max(
                    trailing_data['trailing_stop'],
                    pnl_pct - abs(self.pnl_threshold)
                )
        
        # Check if we've hit the trailing stop
        if pnl_pct <= trailing_data['trailing_stop']:
            # Reset trailing stop data
            del self.trailing_stops[pos_id]
    
    def get_trailing_stop_info(self, position_id: str) -> Optional[Dict[str, float]]:
        """Get trailing stop information for position"""
        return self.trailing_stops.get(position_id)
    
    def reset_trailing_stops(self):
        """Reset all trailing stops"""
        self.trailing_stops.clear()
        self.logger.info("Reset all trailing stops")


class AdaptiveStopLossExitModel(StopLossExitModel):
    """Adaptive stop-loss based on volatility and market conditions"""
    
    def __init__(self, base_stop_loss_pct: float = 0.02, 
                 vol_multiplier: float = 2.0,
                 max_stop_loss_pct: float = 0.05):
        """
        Initialize adaptive stop-loss model
        
        Args:
            base_stop_loss_pct: Base stop loss percentage
            vol_multiplier: Volatility multiplier for stop loss
            max_stop_loss_pct: Maximum stop loss percentage
        """
        super().__init__(stop_loss_pct=base_stop_loss_pct, trailing=True)
        self.base_stop_loss_pct = base_stop_loss_pct
        self.vol_multiplier = vol_multiplier
        self.max_stop_loss_pct = max_stop_loss_pct
        
        self.parameters.update({
            'base_stop_loss_pct': base_stop_loss_pct,
            'vol_multiplier': vol_multiplier,
            'max_stop_loss_pct': max_stop_loss_pct
        })
        
        self.logger.info(f"Initialized adaptive stop-loss with base={base_stop_loss_pct:.1%}, "
                        f"vol_mult={vol_multiplier}, max={max_stop_loss_pct:.1%}")
    
    def _calculate_adaptive_stop_loss(self, position: Position, 
                                     market_data: Dict) -> float:
        """Calculate adaptive stop loss based on volatility"""
        # Get current volatility
        current_vol = require(market_data, 'implied_vol')
        entry_vol = require_entry(position, 'implied_vol')
        
        # Calculate volatility-adjusted stop loss
        vol_ratio = current_vol / entry_vol if entry_vol > 0 else 1.0
        adaptive_stop = self.base_stop_loss_pct * (1 + (vol_ratio - 1) * self.vol_multiplier)
        
        # Cap at maximum stop loss
        adaptive_stop = min(adaptive_stop, self.max_stop_loss_pct)
        
        return adaptive_stop
    
    def generate_exit_signals(self, W_t: WindowTensor, 
                             open_positions: List[Position],
                             market_data: Dict) -> List[SignalOutput]:
        """Generate adaptive stop-loss exit signals"""
        exit_signals = []
        current_price = require(market_data, 'spot_price')
        
        for position in open_positions:
            position.update_pnl(current_price)
            
            # Calculate adaptive stop loss
            adaptive_stop = self._calculate_adaptive_stop_loss(position, market_data)
            
            # Check if adaptive stop loss hit
            pnl_pct = position.get_pnl_pct()
            should_exit = pnl_pct <= -adaptive_stop
            
            if should_exit:
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,
                    kappa=abs(pnl_pct) / adaptive_stop,
                    confidence=1.0,
                    horizon=0.0,
                    position_id=position.position_id,
                    exit_reason='adaptive_stop_loss',
                    meta={
                        'loss_pct': pnl_pct,
                        'adaptive_stop_pct': adaptive_stop,
                        'base_stop_pct': self.base_stop_loss_pct,
                        'entry_price': position.entry_price,
                        'current_price': current_price,
                        'vol_ratio': require(market_data, 'implied_vol') / 
                                   require_entry(position, 'implied_vol')
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated adaptive stop-loss exit for position {position.position_id}: "
                                f"loss={pnl_pct:.2%}, adaptive_stop={adaptive_stop:.2%}")
        
        return exit_signals


class VolatilityStopLossExitModel(StopLossExitModel):
    """Stop-loss based on volatility breakouts"""
    
    def __init__(self, vol_breakout_threshold: float = 2.0,
                 stop_loss_multiplier: float = 1.5):
        """
        Initialize volatility-based stop-loss model
        
        Args:
            vol_breakout_threshold: Volatility breakout threshold (in standard deviations)
            stop_loss_multiplier: Multiplier for stop loss based on volatility
        """
        super().__init__(stop_loss_pct=0.02, trailing=False)
        self.vol_breakout_threshold = vol_breakout_threshold
        self.stop_loss_multiplier = stop_loss_multiplier
        
        self.parameters.update({
            'vol_breakout_threshold': vol_breakout_threshold,
            'stop_loss_multiplier': stop_loss_multiplier
        })
        
        self.logger.info(f"Initialized volatility stop-loss with threshold={vol_breakout_threshold}, "
                        f"multiplier={stop_loss_multiplier}")
    
    def _detect_volatility_breakout(self, position: Position, 
                                   market_data: Dict) -> Tuple[bool, float]:
        """Detect volatility breakout"""
        current_vol = require(market_data, 'implied_vol')
        entry_vol = require_entry(position, 'implied_vol')
        
        # Calculate volatility change in standard deviations
        vol_change = (current_vol - entry_vol) / entry_vol if entry_vol > 0 else 0.0
        vol_breakout = abs(vol_change) >= self.vol_breakout_threshold
        
        return vol_breakout, vol_change
    
    def generate_exit_signals(self, W_t: WindowTensor, 
                             open_positions: List[Position],
                             market_data: Dict) -> List[SignalOutput]:
        """Generate volatility-based stop-loss exit signals"""
        exit_signals = []
        current_price = require(market_data, 'spot_price')
        
        for position in open_positions:
            position.update_pnl(current_price)
            
            # Detect volatility breakout
            vol_breakout, vol_change = self._detect_volatility_breakout(position, market_data)
            
            if vol_breakout:
                # Calculate dynamic stop loss based on volatility
                dynamic_stop = abs(self.pnl_threshold) * self.stop_loss_multiplier
                pnl_pct = position.get_pnl_pct()
                
                # Exit if loss exceeds dynamic stop
                if pnl_pct <= -dynamic_stop:
                    exit_signal = SignalOutput(
                        signal_type='exit',
                        direction=0,
                        kappa=abs(pnl_pct) / dynamic_stop,
                        confidence=0.9,  # High confidence on volatility breakout
                        horizon=0.0,
                        position_id=position.position_id,
                        exit_reason='volatility_stop_loss',
                        meta={
                            'loss_pct': pnl_pct,
                            'dynamic_stop_pct': dynamic_stop,
                            'vol_change': vol_change,
                            'vol_breakout': vol_breakout,
                            'entry_price': position.entry_price,
                            'current_price': current_price
                        }
                    )
                    exit_signals.append(exit_signal)
                    
                    self.logger.debug(f"Generated volatility stop-loss exit for position {position.position_id}: "
                                    f"loss={pnl_pct:.2%}, vol_change={vol_change:.2%}")
        
        return exit_signals


class TimeDecayStopLossExitModel(StopLossExitModel):
    """Stop-loss that tightens over time (for options)"""
    
    def __init__(self, initial_stop_loss_pct: float = 0.03,
                 time_decay_rate: float = 0.1,
                 min_stop_loss_pct: float = 0.01):
        """
        Initialize time-decay stop-loss model
        
        Args:
            initial_stop_loss_pct: Initial stop loss percentage
            time_decay_rate: Rate at which stop loss tightens over time
            min_stop_loss_pct: Minimum stop loss percentage
        """
        super().__init__(stop_loss_pct=initial_stop_loss_pct, trailing=False)
        self.initial_stop_loss_pct = initial_stop_loss_pct
        self.time_decay_rate = time_decay_rate
        self.min_stop_loss_pct = min_stop_loss_pct
        
        self.parameters.update({
            'initial_stop_loss_pct': initial_stop_loss_pct,
            'time_decay_rate': time_decay_rate,
            'min_stop_loss_pct': min_stop_loss_pct
        })
        
        self.logger.info(f"Initialized time-decay stop-loss with initial={initial_stop_loss_pct:.1%}, "
                        f"decay_rate={time_decay_rate}, min={min_stop_loss_pct:.1%}")
    
    def _calculate_time_decay_stop_loss(self, position: Position) -> float:
        """Calculate time-decay adjusted stop loss"""
        holding_hours = position.get_holding_hours()
        
        # Stop loss tightens over time
        time_factor = np.exp(-self.time_decay_rate * holding_hours / 24.0)  # Daily decay
        dynamic_stop = self.initial_stop_loss_pct * time_factor
        
        # Cap at minimum stop loss
        dynamic_stop = max(dynamic_stop, self.min_stop_loss_pct)
        
        return dynamic_stop
    
    def generate_exit_signals(self, W_t: WindowTensor, 
                             open_positions: List[Position],
                             market_data: Dict) -> List[SignalOutput]:
        """Generate time-decay stop-loss exit signals"""
        exit_signals = []
        current_price = require(market_data, 'spot_price')
        
        for position in open_positions:
            position.update_pnl(current_price)
            
            # Calculate time-decay adjusted stop loss
            dynamic_stop = self._calculate_time_decay_stop_loss(position)
            pnl_pct = position.get_pnl_pct()
            
            # Exit if loss exceeds dynamic stop
            if pnl_pct <= -dynamic_stop:
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,
                    kappa=abs(pnl_pct) / dynamic_stop,
                    confidence=0.8,
                    horizon=0.0,
                    position_id=position.position_id,
                    exit_reason='time_decay_stop_loss',
                    meta={
                        'loss_pct': pnl_pct,
                        'dynamic_stop_pct': dynamic_stop,
                        'holding_hours': position.get_holding_hours(),
                        'time_factor': np.exp(-self.time_decay_rate * position.get_holding_hours() / 24.0),
                        'entry_price': position.entry_price,
                        'current_price': current_price
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated time-decay stop-loss exit for position {position.position_id}: "
                                f"loss={pnl_pct:.2%}, dynamic_stop={dynamic_stop:.2%}")
        
        return exit_signals
