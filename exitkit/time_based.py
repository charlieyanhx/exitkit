"""
Time-based exit models for position management.

This module implements various time-based exit strategies including fixed
holding periods, time decay exits, and adaptive time-based exits based on
market conditions and position performance.
"""

import time

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from .types import WindowTensor, SignalOutput
from .marketdata import require, require_entry
from .types import Position
from .base import TimeBasedExitModel


class FixedTimeExitModel(TimeBasedExitModel):
    """Exit after fixed holding period"""
    
    def __init__(self, max_holding_hours: float = 24.0):
        """
        Initialize fixed time exit model
        
        Args:
            max_holding_hours: Maximum holding period in hours
        """
        super().__init__("FixedTimeExit", max_holding_hours)
        self.logger = logging.getLogger(__name__)
    
    def generate_exit_signals(self, open_positions: List[Position],
                             market_data: Dict,
                             W_t: Optional[WindowTensor] = None) -> List[SignalOutput]:
        """Generate fixed time exit signals"""
        exit_signals = []
        
        for position in open_positions:
            should_exit, holding_hours = self.check_time_exit(position)
            
            if should_exit:
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,
                    kappa=holding_hours / self.max_holding_hours,
                    confidence=0.7,  # Medium confidence on time exit
                    horizon=0.0,
                    position_id=position.position_id,
                    exit_reason='time_limit',
                    meta={
                        'holding_hours': holding_hours,
                        'max_holding_hours': self.max_holding_hours,
                        'entry_time': position.entry_time,
                        'current_pnl': position.current_pnl,
                        'pnl_pct': position.get_pnl_pct()
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated fixed time exit for position {position.position_id}: "
                                f"holding={holding_hours:.1f}h")
        
        return exit_signals


class TimeDecayExitModel(TimeBasedExitModel):
    """Exit based on time decay (for options)"""
    
    def __init__(self, max_holding_hours: float = 24.0,
                 time_decay_threshold: float = 0.5):
        """
        Initialize time decay exit model
        
        Args:
            max_holding_hours: Maximum holding period in hours
            time_decay_threshold: Time decay threshold for exit
        """
        super().__init__("TimeDecayExit", max_holding_hours)
        self.time_decay_threshold = time_decay_threshold
        
        self.parameters.update({
            'time_decay_threshold': time_decay_threshold
        })
        
        self.logger = logging.getLogger(__name__)
    
    def _calculate_time_decay(self, position: Position, 
                             market_data: Dict) -> float:
        """Calculate time decay factor"""
        holding_hours = position.get_holding_hours()
        
        # Time decay increases with holding period
        time_decay = 1 - np.exp(-holding_hours / 24.0)  # Daily decay
        
        return time_decay
    
    def generate_exit_signals(self, open_positions: List[Position],
                             market_data: Dict,
                             W_t: Optional[WindowTensor] = None) -> List[SignalOutput]:
        """Generate time decay exit signals"""
        exit_signals = []
        
        for position in open_positions:
            # Check fixed time limit first
            should_exit_time, holding_hours = self.check_time_exit(position)
            
            # Check time decay threshold
            time_decay = self._calculate_time_decay(position, market_data)
            should_exit_decay = time_decay >= self.time_decay_threshold
            
            if should_exit_time or should_exit_decay:
                exit_reason = 'time_decay' if should_exit_decay else 'time_limit'
                
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,
                    kappa=max(holding_hours / self.max_holding_hours, 
                             time_decay / self.time_decay_threshold),
                    confidence=0.8,
                    horizon=0.0,
                    position_id=position.position_id,
                    exit_reason=exit_reason,
                    meta={
                        'holding_hours': holding_hours,
                        'max_holding_hours': self.max_holding_hours,
                        'time_decay': time_decay,
                        'time_decay_threshold': self.time_decay_threshold,
                        'entry_time': position.entry_time,
                        'current_pnl': position.current_pnl
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated time decay exit for position {position.position_id}: "
                                f"holding={holding_hours:.1f}h, decay={time_decay:.3f}")
        
        return exit_signals


class AdaptiveTimeExitModel(TimeBasedExitModel):
    """Adaptive time exit based on market conditions"""
    
    def __init__(self, base_holding_hours: float = 24.0,
                 volatility_adjustment: float = 0.5,
                 performance_adjustment: float = 0.3):
        """
        Initialize adaptive time exit model
        
        Args:
            base_holding_hours: Base holding period in hours
            volatility_adjustment: Volatility adjustment factor
            performance_adjustment: Performance adjustment factor
        """
        super().__init__("AdaptiveTimeExit", base_holding_hours)
        self.base_holding_hours = base_holding_hours
        self.volatility_adjustment = volatility_adjustment
        self.performance_adjustment = performance_adjustment
        
        self.parameters.update({
            'base_holding_hours': base_holding_hours,
            'volatility_adjustment': volatility_adjustment,
            'performance_adjustment': performance_adjustment
        })
        
        self.logger = logging.getLogger(__name__)
    
    def _calculate_adaptive_holding_period(self, position: Position, 
                                         market_data: Dict) -> float:
        """Calculate adaptive holding period based on market conditions"""
        # Get current market conditions
        current_vol = require(market_data, 'implied_vol')
        entry_vol = require_entry(position, 'implied_vol')
        
        # Volatility adjustment
        vol_ratio = current_vol / entry_vol if entry_vol > 0 else 1.0
        vol_adjustment = 1 + (vol_ratio - 1) * self.volatility_adjustment
        
        # Performance adjustment
        pnl_pct = position.get_pnl_pct()
        perf_adjustment = 1 + pnl_pct * self.performance_adjustment
        
        # Calculate adaptive holding period
        adaptive_holding = (self.base_holding_hours * 
                           vol_adjustment * perf_adjustment)
        
        return adaptive_holding
    
    def generate_exit_signals(self, open_positions: List[Position],
                             market_data: Dict,
                             W_t: Optional[WindowTensor] = None) -> List[SignalOutput]:
        """Generate adaptive time exit signals"""
        exit_signals = []
        
        for position in open_positions:
            # Calculate adaptive holding period
            adaptive_holding = self._calculate_adaptive_holding_period(position, market_data)
            holding_hours = position.get_holding_hours()
            
            # Check if adaptive time limit exceeded
            should_exit = holding_hours >= adaptive_holding
            
            if should_exit:
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,
                    kappa=holding_hours / adaptive_holding,
                    confidence=0.7,
                    horizon=0.0,
                    position_id=position.position_id,
                    exit_reason='adaptive_time_limit',
                    meta={
                        'holding_hours': holding_hours,
                        'adaptive_holding_hours': adaptive_holding,
                        'base_holding_hours': self.base_holding_hours,
                        'vol_adjustment': require(market_data, 'implied_vol') / require_entry(position, 'implied_vol'),
                        'perf_adjustment': 1 + position.get_pnl_pct() * self.performance_adjustment,
                        'entry_time': position.entry_time,
                        'current_pnl': position.current_pnl
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated adaptive time exit for position {position.position_id}: "
                                f"holding={holding_hours:.1f}h, adaptive={adaptive_holding:.1f}h")
        
        return exit_signals


class MarketHoursExitModel(TimeBasedExitModel):
    """Exit based on market hours and trading sessions"""
    
    def __init__(self, max_holding_hours: float = 24.0,
                 market_close_exit: bool = True,
                 weekend_exit: bool = True):
        """
        Initialize market hours exit model
        
        Args:
            max_holding_hours: Maximum holding period in hours
            market_close_exit: Exit at market close
            weekend_exit: Exit before weekend
        """
        super().__init__("MarketHoursExit", max_holding_hours)
        self.market_close_exit = market_close_exit
        self.weekend_exit = weekend_exit
        
        self.parameters.update({
            'market_close_exit': market_close_exit,
            'weekend_exit': weekend_exit
        })
        
        self.logger = logging.getLogger(__name__)
    
    def _is_market_close_time(self, current_time: float) -> bool:
        """Check if current time is near market close"""
        # Simple implementation - in practice, would use proper market hours
        import datetime
        dt = datetime.datetime.fromtimestamp(current_time)
        
        # Market close at 4:00 PM ET (simplified)
        market_close_hour = 16
        return dt.hour >= market_close_hour - 1  # 1 hour before close
    
    def _is_weekend(self, current_time: float) -> bool:
        """Check if current time is weekend"""
        import datetime
        dt = datetime.datetime.fromtimestamp(current_time)
        return dt.weekday() >= 5  # Saturday = 5, Sunday = 6
    
    def generate_exit_signals(self, open_positions: List[Position],
                             market_data: Dict,
                             W_t: Optional[WindowTensor] = None) -> List[SignalOutput]:
        """Generate market hours exit signals"""
        exit_signals = []
        current_time = require(market_data, 'timestamp')
        
        for position in open_positions:
            should_exit_time, holding_hours = self.check_time_exit(position)
            
            # Check market close exit
            should_exit_market_close = (self.market_close_exit and 
                                     self._is_market_close_time(current_time))
            
            # Check weekend exit
            should_exit_weekend = (self.weekend_exit and 
                                 self._is_weekend(current_time))
            
            if should_exit_time or should_exit_market_close or should_exit_weekend:
                if should_exit_market_close:
                    exit_reason = 'market_close'
                elif should_exit_weekend:
                    exit_reason = 'weekend'
                else:
                    exit_reason = 'time_limit'
                
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,
                    kappa=holding_hours / self.max_holding_hours,
                    confidence=0.8,
                    horizon=0.0,
                    position_id=position.position_id,
                    exit_reason=exit_reason,
                    meta={
                        'holding_hours': holding_hours,
                        'max_holding_hours': self.max_holding_hours,
                        'market_close_exit': should_exit_market_close,
                        'weekend_exit': should_exit_weekend,
                        'current_time': current_time,
                        'entry_time': position.entry_time,
                        'current_pnl': position.current_pnl
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated market hours exit for position {position.position_id}: "
                                f"reason={exit_reason}")
        
        return exit_signals


class PerformanceTimeExitModel(TimeBasedExitModel):
    """Time exit based on position performance"""
    
    def __init__(self, max_holding_hours: float = 24.0,
                 profit_time_reduction: float = 0.5,
                 loss_time_extension: float = 1.5):
        """
        Initialize performance-based time exit model
        
        Args:
            max_holding_hours: Base maximum holding period
            profit_time_reduction: Time reduction factor for profitable positions
            loss_time_extension: Time extension factor for losing positions
        """
        super().__init__("PerformanceTimeExit", max_holding_hours)
        self.profit_time_reduction = profit_time_reduction
        self.loss_time_extension = loss_time_extension
        
        self.parameters.update({
            'profit_time_reduction': profit_time_reduction,
            'loss_time_extension': loss_time_extension
        })
        
        self.logger = logging.getLogger(__name__)
    
    def _calculate_performance_adjusted_time(self, position: Position) -> float:
        """Calculate performance-adjusted holding time"""
        pnl_pct = position.get_pnl_pct()
        
        if pnl_pct > 0:  # Profitable position
            # Reduce holding time for profitable positions
            adjusted_time = self.max_holding_hours * self.profit_time_reduction
        else:  # Losing position
            # Extend holding time for losing positions
            adjusted_time = self.max_holding_hours * self.loss_time_extension
        
        return adjusted_time
    
    def generate_exit_signals(self, open_positions: List[Position],
                             market_data: Dict,
                             W_t: Optional[WindowTensor] = None) -> List[SignalOutput]:
        """Generate performance-based time exit signals"""
        exit_signals = []
        
        for position in open_positions:
            # Calculate performance-adjusted time
            adjusted_time = self._calculate_performance_adjusted_time(position)
            holding_hours = position.get_holding_hours()
            
            # Check if performance-adjusted time limit exceeded
            should_exit = holding_hours >= adjusted_time
            
            if should_exit:
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,
                    kappa=holding_hours / adjusted_time,
                    confidence=0.7,
                    horizon=0.0,
                    position_id=position.position_id,
                    exit_reason='performance_time_limit',
                    meta={
                        'holding_hours': holding_hours,
                        'adjusted_holding_hours': adjusted_time,
                        'base_holding_hours': self.max_holding_hours,
                        'pnl_pct': position.get_pnl_pct(),
                        'profit_time_reduction': self.profit_time_reduction,
                        'loss_time_extension': self.loss_time_extension,
                        'entry_time': position.entry_time,
                        'current_pnl': position.current_pnl
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated performance time exit for position {position.position_id}: "
                                f"holding={holding_hours:.1f}h, adjusted={adjusted_time:.1f}h, "
                                f"pnl={position.get_pnl_pct():.2%}")
        
        return exit_signals
