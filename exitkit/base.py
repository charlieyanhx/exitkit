"""
Base exit signal models for the signal layer.

This module provides abstract base classes and interfaces for exit signal
generation, supporting various exit strategies including convergence-based,
stop-loss, take-profit, time-based, volatility-based, and signal reversal exits.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import logging

from .types import WindowTensor, SignalOutput
from .marketdata import require, require_entry
from .types import Position


class ExitSignalModel(ABC):
    """Abstract base class for exit signal generation"""
    
    def __init__(self, name: str = "ExitSignalModel"):
        self.name = name
        self.parameters = {}
        self.exit_history = []
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def generate_exit_signals(self, open_positions: List[Position],
                             market_data: Dict,
                             W_t: Optional[WindowTensor] = None) -> List[SignalOutput]:
        """
        Generate exit signals for open positions
        
        Args:
            open_positions: Positions to evaluate for exit
            market_data: Current market data. Fields a model needs are
                required - missing ones raise MissingMarketData rather than
                defaulting to a fabricated value.
            W_t: Optional feature window. No model in this package reads it
                today; it is kept so window-aware models can be added without
                changing the interface.
            
        Returns:
            List of exit SignalOutput objects
        """
        pass
    
    def should_exit(self, position: Position, 
                   current_data: Dict) -> Tuple[bool, str]:
        """
        Check if position should be exited
        
        Args:
            position: Position to evaluate
            current_data: Current market data
            
        Returns:
            (should_exit, exit_reason)
        """
        # Default implementation - override in subclasses
        return False, "no_exit"
    
    def update_performance(self, exit_signal: SignalOutput, 
                          actual_pnl: float):
        """Update model performance based on exit outcome"""
        self.exit_history.append({
            'exit_signal': exit_signal,
            'actual_pnl': actual_pnl,
            'timestamp': len(self.exit_history)
        })
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get exit model performance statistics"""
        if not self.exit_history:
            return {
                'total_exits': 0,
                'avg_pnl': 0.0,
                'success_rate': 0.0,
                'total_pnl': 0.0
            }
        
        pnls = [entry['actual_pnl'] for entry in self.exit_history]
        successes = [1 if pnl > 0 else 0 for pnl in pnls]
        
        return {
            'total_exits': len(self.exit_history),
            'avg_pnl': sum(pnls) / len(pnls),
            'success_rate': sum(successes) / len(successes),
            'total_pnl': sum(pnls),
            'min_pnl': min(pnls),
            'max_pnl': max(pnls)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and parameters"""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'performance': self.get_performance_stats()
        }
    
    def reset_performance(self):
        """Reset performance tracking"""
        self.exit_history.clear()
        self.logger.info(f"Reset performance for {self.name}")


class ThresholdExitModel(ExitSignalModel):
    """Base class for threshold-based exit models"""
    
    def __init__(self, name: str, threshold: float, 
                 comparison_type: str = 'greater'):
        """
        Initialize threshold exit model
        
        Args:
            name: Model name
            threshold: Threshold value
            comparison_type: 'greater', 'less', 'abs_greater', 'abs_less'
        """
        super().__init__(name)
        self.threshold = threshold
        self.comparison_type = comparison_type
        self.parameters = {
            'threshold': threshold,
            'comparison_type': comparison_type
        }
    
    def check_threshold(self, value: float) -> bool:
        """Check if value meets threshold condition"""
        if self.comparison_type == 'greater':
            return value > self.threshold
        elif self.comparison_type == 'less':
            return value < self.threshold
        elif self.comparison_type == 'abs_greater':
            return abs(value) > self.threshold
        elif self.comparison_type == 'abs_less':
            return abs(value) < self.threshold
        else:
            return False


class TimeBasedExitModel(ExitSignalModel):
    """Base class for time-based exit models"""
    
    def __init__(self, name: str, max_holding_hours: float):
        """
        Initialize time-based exit model
        
        Args:
            name: Model name
            max_holding_hours: Maximum holding period in hours
        """
        super().__init__(name)
        self.max_holding_hours = max_holding_hours
        self.parameters = {
            'max_holding_hours': max_holding_hours
        }
    
    def check_time_exit(self, position: Position,
                        now: Optional[float] = None) -> Tuple[bool, float]:
        """
        Check if position should exit based on time
        
        Returns:
            (should_exit, holding_hours)
        """
        holding_hours = position.get_holding_hours(now)
        should_exit = holding_hours >= self.max_holding_hours
        return should_exit, holding_hours


class PnLBasedExitModel(ExitSignalModel):
    """Base class for P&L-based exit models"""
    
    def __init__(self, name: str, pnl_threshold: float, 
                 threshold_type: str = 'percentage'):
        """
        Initialize P&L-based exit model
        
        Args:
            name: Model name
            pnl_threshold: P&L threshold
            threshold_type: 'percentage' or 'absolute'
        """
        super().__init__(name)
        self.pnl_threshold = pnl_threshold
        self.threshold_type = threshold_type
        self.parameters = {
            'pnl_threshold': pnl_threshold,
            'threshold_type': threshold_type
        }
    
    def check_pnl_exit(self, position: Position) -> Tuple[bool, float]:
        """
        Check if position should exit based on P&L
        
        Returns:
            (should_exit, pnl_value)
        """
        if self.threshold_type == 'percentage':
            pnl_value = position.get_pnl_pct()
        else:  # absolute
            pnl_value = position.current_pnl
        
        should_exit = abs(pnl_value) >= abs(self.pnl_threshold)
        return should_exit, pnl_value


class VolatilityBasedExitModel(ExitSignalModel):
    """Base class for volatility-based exit models"""
    
    def __init__(self, name: str, vol_threshold: float, 
                 vol_type: str = 'change_pct'):
        """
        Initialize volatility-based exit model
        
        Args:
            name: Model name
            vol_threshold: Volatility threshold
            vol_type: 'change_pct', 'absolute', 'ratio'
        """
        super().__init__(name)
        self.vol_threshold = vol_threshold
        self.vol_type = vol_type
        self.parameters = {
            'vol_threshold': vol_threshold,
            'vol_type': vol_type
        }
    
    def check_volatility_exit(self, position: Position, 
                             current_vol: float) -> Tuple[bool, float]:
        """
        Check if position should exit based on volatility
        
        Returns:
            (should_exit, vol_metric)
        """
        entry_vol = require_entry(position, 'implied_vol')
        
        if self.vol_type == 'change_pct':
            vol_metric = abs(current_vol - entry_vol) / entry_vol
        elif self.vol_type == 'absolute':
            vol_metric = abs(current_vol - entry_vol)
        elif self.vol_type == 'ratio':
            vol_metric = current_vol / entry_vol
        else:
            vol_metric = 0.0
        
        should_exit = vol_metric >= self.vol_threshold
        return should_exit, vol_metric


class SignalBasedExitModel(ExitSignalModel):
    """Base class for signal-based exit models"""
    
    def __init__(self, name: str, reversal_threshold: float):
        """
        Initialize signal-based exit model
        
        Args:
            name: Model name
            reversal_threshold: Threshold for signal reversal
        """
        super().__init__(name)
        self.reversal_threshold = reversal_threshold
        self.entry_signal_model = None  # Reference to entry model
        self.parameters = {
            'reversal_threshold': reversal_threshold
        }
    
    def set_entry_model(self, entry_model):
        """Set reference to entry signal model"""
        self.entry_signal_model = entry_model
        self.logger.info(f"Set entry model for {self.name}")
    
    def check_signal_reversal(self, position: Position, 
                             current_signals: List[SignalOutput]) -> Tuple[bool, Optional[SignalOutput]]:
        """
        Check if any current signal reverses the entry
        
        Returns:
            (should_exit, reversal_signal)
        """
        entry_direction = position.entry_signal.direction
        
        for signal in current_signals:
            if (signal.direction == -entry_direction and 
                signal.kappa >= self.reversal_threshold):
                return True, signal
        
        return False, None
