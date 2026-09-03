"""
Take-profit exit models for profit realization.

This module implements various take-profit exit strategies including fixed
take-profit, partial profit taking, and adaptive take-profit based on
market conditions and position performance.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from .types import WindowTensor, SignalOutput
from .marketdata import require, require_entry
from .types import Position
from .base import PnLBasedExitModel


class TakeProfitExitModel(PnLBasedExitModel):
    """Exit on favorable price movements"""
    
    def __init__(self, take_profit_pct: float = 0.05, partial: bool = False,
                 partial_ratio: float = 0.5):
        """
        Initialize take-profit exit model
        
        Args:
            take_profit_pct: Take profit percentage (default 5%)
            partial: Whether to use partial profit taking
            partial_ratio: Ratio of position to close on partial profit
        """
        super().__init__(
            name="TakeProfitExit",
            pnl_threshold=take_profit_pct,  # Positive for profits
            threshold_type='percentage'
        )
        self.partial = partial
        self.partial_ratio = partial_ratio
        
        self.parameters.update({
            'partial': partial,
            'partial_ratio': partial_ratio
        })
        
        self.logger = logging.getLogger(__name__)
    
    def generate_exit_signals(self, W_t: WindowTensor, 
                             open_positions: List[Position],
                             market_data: Dict) -> List[SignalOutput]:
        """Generate take-profit exit signals"""
        exit_signals = []
        current_price = require(market_data, 'spot_price')
        
        for position in open_positions:
            position.update_pnl(current_price)
            
            # Check if take profit hit
            should_exit, profit_pct = self.check_pnl_exit(position)
            
            if should_exit and profit_pct >= self.pnl_threshold:
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,
                    kappa=profit_pct / self.pnl_threshold,
                    confidence=0.9,  # High confidence on take profit
                    horizon=0.0,
                    position_id=position.position_id,
                    exit_reason='take_profit',
                    meta={
                        'profit_pct': profit_pct,
                        'take_profit_pct': self.pnl_threshold,
                        'partial': self.partial,
                        'partial_ratio': self.partial_ratio,
                        'entry_price': position.entry_price,
                        'current_price': current_price,
                        'holding_hours': position.get_holding_hours()
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated take-profit exit for position {position.position_id}: "
                                f"profit={profit_pct:.2%}")
        
        return exit_signals


class PartialTakeProfitExitModel(TakeProfitExitModel):
    """Take-profit with partial position closing"""
    
    def __init__(self, take_profit_pct: float = 0.05, 
                 partial_ratio: float = 0.5,
                 max_partial_exits: int = 3):
        """
        Initialize partial take-profit model
        
        Args:
            take_profit_pct: Take profit percentage
            partial_ratio: Ratio of position to close
            max_partial_exits: Maximum number of partial exits
        """
        super().__init__(
            take_profit_pct=take_profit_pct,
            partial=True,
            partial_ratio=partial_ratio
        )
        self.max_partial_exits = max_partial_exits
        self.partial_exit_count = {}  # Track partial exits per position
        
        self.parameters.update({
            'max_partial_exits': max_partial_exits
        })
        
        self.logger.info(f"Initialized partial take-profit with ratio={partial_ratio:.1%}, "
                        f"max_exits={max_partial_exits}")
    
    def generate_exit_signals(self, W_t: WindowTensor, 
                             open_positions: List[Position],
                             market_data: Dict) -> List[SignalOutput]:
        """Generate partial take-profit exit signals"""
        exit_signals = []
        current_price = require(market_data, 'spot_price')
        
        for position in open_positions:
            position.update_pnl(current_price)
            pos_id = position.position_id
            
            # Initialize partial exit counter if not exists
            if pos_id not in self.partial_exit_count:
                self.partial_exit_count[pos_id] = 0
            
            # Check if we can do more partial exits
            if self.partial_exit_count[pos_id] >= self.max_partial_exits:
                continue
            
            # Check if take profit hit
            should_exit, profit_pct = self.check_pnl_exit(position)
            
            if should_exit and profit_pct >= self.pnl_threshold:
                self.partial_exit_count[pos_id] += 1
                
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,
                    kappa=profit_pct / self.pnl_threshold,
                    confidence=0.9,
                    horizon=0.0,
                    position_id=position.position_id,
                    exit_reason='partial_take_profit',
                    meta={
                        'profit_pct': profit_pct,
                        'take_profit_pct': self.pnl_threshold,
                        'partial_ratio': self.partial_ratio,
                        'partial_exit_count': self.partial_exit_count[pos_id],
                        'max_partial_exits': self.max_partial_exits,
                        'entry_price': position.entry_price,
                        'current_price': current_price,
                        'holding_hours': position.get_holding_hours()
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated partial take-profit exit for position {position.position_id}: "
                                f"profit={profit_pct:.2%}, exit_count={self.partial_exit_count[pos_id]}")
        
        return exit_signals
    
    def reset_partial_exits(self):
        """Reset partial exit counters"""
        self.partial_exit_count.clear()
        self.logger.info("Reset partial exit counters")


class AdaptiveTakeProfitExitModel(TakeProfitExitModel):
    """Adaptive take-profit based on market conditions"""
    
    def __init__(self, base_take_profit_pct: float = 0.05,
                 volatility_adjustment: float = 0.5,
                 momentum_adjustment: float = 0.3):
        """
        Initialize adaptive take-profit model
        
        Args:
            base_take_profit_pct: Base take profit percentage
            volatility_adjustment: Volatility adjustment factor
            momentum_adjustment: Momentum adjustment factor
        """
        super().__init__(take_profit_pct=base_take_profit_pct)
        self.base_take_profit_pct = base_take_profit_pct
        self.volatility_adjustment = volatility_adjustment
        self.momentum_adjustment = momentum_adjustment
        
        self.parameters.update({
            'base_take_profit_pct': base_take_profit_pct,
            'volatility_adjustment': volatility_adjustment,
            'momentum_adjustment': momentum_adjustment
        })
        
        self.logger.info(f"Initialized adaptive take-profit with base={base_take_profit_pct:.1%}, "
                        f"vol_adj={volatility_adjustment}, mom_adj={momentum_adjustment}")
    
    def _calculate_adaptive_take_profit(self, position: Position, 
                                       market_data: Dict) -> float:
        """Calculate adaptive take profit based on market conditions"""
        # Get current market conditions
        current_vol = require(market_data, 'implied_vol')
        entry_vol = require_entry(position, 'implied_vol')
        
        # Volatility adjustment
        vol_ratio = current_vol / entry_vol if entry_vol > 0 else 1.0
        vol_adjustment = 1 + (vol_ratio - 1) * self.volatility_adjustment
        
        # Momentum adjustment (based on recent price movement)
        momentum = market_data.get('momentum', 0.0)  # Recent price momentum
        mom_adjustment = 1 + momentum * self.momentum_adjustment
        
        # Calculate adaptive take profit
        adaptive_take_profit = (self.base_take_profit_pct * 
                               vol_adjustment * mom_adjustment)
        
        return adaptive_take_profit
    
    def generate_exit_signals(self, W_t: WindowTensor, 
                             open_positions: List[Position],
                             market_data: Dict) -> List[SignalOutput]:
        """Generate adaptive take-profit exit signals"""
        exit_signals = []
        current_price = require(market_data, 'spot_price')
        
        for position in open_positions:
            position.update_pnl(current_price)
            
            # Calculate adaptive take profit
            adaptive_take_profit = self._calculate_adaptive_take_profit(position, market_data)
            profit_pct = position.get_pnl_pct()
            
            # Exit if profit exceeds adaptive take profit
            if profit_pct >= adaptive_take_profit:
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,
                    kappa=profit_pct / adaptive_take_profit,
                    confidence=0.9,
                    horizon=0.0,
                    position_id=position.position_id,
                    exit_reason='adaptive_take_profit',
                    meta={
                        'profit_pct': profit_pct,
                        'adaptive_take_profit_pct': adaptive_take_profit,
                        'base_take_profit_pct': self.base_take_profit_pct,
                        'vol_adjustment': require(market_data, 'implied_vol') / require_entry(position, 'implied_vol'),
                        'momentum': market_data.get('momentum', 0.0),
                        'entry_price': position.entry_price,
                        'current_price': current_price
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated adaptive take-profit exit for position {position.position_id}: "
                                f"profit={profit_pct:.2%}, adaptive_tp={adaptive_take_profit:.2%}")
        
        return exit_signals


class ScalingTakeProfitExitModel(TakeProfitExitModel):
    """Take-profit with scaling levels"""
    
    def __init__(self, take_profit_levels: List[float] = [0.02, 0.05, 0.10],
                 scaling_ratios: List[float] = [0.3, 0.5, 1.0]):
        """
        Initialize scaling take-profit model
        
        Args:
            take_profit_levels: List of profit levels to trigger exits
            scaling_ratios: List of position ratios to close at each level
        """
        super().__init__(take_profit_pct=take_profit_levels[0] if take_profit_levels else 0.02)
        
        if len(take_profit_levels) != len(scaling_ratios):
            raise ValueError("Take profit levels and scaling ratios must have same length")
        
        self.take_profit_levels = take_profit_levels
        self.scaling_ratios = scaling_ratios
        self.scaling_exits = {}  # Track scaling exits per position
        
        self.parameters.update({
            'take_profit_levels': take_profit_levels,
            'scaling_ratios': scaling_ratios
        })
        
        self.logger.info(f"Initialized scaling take-profit with levels={take_profit_levels}, "
                        f"ratios={scaling_ratios}")
    
    def generate_exit_signals(self, W_t: WindowTensor, 
                             open_positions: List[Position],
                             market_data: Dict) -> List[SignalOutput]:
        """Generate scaling take-profit exit signals"""
        exit_signals = []
        current_price = require(market_data, 'spot_price')
        
        for position in open_positions:
            position.update_pnl(current_price)
            pos_id = position.position_id
            profit_pct = position.get_pnl_pct()
            
            # Initialize scaling exit tracker if not exists
            if pos_id not in self.scaling_exits:
                self.scaling_exits[pos_id] = set()
            
            # Check each scaling level
            for i, (level, ratio) in enumerate(zip(self.take_profit_levels, self.scaling_ratios)):
                if profit_pct >= level and i not in self.scaling_exits[pos_id]:
                    self.scaling_exits[pos_id].add(i)
                    
                    exit_signal = SignalOutput(
                        signal_type='exit',
                        direction=0,
                        kappa=profit_pct / level,
                        confidence=0.9,
                        horizon=0.0,
                        position_id=position.position_id,
                        exit_reason='scaling_take_profit',
                        meta={
                            'profit_pct': profit_pct,
                            'take_profit_level': level,
                            'scaling_ratio': ratio,
                            'level_index': i,
                            'total_levels': len(self.take_profit_levels),
                            'entry_price': position.entry_price,
                            'current_price': current_price,
                            'holding_hours': position.get_holding_hours()
                        }
                    )
                    exit_signals.append(exit_signal)
                    
                    self.logger.debug(f"Generated scaling take-profit exit for position {position.position_id}: "
                                    f"profit={profit_pct:.2%}, level={level:.2%}, ratio={ratio:.1%}")
        
        return exit_signals
    
    def reset_scaling_exits(self):
        """Reset scaling exit trackers"""
        self.scaling_exits.clear()
        self.logger.info("Reset scaling exit trackers")


class MomentumTakeProfitExitModel(TakeProfitExitModel):
    """Take-profit based on momentum indicators"""
    
    def __init__(self, base_take_profit_pct: float = 0.05,
                 momentum_threshold: float = 0.5,
                 momentum_decay: float = 0.1):
        """
        Initialize momentum-based take-profit model
        
        Args:
            base_take_profit_pct: Base take profit percentage
            momentum_threshold: Momentum threshold for take profit
            momentum_decay: Rate of momentum decay
        """
        super().__init__(take_profit_pct=base_take_profit_pct)
        self.momentum_threshold = momentum_threshold
        self.momentum_decay = momentum_decay
        self.position_momentum = {}  # Track momentum per position
        
        self.parameters.update({
            'momentum_threshold': momentum_threshold,
            'momentum_decay': momentum_decay
        })
        
        self.logger.info(f"Initialized momentum take-profit with base={base_take_profit_pct:.1%}, "
                        f"momentum_thresh={momentum_threshold}, decay={momentum_decay}")
    
    def _update_momentum(self, position: Position, market_data: Dict):
        """Update momentum for position"""
        pos_id = position.position_id
        current_momentum = market_data.get('momentum', 0.0)
        
        if pos_id not in self.position_momentum:
            self.position_momentum[pos_id] = {
                'peak_momentum': 0.0,
                'current_momentum': current_momentum
            }
        
        momentum_data = self.position_momentum[pos_id]
        
        # Update peak momentum
        if current_momentum > momentum_data['peak_momentum']:
            momentum_data['peak_momentum'] = current_momentum
        
        # Apply momentum decay
        momentum_data['current_momentum'] = current_momentum
    
    def _calculate_momentum_take_profit(self, position: Position, 
                                      market_data: Dict) -> float:
        """Calculate momentum-based take profit"""
        pos_id = position.position_id
        
        if pos_id not in self.position_momentum:
            return self.pnl_threshold
        
        momentum_data = self.position_momentum[pos_id]
        current_momentum = momentum_data['current_momentum']
        peak_momentum = momentum_data['peak_momentum']
        
        # Take profit when momentum drops below threshold
        if current_momentum < self.momentum_threshold and peak_momentum > self.momentum_threshold:
            # Adjust take profit based on momentum decay
            momentum_factor = 1 + (peak_momentum - current_momentum) * self.momentum_decay
            return self.pnl_threshold * momentum_factor
        
        return self.pnl_threshold
    
    def generate_exit_signals(self, W_t: WindowTensor, 
                             open_positions: List[Position],
                             market_data: Dict) -> List[SignalOutput]:
        """Generate momentum-based take-profit exit signals"""
        exit_signals = []
        current_price = require(market_data, 'spot_price')
        
        for position in open_positions:
            position.update_pnl(current_price)
            
            # Update momentum
            self._update_momentum(position, market_data)
            
            # Calculate momentum-based take profit
            momentum_take_profit = self._calculate_momentum_take_profit(position, market_data)
            profit_pct = position.get_pnl_pct()
            
            # Exit if profit exceeds momentum take profit
            if profit_pct >= momentum_take_profit:
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,
                    kappa=profit_pct / momentum_take_profit,
                    confidence=0.8,
                    horizon=0.0,
                    position_id=position.position_id,
                    exit_reason='momentum_take_profit',
                    meta={
                        'profit_pct': profit_pct,
                        'momentum_take_profit_pct': momentum_take_profit,
                        'current_momentum': self.position_momentum[position.position_id]['current_momentum'],
                        'peak_momentum': self.position_momentum[position.position_id]['peak_momentum'],
                        'momentum_threshold': self.momentum_threshold,
                        'entry_price': position.entry_price,
                        'current_price': current_price
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated momentum take-profit exit for position {position.position_id}: "
                                f"profit={profit_pct:.2%}, momentum_tp={momentum_take_profit:.2%}")
        
        return exit_signals
    
    def reset_momentum_tracking(self):
        """Reset momentum tracking"""
        self.position_momentum.clear()
        self.logger.info("Reset momentum tracking")
