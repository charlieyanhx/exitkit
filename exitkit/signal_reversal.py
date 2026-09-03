"""
Signal reversal exit models for position management.

This module implements various signal reversal exit strategies including
signal direction reversal, signal strength decay, and signal divergence exits.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from .types import WindowTensor, SignalOutput
from .marketdata import require, require_entry
from .types import Position
from .base import SignalBasedExitModel


class SignalReversalExitModel(SignalBasedExitModel):
    """Exit when entry signal reverses"""
    
    def __init__(self, reversal_threshold: float = 0.5,
                 confirmation_periods: int = 1):
        """
        Initialize signal reversal exit model
        
        Args:
            reversal_threshold: Threshold for signal reversal
            confirmation_periods: Number of periods to confirm reversal
        """
        super().__init__("SignalReversalExit", reversal_threshold)
        self.confirmation_periods = confirmation_periods
        self.reversal_history = {}  # Track reversal history per position
        
        self.parameters.update({
            'confirmation_periods': confirmation_periods
        })
        
        self.logger = logging.getLogger(__name__)
    
    def generate_exit_signals(self, open_positions: List[Position],
                             market_data: Dict,
                             W_t: Optional[WindowTensor] = None) -> List[SignalOutput]:
        """Generate signal reversal exit signals"""
        exit_signals = []
        
        if self.entry_signal_model is None:
            self.logger.warning("Entry signal model not set for signal reversal exit")
            return exit_signals
        
        # Generate current signals from entry model
        current_signals = self.entry_signal_model.generate_signals(W_t)
        
        for position in open_positions:
            pos_id = position.position_id
            entry_direction = position.entry_signal.direction
            
            # Initialize reversal history if not exists
            if pos_id not in self.reversal_history:
                self.reversal_history[pos_id] = []
            
            # Check for signal reversals
            reversal_signals = []
            for signal in current_signals:
                if (signal.direction == -entry_direction and 
                    signal.kappa >= self.reversal_threshold):
                    reversal_signals.append(signal)
            
            # Update reversal history
            if reversal_signals:
                self.reversal_history[pos_id].append(True)
            else:
                self.reversal_history[pos_id].append(False)
            
            # Keep only recent history
            if len(self.reversal_history[pos_id]) > self.confirmation_periods * 2:
                self.reversal_history[pos_id] = self.reversal_history[pos_id][-self.confirmation_periods * 2:]
            
            # Check if reversal is confirmed
            if len(self.reversal_history[pos_id]) >= self.confirmation_periods:
                recent_reversals = self.reversal_history[pos_id][-self.confirmation_periods:]
                confirmed_reversal = all(recent_reversals)
                
                if confirmed_reversal and reversal_signals:
                    # Use strongest reversal signal
                    strongest_signal = max(reversal_signals, key=lambda s: s.kappa)
                    
                    exit_signal = SignalOutput(
                        signal_type='exit',
                        direction=0,
                        kappa=strongest_signal.kappa,
                        confidence=strongest_signal.confidence,
                        horizon=0.0,
                        position_id=position.position_id,
                        exit_reason='signal_reversal',
                        meta={
                            'entry_direction': entry_direction,
                            'reversal_direction': strongest_signal.direction,
                            'reversal_kappa': strongest_signal.kappa,
                            'reversal_confidence': strongest_signal.confidence,
                            'confirmation_periods': self.confirmation_periods,
                            'reversal_history': self.reversal_history[pos_id][-self.confirmation_periods:],
                            'entry_price': position.entry_price,
                            'current_pnl': position.current_pnl
                        }
                    )
                    exit_signals.append(exit_signal)
                    
                    self.logger.debug(f"Generated signal reversal exit for position {position.position_id}: "
                                    f"entry_dir={entry_direction}, reversal_dir={strongest_signal.direction}")
        
        return exit_signals


class SignalStrengthDecayExitModel(SignalBasedExitModel):
    """Exit when signal strength decays below threshold"""
    
    def __init__(self, strength_decay_threshold: float = 0.3,
                 decay_periods: int = 3):
        """
        Initialize signal strength decay exit model
        
        Args:
            strength_decay_threshold: Threshold for signal strength decay
            decay_periods: Number of periods to confirm decay
        """
        super().__init__("SignalStrengthDecayExit", strength_decay_threshold)
        self.decay_periods = decay_periods
        self.strength_history = {}  # Track signal strength history per position
        
        self.parameters.update({
            'decay_periods': decay_periods
        })
        
        self.logger = logging.getLogger(__name__)
    
    def generate_exit_signals(self, open_positions: List[Position],
                             market_data: Dict,
                             W_t: Optional[WindowTensor] = None) -> List[SignalOutput]:
        """Generate signal strength decay exit signals"""
        exit_signals = []
        
        if self.entry_signal_model is None:
            self.logger.warning("Entry signal model not set for signal strength decay exit")
            return exit_signals
        
        # Generate current signals from entry model
        current_signals = self.entry_signal_model.generate_signals(W_t)
        
        for position in open_positions:
            pos_id = position.position_id
            entry_direction = position.entry_signal.direction
            
            # Initialize strength history if not exists
            if pos_id not in self.strength_history:
                self.strength_history[pos_id] = []
            
            # Find current signal in same direction as entry
            current_signal = None
            for signal in current_signals:
                if signal.direction == entry_direction:
                    current_signal = signal
                    break
            
            if current_signal:
                # Update strength history
                self.strength_history[pos_id].append(current_signal.kappa)
                
                # Keep only recent history
                if len(self.strength_history[pos_id]) > self.decay_periods * 2:
                    self.strength_history[pos_id] = self.strength_history[pos_id][-self.decay_periods * 2:]
                
                # Check for strength decay
                if len(self.strength_history[pos_id]) >= self.decay_periods:
                    recent_strengths = self.strength_history[pos_id][-self.decay_periods:]
                    
                    # Calculate strength decay
                    initial_strength = recent_strengths[0]
                    current_strength = recent_strengths[-1]
                    strength_decay = (initial_strength - current_strength) / initial_strength if initial_strength > 0 else 0.0
                    
                    # Check if decay exceeds threshold
                    if strength_decay >= self.reversal_threshold:
                        exit_signal = SignalOutput(
                            signal_type='exit',
                            direction=0,
                            kappa=strength_decay,
                            confidence=0.8,
                            horizon=0.0,
                            position_id=position.position_id,
                            exit_reason='signal_strength_decay',
                            meta={
                                'strength_decay': strength_decay,
                                'strength_decay_threshold': self.strength_decay_threshold,
                                'initial_strength': initial_strength,
                                'current_strength': current_strength,
                                'decay_periods': self.decay_periods,
                                'strength_history': self.strength_history[pos_id][-self.decay_periods:],
                                'entry_price': position.entry_price,
                                'current_pnl': position.current_pnl
                            }
                        )
                        exit_signals.append(exit_signal)
                        
                        self.logger.debug(f"Generated signal strength decay exit for position {position.position_id}: "
                                        f"decay={strength_decay:.3f}")
        
        return exit_signals


class SignalDivergenceExitModel(SignalBasedExitModel):
    """Exit on signal divergence from price action"""
    
    def __init__(self, divergence_threshold: float = 0.4,
                 divergence_periods: int = 5):
        """
        Initialize signal divergence exit model
        
        Args:
            divergence_threshold: Threshold for signal divergence
            divergence_periods: Number of periods to check for divergence
        """
        super().__init__("SignalDivergenceExit", divergence_threshold)
        self.divergence_periods = divergence_periods
        self.price_signal_history = {}  # Track price and signal history per position
        
        self.parameters.update({
            'divergence_periods': divergence_periods
        })
        
        self.logger = logging.getLogger(__name__)
    
    def _update_price_signal_history(self, position: Position, market_data: Dict, current_signals: List[SignalOutput]):
        """Update price and signal history for position"""
        pos_id = position.position_id
        current_price = require(market_data, 'spot_price')
        
        # Find current signal in same direction as entry
        current_signal = None
        for signal in current_signals:
            if signal.direction == position.entry_signal.direction:
                current_signal = signal
                break
        
        if pos_id not in self.price_signal_history:
            self.price_signal_history[pos_id] = {
                'prices': [],
                'signals': []
            }
        
        # Update history
        self.price_signal_history[pos_id]['prices'].append(current_price)
        self.price_signal_history[pos_id]['signals'].append(current_signal.kappa if current_signal else 0.0)
        
        # Keep only recent history
        if len(self.price_signal_history[pos_id]['prices']) > self.divergence_periods * 2:
            self.price_signal_history[pos_id]['prices'] = self.price_signal_history[pos_id]['prices'][-self.divergence_periods * 2:]
            self.price_signal_history[pos_id]['signals'] = self.price_signal_history[pos_id]['signals'][-self.divergence_periods * 2:]
    
    def _detect_signal_divergence(self, position: Position) -> Tuple[bool, float]:
        """Detect signal divergence from price action"""
        pos_id = position.position_id
        
        if pos_id not in self.price_signal_history:
            return False, 0.0
        
        price_history = self.price_signal_history[pos_id]['prices']
        signal_history = self.price_signal_history[pos_id]['signals']
        
        if len(price_history) < self.divergence_periods or len(signal_history) < self.divergence_periods:
            return False, 0.0
        
        # Calculate price and signal trends
        recent_prices = price_history[-self.divergence_periods:]
        recent_signals = signal_history[-self.divergence_periods:]
        
        # Calculate trends
        price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0.0
        signal_trend = (recent_signals[-1] - recent_signals[0]) / recent_signals[0] if recent_signals[0] > 0 else 0.0
        
        # Detect divergence (opposite trends)
        divergence = abs(price_trend - signal_trend)
        is_divergence = divergence >= self.divergence_threshold
        
        return is_divergence, divergence
    
    def generate_exit_signals(self, open_positions: List[Position],
                             market_data: Dict,
                             W_t: Optional[WindowTensor] = None) -> List[SignalOutput]:
        """Generate signal divergence exit signals"""
        exit_signals = []
        
        if self.entry_signal_model is None:
            self.logger.warning("Entry signal model not set for signal divergence exit")
            return exit_signals
        
        # Generate current signals from entry model
        current_signals = self.entry_signal_model.generate_signals(W_t)
        
        for position in open_positions:
            # Update price and signal history
            self._update_price_signal_history(position, market_data, current_signals)
            
            # Detect divergence
            divergence, divergence_strength = self._detect_signal_divergence(position)
            
            if divergence:
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,
                    kappa=divergence_strength,
                    confidence=0.7,
                    horizon=0.0,
                    position_id=position.position_id,
                    exit_reason='signal_divergence',
                    meta={
                        'divergence_strength': divergence_strength,
                        'divergence_threshold': self.divergence_threshold,
                        'divergence_periods': self.divergence_periods,
                        'price_history': self.price_signal_history[position.position_id]['prices'][-self.divergence_periods:],
                        'signal_history': self.price_signal_history[position.position_id]['signals'][-self.divergence_periods:],
                        'entry_price': position.entry_price,
                        'current_pnl': position.current_pnl
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated signal divergence exit for position {position.position_id}: "
                                f"divergence={divergence_strength:.3f}")
        
        return exit_signals


class SignalConsistencyExitModel(SignalBasedExitModel):
    """Exit when signal consistency drops below threshold"""
    
    def __init__(self, consistency_threshold: float = 0.6,
                 consistency_periods: int = 5):
        """
        Initialize signal consistency exit model
        
        Args:
            consistency_threshold: Threshold for signal consistency
            consistency_periods: Number of periods to check for consistency
        """
        super().__init__("SignalConsistencyExit", consistency_threshold)
        self.consistency_periods = consistency_periods
        self.signal_direction_history = {}  # Track signal direction history per position
        
        self.parameters.update({
            'consistency_periods': consistency_periods
        })
        
        self.logger = logging.getLogger(__name__)
    
    def _update_signal_direction_history(self, position: Position, current_signals: List[SignalOutput]):
        """Update signal direction history for position"""
        pos_id = position.position_id
        entry_direction = position.entry_signal.direction
        
        # Find current signal in same direction as entry
        current_signal = None
        for signal in current_signals:
            if signal.direction == entry_direction:
                current_signal = signal
                break
        
        if pos_id not in self.signal_direction_history:
            self.signal_direction_history[pos_id] = []
        
        # Record signal direction consistency
        if current_signal:
            # Signal is consistent with entry direction
            self.signal_direction_history[pos_id].append(True)
        else:
            # No signal in entry direction
            self.signal_direction_history[pos_id].append(False)
        
        # Keep only recent history
        if len(self.signal_direction_history[pos_id]) > self.consistency_periods * 2:
            self.signal_direction_history[pos_id] = self.signal_direction_history[pos_id][-self.consistency_periods * 2:]
    
    def _calculate_signal_consistency(self, position: Position) -> Tuple[bool, float]:
        """Calculate signal consistency"""
        pos_id = position.position_id
        
        if pos_id not in self.signal_direction_history:
            return False, 0.0
        
        if len(self.signal_direction_history[pos_id]) < self.consistency_periods:
            return False, 0.0
        
        recent_consistency = self.signal_direction_history[pos_id][-self.consistency_periods:]
        consistency_ratio = sum(recent_consistency) / len(recent_consistency)
        
        # Check if consistency is below threshold
        low_consistency = consistency_ratio < self.consistency_threshold
        
        return low_consistency, consistency_ratio
    
    def generate_exit_signals(self, open_positions: List[Position],
                             market_data: Dict,
                             W_t: Optional[WindowTensor] = None) -> List[SignalOutput]:
        """Generate signal consistency exit signals"""
        exit_signals = []
        
        if self.entry_signal_model is None:
            self.logger.warning("Entry signal model not set for signal consistency exit")
            return exit_signals
        
        # Generate current signals from entry model
        current_signals = self.entry_signal_model.generate_signals(W_t)
        
        for position in open_positions:
            # Update signal direction history
            self._update_signal_direction_history(position, current_signals)
            
            # Calculate consistency
            low_consistency, consistency_ratio = self._calculate_signal_consistency(position)
            
            if low_consistency:
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,
                    kappa=1.0 - consistency_ratio,  # Higher kappa for lower consistency
                    confidence=0.6,
                    horizon=0.0,
                    position_id=position.position_id,
                    exit_reason='signal_consistency',
                    meta={
                        'consistency_ratio': consistency_ratio,
                        'consistency_threshold': self.consistency_threshold,
                        'consistency_periods': self.consistency_periods,
                        'direction_history': self.signal_direction_history[position.position_id][-self.consistency_periods:],
                        'entry_price': position.entry_price,
                        'current_pnl': position.current_pnl
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated signal consistency exit for position {position.position_id}: "
                                f"consistency={consistency_ratio:.3f}")
        
        return exit_signals
