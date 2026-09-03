"""
Volatility-based exit models for regime change detection.

This module implements various volatility-based exit strategies including
volatility breakout exits, regime change exits, and volatility mean reversion exits.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from .types import WindowTensor, SignalOutput
from .marketdata import require, require_entry
from .types import Position
from .base import VolatilityBasedExitModel


class VolatilityBreakoutExitModel(VolatilityBasedExitModel):
    """Exit on volatility breakouts"""
    
    def __init__(self, vol_threshold: float = 0.5, 
                 vol_type: str = 'change_pct',
                 breakout_direction: str = 'both'):
        """
        Initialize volatility breakout exit model
        
        Args:
            vol_threshold: Volatility threshold for breakout
            vol_type: Type of volatility measure ('change_pct', 'absolute', 'ratio')
            breakout_direction: Direction of breakout ('up', 'down', 'both')
        """
        super().__init__("VolatilityBreakoutExit", vol_threshold, vol_type)
        self.breakout_direction = breakout_direction
        
        self.parameters.update({
            'breakout_direction': breakout_direction
        })
        
        self.logger = logging.getLogger(__name__)
    
    def _detect_volatility_breakout(self, position: Position, 
                                   market_data: Dict) -> Tuple[bool, float, str]:
        """Detect volatility breakout"""
        current_vol = require(market_data, 'implied_vol')
        entry_vol = require_entry(position, 'implied_vol')
        
        # Calculate volatility change
        if self.vol_type == 'change_pct':
            vol_change = (current_vol - entry_vol) / entry_vol if entry_vol > 0 else 0.0
        elif self.vol_type == 'absolute':
            vol_change = current_vol - entry_vol
        elif self.vol_type == 'ratio':
            vol_change = current_vol / entry_vol if entry_vol > 0 else 1.0
        else:
            vol_change = 0.0
        
        # Determine breakout direction
        if self.breakout_direction == 'up':
            breakout = vol_change >= self.vol_threshold
            direction = 'up' if breakout else 'none'
        elif self.breakout_direction == 'down':
            breakout = vol_change <= -self.vol_threshold
            direction = 'down' if breakout else 'none'
        else:  # both
            breakout = abs(vol_change) >= self.vol_threshold
            direction = 'up' if vol_change > 0 else 'down' if vol_change < 0 else 'none'
        
        return breakout, vol_change, direction
    
    def generate_exit_signals(self, open_positions: List[Position],
                             market_data: Dict,
                             W_t: Optional[WindowTensor] = None) -> List[SignalOutput]:
        """Generate volatility breakout exit signals"""
        exit_signals = []
        
        for position in open_positions:
            breakout, vol_change, direction = self._detect_volatility_breakout(position, market_data)
            
            if breakout:
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,
                    kappa=abs(vol_change) / self.vol_threshold,
                    confidence=0.8,
                    horizon=0.0,
                    position_id=position.position_id,
                    exit_reason='volatility_breakout',
                    meta={
                        'vol_change': vol_change,
                        'vol_threshold': self.vol_threshold,
                        'breakout_direction': direction,
                        'entry_vol': require_entry(position, 'implied_vol'),
                        'current_vol': require(market_data, 'implied_vol'),
                        'vol_type': self.vol_type,
                        'entry_price': position.entry_price,
                        'current_pnl': position.current_pnl
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated volatility breakout exit for position {position.position_id}: "
                                f"vol_change={vol_change:.3f}, direction={direction}")
        
        return exit_signals


class VolatilityRegimeExitModel(VolatilityBasedExitModel):
    """Exit on volatility regime changes"""
    
    def __init__(self, vol_threshold: float = 0.3,
                 regime_persistence: int = 3,
                 regime_type: str = 'volatility'):
        """
        Initialize volatility regime exit model
        
        Args:
            vol_threshold: Volatility threshold for regime change
            regime_persistence: Number of periods to confirm regime change
            regime_type: Type of regime ('volatility', 'trend', 'mean_reversion')
        """
        super().__init__("VolatilityRegimeExit", vol_threshold, 'change_pct')
        self.regime_persistence = regime_persistence
        self.regime_type = regime_type
        self.regime_history = {}  # Track regime history per position
        
        self.parameters.update({
            'regime_persistence': regime_persistence,
            'regime_type': regime_type
        })
        
        self.logger = logging.getLogger(__name__)
    
    def _update_regime_history(self, position: Position, market_data: Dict):
        """Update regime history for position"""
        pos_id = position.position_id
        current_vol = require(market_data, 'implied_vol')
        entry_vol = require_entry(position, 'implied_vol')
        
        vol_change = (current_vol - entry_vol) / entry_vol if entry_vol > 0 else 0.0
        
        if pos_id not in self.regime_history:
            self.regime_history[pos_id] = []
        
        # Determine current regime
        if vol_change > self.vol_threshold:
            current_regime = 'high_vol'
        elif vol_change < -self.vol_threshold:
            current_regime = 'low_vol'
        else:
            current_regime = 'normal_vol'
        
        self.regime_history[pos_id].append(current_regime)
        
        # Keep only recent history
        if len(self.regime_history[pos_id]) > self.regime_persistence * 2:
            self.regime_history[pos_id] = self.regime_history[pos_id][-self.regime_persistence * 2:]
    
    def _detect_regime_change(self, position: Position) -> Tuple[bool, str]:
        """Detect regime change based on persistence"""
        pos_id = position.position_id
        
        if pos_id not in self.regime_history or len(self.regime_history[pos_id]) < self.regime_persistence:
            return False, 'insufficient_data'
        
        recent_regimes = self.regime_history[pos_id][-self.regime_persistence:]
        
        # Check if regime has persisted
        if len(set(recent_regimes)) == 1:  # All same regime
            current_regime = recent_regimes[0]
            
            # Check if this is different from initial regime
            if len(self.regime_history[pos_id]) >= self.regime_persistence * 2:
                initial_regimes = self.regime_history[pos_id][:self.regime_persistence]
                initial_regime = initial_regimes[0] if len(set(initial_regimes)) == 1 else 'mixed'
                
                if current_regime != initial_regime and current_regime != 'normal_vol':
                    return True, f"{initial_regime}_to_{current_regime}"
        
        return False, 'no_change'
    
    def generate_exit_signals(self, open_positions: List[Position],
                             market_data: Dict,
                             W_t: Optional[WindowTensor] = None) -> List[SignalOutput]:
        """Generate volatility regime exit signals"""
        exit_signals = []
        
        for position in open_positions:
            # Update regime history
            self._update_regime_history(position, market_data)
            
            # Detect regime change
            regime_change, change_type = self._detect_regime_change(position)
            
            if regime_change:
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,
                    kappa=1.0,  # High confidence on regime change
                    confidence=0.9,
                    horizon=0.0,
                    position_id=position.position_id,
                    exit_reason='volatility_regime_change',
                    meta={
                        'regime_change_type': change_type,
                        'regime_persistence': self.regime_persistence,
                        'regime_type': self.regime_type,
                        'regime_history': self.regime_history[position.position_id][-self.regime_persistence:],
                        'entry_price': position.entry_price,
                        'current_pnl': position.current_pnl
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated volatility regime exit for position {position.position_id}: "
                                f"change_type={change_type}")
        
        return exit_signals


class VolatilityMeanReversionExitModel(VolatilityBasedExitModel):
    """Exit on volatility mean reversion"""
    
    def __init__(self, vol_threshold: float = 0.2,
                 reversion_threshold: float = 0.1,
                 lookback_periods: int = 10):
        """
        Initialize volatility mean reversion exit model
        
        Args:
            vol_threshold: Volatility threshold for mean reversion
            reversion_threshold: Threshold for reversion detection
            lookback_periods: Number of periods to look back for mean calculation
        """
        super().__init__("VolatilityMeanReversionExit", vol_threshold, 'change_pct')
        self.reversion_threshold = reversion_threshold
        self.lookback_periods = lookback_periods
        self.vol_history = {}  # Track volatility history per position
        
        self.parameters.update({
            'reversion_threshold': reversion_threshold,
            'lookback_periods': lookback_periods
        })
        
        self.logger = logging.getLogger(__name__)
    
    def _update_volatility_history(self, position: Position, market_data: Dict):
        """Update volatility history for position"""
        pos_id = position.position_id
        current_vol = require(market_data, 'implied_vol')
        
        if pos_id not in self.vol_history:
            self.vol_history[pos_id] = []
        
        self.vol_history[pos_id].append(current_vol)
        
        # Keep only recent history
        if len(self.vol_history[pos_id]) > self.lookback_periods:
            self.vol_history[pos_id] = self.vol_history[pos_id][-self.lookback_periods:]
    
    def _detect_mean_reversion(self, position: Position) -> Tuple[bool, float]:
        """Detect volatility mean reversion"""
        pos_id = position.position_id
        
        if pos_id not in self.vol_history or len(self.vol_history[pos_id]) < self.lookback_periods:
            return False, 0.0
        
        vol_history = self.vol_history[pos_id]
        current_vol = vol_history[-1]
        mean_vol = np.mean(vol_history[:-1])  # Mean excluding current
        std_vol = np.std(vol_history[:-1]) if len(vol_history) > 1 else 0.0
        
        # Calculate z-score
        if std_vol > 0:
            z_score = (current_vol - mean_vol) / std_vol
        else:
            z_score = 0.0
        
        # Detect reversion (volatility moving back toward mean)
        reversion = abs(z_score) <= self.reversion_threshold
        
        return reversion, z_score
    
    def generate_exit_signals(self, open_positions: List[Position],
                             market_data: Dict,
                             W_t: Optional[WindowTensor] = None) -> List[SignalOutput]:
        """Generate volatility mean reversion exit signals"""
        exit_signals = []
        
        for position in open_positions:
            # Update volatility history
            self._update_volatility_history(position, market_data)
            
            # Detect mean reversion
            reversion, z_score = self._detect_mean_reversion(position)
            
            if reversion:
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,
                    kappa=1.0 - abs(z_score),  # Higher kappa for closer to mean
                    confidence=0.7,
                    horizon=0.0,
                    position_id=position.position_id,
                    exit_reason='volatility_mean_reversion',
                    meta={
                        'z_score': z_score,
                        'reversion_threshold': self.reversion_threshold,
                        'lookback_periods': self.lookback_periods,
                        'vol_history': self.vol_history[position.position_id][-5:],  # Last 5 periods
                        'entry_price': position.entry_price,
                        'current_pnl': position.current_pnl
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated volatility mean reversion exit for position {position.position_id}: "
                                f"z_score={z_score:.3f}")
        
        return exit_signals


class VolatilityClusteringExitModel(VolatilityBasedExitModel):
    """Exit on volatility clustering patterns"""
    
    def __init__(self, vol_threshold: float = 0.3,
                 clustering_threshold: float = 0.8,
                 clustering_periods: int = 5):
        """
        Initialize volatility clustering exit model
        
        Args:
            vol_threshold: Volatility threshold for clustering
            clustering_threshold: Threshold for clustering detection
            clustering_periods: Number of periods to check for clustering
        """
        super().__init__("VolatilityClusteringExit", vol_threshold, 'change_pct')
        self.clustering_threshold = clustering_threshold
        self.clustering_periods = clustering_periods
        self.vol_clusters = {}  # Track volatility clusters per position
        
        self.parameters.update({
            'clustering_threshold': clustering_threshold,
            'clustering_periods': clustering_periods
        })
        
        self.logger = logging.getLogger(__name__)
    
    def _update_volatility_clusters(self, position: Position, market_data: Dict):
        """Update volatility clusters for position"""
        pos_id = position.position_id
        current_vol = require(market_data, 'implied_vol')
        entry_vol = require_entry(position, 'implied_vol')
        
        vol_change = (current_vol - entry_vol) / entry_vol if entry_vol > 0 else 0.0
        
        if pos_id not in self.vol_clusters:
            self.vol_clusters[pos_id] = []
        
        # Determine if current period is high volatility
        is_high_vol = abs(vol_change) >= self.vol_threshold
        self.vol_clusters[pos_id].append(is_high_vol)
        
        # Keep only recent clusters
        if len(self.vol_clusters[pos_id]) > self.clustering_periods * 2:
            self.vol_clusters[pos_id] = self.vol_clusters[pos_id][-self.clustering_periods * 2:]
    
    def _detect_volatility_clustering(self, position: Position) -> Tuple[bool, float]:
        """Detect volatility clustering"""
        pos_id = position.position_id
        
        if pos_id not in self.vol_clusters or len(self.vol_clusters[pos_id]) < self.clustering_periods:
            return False, 0.0
        
        recent_clusters = self.vol_clusters[pos_id][-self.clustering_periods:]
        
        # Calculate clustering ratio
        high_vol_count = sum(recent_clusters)
        clustering_ratio = high_vol_count / len(recent_clusters)
        
        # Detect clustering
        clustering = clustering_ratio >= self.clustering_threshold
        
        return clustering, clustering_ratio
    
    def generate_exit_signals(self, open_positions: List[Position],
                             market_data: Dict,
                             W_t: Optional[WindowTensor] = None) -> List[SignalOutput]:
        """Generate volatility clustering exit signals"""
        exit_signals = []
        
        for position in open_positions:
            # Update volatility clusters
            self._update_volatility_clusters(position, market_data)
            
            # Detect clustering
            clustering, clustering_ratio = self._detect_volatility_clustering(position)
            
            if clustering:
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,
                    kappa=clustering_ratio,
                    confidence=0.8,
                    horizon=0.0,
                    position_id=position.position_id,
                    exit_reason='volatility_clustering',
                    meta={
                        'clustering_ratio': clustering_ratio,
                        'clustering_threshold': self.clustering_threshold,
                        'clustering_periods': self.clustering_periods,
                        'vol_clusters': self.vol_clusters[position.position_id][-self.clustering_periods:],
                        'entry_price': position.entry_price,
                        'current_pnl': position.current_pnl
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated volatility clustering exit for position {position.position_id}: "
                                f"clustering_ratio={clustering_ratio:.3f}")
        
        return exit_signals
