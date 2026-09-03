"""
Convergence-based exit models implementing Conv1, Conv2, Conv3 strategies.

This module implements exit signals based on implied volatility convergence
to SVI surfaces, following the three convergence definitions from the paper:
- Conv1: Absolute reversion to local same-day SVI fit
- Conv2: Reversion to previous-day terminal SVI
- Conv3: Monotonic convergence to rolling upper envelope
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from .types import WindowTensor, SignalOutput
from .marketdata import require, require_entry
from .types import Position
from .base import ExitSignalModel


class ConvergenceExitModel(ExitSignalModel):
    """
    Exit based on IV convergence to SVI surfaces (Conv1, Conv2, Conv3)
    
    From the convergence paper:
    - Conv1: Absolute reversion to local same-day SVI fit
    - Conv2: Reversion to previous-day terminal SVI
    - Conv3: Monotonic convergence to rolling upper envelope
    """
    
    def __init__(self, convergence_type: str = 'Conv1', epsilon: float = 0.01,
                 kappa: float = 1.0, sigma: float = 0.1):
        """
        Initialize convergence exit model
        
        Args:
            convergence_type: 'Conv1', 'Conv2', or 'Conv3'
            epsilon: Convergence tolerance
            kappa: OU reversion speed parameter
            sigma: OU diffusion parameter
        """
        super().__init__(f"ConvergenceExit_{convergence_type}")
        self.convergence_type = convergence_type
        self.epsilon = epsilon
        self.kappa = kappa
        self.sigma = sigma
        
        self.parameters = {
            'convergence_type': convergence_type,
            'epsilon': epsilon,
            'kappa': kappa,
            'sigma': sigma
        }
        
        self.logger = logging.getLogger(__name__)
    
    def generate_exit_signals(self, open_positions: List[Position],
                             market_data: Dict,
                             W_t: Optional[WindowTensor] = None) -> List[SignalOutput]:
        """Generate convergence-based exit signals"""
        exit_signals = []
        
        for position in open_positions:
            # Calculate residual: r_t = σ_IV - σ_SVI
            sigma_iv = require(market_data, 'implied_vol')
            sigma_svi = self._get_svi_benchmark(position, market_data)
            residual = sigma_iv - sigma_svi
            
            # Check convergence
            converged, ttc = self._check_convergence(residual, sigma_iv, sigma_svi)
            
            if converged:
                exit_signal = SignalOutput(
                    signal_type='exit',
                    direction=0,  # Close position
                    kappa=abs(residual) / self.epsilon,  # Strength based on how close
                    confidence=self._calculate_convergence_confidence(ttc),
                    horizon=ttc,
                    position_id=position.position_id,
                    exit_reason=f'convergence_{self.convergence_type}',
                    meta={
                        'residual': residual,
                        'sigma_iv': sigma_iv,
                        'sigma_svi': sigma_svi,
                        'ttc': ttc,
                        'convergence_type': self.convergence_type,
                        'epsilon': self.epsilon,
                        'exit_price': require(market_data, 'spot_price')
                    }
                )
                exit_signals.append(exit_signal)
                
                self.logger.debug(f"Generated convergence exit for position {position.position_id}: "
                                f"residual={residual:.4f}, ttc={ttc:.2f}")
        
        return exit_signals
    
    def _get_svi_benchmark(self, position: Position, market_data: Dict) -> float:
        """Get SVI benchmark based on convergence type"""
        if self.convergence_type == 'Conv1':
            # Local same-day SVI fit
            return require(market_data, 'svi_fit_current')
        elif self.convergence_type == 'Conv2':
            # Previous-day terminal SVI
            return require(market_data, 'svi_fit_previous_day')
        elif self.convergence_type == 'Conv3':
            # Rolling upper envelope
            return require(market_data, 'svi_envelope_max')
        else:
            return 0.2  # Default
    
    def _check_convergence(self, residual: float, sigma_iv: float, 
                          sigma_svi: float) -> Tuple[bool, float]:
        """
        Check if convergence criterion is met
        
        Returns:
            (converged, estimated_ttc)
        """
        converged = abs(residual) <= self.epsilon
        
        # Estimate time-to-convergence using OU scaling law
        if abs(residual) > self.epsilon:
            ttc = (1.0 / self.kappa) * np.log(abs(residual) / self.epsilon)
        else:
            ttc = 0.0
        
        return converged, ttc
    
    def _calculate_convergence_confidence(self, ttc: float) -> float:
        """Calculate confidence based on time-to-convergence"""
        # Higher confidence for faster convergence
        return np.exp(-0.1 * ttc)
    
    def get_convergence_stats(self, positions: List[Position], 
                             market_data: Dict) -> Dict[str, Any]:
        """Get convergence statistics for positions"""
        stats = {
            'total_positions': len(positions),
            'converged_positions': 0,
            'avg_residual': 0.0,
            'avg_ttc': 0.0,
            'convergence_rate': 0.0
        }
        
        if not positions:
            return stats
        
        residuals = []
        ttcs = []
        converged_count = 0
        
        for position in positions:
            sigma_iv = require(market_data, 'implied_vol')
            sigma_svi = self._get_svi_benchmark(position, market_data)
            residual = sigma_iv - sigma_svi
            residuals.append(residual)
            
            converged, ttc = self._check_convergence(residual, sigma_iv, sigma_svi)
            ttcs.append(ttc)
            
            if converged:
                converged_count += 1
        
        stats.update({
            'converged_positions': converged_count,
            'avg_residual': np.mean(residuals),
            'avg_ttc': np.mean(ttcs),
            'convergence_rate': converged_count / len(positions)
        })
        
        return stats


class Conv1ExitModel(ConvergenceExitModel):
    """Conv1: Absolute reversion to local same-day SVI fit"""
    
    def __init__(self, epsilon: float = 0.01, kappa: float = 1.0):
        """
        Initialize Conv1 exit model
        
        Args:
            epsilon: Convergence tolerance (default 0.01)
            kappa: OU reversion speed (default 1.0)
        """
        super().__init__(
            convergence_type='Conv1',
            epsilon=epsilon,
            kappa=kappa
        )
        self.logger.info(f"Initialized Conv1 exit model with epsilon={epsilon}, kappa={kappa}")


class Conv2ExitModel(ConvergenceExitModel):
    """Conv2: Reversion to previous-day terminal SVI"""
    
    def __init__(self, epsilon: float = 0.01, kappa: float = 0.8):
        """
        Initialize Conv2 exit model
        
        Args:
            epsilon: Convergence tolerance (default 0.01)
            kappa: OU reversion speed (default 0.8, slower than Conv1)
        """
        super().__init__(
            convergence_type='Conv2',
            epsilon=epsilon,
            kappa=kappa
        )
        self.logger.info(f"Initialized Conv2 exit model with epsilon={epsilon}, kappa={kappa}")


class Conv3ExitModel(ConvergenceExitModel):
    """Conv3: Convergence to rolling upper envelope"""
    
    def __init__(self, epsilon: float = 0.01, kappa: float = 0.6):
        """
        Initialize Conv3 exit model
        
        Args:
            epsilon: Convergence tolerance (default 0.01)
            kappa: OU reversion speed (default 0.6, slowest due to moving target)
        """
        super().__init__(
            convergence_type='Conv3',
            epsilon=epsilon,
            kappa=kappa
        )
        self.logger.info(f"Initialized Conv3 exit model with epsilon={epsilon}, kappa={kappa}")
    
    def _check_convergence(self, residual: float, sigma_iv: float, 
                          sigma_svi: float) -> Tuple[bool, float]:
        """
        Conv3-specific convergence check with moving envelope
        
        Conv3 uses a nondecreasing boundary, so convergence is harder
        """
        # For Conv3, we need to check if we're close to the rolling maximum
        converged = abs(residual) <= self.epsilon
        
        # Conv3 has heavier right tails in TTC due to moving target
        if abs(residual) > self.epsilon:
            # Adjust TTC calculation for moving envelope
            ttc = (1.0 / self.kappa) * np.log(abs(residual) / self.epsilon) * 1.5
        else:
            ttc = 0.0
        
        return converged, ttc


class MultiConvergenceExitModel(ExitSignalModel):
    """
    Combined convergence model using all three Conv1, Conv2, Conv3
    
    This model can trigger exits based on any of the three convergence
    definitions, with configurable priorities and weights.
    """
    
    def __init__(self, conv1_weight: float = 0.4, conv2_weight: float = 0.3, 
                 conv3_weight: float = 0.3, epsilon: float = 0.01):
        """
        Initialize multi-convergence exit model
        
        Args:
            conv1_weight: Weight for Conv1 signals
            conv2_weight: Weight for Conv2 signals  
            conv3_weight: Weight for Conv3 signals
            epsilon: Convergence tolerance
        """
        super().__init__("MultiConvergenceExit")
        
        self.conv1_model = Conv1ExitModel(epsilon=epsilon)
        self.conv2_model = Conv2ExitModel(epsilon=epsilon)
        self.conv3_model = Conv3ExitModel(epsilon=epsilon)
        
        self.weights = {
            'Conv1': conv1_weight,
            'Conv2': conv2_weight,
            'Conv3': conv3_weight
        }
        
        self.parameters = {
            'conv1_weight': conv1_weight,
            'conv2_weight': conv2_weight,
            'conv3_weight': conv3_weight,
            'epsilon': epsilon
        }
        
        self.logger = logging.getLogger(__name__)
    
    def generate_exit_signals(self, open_positions: List[Position],
                             market_data: Dict,
                             W_t: Optional[WindowTensor] = None) -> List[SignalOutput]:
        """Generate exit signals using all three convergence models"""
        all_exit_signals = []
        
        # Get signals from each convergence model
        conv1_signals = self.conv1_model.generate_exit_signals(open_positions, market_data)
        conv2_signals = self.conv2_model.generate_exit_signals(open_positions, market_data)
        conv3_signals = self.conv3_model.generate_exit_signals(open_positions, market_data)
        
        # Combine signals with weights
        for signal in conv1_signals:
            signal.kappa *= self.weights['Conv1']
            signal.meta['convergence_weight'] = self.weights['Conv1']
            all_exit_signals.append(signal)
        
        for signal in conv2_signals:
            signal.kappa *= self.weights['Conv2']
            signal.meta['convergence_weight'] = self.weights['Conv2']
            all_exit_signals.append(signal)
        
        for signal in conv3_signals:
            signal.kappa *= self.weights['Conv3']
            signal.meta['convergence_weight'] = self.weights['Conv3']
            all_exit_signals.append(signal)
        
        # Remove duplicates (same position_id) by keeping highest kappa
        unique_signals = {}
        for signal in all_exit_signals:
            pos_id = signal.position_id
            if pos_id not in unique_signals or signal.kappa > unique_signals[pos_id].kappa:
                unique_signals[pos_id] = signal
        
        return list(unique_signals.values())
    
    def get_convergence_analysis(self, positions: List[Position], 
                                market_data: Dict) -> Dict[str, Any]:
        """Get comprehensive convergence analysis"""
        analysis = {
            'conv1_stats': self.conv1_model.get_convergence_stats(positions, market_data),
            'conv2_stats': self.conv2_model.get_convergence_stats(positions, market_data),
            'conv3_stats': self.conv3_model.get_convergence_stats(positions, market_data),
            'weights': self.weights
        }
        
        # Calculate combined metrics
        total_positions = len(positions)
        if total_positions > 0:
            conv1_rate = analysis['conv1_stats']['convergence_rate']
            conv2_rate = analysis['conv2_stats']['convergence_rate']
            conv3_rate = analysis['conv3_stats']['convergence_rate']
            
            analysis['weighted_convergence_rate'] = (
                conv1_rate * self.weights['Conv1'] +
                conv2_rate * self.weights['Conv2'] +
                conv3_rate * self.weights['Conv3']
            )
        
        return analysis
