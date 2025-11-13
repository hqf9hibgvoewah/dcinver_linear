"""
频散曲线反演的Python实现
使用disba进行正演计算，scipy.optimize进行反演优化
"""

import numpy as np
import scipy.optimize as opt
from disba import PhaseDispersion
from disba import GroupDispersion
import warnings
from typing import Tuple, Optional

class DispersionInverter:
    """
    频散曲线反演器
    基于Rayleigh波频散数据反演1D速度结构
    """
    
    def __init__(self, vp_vs_ratio: float = 1.9, density_vp_ratio: float = 0.32):
        """
        初始化反演器
        
        Parameters:
        -----------
        vp_vs_ratio : float, default=1.9
            Vp/Vs比值
        density_vp_ratio : float, default=0.32  
            ρ/Vp比值 (g/cm³ per km/s)
        """
        self.vp_vs_ratio = vp_vs_ratio
        self.density_vp_ratio = density_vp_ratio
        
    def forward_modeling(self, periods: np.ndarray, model: np.ndarray, 
                        wave_type: str = 'rayleigh',forward_type: str = 'phase') -> np.ndarray:
        """
        Forward modeling for dispersion curves
        
        Parameters:
        -----------
        periods : np.ndarray
            Period array (s)
        model : np.ndarray
            Velocity model [thickness(km), Vp(km/s), Vs(km/s), density(g/cm³)]
        wave_type : str, default='rayleigh'
            Wave type ('rayleigh' or 'love')
        forward_type : str, default='phase'
            Forward type ('phase' or 'group')
            
        Returns:
        --------
        phase_velocity : np.ndarray
            Predicted phase velocity (km/s) or group velocity (km/s)
        """
        # Extract model parameters
        thickness = model[:, 0]
        vp = model[:, 1]
        vs = model[:, 2]
        density = model[:, 3]
        
        # Validate model parameters
        if np.any(vs <= 0) or np.any(vp <= 0) or np.any(density <= 0):
            warnings.warn("Invalid model parameters: negative or zero values detected")
            return np.full_like(periods, np.nan)
        
        # Check for physically unrealistic values
        if np.any(vs > 5.0) or np.any(vp > 9.0) or np.any(density > 3.5):
            warnings.warn("Model parameters exceed typical physical ranges")
            # Continue but warn
        
        # Use disba for forward modeling with enhanced error handling
        try:
            # Prepare model for disba
            prepared_model = self._prepare_model(thickness, vp, vs, density)
            
            # Additional validation for disba requirements
            if len(prepared_model[0]) < 2:
                warnings.warn("Model must have at least 2 layers")
                return np.full_like(periods, np.nan)
            
            # Check if Vs < Vp (physical requirement)
            if np.any(prepared_model[2] >= prepared_model[1]):
                warnings.warn("Vs must be less than Vp for physical consistency")
                # Adjust Vs to be slightly less than Vp
                vs_adjusted = np.minimum(prepared_model[2], prepared_model[1] * 0.99)
                prepared_model = (prepared_model[0], prepared_model[1], vs_adjusted, prepared_model[3])
            
            if forward_type.lower() == 'phase':
                pd = PhaseDispersion(*prepared_model)
            elif forward_type.lower() == 'group':
                pd = GroupDispersion(*prepared_model)
            
            if wave_type.lower() == 'rayleigh':
                dispersion = pd(periods, wave='rayleigh', mode=0)
            else:
                dispersion = pd(periods, wave='love', mode=0)
            
            # Check for valid results
            if np.any(np.isnan(dispersion.velocity)) or np.any(dispersion.velocity <= 0):
                warnings.warn("Forward modeling produced invalid phase velocities")
                return np.full_like(periods, np.nan)
                
            return dispersion.velocity
            
        except ValueError as e:
            # Handle specific disba errors
            if "No solution found" in str(e):
                warnings.warn("No dispersion solution found for given model parameters")
                return np.full_like(periods, np.nan)
            elif "singular matrix" in str(e).lower():
                warnings.warn("Numerical instability in forward modeling")
                return np.full_like(periods, np.nan)
            else:
                warnings.warn(f"ValueError in forward modeling: {e}")
                return np.full_like(periods, np.nan)
                
        except Exception as e:
            warnings.warn(f"Unexpected error in forward modeling: {e}")
            return np.full_like(periods, np.nan)
    
    def _prepare_model(self, thickness: np.ndarray, vp: np.ndarray, 
                      vs: np.ndarray, density: np.ndarray) -> Tuple:
        """Prepare model format for disba"""
        # Ensure bottom layer is half-space
        if thickness[-1] > 0:
            thickness = np.append(thickness, 0)
            vp = np.append(vp, vp[-1])
            vs = np.append(vs, vs[-1])
            density = np.append(density, density[-1])
        
        return thickness, vp, vs, density
    
    def objective_function(self, vs_params: np.ndarray, fixed_model: np.ndarray,
                          periods: np.ndarray, observed_phv: np.ndarray, 
                          phv_std: np.ndarray, smoothing_weights: np.ndarray,
                          damping: float = 0.1, forward_type: str = 'phase') -> float:
        """
        Enhanced objective function with professional dispersion curve similarity measures
        
        Parameters:
        -----------
        vs_params : np.ndarray
            Vs parameters to optimize
        fixed_model : np.ndarray
            Fixed parameter model [thickness, Vp, density]
        periods : np.ndarray
            Period array
        observed_phv : np.ndarray
            Observed phase velocity or group velocity
        phv_std : np.ndarray
            Observation error
        smoothing_weights : np.ndarray
            Smoothing weights
        damping : float, default=0.1
            Damping coefficient
        forward_type : str, default='phase'
            Forward type ('phase' or 'group')
            
        Returns:
        --------
        misfit : float
            Objective function value
        """
        # Build complete model
        full_model = self._build_model(vs_params, fixed_model)
        
        # Forward modeling with enhanced error handling
        predicted_phv = self.forward_modeling(periods, full_model, forward_type=forward_type)
        
        # Check for invalid results
        if np.any(np.isnan(predicted_phv)) or np.any(predicted_phv <= 0):
            return 1e10  # Return large value for invalid solution
        
        # Get valid data points
        valid_indices = ~np.isnan(observed_phv) & ~np.isnan(predicted_phv)
        if np.sum(valid_indices) == 0:
            return 1e10  # No valid data points
        
        valid_observed = observed_phv[valid_indices]
        valid_predicted = predicted_phv[valid_indices]
        valid_std = phv_std[valid_indices]
        valid_periods = periods[valid_indices]
        
        # Enhanced misfit calculation with multiple similarity measures
        misfit = self._calculate_enhanced_misfit(
            valid_observed, valid_predicted, valid_std, valid_periods)
        
        # Model smoothness term (first-order difference)
        if len(vs_params) > 1:
            model_smoothness = np.sum(smoothing_weights * np.diff(vs_params) ** 2)
        else:
            model_smoothness = 0.0
        
        # Total objective function
        total_misfit = misfit + damping * model_smoothness
        
        return total_misfit
    
    def _calculate_enhanced_misfit(self, observed: np.ndarray, predicted: np.ndarray,
                                 std: np.ndarray, periods: np.ndarray) -> float:
        """
        Calculate enhanced misfit using multiple professional similarity measures
        
        Parameters:
        -----------
        observed : np.ndarray
            Observed phase velocities
        predicted : np.ndarray
            Predicted phase velocities
        std : np.ndarray
            Observation errors
        periods : np.ndarray
            Period array
            
        Returns:
        --------
        combined_misfit : float
            Combined misfit value
        """
        n_points = len(observed)
        if n_points < 3:
            return 1e10  # Not enough data points
        
        # 1. Weighted L2 norm (traditional but robust)
        weighted_l2 = np.sum(((predicted - observed) / std) ** 2)
        
        # 2. Normalized RMS misfit (scale-independent)
        rms_misfit = np.sqrt(np.mean(((predicted - observed) / observed) ** 2))
        
        # 3. Correlation coefficient based misfit
        correlation = np.corrcoef(observed, predicted)[0, 1]
        if np.isnan(correlation):
            correlation_misfit = 1.0
        else:
            correlation_misfit = 1.0 - correlation  # Higher correlation = lower misfit
        
        # 4. Frequency-weighted misfit (emphasize longer periods)
        period_weights = periods / np.max(periods)  # Weight by period
        freq_weighted_misfit = np.sum(period_weights * ((predicted - observed) / std) ** 2)
        
        # 5. Shape-based misfit (focus on curve shape rather than absolute values)
        # Normalize curves to zero mean and unit variance
        obs_norm = (observed - np.mean(observed)) / np.std(observed)
        pred_norm = (predicted - np.mean(predicted)) / np.std(predicted)
        shape_misfit = np.sum((obs_norm - pred_norm) ** 2)
        
        # 6. Derivative-based misfit (match slope of dispersion curve)
        if n_points >= 4:
            # Calculate first derivatives (slope)
            obs_deriv = np.gradient(observed, periods)
            pred_deriv = np.gradient(predicted, periods)
            deriv_misfit = np.sum(((pred_deriv - obs_deriv) / (np.abs(obs_deriv) + 0.01)) ** 2)
        else:
            deriv_misfit = 0.0
        
        # 7. Logarithmic misfit (better for wide velocity ranges)
        log_misfit = np.sum((np.log(predicted) - np.log(observed)) ** 2)
        
        # Combine all misfit measures with appropriate weights
        misfit_components = {
            'weighted_l2': weighted_l2,
            'rms_misfit': rms_misfit * n_points,  # Scale to be comparable
            'correlation': correlation_misfit * n_points,
            'frequency_weighted': freq_weighted_misfit,
            'shape': shape_misfit,
            'derivative': deriv_misfit,
            'logarithmic': log_misfit
        }
        
        # Adaptive weighting based on data quality and curve characteristics
        weights = self._calculate_adaptive_weights(observed, predicted, periods)
        
        # Combine misfit components
        combined_misfit = 0.0
        for key, misfit_val in misfit_components.items():
            combined_misfit += weights[key] * misfit_val
        
        return combined_misfit
    
    def _calculate_adaptive_weights(self, observed: np.ndarray, predicted: np.ndarray,
                                  periods: np.ndarray) -> dict:
        """
        Calculate adaptive weights for different misfit components
        
        Parameters:
        -----------
        observed : np.ndarray
            Observed phase velocities
        predicted : np.ndarray
            Predicted phase velocities
        periods : np.ndarray
            Period array
            
        Returns:
        --------
        weights : dict
            Dictionary of weights for each misfit component
        """
        n_points = len(observed)
        
        # Base weights (can be adjusted based on specific requirements)
        base_weights = {
            'weighted_l2': 0.3,      # Traditional measure
            'rms_misfit': 0.2,       # Scale-independent
            'correlation': 0.15,     # Curve similarity
            'frequency_weighted': 0.1, # Emphasize longer periods
            'shape': 0.1,            # Shape matching
            'derivative': 0.1,       # Slope matching
            'logarithmic': 0.05      # Logarithmic scale
        }
        
        # Adaptive adjustments based on data characteristics
        
        # If data has high variance, emphasize shape and correlation
        obs_variance = np.var(observed)
        if obs_variance > 0.1:  # High variance
            base_weights['shape'] *= 1.5
            base_weights['correlation'] *= 1.3
        
        # If periods cover wide range, emphasize frequency weighting
        period_range = np.max(periods) - np.min(periods)
        if period_range > 10:  # Wide period range
            base_weights['frequency_weighted'] *= 1.5
            base_weights['derivative'] *= 1.2
        
        # Normalize weights to sum to 1
        total_weight = sum(base_weights.values())
        normalized_weights = {k: v/total_weight for k, v in base_weights.items()}
        
        return normalized_weights
    
    def _build_model(self, vs_params: np.ndarray, fixed_model: np.ndarray) -> np.ndarray:
        """构建完整的速度模型"""
        thickness = fixed_model[:, 0]
        vp = self.vp_vs_ratio * vs_params
        density = self.density_vp_ratio * vp
        
        return np.column_stack([thickness, vp, vs_params, density])
    
    def set_smoothing_weights(self, model: np.ndarray, water_depth: float = 0,
                            crust_thickness: float = -1) -> np.ndarray:
        """
        设置平滑权重，允许在特定界面处有较大变化
        
        Parameters:
        -----------
        model : np.ndarray
            速度模型
        water_depth : float, default=0
            水深 (km)
        crust_thickness : float, default=-1
            地壳厚度 (km)，<0表示不使用
            
        Returns:
        --------
        weights : np.ndarray
            平滑权重数组
        """
        n_layers = len(model)
        weights = np.ones(n_layers - 1)  # 差分项数量
        
        # 计算深度剖面
        depths = np.cumsum(model[:, 0])
        
        # 初始化海底界面索引
        seafloor_idx = -1
        
        # 设置海底界面处的权重
        if water_depth > 0:
            seafloor_idx = np.argmin(np.abs(depths - water_depth))
            if seafloor_idx < len(weights):
                weights[seafloor_idx] = 0.1  # 小权重允许大变化
        
        # 设置莫霍面处的权重
        if crust_thickness > 0:
            moho_depth = water_depth + crust_thickness
            moho_idx = np.argmin(np.abs(depths - moho_depth))
            if moho_idx < len(weights):
                weights[moho_idx] = 0.1  # 小权重允许大变化
                
                # 地壳内部中等平滑（仅在海底界面存在时应用）
                if seafloor_idx >= 0 and seafloor_idx < moho_idx - 2:
                    weights[seafloor_idx+1:moho_idx-1] = 0.5
                
                # 莫霍面附近较强平滑
                if moho_idx > 1 and moho_idx < len(weights) - 1:
                    weights[moho_idx-1:moho_idx+1] = 0.2
        
        return weights
    
    def invert(self, periods: np.ndarray, observed_phv: np.ndarray, 
               phv_std: np.ndarray, initial_model: np.ndarray, forward_type: str = 'phase',
               water_depth: float = 0, crust_thickness: float = -1,
               n_iterations: int = 5, damping: float = 0.1,
               vary_thickness: bool = False,
               thickness_bounds_ratio: float = 0.5,
               bounds: Optional[Tuple] = None,
               rms_error_threshold: float = None,
               max_consecutive_failures: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        支持层厚变化的频散曲线反演
        
        Parameters:
        -----------
        periods : np.ndarray
            周期数组 (s)
        observed_phv : np.ndarray
            观测相速度 (km/s)
        phv_std : np.ndarray
            观测误差 (km/s)
        initial_model : np.ndarray
            初始模型 [厚度(km), Vp(km/s), Vs(km/s), 密度(g/cm³)]
        forward_type : str, default='phase'
            正演类型 ('phase' or 'group')
        water_depth : float, default=0
            水深 (km)
        crust_thickness : float, default=-1
            地壳厚度 (km)
        n_iterations : int, default=5
            迭代次数
        damping : float, default=0.1
            阻尼系数
        vary_thickness : bool, default=False
            是否允许层厚变化
        thickness_bounds_ratio : float, default=0.5
            层厚变化范围比例 (0-1)
        bounds : tuple, optional
            参数边界
        rms_error_threshold : float, optional
            RMS误差停止阈值
max_consecutive_failures : int, default=3
            最大连续失败次数
            
        Returns:
        --------
        inverted_model : np.ndarray
            反演模型
        predicted_phv : np.ndarray
            预测相速度
        """
        print("=========== 1D Rayleigh Wave Dispersion Inversion (with Thickness Variation) ============")
        print(f"Thickness variation enabled: {vary_thickness}")
        
        # 验证输入数据
        if len(periods) != len(observed_phv) or len(periods) != len(phv_std):
            raise ValueError("输入数组长度必须相同")
        
        if np.any(phv_std <= 0):
            warnings.warn("检测到无效的标准差，使用默认值")
            phv_std = np.full_like(phv_std, 0.05)  # 默认5%误差
        
        # 设置平滑权重
        smoothing_weights = self.set_smoothing_weights(
            initial_model, water_depth, crust_thickness)
        
        # 根据是否允许层厚变化，设置不同的参数化策略
        if vary_thickness:
            # 同时优化Vs和层厚
            initial_thickness = initial_model[:, 0].copy()
            initial_vs = initial_model[:, 2].copy()
            
            # 合并初始参数 [厚度1, 厚度2, ..., Vs1, Vs2, ...]
            initial_params = np.concatenate([initial_thickness, initial_vs])
            
            # 设置参数边界
            if bounds is None:
                bounds = []
                n_layers = len(initial_thickness)
                
                # 层厚边界：允许在初始值的±thickness_bounds_ratio范围内变化
                for i in range(n_layers):
                    min_thickness = max(0.1, initial_thickness[i] * (1 - thickness_bounds_ratio))
                    max_thickness = initial_thickness[i] * (1 + thickness_bounds_ratio)
                    bounds.append((min_thickness, max_thickness))
                
                # Vs边界：保持物理约束
                for i in range(n_layers):
                    vp_val = initial_model[i, 1]
                    min_vs = max(0.1, vp_val * 0.4)  # Vs至少为Vp的40%
                    max_vs = min(5.0, vp_val * 0.99)  # Vs小于Vp
                    bounds.append((min_vs, max_vs))
            
            # 固定参数：Vp和密度
            fixed_vp = initial_model[:, 1].copy()
            fixed_density = initial_model[:, 3].copy()
            
        else:
            # 只优化Vs（保持与原始版本兼容）
            initial_thickness = initial_model[:, 0]
            initial_vs = initial_model[:, 2].copy()
            initial_params = initial_vs.copy()
            
            # 固定参数
            fixed_model = np.column_stack([initial_thickness, 
                                         initial_model[:, 1],  # Vp
                                         initial_model[:, 3]]) # density
            
            # 设置Vs边界
            if bounds is None:
                bounds = []
                for i, vs_val in enumerate(initial_vs):
                    vp_val = initial_model[i, 1]
                    min_vs = max(0.1, vp_val * 0.4)
                    max_vs = min(5.0, vp_val * 0.99)
                    bounds.append((min_vs, max_vs))
        
        # 增强的迭代反演监控
        current_params = initial_params.copy()
        best_params = initial_params.copy()
        best_misfit = float('inf')
        consecutive_failures = 0
        convergence_history = []
        
        print(f"Starting inversion with {n_iterations} iterations")
        print(f"Data points: {len(periods)}, Model layers: {len(initial_model)}")
        print(f"Parameters to optimize: {len(initial_params)}")
        
        for iteration in range(n_iterations):
            print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
            
            try:
                # 构建当前模型
                if vary_thickness:
                    n_layers = len(initial_model)
                    current_thickness = current_params[:n_layers]
                    current_vs = current_params[n_layers:]
                    current_model = np.column_stack([current_thickness, fixed_vp, current_vs, fixed_density])
                else:
                    current_vs = current_params
                    current_model = self._build_model(current_vs, fixed_model)
                
                # 测试正演模拟
                test_phv = self.forward_modeling(periods, current_model, forward_type=forward_type)
                
                if np.any(np.isnan(test_phv)):
                    warnings.warn(f"Iteration {iteration+1}: 正演模拟失败，使用先前解")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        warnings.warn("连续失败次数过多，停止反演")
                        break
                    continue
                
                # 成功时重置失败计数器
                consecutive_failures = 0
                
                # 自适应优化参数
                if iteration == 0:
                    opt_options = {'maxiter': 20, 'ftol': 1e-3, 'gtol': 1e-3, 'eps': 1e-5, 'maxls': 10}
                elif iteration < 3:
                    opt_options = {'maxiter': 30, 'ftol': 5e-4, 'gtol': 5e-4, 'eps': 1e-6, 'maxls': 15}
                else:
                    opt_options = {'maxiter': 50, 'ftol': 1e-4, 'gtol': 1e-4, 'eps': 1e-6, 'maxls': 20}
                
                # 自适应阻尼
                current_damping = damping * (0.8 ** iteration)
                current_damping = max(current_damping, 0.01)
                
                # 优化（使用不同的目标函数）
                if vary_thickness:
                    result = opt.minimize(
                        self.objective_function_with_thickness,
                        current_params,
                        args=(fixed_vp, fixed_density, periods, observed_phv, phv_std, 
                              smoothing_weights, current_damping, water_depth, crust_thickness),
                        bounds=bounds,
                        method='L-BFGS-B',
                        options=opt_options
                    )
                else:
                    result = opt.minimize(
                        self.objective_function,
                        current_params,
                        args=(fixed_model, periods, observed_phv, phv_std, 
                              smoothing_weights, current_damping, forward_type),
                        bounds=bounds,
                        method='L-BFGS-B',
                        options=opt_options
                    )
                
                if result.success:
                    current_params = result.x
                    current_misfit = result.fun
                    
                    # 构建最终模型计算质量指标
                    if vary_thickness:
                        n_layers = len(initial_model)
                        final_thickness = current_params[:n_layers]
                        final_vs = current_params[n_layers:]
                        final_model = np.column_stack([final_thickness, fixed_vp, final_vs, fixed_density])
                    else:
                        final_vs = current_params
                        final_model = self._build_model(final_vs, fixed_model)
                    
                    final_pred = self.forward_modeling(periods, final_model, forward_type=forward_type)
                    valid_indices = ~np.isnan(observed_phv) & ~np.isnan(final_pred)
                    
                    if np.sum(valid_indices) > 0:
                        valid_obs = observed_phv[valid_indices]
                        valid_pred = final_pred[valid_indices]
                        
                        correlation = np.corrcoef(valid_obs, valid_pred)[0, 1]
                        rms_error = np.sqrt(np.mean(((valid_pred - valid_obs) / valid_obs) ** 2))
                        max_error = np.max(np.abs(valid_pred - valid_obs))
                        
                        print(f"  目标函数: {current_misfit:.6f}")
                        print(f"  相关系数: {correlation:.4f}")
                        print(f"  RMS相对误差: {rms_error:.4f}")
                        print(f"  最大绝对误差: {max_error:.4f} km/s")
                        print(f"  当前阻尼: {current_damping:.4f}")
                        
                        # 如果允许层厚变化，显示层厚变化信息
                        if vary_thickness:
                            thickness_change = np.mean(np.abs(final_thickness - initial_thickness) / initial_thickness)
                            print(f"  平均层厚变化: {thickness_change:.4f}")
                    
                    # 跟踪最佳解
                    if current_misfit < best_misfit:
                        best_misfit = current_misfit
                        best_params = current_params.copy()
                        print("  *** 找到新的最佳解 ***")
                    
                    # 存储收敛历史
                    convergence_history.append(current_misfit)
                    
                    # 收敛检测（与原始版本相同）
                    if self._check_convergence(convergence_history, rms_error, rms_error_threshold):
                        break
                    
                else:
                    warnings.warn(f"Iteration {iteration+1} 优化失败: {result.message}")
                    # 回退策略
                    if best_misfit < float('inf'):
                        current_params = best_params.copy()
                        if vary_thickness:
                            # 添加小扰动避免局部最优
                            n_layers = len(initial_model)
                            current_params[:n_layers] += 0.01 * np.random.randn(n_layers)
                        print("  回退到最佳解")
                    
            except Exception as e:
                warnings.warn(f"Iteration {iteration+1} 遇到错误: {str(e)}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    warnings.warn("连续失败次数过多，停止反演")
                    break
                if best_misfit < float('inf'):
                    current_params = best_params.copy()
                    print("  错误后回退到最佳解")
        
        # 最终解选择
        if best_misfit < float('inf'):
            final_params = best_params
            print(f"\n找到最佳解，目标函数值: {best_misfit:.6f}")
        else:
            warnings.warn("未找到有效解，使用初始模型")
            final_params = initial_params.copy()
            print("使用初始模型作为回退")
        
        # 构建最终模型
        if vary_thickness:
            n_layers = len(initial_model)
            final_thickness = final_params[:n_layers]
            final_vs = final_params[n_layers:]
            inverted_model = np.column_stack([final_thickness, fixed_vp, final_vs, fixed_density])
        else:
            final_vs = final_params
            inverted_model = self._build_model(final_vs, fixed_model)
        
        # 最终预测和质量评估
        predicted_phv = self.forward_modeling(periods, inverted_model, forward_type=forward_type)
        
        # 最终验证
        valid_indices = ~np.isnan(observed_phv) & ~np.isnan(predicted_phv)
        if np.sum(valid_indices) > 0:
            valid_obs = observed_phv[valid_indices]
            valid_pred = predicted_phv[valid_indices]
            
            correlation = np.corrcoef(valid_obs, valid_pred)[0, 1] if len(valid_obs) > 1 else 1.0
            rms_relative = np.sqrt(np.mean(((valid_pred - valid_obs) / valid_obs) ** 2))
            mean_absolute = np.mean(np.abs(valid_pred - valid_obs))
            
            print("\n================= 最终结果 ===================")
            print(f"相关系数: {correlation:.4f}")
            print(f"RMS相对误差: {rms_relative:.4f}")
            print(f"平均绝对误差: {mean_absolute:.4f} km/s")
            
            if vary_thickness:
                thickness_change = np.mean(np.abs(final_thickness - initial_thickness) / initial_thickness)
                print(f"平均层厚变化: {thickness_change:.4f}")
                print(f"最终层厚: {final_thickness}")
        
        print("================= 反演完成 ===================")
        
        return inverted_model, predicted_phv

    # def objective_function(self, vs_params: np.ndarray, fixed_model: np.ndarray,
    #                       periods: np.ndarray, observed_phv: np.ndarray, 
    #                       phv_std: np.ndarray, smoothing_weights: np.ndarray,
    #                       damping: float = 0.1, forward_type: str = 'phase') -> float:
    #     """
    #     Enhanced objective function with professional dispersion curve similarity measures
        
    #     Parameters:
    #     -----------
    #     vs_params : np.ndarray
    #         Vs parameters to optimize
    #     fixed_model : np.ndarray
    #         Fixed parameter model [thickness, Vp, density]
    #     periods : np.ndarray
    #         Period array
    #     observed_phv : np.ndarray
    #         Observed phase velocity or group velocity
    #     phv_std : np.ndarray
    #         Observation error
    #     smoothing_weights : np.ndarray
    #         Smoothing weights
    #     damping : float, default=0.1
    #         Damping coefficient
    #     forward_type : str, default='phase'
    #         Forward type ('phase' or 'group')
            
    #     Returns:
    #     --------
    #     misfit : float
    #         Objective function value
    #     """
    #     # Build complete model
    #     full_model = self._build_model(vs_params, fixed_model)
        
    #     # Forward modeling with enhanced error handling
    #     predicted_phv = self.forward_modeling(periods, full_model, forward_type=forward_type)
        
    #     # 数据拟合项
    #     valid_indices = ~np.isnan(observed_phv) & ~np.isnan(predicted_phv)
    #     if np.sum(valid_indices) == 0:
    #         return 1e10
        
    #     valid_obs = observed_phv[valid_indices]
    #     valid_pred = predicted_phv[valid_indices]
    #     valid_std = phv_std[valid_indices]
        
    #     # 加权L2范数
    #     data_misfit = np.sum(((valid_pred - valid_obs) / valid_std) ** 2)
        
    #     # 模型平滑项（针对Vs）
    #     vs_smoothness = 0
    #     if len(vs) > 1:
    #         vs_diff = np.diff(vs)
    #         vs_smoothness = np.sum((vs_diff ** 2) * smoothing_weights)
        
    #     # 层厚平滑项（如果允许层厚变化）
    #     thickness_smoothness = 0
    #     if len(thickness) > 1:
    #         # 层厚变化应相对平滑
    #         thickness_diff = np.diff(np.log(thickness))  # 对数尺度更合理
    #         thickness_smoothness = np.sum(thickness_diff ** 2)
        
    #     # 总目标函数
    #     total_misfit = data_misfit + damping * (vs_smoothness + 0.5 * thickness_smoothness)
        
    #     return total_misfit

    def objective_function_with_thickness(self, params, fixed_vp, fixed_density, 
                                        periods, observed_phv, phv_std, 
                                        smoothing_weights, damping, forward_type,
                                        water_depth, crust_thickness):
        """
        支持层厚变化的目标函数
        
        Parameters:
        -----------
        params : np.ndarray
            优化参数 [厚度1, 厚度2, ..., Vs1, Vs2, ...]
        fixed_vp : np.ndarray
            固定的Vp值
        fixed_density : np.ndarray
            固定的密度值
        periods : np.ndarray
            周期数组
        observed_phv : np.ndarray
            观测相速度或群速度
        phv_std : np.ndarray
            观测误差
        smoothing_weights : np.ndarray
            平滑权重
        damping : float
            阻尼系数
        forward_type : str
            正演类型 ('phase' or 'group')
        water_depth : float
            水深
        crust_thickness : float
            地壳厚度
            
        Returns:
        --------
        total_misfit : float
            总目标函数值
        """
        n_layers = len(fixed_vp)
        thickness = params[:n_layers]
        vs = params[n_layers:]
        
        # 构建模型
        model = np.column_stack([thickness, fixed_vp, vs, fixed_density])
        
        # 计算预测值（支持群速度反演）
        predicted_phv = self.forward_modeling(periods, model, forward_type=forward_type)
        
        if np.any(np.isnan(predicted_phv)):
            return 1e10  # 返回大值表示失败
        
        # 数据拟合项
        valid_indices = ~np.isnan(observed_phv) & ~np.isnan(predicted_phv)
        if np.sum(valid_indices) == 0:
            return 1e10
        
        valid_obs = observed_phv[valid_indices]
        valid_pred = predicted_phv[valid_indices]
        valid_std = phv_std[valid_indices]
        
        # 加权L2范数
        data_misfit = np.sum(((valid_pred - valid_obs) / valid_std) ** 2)
        
        # 模型平滑项（针对Vs）
        vs_smoothness = 0
        if len(vs) > 1:
            vs_diff = np.diff(vs)
            vs_smoothness = np.sum((vs_diff ** 2) * smoothing_weights)
        
        # 层厚平滑项（如果允许层厚变化）
        thickness_smoothness = 0
        if len(thickness) > 1:
            # 层厚变化应相对平滑
            thickness_diff = np.diff(np.log(thickness))  # 对数尺度更合理
            thickness_smoothness = np.sum(thickness_diff ** 2)
        
        # 总目标函数
        total_misfit = data_misfit + damping * (vs_smoothness + 0.5 * thickness_smoothness)
        
        return total_misfit

    def _check_convergence(self, convergence_history, rms_error, rms_error_threshold):
        """检查收敛条件"""
        if len(convergence_history) < 2:
            return False
        
        # 检查相对改进
        recent_improvement = (convergence_history[-2] - convergence_history[-1]) / convergence_history[-2]
        if abs(recent_improvement) < 5e-6:
            print("  收敛达成（改进极小）")
            return True
        
        # 检查RMS误差阈值
        if rms_error_threshold is not None and rms_error <= rms_error_threshold:
            print(f"  收敛达成（RMS误差 ≤ {rms_error_threshold})")
            return True
        
        # 检查稳定性（6次迭代窗口）
        if len(convergence_history) >= 6:
            recent_changes = np.diff(convergence_history[-6:])
            if np.all(np.abs(recent_changes) < 5e-5):
                print("  收敛达成（解稳定）")
                return True
        
        # 检查连续无改进
        if len(convergence_history) >= 4:
            recent_misfits = convergence_history[-4:]
            if np.all(np.diff(recent_misfits) >= 0):
                print("  收敛达成（连续4次无改进）")
                return True
        
        return False

# Convenience function
def invdispR(periods: np.ndarray, observed_phv: np.ndarray, 
             phv_std: np.ndarray, initial_model: np.ndarray,
             water_depth: float = 0, crust_thickness: float = -1,
             n_iterations: int = 5, forward_type: str = 'phase', **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dispersion curve inversion function (MATLAB compatible interface)
    
    Parameters:
    -----------
    periods : Period array
    observed_phv : Observed phase velocity or group velocity
    phv_std : Observation error
    initial_model : Initial model [thickness, Vp, Vs, density]
    water_depth : Water depth
    crust_thickness : Crust thickness
    n_iterations : Number of iterations
    forward_type : str, default='phase'
        Forward type ('phase' or 'group')
    
    Returns:
    --------
    inverted_model : Inverted model
    predicted_phv : Predicted phase velocity or group velocity
    """
    inverter = DispersionInverter()
    return inverter.invert(periods, observed_phv, phv_std, initial_model,
                          forward_type=forward_type, water_depth=water_depth, 
                          crust_thickness=crust_thickness, n_iterations=n_iterations, **kwargs)