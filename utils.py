"""
辅助工具函数
"""

import numpy as np
from typing import Union
from scipy.interpolate import interp1d

def vr2vs(vr,sita): # Vr transform vs, sita is Piosson Ratio
    vs = ((1 + sita) * vr)/(0.87 + 1.12 * sita)
    return vs
def vr2vp(vr,sita): # Vr transform vp
    vs = vr2vs(vr,sita)
    vp = 1.73 * vs
    return vp
def vr2ly(vr,freq,a): # Vr transform layer
    ly = a * vr / freq
    return ly
def vr2ly_period(vr,period,a): # Vr transform layer
    ly = a * vr * period
    return ly
def SmoothLine(x,y,n): # scatter line smooth.
    new_inter = interp1d(x,y,kind='cubic')
    # new_x = np.linspace(x[0],x[-1],multiple*len(x))
    new_x = np.linspace(x[0],x[-1],n)
    new_y = new_inter(new_x)
    return new_x,new_y
def empirical_initial_model(vr,period,a,manual_ly=None,c_model=1):
    """
    基于 vr 和 period 计算初始模型
    """
    # 计算 vp 和 vs
    vp = vr2vp(vr,0.25)
    vs = vr2vs(vr,0.25)
    ly = vr2ly_period(vr,period,a)    
    if c_model == 1:
        return vp, vs, ly
    elif c_model == 2 and manual_ly is not None:
        new_ly,new_vs = SmoothLine(ly,vs,len(manual_ly))
        new_vp = vr2vp(new_vs,0.25)
        return new_vp, new_vs

def create_layered_model(depths: np.ndarray, vp: Union[float, np.ndarray], 
                        vs: Union[float, np.ndarray], 
                        density: Union[float, np.ndarray] = None,
                        vp_vs_ratio: float = 1.9, 
                        density_vp_ratio: float = 0.32) -> np.ndarray:
    """
    创建分层速度模型
    
    Parameters:
    -----------
    depths : np.ndarray
        界面深度数组 (km)，从地表开始
    vp : float or np.ndarray
        P波速度或速度数组 (km/s)
    vs : float or np.ndarray  
        S波速度或速度数组 (km/s)
    density : float or np.ndarray, optional
        密度或密度数组 (g/cm³)
    vp_vs_ratio : float, default=1.9
        Vp/Vs比值（当vs为标量时使用）
    density_vp_ratio : float, default=0.32
        ρ/Vp比值（当density为None时使用）
        
    Returns:
    --------
    model : np.ndarray
        速度模型 [厚度, Vp, Vs, 密度]
    """
    # 计算层厚度
    thickness = np.diff(np.append(0, depths))
    
    # 处理速度参数
    if np.isscalar(vp):
        vp = np.full(len(thickness), vp)
    if np.isscalar(vs):
        vs = np.full(len(thickness), vs)
    
    # 处理密度参数
    if density is None:
        density = density_vp_ratio * vp
    elif np.isscalar(density):
        density = np.full(len(thickness), density)
    
    return np.column_stack([thickness, vp, vs, density])

def calculate_enhanced_misfit(observed: np.ndarray, predicted: np.ndarray,
                            errors: np.ndarray = None, 
                            periods: np.ndarray = None) -> dict:
    """
    Calculate enhanced misfit metrics for dispersion curve comparison
    
    Parameters:
    -----------
    observed : np.ndarray
        Observed phase velocities
    predicted : np.ndarray
        Predicted phase velocities
    errors : np.ndarray, optional
        Observation errors
    periods : np.ndarray, optional
        Period array for frequency weighting
        
    Returns:
    --------
    metrics : dict
        Dictionary of misfit metrics
    """
    valid_indices = ~np.isnan(observed) & ~np.isnan(predicted)
    if np.sum(valid_indices) < 2:
        return {'error': 'Insufficient valid data points'}
    
    obs_valid = observed[valid_indices]
    pred_valid = predicted[valid_indices]
    n_points = len(obs_valid)
    
    metrics = {}
    
    # 1. Weighted L2 norm
    if errors is not None:
        err_valid = errors[valid_indices]
        metrics['weighted_l2'] = np.sum(((pred_valid - obs_valid) / err_valid) ** 2)
    else:
        metrics['weighted_l2'] = np.sum((pred_valid - obs_valid) ** 2)
    
    # 2. RMS relative error
    metrics['rms_relative'] = np.sqrt(np.mean(((pred_valid - obs_valid) / obs_valid) ** 2))
    
    # 3. Correlation coefficient
    if n_points >= 2:
        correlation = np.corrcoef(obs_valid, pred_valid)[0, 1]
        metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0
    else:
        metrics['correlation'] = 1.0
    
    # 4. Mean absolute error
    metrics['mean_absolute'] = np.mean(np.abs(pred_valid - obs_valid))
    
    # 5. Maximum absolute error
    metrics['max_absolute'] = np.max(np.abs(pred_valid - obs_valid))
    
    # 6. Frequency-weighted misfit (if periods provided)
    if periods is not None:
        period_valid = periods[valid_indices]
        if len(period_valid) > 0:
            period_weights = period_valid / np.max(period_valid)
            if errors is not None:
                freq_weighted = np.sum(period_weights * ((pred_valid - obs_valid) / err_valid) ** 2)
            else:
                freq_weighted = np.sum(period_weights * (pred_valid - obs_valid) ** 2)
            metrics['frequency_weighted'] = freq_weighted
    
    # 7. Shape-based misfit
    obs_norm = (obs_valid - np.mean(obs_valid)) / np.std(obs_valid)
    pred_norm = (pred_valid - np.mean(pred_valid)) / np.std(pred_valid)
    metrics['shape_misfit'] = np.sum((obs_norm - pred_norm) ** 2)
    
    # 8. Derivative-based misfit (if enough points and periods provided)
    if n_points >= 4 and periods is not None:
        period_valid = periods[valid_indices]
        obs_deriv = np.gradient(obs_valid, period_valid)
        pred_deriv = np.gradient(pred_valid, period_valid)
        deriv_misfit = np.sum(((pred_deriv - obs_deriv) / (np.abs(obs_deriv) + 0.01)) ** 2)
        metrics['derivative_misfit'] = deriv_misfit
    
    # 9. Logarithmic misfit
    metrics['log_misfit'] = np.sum((np.log(pred_valid) - np.log(obs_valid)) ** 2)
    
    # 10. Combined quality score (0-1, higher is better)
    quality_score = (metrics['correlation'] + 
                   1.0 / (1.0 + metrics['rms_relative']) + 
                   1.0 / (1.0 + metrics['mean_absolute'])) / 3.0
    metrics['quality_score'] = quality_score
    
    return metrics

def print_misfit_summary(metrics: dict):
    """
    Print a summary of misfit metrics
    
    Parameters:
    -----------
    metrics : dict
        Misfit metrics from calculate_enhanced_misfit
    """
    print("\n=== Misfit Metrics Summary ===")
    
    if 'error' in metrics:
        print(f"Error: {metrics['error']}")
        return
    
    print(f"Correlation coefficient: {metrics.get('correlation', 0):.4f}")
    print(f"RMS relative error: {metrics.get('rms_relative', 0):.4f}")
    print(f"Mean absolute error: {metrics.get('mean_absolute', 0):.4f} km/s")
    print(f"Maximum absolute error: {metrics.get('max_absolute', 0):.4f} km/s")
    print(f"Quality score: {metrics.get('quality_score', 0):.4f}")
    
    # Quality assessment
    correlation = metrics.get('correlation', 0)
    if correlation > 0.95:
        print("Fit quality: Excellent")
    elif correlation > 0.85:
        print("Fit quality: Good")
    elif correlation > 0.70:
        print("Fit quality: Fair")
    else:
        print("Fit quality: Poor")

def depth_interp(depth_profile: np.ndarray, model_thickness: np.ndarray,
                model_vs: np.ndarray) -> np.ndarray:
    """
    在深度剖面上插值速度
    
    Parameters:
    -----------
    depth_profile : 目标深度数组
    model_thickness : 模型层厚度
    model_vs : 模型S波速度
    
    Returns:
    --------
    interpolated_vs : 插值后的速度
    """
    depths = np.cumsum(model_thickness)
    depths = np.insert(depths, 0, 0)  # 添加地表
    
    # 扩展速度数组（地表值）
    extended_vs = np.insert(model_vs, 0, model_vs[0])
    
    # 在目标深度处插值
    interpolated_vs = np.interp(depth_profile, depths, extended_vs,
                               left=extended_vs[0], right=extended_vs[-1])
    
    return interpolated_vs


def calculate_rms_misfit(observed: np.ndarray, predicted: np.ndarray,
                        errors: np.ndarray = None) -> float:
    """
    计算RMS misfit
    
    Parameters:
    -----------
    observed : 观测值
    predicted : 预测值
    errors : 误差（加权时使用）
    
    Returns:
    --------
    rms : RMS misfit
    """
    valid_indices = ~np.isnan(observed) & ~np.isnan(predicted)
    if np.sum(valid_indices) == 0:
        return float('inf')  # 没有有效数据点
    
    obs_valid = observed[valid_indices]
    pred_valid = predicted[valid_indices]
    
    if errors is not None:
        err_valid = errors[valid_indices]
        weights = 1.0 / err_valid**2
        misfit = np.sqrt(np.sum(weights * (obs_valid - pred_valid)**2) / np.sum(weights))
    else:
        misfit = np.sqrt(np.mean((obs_valid - pred_valid)**2))
    
    return misfit
def conservative_phv_std_strategy(observed_phv, periods, noise_level = 0.03):
    """保守策略：使用较大的标准差提高稳定性"""
    # 基础误差设为3%
    base_std = noise_level * np.mean(observed_phv)
    
    # 根据周期调整：长周期数据误差更大
    period_weights = periods / np.max(periods)
    phv_std = base_std * (1 + 0.1 * period_weights)  # 长周期误差增加50%
    
    return phv_std