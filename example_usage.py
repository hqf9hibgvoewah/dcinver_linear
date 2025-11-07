"""
Example: Python implementation of dispersion curve inversion
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from dispersion_inversion import DispersionInverter, invdispR
from utils import create_layered_model, calculate_rms_misfit, calculate_enhanced_misfit, print_misfit_summary

def example_1d_inversion():
    """1D inversion example with enhanced misfit evaluation"""
    
    print("Starting 1D dispersion curve inversion example...")
    
    # Generate synthetic data
    periods = np.linspace(5, 40, 20)  # 5-40 seconds period
    
    # True model (3-layer structure)
    true_depths = np.array([2, 10, 35])  # Interface depths
    true_vp = np.array([4.0, 6.0, 8.0])  # P-wave velocity
    true_vs = np.array([2.3, 3.5, 4.5])  # S-wave velocity
    
    true_model = create_layered_model(true_depths, true_vp, true_vs)
    
    # Forward modeling for "observed" data
    inverter = DispersionInverter()
    true_phv = inverter.forward_modeling(periods, true_model)
    
    # Add noise
    noise_level = 0.02  # 2% noise
    observed_phv = true_phv * (1 + noise_level * np.random.randn(len(periods)))
    phv_std = np.full_like(periods, noise_level * np.mean(true_phv))
    
    # Initial model (simpler than true model)
    init_depths = np.array([5, 20, 50])
    init_vp = np.array([4.5, 6.5, 8.0])
    init_vs = np.array([2.0, 3.0, 4.0])
    
    initial_model = create_layered_model(init_depths, init_vp, init_vs)
    
    try:
        # Execute inversion with improved settings
        print("Starting inversion...")
        inverted_model, predicted_phv = invdispR(
            periods, observed_phv, phv_std, initial_model,
            water_depth=0, crust_thickness=30, n_iterations=200,  # Reduced iterations for testing
            damping=0.05  # Reduced damping for better convergence
        )
        
        # Enhanced misfit evaluation
        metrics = calculate_enhanced_misfit(observed_phv, predicted_phv, phv_std, periods)
        print_misfit_summary(metrics)
        
        # Traditional RMS misfit for comparison
        rms = calculate_rms_misfit(observed_phv, predicted_phv, phv_std)
        print(f"Traditional RMS misfit: {rms:.4f} km/s")
        
        # Plot results
        plot_results(periods, observed_phv, true_phv, predicted_phv, 
                    true_model, initial_model, inverted_model)
        
    except Exception as e:
        print(f"Error during inversion: {e}")
        # Fallback: use forward modeling with initial model
        predicted_phv = inverter.forward_modeling(periods, initial_model)
        
        # Calculate fallback misfit
        metrics = calculate_enhanced_misfit(observed_phv, predicted_phv, phv_std, periods)
        print_misfit_summary(metrics)
        
        rms = calculate_rms_misfit(observed_phv, predicted_phv, phv_std)
        print(f"Fallback RMS misfit: {rms:.4f} km/s")
        
        plot_results(periods, observed_phv, true_phv, predicted_phv, 
                    true_model, initial_model, initial_model)

def test_enhanced_misfit():
    """Test the enhanced misfit calculation with different scenarios"""
    print("\n=== Testing Enhanced Misfit Calculation ===")
    
    # Create test data
    periods = np.linspace(5, 40, 10)
    observed = np.array([3.0, 3.2, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1])
    std = np.full_like(observed, 0.05)
    
    # Test 1: Perfect match
    predicted1 = observed.copy()
    metrics1 = calculate_enhanced_misfit(observed, predicted1, std, periods)
    print("Test 1 - Perfect match:")
    print_misfit_summary(metrics1)
    
    # Test 2: Small offset
    predicted2 = observed + 0.1
    metrics2 = calculate_enhanced_misfit(observed, predicted2, std, periods)
    print("\nTest 2 - Small offset:")
    print_misfit_summary(metrics2)
    
    # Test 3: Different shape
    predicted3 = np.array([3.0, 3.3, 3.3, 3.6, 3.6, 3.9, 3.9, 4.2, 4.2, 4.5],dtype=float)
    metrics3 = calculate_enhanced_misfit(observed, predicted3, std, periods)
    print("\nTest 3 - Different shape:")
    print_misfit_summary(metrics3)
    
    # Test 4: Large difference
    predicted4 = observed * 1.2
    metrics4 = calculate_enhanced_misfit(observed, predicted4, std, periods)
    print("\nTest 4 - Large difference:")
    print_misfit_summary(metrics4)

def plot_results(periods, observed_phv, true_phv, predicted_phv,
                true_model, initial_model, inverted_model):
    """Plot inversion results"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Dispersion curve fitting plot
    ax1.plot(periods, true_phv, 'k-', linewidth=2, label='True Model')
    ax1.errorbar(periods, observed_phv, yerr=0.02, fmt='ro', 
                markersize=6, label='Observed Data')
    ax1.plot(periods, predicted_phv, 'b--', linewidth=2, label='Inverted Result')
    ax1.set_xlabel('Period (s)')
    ax1.set_ylabel('Phase Velocity (km/s)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Dispersion Curve Fitting')
    
    # Velocity model comparison plot
    plot_velocity_models(ax2, true_model, initial_model, inverted_model)
    ax2.set_title('Velocity Model Comparison')
    
    plt.tight_layout()
    plt.show()

def plot_velocity_models(ax, true_model, initial_model, inverted_model):
    """Plot velocity model comparison"""
    
    # Calculate depth profiles
    true_depths = np.cumsum(true_model[:, 0])
    init_depths = np.cumsum(initial_model[:, 0])
    inv_depths = np.cumsum(inverted_model[:, 0])
    
    # Extend velocity arrays for step plot
    true_vs = np.repeat(true_model[:, 2], 2)
    init_vs = np.repeat(initial_model[:, 2], 2)
    inv_vs = np.repeat(inverted_model[:, 2], 2)
    
    true_depths_plot = np.repeat(np.insert(true_depths, 0, 0), 2)[1:-1]
    init_depths_plot = np.repeat(np.insert(init_depths, 0, 0), 2)[1:-1]
    inv_depths_plot = np.repeat(np.insert(inv_depths, 0, 0), 2)[1:-1]
    
    ax.plot(true_vs, true_depths_plot, 'k-', linewidth=2, label='True Model')
    ax.plot(init_vs, init_depths_plot, 'r--', linewidth=2, label='Initial Model')
    ax.plot(inv_vs, inv_depths_plot, 'b-', linewidth=2, label='Inverted Model')
    
    ax.set_xlabel('S-wave Velocity (km/s)')
    ax.set_ylabel('Depth (km)')
    ax.invert_yaxis()  # Depth increases downward
    ax.legend()
    ax.grid(True, alpha=0.3)
# %%
if __name__ == "__main__":
    # %%Test enhanced misfit calculation
    test_enhanced_misfit()
    
    # Run inversion example
    example_1d_inversion()
# %%
