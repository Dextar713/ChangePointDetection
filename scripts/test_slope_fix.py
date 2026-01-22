import numpy as np
from cost_computers import LinearCostComputer

# Create synthetic data with known slope
np.random.seed(42)
slope_true = 2.5  # True slope
intercept_true = 5.0  # True intercept
x = np.arange(0, 20)
y = intercept_true + slope_true * x + np.random.normal(0, 0.1, size=20)  # Add small noise

print(f"True slope: {slope_true}")
print(f"True intercept: {intercept_true}")
print()

# Test different segments
computer = LinearCostComputer(y)

# Test 1: Full segment from 0 to 20
slope, intercept = computer.get_slope_intercept(0, 20)
print(f"Segment [0, 20]:")
print(f"  Detected slope: {slope:.6f}")
print(f"  Detected intercept: {intercept:.6f}")
print(f"  Angle: {np.degrees(np.arctan(slope)):.2f}°")
print()

# Test 2: Middle segment from 5 to 15
slope, intercept = computer.get_slope_intercept(5, 15)
print(f"Segment [5, 15]:")
print(f"  Detected slope: {slope:.6f}")
print(f"  Detected intercept: {intercept:.6f}")
print(f"  Angle: {np.degrees(np.arctan(slope)):.2f}°")
print()

# Test 3: Small segment from 10 to 15
slope, intercept = computer.get_slope_intercept(10, 15)
print(f"Segment [10, 15]:")
print(f"  Detected slope: {slope:.6f}")
print(f"  Detected intercept: {intercept:.6f}")
print(f"  Angle: {np.degrees(np.arctan(slope)):.2f}°")
print()

# Verify with numpy polyfit
print("=" * 70)
print("NumPy polyfit (absolute coordinates - x is the actual index):")
for start, end in [(0, 20), (5, 15), (10, 15)]:
    x_segment = np.arange(start, end)
    y_segment = y[start:end]
    coeffs = np.polyfit(x_segment, y_segment, 1)
    print(f"NumPy polyfit [{start}, {end}]: slope={coeffs[0]:.6f}, intercept={coeffs[1]:.6f}")
    
    # Convert NumPy's absolute intercept to relative intercept
    intercept_relative = coeffs[1] + coeffs[0] * start
    print(f"  → Converted to relative coords: intercept_relative={intercept_relative:.6f}")
    print()

print("=" * 70)
print("Function intercepts are in RELATIVE coordinates (x=0 at segment start)")
print("NumPy intercepts are in ABSOLUTE coordinates (x is the actual index)")
