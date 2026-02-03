import numpy as np
import matplotlib.pyplot as plt


def fit_and_plot(y: np.ndarray):
    """
    Fits a linear regression line (degree=1) to a 1D numpy array
    and plots the data along with the fitted line.
    """
    x = np.arange(len(y))

    # Fit linear regression: y = ax + b
    a, b = np.polyfit(x, y, deg=1)

    # Fitted values
    y_fit = a * x + b

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label='Data')
    plt.plot(x, y_fit, label=f'Fit: y = {a:.4f}x + {b:.4f}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Linear Regression Fit (np.polyfit)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return a, b


# Example usage
if __name__ == "__main__":
    data = np.array([4140.06005859, 4122.47021484, 4210.24023438, 4207.27001953, 4280.14990234,
 4297.14013672, 4305.20019531, 4274.04003906, 4283.74023438, 4228.47998047,
 4137.99023438, 4128.72998047])
    slope, intercept = fit_and_plot(data)
    print(f"Slope: {slope:.6f}, Intercept: {intercept:.6f}") 