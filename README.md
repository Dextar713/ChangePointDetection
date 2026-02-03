# Change Point Detection for Financial Time Series

This project explores **change-point detection methods** and their application to **algorithmic trading strategies** on financial time series. The work combines theoretical foundations, efficient algorithmic implementations, and practical evaluation through backtesting on real market data.

The core idea is to detect **structural breaks in price dynamics** (trend changes) using linear models and cost-based segmentation algorithms, then exploit these regime changes in a systematic trading strategy.

---

## Project Goals

* Implement **fast cost computation** for linear regression–based change-point detection
* Study and implement classical algorithms such as:

  * Binary Segmentation
  * Optimal Partitioning
* Build an **online change-point detection wrapper** suitable for streaming data
* Design and evaluate **trend-following trading strategies** based on detected change points
* Analyze robustness, parameter sensitivity, and limitations of change-point–based trading

---

## Repository Structure

```
ChangePointDetection/
│
├── cost_computers.py        # Efficient Cost Computers (RSS, least squares)
├── bin_seg.py               # Binary segmentation
├── opt_seg.py               # Optimal Partitioning
├── online_detector.py       # Online / streaming change-point detection wrappers
├── strategy_improved.py     # Trading strategy based on change points 
├── data/                    # Data for stock prices
├── prepare_data.py          # Data loading and preprocessing
├── run_test.py              # Testing detection algorithms on time series
├── visuals/                 # Generated plots and visualizations
├── report/                  # LaTeX report and bibliography
└── README.md
```


---

## Methodology Overview

### 1. Linear Cost Model

Segments are modeled using **simple linear regression**. For a segment ([a, b]), the cost is defined as the **Residual Sum of Squares (RSS)**:

$$
\mathrm{RSS}(a,b) = \sum_{t=a}^{b} (x_t - \hat{x}_t)^2
$$


Fast cost computation techniques are used to allow efficient segmentation over long time series.

---

### 2. Change-Point Detection Algorithms

The project implements and studies:

* **Binary Segmentation**
  A greedy, recursive approach suitable for fast offline detection.

* **Optimal Partitioning**
  A dynamic programming method that finds a globally optimal segmentation under a given cost + penalty formulation.

Both methods rely on the same linear cost function, enabling direct comparison.

---

### 3. Online Change-Point Detection

An **online wrapper** is built on top of offline algorithms to enable:

* Sequential data processing
* Detection with limited lookahead (horizon-based)
* Practical applicability to live trading scenarios

This allows the system to approximate real-time regime detection while remaining computationally feasible.

---

### 4. Trading Strategy Design

The trading strategy is based on:

* Detecting **trend change points** in price series
* Estimating **local and global trend slopes** using linear regression
* Trading only when local and global trends agree
* Dynamic filtering using:

  * Volatility-normalized thresholds
  * Exponential averaging of trend strength
  * Volume-based filters

Both **long and short** positions are supported, with risk management via stop-loss and take-profit rules.

---

## Backtesting & Evaluation

* Backtests are performed using historical market data (e.g. S&P 500, AMD)
* Performance is evaluated using:

  * Sharpe ratio
  * Annualized returns
  * Comparison against Buy & Hold benchmark

### Key Observations

* The strategy can **outperform Buy & Hold** on certain regimes (e.g. S&P 500, 2020–2026)
* Performance is **highly sensitive** to parameters such as minimum segment length
* Results vary significantly across assets and time periods
* Small parameter changes can lead to large performance differences, highlighting **model instability and regime dependence**

---

## Limitations

* Strong sensitivity to hyperparameters (e.g. minimum distance between change points)
* Non-stationarity of financial markets reduces robustness
* Change-point detection does not guarantee predictive power
* Risk of overfitting when tuning parameters on historical data

These findings suggest that change-point detection should be treated as a **contextual signal**, not a standalone trading edge.

---

## Technologies Used

* Python
* NumPy / SciPy
* Pandas
* Matplotlib
* Backtesting.py
* LaTeX (for documentation)

---

## References

The theoretical background and algorithms are based on literature covering:

* Linear regression and least squares fitting
* Offline and online change-point detection
* Binary segmentation and optimal partitioning
* Bayesian and cost-based approaches to regime detection

(See `ref.bib` in the report directory for full references.)

---

## Disclaimer

This project is for **educational and research purposes only**.
It does not constitute financial advice, and the strategies presented should not be used for real trading without thorough independent validation.

---

## Author

Developed by **Dextar713** as a research and engineering project on change-point detection
