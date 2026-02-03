import datetime
from backtesting import Strategy, Backtest
from matplotlib import pyplot as plt
import numpy as np 
from online_detector import NaiveOnlineDetector, FastOnlineDetector
from prepare_data import load_data, prepare_data
import pandas as pd
import os 

all_change_points = []
trend_noise = []

class BuyAndHoldStrategy(Strategy):
    def init(self):
        self.has_bought = False

    def next(self):
        if not self.has_bought:
            self.buy(size=0.95)  # Buy with 95% of available cash
            self.has_bought = True


class VolatilityNormalizedTrendCPStrategy(Strategy):
    """
    Improved strategy with volatility-normalized thresholds.
    Thresholds automatically adjust based on stock's recent volatility,
    making the strategy work better across different stocks.
    """
    
    def init(self):
        self.min_dist = 17
        self.detector_horizon = 30
        self.lookback_period = 50
        self.cur_trend_delay = 3
                
        # Risk management
        self.stop_loss_pct = -90.0
        self.take_profit_pct = 95.0
        self.position_size = 0.4
        self.avg_trend = None
        self.alpha = 0.7
        self.volume_window = 40
        
        self.detector = NaiveOnlineDetector(
            cost_type='linear', 
            model_type='opt', 
            min_dist=self.min_dist, 
            horizon_size=self.detector_horizon
        )
        
        self.entry_price = 0.0
        self.initial_equity = self.equity
        self.data_count = 0
        self.init_trade = None
        self.prev_trade = None


    def next(self):
        self.data_count += 1
        
        target_col = 'log_close'
        current_price = self.data['Close'][-1]
        
        # Position management: stop loss and take profit
        if self.position:
            if self.position.is_long:
                diff_pct = (current_price - self.entry_price) / self.entry_price * 100
                if diff_pct < self.stop_loss_pct or diff_pct > self.take_profit_pct:
                    print(f'Close long SL/TP ({diff_pct:.2f}%) date: {self.data.index[-1]}')
                    self.position.close()
            elif self.position.is_short:
                diff_pct = (self.entry_price - current_price) / self.entry_price * 100
                if diff_pct < self.stop_loss_pct or diff_pct > self.take_profit_pct:
                    print(f'Close short SL/TP ({diff_pct:.2f}%) date: {self.data.index[-1]}')
                    self.position.close()

        # Detect change points in the trend
        is_detected = self.detector.update(current_price)

        # if self.init_trade is None:
        #     self.init_trade = True
        #     self.buy(size=0.5*self.initial_equity)
        #     self.init_trade.cl
        # if is_detected:
        #     print(f'Change point detected at index {self.data_count} date: {self.data.index[-1]}')
        if not is_detected:
            return 

        if self.data_count < self.detector_horizon:
            return
        
        # Analyze local trend (after change point)
        last_change_point = self.detector.last_change_point
        all_change_points.append(self.detector.last_change_point + self.detector.offset)
        end_point = self.detector.n_samples - self.detector.offset
        self.cur_trend_delay = int((end_point - last_change_point) / self.min_dist*2)
        range_start = last_change_point + self.cur_trend_delay + self.detector.offset
        range_end = end_point + self.detector.offset
        # range_start = max(0, range_end - self.min_dist)
        
        data_range = self.data[target_col][range_start:range_end]
            
        slope_local = np.polyfit(np.arange(range_end - range_start), data_range, 1)[0]
        # angle_local = np.mean(np.diff(data_range)) * 100  # Approximate slope as mean daily change

        angle_local = np.degrees(np.arctan(slope_local))  # Slope angle in degrees
        # angle_local = (np.exp(slope_local) - 1) * 100
        # angles_local.append(abs(angle_local))  # For global tracking
        
        # Analyze global trend (longer-term context)
        global_range_start = max(end_point + self.detector.offset - self.lookback_period, 0)
        # print(range_end-range_start)
        
        global_data_range = self.data[target_col][global_range_start:range_end]
        if len(global_data_range) < self.min_dist:
            angle_global = angle_local
        else:
            slope_global = np.polyfit(
                np.arange(range_end - global_range_start), 
                global_data_range, 
                1
            )[0]
            # angle_global = (np.exp(slope_global) - 1) * 100
            angle_global = np.degrees(np.arctan(slope_global))  # Slope angle in degrees
            # angle_global = np.mean(np.diff(global_data_range)) * 100  # Approximate slope as mean daily change
        
        returns = np.diff(np.log(self.data['Close'][-self.lookback_period:]))
        trend = abs(np.mean(returns))
        noise = np.std(returns)

        regime_score = trend / (noise + 1e-8)
        trend_noise.append(regime_score)
        # self.position_size = np.clip(a=0.5-regime_score, a_min=0.05, a_max=0.9)
        self.position_size = np.clip(0.1 + 2*regime_score, 0.1, 0.9)

        # print(f'Regime score: {regime_score:.4f}')
        # if regime_score < 0.09:
        #     return # Skip trading in low regime score (high noise) conditions

        if self.avg_trend is None:
            self.avg_trend = abs(angle_local)
        
        self.avg_trend = (1 - self.alpha) * self.avg_trend + self.alpha * abs(angle_local)

        if angle_local > 0:
            if self.position.is_short:
                print(f'Close short (reversal) angle={angle_local:.3f}% date: {self.data.index[-1]}')
                self.position.close()
        elif angle_local < 0:
            if self.position.is_long:
                print(f'Close long (reversal) angle={angle_local:.3f}% date: {self.data.index[-1]}')
                self.position.close()

        

        if abs(angle_local) < 0.9*self.avg_trend:
            # pass
            return

        # Filter: only trade if local and global trends agree

        if len(data_range)<2*self.min_dist and np.sign(angle_local) != np.sign(angle_global) and abs(angle_local) < abs(angle_global):
            # pass
            return
        
        
        
        if self.data_count >= self.volume_window:
            avg_volume = self.data['Volume'][-self.volume_window:].mean()
            # print(f'Avg volume over last {self.volume_window} days: {avg_volume:.0f}')
            if self.data['Volume'][-1] < avg_volume * 0.75:
                print(f'Low volume day ({self.data["Volume"][-1]:.0f} < {avg_volume*0.75:.0f}), skipping trade date: {self.data.index[-1]}')
                # Skip trading on low volume days
                return 
                # pass
    
        
        # Entry logic: buy on uptrend, sell on downtrend
        if angle_local > 0:
            if not self.position.is_long:
                # print(f'Open long angle={angle_local:.3f}% date: {self.data.index[-1]}')
                self.position.close()
                cur_pos_size = self.position_size * self.initial_equity / max(self.initial_equity, self.equity)
                self.buy(size=cur_pos_size)
                # self.sell(size=cur_pos_size)
                self.entry_price = current_price
                
        elif angle_local < 0:
            if not self.position.is_short:
                # print(f'Open short angle={angle_local:.3f}% date: {self.data.index[-1]}')
                # print('Open short angle local:', angle_local, 'Angle global:', angle_global)
                self.position.close()
                cur_pos_size = self.position_size * self.initial_equity / max(self.initial_equity, self.equity)
                self.sell(size=cur_pos_size)
                # self.buy(size=cur_pos_size)
                self.entry_price = current_price



def test_buy_hold(data: pd.DataFrame):
    """Run buy-and-hold strategy as benchmark."""
    bt = Backtest(
        data,
        BuyAndHoldStrategy,
        margin=1.0,
        cash=10_000,
        commission=0.001,
        finalize_trades=True
    )
    stats = bt.run()
    print("\n" + "="*60)
    print("BUY AND HOLD STRATEGY")
    print("="*60)
    print(stats['Sharpe Ratio'])
    # bt.plot()
    return stats


def test_improved_strategy(data: pd.DataFrame):
    """Run improved volatility-normalized strategy."""
    # Reset global state
    global all_change_points
    all_change_points = []
    
    bt = Backtest(
        data,
        VolatilityNormalizedTrendCPStrategy,
        cash=100_000,
        commission=0.001,
        finalize_trades=True,
        # trade_on_close=True
    )
    stats = bt.run()
    print("\n" + "="*60)
    print("VOLATILITY-NORMALIZED STRATEGY")
    print("="*60)
    print(stats['Sharpe Ratio'])
    # bt.plot()
    
    return stats


def plot_changepoints(data: pd.DataFrame):
    """Visualize detected change points on price chart."""
    plt.figure(figsize=(14, 6))
    plt.plot(data.index, data['Close'].values, label='Close Price', linewidth=2)
    
    # Mark change points
    for point in all_change_points:
        if point < len(data):
            plt.axvline(x=data.index[point], color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.title('Stock Price with Detected Change Points')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(os.path.dirname(__file__), '../visuals/strategy_improved_cp.png')
    plt.savefig(save_path)
    print(f"\nChart saved to {save_path}")


if __name__ == '__main__':
    # Test on different stocks
    ticker = '^GSPC'  # Try: 'AMD', 'AAPL', 'NVDA'
    
    print(f"\nLoading data for {ticker}...")
    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2020, 1, 1)
    data = load_data(ticker=ticker, start_date=start_date, end_date=end_date, interval='1d')
    data = prepare_data(data)
    
    ranges = [(-800, -200), (-550, -150), (-1050, -470), (0, len(data)), (-300, -80), (-1100, -350), (-750, -400), (-900, -540), (-600, -350)]
    all_sharpe_ratios = []
    all_ann_returns = []
    buy_hold_sharpe = []
    buy_hold_returns = []

    for i in range(0, 9):
        print("\n" + "="*60)
        print(f"Running strategy test {i+1}/{len(ranges)}...")
        start, end = ranges[i]
        
        data_slice = data[start:end]
        
        print(f"\nTesting on {ticker} from {data_slice.index[0].date()} to {data_slice.index[-1].date()}")
        print(f"Num data points: {len(data_slice)}")
        
        
        print("\nStep 2: Running backtest...")
        print("\n" + "="*60) 
        stats_buy_hold = test_buy_hold(data_slice)
        stats_improved = test_improved_strategy(data_slice)
        # print(stats_improved)
        all_sharpe_ratios.append(stats_improved['Sharpe Ratio'])
        all_ann_returns.append(stats_improved['Return (Ann.) [%]'])
        buy_hold_sharpe.append(stats_buy_hold['Sharpe Ratio'])
        buy_hold_returns.append(stats_buy_hold['Return (Ann.) [%]'])
    
    # STEP 3: Plot results
    print("\nStep 3: Plotting results...")
    plot_changepoints(data_slice)
     
    print("\n" + "="*60)
    print(f"Strategy completed for {ticker}")
    print('Mean cp trend sharpe ratio:', np.mean(all_sharpe_ratios))
    print('Min cp trend sharpe ratio:', np.min(all_sharpe_ratios))
    print('Mean cp trend annual return:', np.mean(all_ann_returns))
    print('Buy and hold mean sharpe ratio:', np.mean(buy_hold_sharpe))
    print('Buy and hold min sharpe ratio:', np.min(buy_hold_sharpe))
    print('Buy and hold mean annual return:', np.mean(buy_hold_returns))
    print('Mean regime score: ', np.mean(trend_noise))