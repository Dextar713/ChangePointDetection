import datetime
from backtesting import Strategy, Backtest
from matplotlib import pyplot as plt
import numpy as np 
from online_detector import NaiveOnlineDetector
from prepare_data import load_data, prepare_data
import pandas as pd
import os 

all_change_points = []
angles_local = []

class BuyAndHoldStrategy(Strategy):
    def init(self):
        self.has_bought = False

    def next(self):
        if not self.has_bought:
            self.buy(size=0.95)  # Buy with 95% of available cash
            self.has_bought = True

# AMD 20-25 min slope, 60-70 max slope, 11-12 min dist
# AAPL 12 min slope, 60 max slope, 13 min dist
# NVDA 4 min slope, 15 max slope, 12 min dist
class LinearTrendCPStrategy(Strategy):
    def init(self):
        self.min_dist = 12  # 11 - 12 good range
        self.detector_horizon = 30   # 30 optimal
        self.cur_trend_delay = 2  # 2 is optimal
        self.min_slope_angle = 0.25  # 25 optimal       # 0.5 AMD   #0.1 NVDA   
        self.max_slope_angle = 2.0 # 70 optimal      # 6.2
        self.min_exit_angle = 0.5  # 25 optimal      # 0.5 AMD   #0.2 NVDA
        self.stop_loss_pct = -10.0  # Stop loss at -10%
        self.take_profit_pct = 15.0  # 15.0 Take profit at 15%
        # self.min_log_return = 0.01
        self.position_size = 0.35  # 75% of current equity per trade
        self.detector = NaiveOnlineDetector(cost_type='linear', model_type='opt', 
                                            min_dist=self.min_dist, horizon_size=self.detector_horizon)
        self.entry_price = 0.0
        self.initial_equity = self.equity

    def next(self):
        

        target_col = 'log_close'
        # log_price = self.data[target_col][-1]
        current_price = self.data['Close'][-1]  
        
        # self.prices.append(price)        
        if self.position:
            if self.position.is_long:
                diff_pct = (current_price - self.entry_price) / self.entry_price * 100
                if diff_pct < self.stop_loss_pct or diff_pct > self.take_profit_pct:
                    print(f'Close long SL/TP date: {self.data.index[-1]}')
                    self.position.close()
            elif self.position.is_short:
                diff_pct = (self.entry_price - current_price) / self.entry_price * 100
                if diff_pct < self.stop_loss_pct or diff_pct > self.take_profit_pct:
                    print(f'Close short SL/TP date: {self.data.index[-1]}')
                    self.position.close()

        is_detected = self.detector.update(current_price)
        if not is_detected:
            return 
        # print(np.exp(log_price))
        # print(current_price)
        # print('-----------')
        last_change_point = self.detector.last_change_point
        all_change_points.append(self.detector.last_change_point+self.detector.offset)
        end_point = len(self.detector.signal_buffer)
        range_start = max(last_change_point + self.cur_trend_delay + self.detector.offset, 0)
        range_end = end_point+self.detector.offset
        data_range = self.data[target_col][range_start:range_end]
        slope_local = np.polyfit(np.arange(range_end-range_start), data_range ,1)[0]
        # print('Slope local: ', slope_local)
        daily_return_pct = (np.exp(slope_local) - 1) * 100
        angle_local = daily_return_pct
        # angle_local = np.degrees(np.arctan(slope_local))
        angles_local.append(abs(angle_local))
        global_range_start = end_point + self.detector.offset - 3*self.detector_horizon
        if global_range_start < 0:
            angle_global = angle_local
        else:
            global_data_range = self.data[target_col][global_range_start:range_end]
            slope_global = np.polyfit(np.arange(range_end-global_range_start), global_data_range ,1)[0]
            # angle_global = np.degrees(np.arctan(slope_global))
            daily_return_pct_global = (np.exp(slope_global) - 1) * 100
            angle_global = daily_return_pct_global
        # print(f'Start: {last_change_point+self.detector.offset}, End: {end_point+self.detector.offset}')
        # print('Angle: ', angle_local)

        if np.sign(angle_local) != np.sign(angle_global) and abs(angle_local) < abs(angle_global):
            # pass
            return

        if angle_local > self.min_exit_angle or angle_local < -self.max_slope_angle:
            if self.position.is_short:
                print(f'Close short with angle {angle_local} date: {self.data.index[-1]}')
                self.position.close()
        elif angle_local < -self.min_exit_angle or angle_local > self.max_slope_angle:
            if self.position.is_long:
                print(f'Close long with angle {angle_local} date: {self.data.index[-1]}')
                self.position.close()

        
        if self.min_slope_angle < angle_local < self.max_slope_angle:
            if not self.position.is_long:
                print(f'Open long with angle {angle_local} date: {self.data.index[-1]}')
                self.position.close()
                # cur_pos_size = self.position_size * self.initial_equity / self.equity
                cur_pos_size = self.position_size * self.initial_equity / max(self.initial_equity, self.equity)
                self.buy(size=cur_pos_size)
                self.entry_price = current_price
        elif -self.max_slope_angle < angle_local < -self.min_slope_angle:
            if not self.position.is_short:
                print(f'Open short with angle {angle_local} date: {self.data.index[-1]}')
                self.position.close()
                # cur_pos_size = self.position_size * self.initial_equity / self.equity
                cur_pos_size = self.position_size * self.initial_equity / max(self.initial_equity, self.equity)
                self.sell(size=cur_pos_size)
                self.entry_price = current_price

def test_buy_hold(data: pd.DataFrame):
    bt = Backtest(
        data,
        BuyAndHoldStrategy,
        cash=10_000,
        commission=0.001
    )
    stats = bt.run()
    print(stats)
    bt.plot()

def test_strategy(data: pd.DataFrame):
    bt = Backtest(
        data,
        LinearTrendCPStrategy,
        cash=10_000,
        commission=0.001
    )
    stats = bt.run()
    print(stats)
    # print(stats['_trades'])
    # with pd.ExcelWriter('strategy_results.xlsx') as writer:
    #     stats.to_frame(name='value').to_excel(writer, sheet_name='summary')
    #     stats['_trades'].to_excel(writer, sheet_name='trades')
    #     stats['_equity_curve'].to_excel(writer, sheet_name='equity')
    bt.plot()
    print('Angles local stats: ', min(angles_local), np.percentile(angles_local, 25), np.median(angles_local), np.percentile(angles_local, 75), max(angles_local))


if __name__ == '__main__':
    data = load_data(ticker='AMD', start_date=datetime.datetime(2020, 1, 1), interval='1d')
    data = prepare_data(data)
    target_col = 'Close'
    # start, end = -1000, -700
    # start, end = -700, -300
    # start, end = 0, len(data)
    # start, end = -450, -50
    start, end = -500, -100
    # start, end = -800, -200
    # start, end = -500, -10
    # start, end = -200, -70
    # print(data[start:end].index[:5])

    data_slice = data[start:end]
    test_strategy(data_slice)
    # test_buy_hold(data_slice)

    plt.figure(figsize=(12, 6))
    plt.plot(data_slice.index, data_slice[target_col].values)
    
    # Convert numeric changepoints to datetime using the data slice's index
    for point in all_change_points:
        if point < len(data_slice):
            plt.axvline(x=data_slice.index[point], color='red', linestyle='--', linewidth=1)
    
    save_path = os.path.join(os.path.dirname(__file__), '../visuals/trading_cp.png')
    plt.savefig(save_path)

        
