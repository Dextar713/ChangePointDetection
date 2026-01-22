import datetime
from backtesting import Strategy, Backtest
from matplotlib import pyplot as plt
import numpy as np 
from online_detector import NaiveOnlineDetector
from prepare_data import load_data, prepare_data
import pandas as pd
import os 

all_change_points = []

class LinearTrendCPStrategy(Strategy):
    def init(self):
        self.min_dist = 10
        self.min_slope_angle = 25
        self.max_slope_angle = 70
        self.min_exit_angle = 25
        self.stop_loss_pct = -10.0  # Stop loss at -7%
        self.take_profit_pct = 15.0
        # self.min_log_return = 0.01
        self.position_size = 0.75  # 30% of current equity per trade
        self.detector = NaiveOnlineDetector(cost_type='linear', model_type='opt', min_dist=self.min_dist)
        self.entry_price = 0.0
        self.initial_equity = self.equity

    def next(self):
        

        target_col = 'Close'
        log_price = self.data[target_col][-1]
        current_price = self.data['Close'][-1]  
        
        # self.prices.append(price)        
        if self.position:
            if self.position.is_long:
                diff_pct = (current_price - self.entry_price) / self.entry_price * 100
                if diff_pct < self.stop_loss_pct or diff_pct > self.take_profit_pct:
                    self.position.close()
            elif self.position.is_short:
                diff_pct = (self.entry_price - current_price) / self.entry_price * 100
                if diff_pct < self.stop_loss_pct or diff_pct > self.take_profit_pct:
                    self.position.close()

        is_detected = self.detector.update(log_price)
        if not is_detected:
            return 
        # print(np.exp(log_price))
        # print(current_price)
        # print('-----------')
        last_change_point = self.detector.last_change_point
        all_change_points.append(self.detector.last_change_point+self.detector.offset)
        end_point = len(self.detector.signal_buffer)
        range_start = max(last_change_point + 2+ self.detector.offset, 0)
        range_end = end_point+self.detector.offset
        data_range = self.data[target_col][range_start:range_end]
        slope_log = np.polyfit(np.arange(range_end-range_start), data_range ,1)[0]
        # print('Slope: ', slope_log)
        angle_log = np.degrees(np.arctan(slope_log))
        # print(f'Start: {last_change_point+self.detector.offset}, End: {end_point+self.detector.offset}')
        print('Angle: ', angle_log)
        if angle_log > self.min_exit_angle or angle_log < -self.max_slope_angle:
            if self.position.is_short:
                print(f'Close short with angle {angle_log} date: {self.data.index[-1]}')
                self.position.close()
        elif angle_log < -self.min_exit_angle or angle_log > self.max_slope_angle:
            if self.position.is_long:
                print(f'Close long with angle {angle_log} date: {self.data.index[-1]}')
                self.position.close()

        if self.min_slope_angle < angle_log < self.max_slope_angle:
            if not self.position.is_long:
                print(f'Open long with angle {angle_log} date: {self.data.index[-1]}')
                self.position.close()
                cur_pos_size = self.position_size * self.initial_equity / self.equity
                self.buy(size=cur_pos_size)
                self.entry_price = current_price
        elif -self.max_slope_angle < angle_log < -self.min_slope_angle:
            if not self.position.is_short:
                print(f'Open short with angle {angle_log} date: {self.data.index[-1]}')
                self.position.close()
                cur_pos_size = self.position_size * self.initial_equity / self.equity

                self.sell(size=cur_pos_size)
                self.entry_price = current_price


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


if __name__ == '__main__':
    data = load_data(ticker='AMD', start_date=datetime.datetime(2020, 1, 1), interval='1d')
    data = prepare_data(data)
    target_col = 'Close'
    start, end = -1000, -100
    # start, end = -700, -300
    # print(data[start:end].index[:5])

    data_slice = data[start:end]
    test_strategy(data_slice)

    plt.figure(figsize=(12, 6))
    plt.plot(data_slice.index, data_slice[target_col].values)
    
    # Convert numeric changepoints to datetime using the data slice's index
    for point in all_change_points:
        if point < len(data_slice):
            plt.axvline(x=data_slice.index[point], color='red', linestyle='--', linewidth=1)
    
    save_path = os.path.join(os.path.dirname(__file__), '../visuals/trading_cp.png')
    plt.savefig(save_path)

        
