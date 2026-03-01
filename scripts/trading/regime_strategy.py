from backtesting import Strategy
from backtesting.lib import SMA, EMA, RSI, ATR

class RegimeStrategy(Strategy):

    risk_per_trade = 0.005

    def init(self):
        self.sma200 = self.I(SMA, self.data.Close, 200)
        self.ema20 = self.I(EMA, self.data.Close, 20)
        self.rsi2 = self.I(RSI, self.data.Close, 2)
        self.atr = self.I(ATR, self.data.High,
                          self.data.Low,
                          self.data.Close, 14)

        # self.cp_slope = self.I(my_cp_slope_indicator)

    def next(self):

        price = self.data.Close[-1]
        sma200 = self.sma200[-1]
        # cp = self.cp_slope[-1]

        bull = price > sma200 and cp > 0
        bear = price < sma200 and cp < 0

        if not self.position:

            if bull and self.rsi2[-1] < 10:
                size = self._position_size()
                self.buy(size=size)

            elif bear and self.rsi2[-1] > 90:
                size = self._position_size()
                self.sell(size=size)

        else:
            if self.position.is_long:
                if price > self.ema20[-1]:
                    self.position.close()

            elif self.position.is_short:
                if price < self.ema20[-1]:
                    self.position.close()

    def _position_size(self):
        risk = self.equity * self.risk_per_trade
        stop_distance = 2 * self.atr[-1]
        return risk / stop_distance