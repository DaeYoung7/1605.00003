import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('AAPL.csv', index_col='Date', parse_dates=True)
price = data['Adj Close']
vol = data['Volume']

# RSI
price_changed = (price.shift(1) - price)
up, down = price_changed.copy(), price_changed.copy()
up[up < 0.] = np.nan
down[down > 0.] = np.nan
au = up.rolling(14, min_periods=1).mean().ffill()
ad = abs(down.rolling(14, min_periods=1).mean().ffill())
rsi = au / (au + ad) * 100

# stochastic oscillator
l14 = price.rolling(14, min_periods=1).min()
h14 = price.rolling(14, min_periods=1).max()
so = (price - l14) / (h14 - l14) * 100

# william %R
wr = (h14 - price) / (h14 - l14) * -100

# moving average convergence divergence
ema12 = price.ewm(span=12).mean()
ema26 = price.ewm(span=26).mean()
mach = ema12 - ema26

# price rate of change
proc = price.shift(14) / price - 1

# on balance volume
down = price_changed < 0.
vol_ = vol.copy()
vol_[down] *= -1
obv = vol_[1:].cumsum()

X = pd.concat([rsi, so, wr, mach, proc, obv], axis=1)
X.columns = ['RSI', 'SO', 'WR', 'MACH', 'PROC', 'OBV']
X = X[X.index.year > 1981]
label = price / price.shift(-30) - 1
label[label > 0.] = 1
label[label < 0.] = 0
label = label[label.index.year > 1981]
label = label.dropna()
X = X.loc[label.index]

criterion = pd.Timestamp(2017, 1, 1)

model = RandomForestClassifier()
model.fit(X[X.index < criterion].values, label[label.index < criterion].values)
y_pred = model.predict(X[X.index > criterion])
print(accuracy_score(y_pred, label[label.index > criterion]))
print(classification_report(y_pred, label[label.index > criterion]))