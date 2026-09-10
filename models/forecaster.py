"""
Aegis - ARIMA + XGBoost hybrid forecaster for oil price / supply-chain
disruption scenarios. Runs on GPU (CUDA) if available, falls back to CPU.
"""
import subprocess
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from statsmodels.tsa.arima.model import ARIMA
    HAS_MODELS = True
except ImportError:
    HAS_MODELS = False


def _detect_gpu():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            return True, "cuda"
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        pass
    return False, "cpu"


GPU_AVAILABLE, XGB_DEVICE = _detect_gpu()


def _gen_history(n=120, base=82.0, seed=42):
    """Synthetic oil-price history used to seed the model when no real
    price feed is wired in. Replace with a live price feed for production use."""
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.today(), periods=n, freq="D")
    prices = [base]
    for _ in range(n - 1):
        shock = np.random.normal(0, 1.2)
        drift = 0.05 * (base - prices[-1])
        prices.append(max(prices[-1] + drift + shock, 40))
    return pd.DataFrame({"price": prices}, index=dates)


def _features(df):
    df = df.copy()
    for lag in [1, 3, 7, 14]:
        df[f"lag_{lag}"] = df["price"].shift(lag)
    df["rm7"] = df["price"].rolling(7).mean()
    df["rs7"] = df["price"].rolling(7).std()
    df["rm14"] = df["price"].rolling(14).mean()
    df["pct3"] = df["price"].pct_change(3)
    df["dow"] = df.index.dayofweek
    return df.dropna()


FEAT = ["lag_1", "lag_3", "lag_7", "lag_14", "rm7", "rs7", "rm14", "pct3", "dow"]


class AegisForecaster:
    def __init__(self):
        self.arima = self.xgb = self.scaler = None
        self.hist = None
        self.fitted = False
        self.gpu = GPU_AVAILABLE
        self.device = XGB_DEVICE

    def fit(self, df=None):
        self.hist = df if df is not None else _gen_history()
        if not HAS_MODELS:
            self.fitted = True
            return self
        try:
            self.arima = ARIMA(self.hist["price"], order=(2, 1, 2)).fit()
        except Exception:
            self.arima = None

        fd = _features(self.hist)
        X = fd[FEAT].values
        y = fd["price"].values
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        params = dict(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
        )
        if self.gpu:
            params["device"] = "cuda"
        params["tree_method"] = "hist"

        self.xgb = xgb.XGBRegressor(**params).fit(Xs, y)
        self.fitted = True
        return self

    def forecast(self, horizon_days=14, crisis_shock=0.0, disruption_factor=0.0, blend_weight=None):
        """
        blend_weight overrides the ARIMA/XGBoost blend ratio (0.0 = pure
        ARIMA, 1.0 = pure XGBoost). Left as None in normal use, where the
        weight is chosen automatically based on crisis_shock (see below).
        This override exists so experiments/run_baseline_comparison.py can
        isolate each model's contribution for an ablation comparison
        without duplicating the forecasting logic.
        """
        if not self.fitted:
            self.fit()
        base = float(self.hist["price"].iloc[-1])
        shocked = base * (1 + crisis_shock / 100)
        dates = [datetime.today() + timedelta(days=i + 1) for i in range(horizon_days)]

        if self.arima:
            arima_p = list(self.arima.forecast(steps=horizon_days))
        else:
            arima_p = [base + np.random.normal(0, 1) * (i + 1) ** 0.5 for i in range(horizon_days)]

        rw = list(self.hist["price"].tail(14).values)
        if crisis_shock > 0:
            rw[-1] = shocked

        xgb_p = []
        for step in range(horizon_days):
            pw = rw[-14:]
            f = np.array([[
                pw[-1], pw[-3], pw[-7], pw[0],
                np.mean(pw[-7:]), np.std(pw[-7:]), np.mean(pw),
                (pw[-1] - pw[-4]) / pw[-4] if pw[-4] != 0 else 0,
                (datetime.today().weekday() + step + 1) % 7,
            ]])
            pred = float(self.xgb.predict(self.scaler.transform(f))[0])
            pred *= (1 + disruption_factor * 0.8 * (1 - step / horizon_days))
            xgb_p.append(pred)
            rw.append(pred)

        w = blend_weight if blend_weight is not None else (0.7 if crisis_shock > 0 else 0.4)
        prices = [round(w * xgb_p[i] + (1 - w) * arima_p[i], 2) for i in range(horizon_days)]
        std = float(self.hist["price"].pct_change().std()) * base
        lower = [round(p - 1.96 * std * ((i + 1) ** 0.4), 2) for i, p in enumerate(prices)]
        upper = [round(p + 1.96 * std * ((i + 1) ** 0.4), 2) for i, p in enumerate(prices)]

        fp = prices[-1]
        pct = round((fp - base) / base * 100, 1)

        return {
            "base_price": round(base, 2),
            "shocked_price": round(shocked, 2),
            "horizon_days": horizon_days,
            "gpu_accelerated": self.gpu,
            "device": "GPU (CUDA)" if self.gpu else "CPU",
            "forecast": [
                {"date": d.strftime("%Y-%m-%d"), "price": p, "lower": lo, "upper": u}
                for d, p, lo, u in zip(dates, prices, lower, upper)
            ],
            "summary": {
                "final_price": fp,
                "pct_change": pct,
                "peak_price": round(max(prices), 2),
                "peak_day": prices.index(max(prices)) + 1,
                "risk_score": min(100, round(abs(pct) * 1.5 + disruption_factor * 40 + (crisis_shock / 100) * 30, 1)),
                "delay_prob": min(99, round(disruption_factor * 65 + (pct / 100) * 20, 1)),
                "cost_impact": round(pct * 0.35 + disruption_factor * 18, 1),
            },
            "model": "ARIMA+XGBoost hybrid (GPU)" if self.gpu else "ARIMA+XGBoost hybrid (CPU)",
        }


_fc = None


def get_forecaster():
    global _fc
    if _fc is None:
        _fc = AegisForecaster().fit()
    return _fc
