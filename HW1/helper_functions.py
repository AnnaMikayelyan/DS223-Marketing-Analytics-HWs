import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def bass_new_adopters(params, y):
    p, q, M = params
    T = len(y)
    Y_cum = 0.0
    y_hat = np.zeros(T)
    for t in range(T):
        y_hat[t] = (p + q * (Y_cum / max(M, 1e-9))) * max(M - Y_cum, 0.0)
        Y_cum += y_hat[t]
    return y_hat


def bass_model(t, p, q, M):
    return M * (1 - np.exp(-(p + q) * t)) / (1 + (q/p) * np.exp(-(p + q) * t))


def residuals(params, y):
    return bass_new_adopters(params, y) - y


def fit_bass_model(y):
    y = np.array(y)
    M0 = y.sum() * 2
    p0, q0 = 0.01, 0.4
    x0 = np.array([p0, q0, M0])
    lower = np.array([1e-6, 1e-6, y.sum()*1.01])
    upper = np.array([1.0, 1.5, y.sum()*50])
    res = least_squares(residuals, x0, bounds=(lower, upper), args=(y,))
    p_hat, q_hat, M_hat = res.x
    y_hat = bass_new_adopters(res.x, y)
    sse = np.sum((y_hat - y)**2)
    sst = np.sum((y - y.mean())**2)
    r2 = 1 - sse/sst if sst > 0 else np.nan
    return p_hat, q_hat, M_hat, r2, y_hat


def fit_bass_model_a(df, col_name):
    t = df["Year"] - df["Year"].min()
    y = df[col_name]
    params, _ = curve_fit(bass_model, t, y, p0=[0.03, 0.38, max(y)])
    p, q, M = params
    y_pred = bass_model(t, p, q, M)
    return p, q, M, y_pred


def plot_bass_fit(df, actual_col, predicted, title, ylabel):
    plt.plot(df["Year"], df[actual_col], "o-", label=f"Actual {ylabel}")
    plt.plot(df["Year"], predicted, "s--", label="Bass Model Prediction")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.legend()