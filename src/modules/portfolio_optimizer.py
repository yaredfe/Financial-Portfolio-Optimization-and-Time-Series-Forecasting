import numpy as np
import scipy.optimize as sco

def calculate_returns_and_covariance(forecast_df):
    returns = forecast_df.pct_change().dropna()
    mean_returns = returns.mean() * 252  # Annualized returns
    covariance_matrix = returns.cov() * 252  # Annualized covariance
    return mean_returns, covariance_matrix

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    p_std, p_ret = portfolio_annualized_performance(weights, mean_returns, cov_matrix)
    return -((p_ret - risk_free_rate) / p_std)

def portfolio_annualized_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return std, returns

def optimize_portfolio(mean_returns, covariance_matrix, num_assets):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = sco.minimize(negative_sharpe_ratio, num_assets * [1. / num_assets,],
                          args=(mean_returns, covariance_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x