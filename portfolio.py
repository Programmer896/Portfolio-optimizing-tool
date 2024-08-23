import numpy as np
import pandas as pd
import scipy.optimize as sco

# Sample data: Daily returns for three assets
data = pd.DataFrame({
    'Asset1': np.random.normal(0.001, 0.02, 252),
    'Asset2': np.random.normal(0.0015, 0.015, 252),
    'Asset3': np.random.normal(0.002, 0.025, 252)
})

# Calculate expected returns, covariance matrix, and number of assets
returns = data.mean()
cov_matrix = data.cov()
num_assets = len(data.columns)

# Define portfolio statistics functions
def portfolio_performance(weights, returns, cov_matrix):
    portfolio_return = np.sum(returns * weights) * 252
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return portfolio_return, portfolio_stddev

def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.01):
    p_return, p_stddev = portfolio_performance(weights, returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_stddev

def max_sharpe_ratio(returns, cov_matrix):
    num_assets = len(returns)
    args = (returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = sco.minimize(negative_sharpe_ratio, num_assets * [1. / num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Run optimization
optimal_portfolio = max_sharpe_ratio(returns, cov_matrix)
optimal_weights = optimal_portfolio['x']

# Output results
print("Optimal Weights:", optimal_weights)
print("Expected Portfolio Return:", portfolio_performance(optimal_weights, returns, cov_matrix)[0])
print("Expected Portfolio Volatility:", portfolio_performance(optimal_weights, returns, cov_matrix)[1])

