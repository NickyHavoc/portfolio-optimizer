from typing import Any, Sequence
import numpy as np
from numpy.typing import NDArray

import pandas as pd
from pydantic import BaseModel
from scipy.optimize import minimize
from scipy.optimize._optimize import OptimizeResult

from stock_getter.stock_getter import StockGetter


class PortfolioSecurity(BaseModel):
    ticker_symbol: str
    weight: float
    annual_return: float
    annual_risk: float


class PortfolioPerformance(BaseModel):
    sharpe: float
    annual_return: float
    annual_risk: float
    risk_free_rate: float


class Portfolio(BaseModel):
    securities: Sequence[PortfolioSecurity]
    performance: PortfolioPerformance

    def security(self, ticker_symbol: str) -> PortfolioSecurity | None:
        return next(
            (
                security for security in self.securities if (
                    security.ticker_symbol == ticker_symbol
                )
            ), None
        )


class PortfolioOptimizer:
    def __init__(self, stock_getter: StockGetter) -> None:
        self.stock_getter = stock_getter
        self.trade_days_per_year = 252

    def stocks(
        self,
        ticker_symbols: Sequence[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        # consider caching
        return self.stock_getter.get_stocks(
            ticker_symbols, start_date, end_date
        )

    def find_optimal_portfolio(
        self,
        ticker_symbols: Sequence[str],
        start_date: str,
        end_date: str
    ) -> Portfolio:
        stocks = self.stocks(ticker_symbols, start_date, end_date)
        
        returns_day_to_day = stocks.pct_change().dropna() # dropna before?
        mean_returns_annualized = self._calculate_annualized_return(stocks)

        stdev_day_to_day = returns_day_to_day.std()
        stdev_annualized = stdev_day_to_day * np.sqrt(self.trade_days_per_year)

        covariance_matrix = returns_day_to_day.cov()
        num_assets = len(stocks.columns)

        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        # Bounds: asset weights between 0 and 1 (no short selling)
        bounds = tuple((0, 1) for asset in range(num_assets))
        # Initial guess: equal allocation
        initial_weights = np.ones(num_assets) / num_assets
        # Perform optimization to minimize the negative Sharpe ratio
        result: OptimizeResult = minimize(self._negative_sharpe_ratio, initial_weights, args=(
            mean_returns_annualized, covariance_matrix
        ), method='SLSQP', bounds=bounds, constraints=constraints)

        # Get the optimal portfolio weights
        optimal_weights = result['x']

        optimal_return = np.sum(optimal_weights * mean_returns_annualized)
        optimal_risk = np.sqrt(
            np.dot(
                optimal_weights.T, np.dot(covariance_matrix, optimal_weights)
            )
        ) * np.sqrt(self.trade_days_per_year)

        return Portfolio(
            securities=[
                PortfolioSecurity(
                    ticker_symbol=ticker_symbol,
                    weight=weight,
                    annual_return=annual_return,
                    annual_risk=annual_risk
                ) for ticker_symbol, weight, annual_return, annual_risk in zip(
                    ticker_symbols, optimal_weights, mean_returns_annualized, stdev_annualized
                )
            ],
            performance=PortfolioPerformance(
                annual_return=optimal_return,
                annual_risk=optimal_risk,
                sharpe=optimal_return / optimal_risk,
                risk_free_rate=0.0
            )
        )

    @staticmethod
    def _calculate_annualized_return(stocks: pd.DataFrame) -> NDArray[np.floating[Any]]:
        # this probably should be done somewhere else
        num_years = len(stocks) / 252 # should this be hardcoded?
        return np.array([
            (final / initial) ** (1 / num_years) - 1
            for initial, final in zip(stocks.iloc[0], stocks.iloc[-1])
        ], dtype=np.float64)

    @staticmethod
    def _negative_sharpe_ratio(
        weights: NDArray[np.floating[Any]],
        mean_annualized_returns: NDArray,
        cov_matrix
    ) -> float:
        # replace by more generic opimization function with weights for sharpe, risk, return
        portfolio_return = np.sum(weights * mean_annualized_returns)
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        ) * np.sqrt(252)  # Annualized risk
        return -portfolio_return / portfolio_volatility
