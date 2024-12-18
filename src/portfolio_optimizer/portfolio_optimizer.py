from typing import Any, Sequence
import numpy as np
from numpy.typing import NDArray

import pandas as pd
from pydantic import BaseModel
from scipy.optimize import minimize
from scipy.optimize._optimize import OptimizeResult

from .domain.base_stock_fetcher import BaseStockFetcher


class SecurityPerformance(BaseModel):
    sharpe: float
    annual_return: float
    annual_risk: float


class PortfolioSecurity(BaseModel):
    ticker_symbol: str
    weight: float
    performance: SecurityPerformance


class Portfolio(BaseModel):
    securities: Sequence[PortfolioSecurity]
    performance: SecurityPerformance
    risk_free_rate: float

    def security(self, ticker_symbol: str) -> PortfolioSecurity | None:
        return next(
            (
                security for security in self.securities if (
                    security.ticker_symbol == ticker_symbol
                )
            ), None
        )
    
    def securities_to_dataframe(self) -> pd.DataFrame:
            data = [
                {
                    "Ticker": security.ticker_symbol,
                    "Weight": security.weight,
                    "Sharpe": security.performance.sharpe,
                    "Annual Return (%)": security.performance.annual_return * 100,
                    "Annual Risk (%)": security.performance.annual_risk * 100,
                }
                for security in self.securities
            ]
            df = pd.DataFrame(data)
            return df.sort_values(by="Weight", ascending=False).reset_index(drop=True)


class PortfolioOptimizer:
    def __init__(self, stock_fetcher: BaseStockFetcher) -> None:
        self.stock_fetcher = stock_fetcher

    def stocks(
        self,
        ticker_symbols: Sequence[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        # consider caching
        return self.stock_fetcher.fetch(
            ticker_symbols, start_date, end_date
        )

    def find_optimal_portfolio(
        self,
        ticker_symbols: Sequence[str],
        start_date: str,
        end_date: str,
        weight_return: float = 0.5,
        risk_free_rate: float = 0.0
    ) -> Portfolio:
        assert 0 <= weight_return <= 1

        stocks = self.stocks(ticker_symbols, start_date, end_date)
        
        returns_day_to_day = stocks.pct_change().dropna() # dropna before?
        mean_returns_annualized = self._calculate_annualized_return(stocks)

        stdev_day_to_day = returns_day_to_day.std()
        stdev_annualized = stdev_day_to_day * np.sqrt(self.stock_fetcher.data_frequency.approx_frequency_factor)

        covariance_matrix = returns_day_to_day.cov()
        num_assets = len(stocks.columns)

        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        # Bounds: asset weights between 0 and 1 (no short selling)
        bounds = tuple((0, 1) for _ in range(num_assets))
        # Initial guess: equal allocation
        initial_weights = np.ones(num_assets) / num_assets
        # Perform optimization to minimize the negative Sharpe ratio
        result: OptimizeResult = minimize(self._optimize, initial_weights, args=(
            mean_returns_annualized,
            covariance_matrix,
            weight_return,
            risk_free_rate
        ), method='SLSQP', bounds=bounds, constraints=constraints)

        # Get the optimal portfolio weights
        optimal_weights = result['x']

        # replace with functions for both
        optimal_return = self._calculate_weighted_return(optimal_weights, mean_returns_annualized)
        optimal_risk = self._calculate_portfolio_risk(optimal_weights, covariance_matrix)

        return Portfolio(
            securities=[
                PortfolioSecurity(
                    ticker_symbol=ticker_symbol,
                    weight=weight,
                    performance=SecurityPerformance(
                        sharpe=self._calculate_sharpe_ratio(
                            annual_return, annual_risk, risk_free_rate
                        ),
                        annual_return=annual_return,
                        annual_risk=annual_risk
                    )
                ) for ticker_symbol, weight, annual_return, annual_risk in zip(
                    ticker_symbols, optimal_weights, mean_returns_annualized, stdev_annualized
                )
            ],
            performance=SecurityPerformance(
                sharpe=self._calculate_sharpe_ratio(
                    optimal_return, optimal_risk, risk_free_rate
                ),
                annual_return=optimal_return,
                annual_risk=optimal_risk,
            ),
            risk_free_rate=risk_free_rate
        )

    def _calculate_annualized_return(self, stocks: pd.DataFrame) -> NDArray[np.floating[Any]]:
        num_years = len(stocks) / self.stock_fetcher.data_frequency.approx_frequency_factor
        return np.array([
            (final / initial) ** (1 / num_years) - 1
            for initial, final in zip(stocks.iloc[0], stocks.iloc[-1])
        ], dtype=np.float64)

    def _optimize(
        self,
        weights: NDArray[np.floating[Any]],
        mean_annualized_returns: NDArray,
        cov_matrix: pd.DataFrame,
        weight_return: float,
        risk_free_rate: float
    ) -> float:
        return self._determine_sharpe_ratio_for_optimization(
            weights,
            mean_annualized_returns,
            cov_matrix,
            weight_return,
            risk_free_rate
        )

    def _determine_sharpe_ratio_for_optimization(
        self,
        weights: NDArray[np.floating[Any]],
        mean_annualized_returns: NDArray,
        covariance_matrix: pd.DataFrame,
        weight_return: float,
        risk_free_rate: float
    ) -> float:
        portfolio_return = self._calculate_weighted_return(weights, mean_annualized_returns)
        portfolio_risk = self._calculate_portfolio_risk(weights, covariance_matrix)
        return - self._calculate_sharpe_ratio(
            portfolio_return, portfolio_risk, risk_free_rate, weight_return
        )

    @staticmethod
    def _calculate_weighted_return(
        weights: NDArray[np.floating[Any]],
        mean_annualized_returns: NDArray
    ) -> float:
        return np.sum(weights * mean_annualized_returns)

    def _calculate_portfolio_risk(
        self,
        weights: NDArray[np.floating[Any]],
        covariance_matrix: pd.DataFrame
    ) -> float:
        return np.sqrt(
            np.dot(weights.T, np.dot(covariance_matrix, weights))
        ) * np.sqrt(self.stock_fetcher.data_frequency.approx_frequency_factor)

    @staticmethod
    def _calculate_sharpe_ratio(
        security_return: float,
        security_risk: float,
        risk_free_rate: float,
        # setting default here, too, for standard sharpe
        weight_return: float = 0.5,
    ) -> float:
        return_exponent = (2 * weight_return)
        risk_exponent = (2 * (1 - weight_return ))
        return (
            ( security_return - risk_free_rate ) ** return_exponent
        ) / ( security_risk  ** risk_exponent )
