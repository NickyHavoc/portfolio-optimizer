from typing import Any, Iterable, Sequence
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
    performance: SecurityPerformance | None = None


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
        ticker_symbols: Iterable[str],
        start_date: str,
        end_date: str,
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
        risk_free_rate: float = 0.0,
        fixed_securities: Sequence[PortfolioSecurity] = list(),
        max_weight: float = 0.2  # New parameter
    ) -> Portfolio:
        assert 0 <= weight_return <= 1
        assert 0 < max_weight <= 1, "max_weight must be between 0 and 1"

        fixed_ticker_symbols = [fs.ticker_symbol for fs in fixed_securities]
        fixed_stocks = self.stocks(fixed_ticker_symbols, start_date, end_date)
        fixed_stocks.columns = ['FIXED-' + col for col in fixed_stocks.columns]
        new_stocks = self.stocks(ticker_symbols, start_date, end_date)
        stocks = pd.concat([fixed_stocks, new_stocks], axis=1)
        joined_ticker_symbols = fixed_ticker_symbols + ticker_symbols

        returns_day_to_day = stocks.pct_change().dropna()
        mean_returns_annualized = self._calculate_annualized_return(stocks)
        stdev_day_to_day = returns_day_to_day.std()
        stdev_annualized = stdev_day_to_day * np.sqrt(self.stock_fetcher.data_frequency.approx_frequency_factor)
        covariance_matrix = returns_day_to_day.cov()
        num_assets = len(stocks.columns)

        # Fixed weights for securities (now we handle column name changes for fixed stocks)
        fixed_weights = np.zeros(num_assets)
        for fixed_security in fixed_securities:
            # Match the ticker with the "FIXED-" prefix
            fixed_ticker = 'FIXED-' + fixed_security.ticker_symbol
            idx = stocks.columns.get_loc(fixed_ticker)
            fixed_weights[idx] = fixed_security.weight

        # Remaining budget to optimize
        open_budget = 1 - np.sum(fixed_weights)
        assert open_budget >= 0, "Fixed weights exceed total budget!"

        # Adjust constraints for optimization
        def constraint_function(weights):
            return np.sum(weights) - open_budget

        constraints = [{'type': 'eq', 'fun': constraint_function}]

        # Bounds only apply to optimizable securities (those not fixed)
        bounds = [
            (0, max_weight) if fixed_weights[i] == 0 else (0, 0)
            for i in range(num_assets)
        ]
        initial_weights = np.ones(num_assets) * (open_budget / np.count_nonzero(fixed_weights == 0))

        # Perform optimization
        result: OptimizeResult = minimize(
            self._optimize,
            initial_weights,
            args=(mean_returns_annualized, covariance_matrix, weight_return, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        # Combine fixed and optimized weights
        optimized_weights = result['x']
        final_weights = fixed_weights + optimized_weights

        # Portfolio statistics
        optimal_return = self._calculate_weighted_return(final_weights, mean_returns_annualized)
        optimal_risk = self._calculate_portfolio_risk(final_weights, covariance_matrix)

        # Return the optimal portfolio with performance statistics
        return Portfolio(
            securities=[
                PortfolioSecurity(
                    ticker_symbol=ticker_symbol,
                    weight=weight,
                    performance=SecurityPerformance(
                        sharpe=self._calculate_sharpe_ratio(annual_return, annual_risk, risk_free_rate),
                        annual_return=annual_return,
                        annual_risk=annual_risk
                    )
                ) for ticker_symbol, weight, annual_return, annual_risk in zip(
                    joined_ticker_symbols, final_weights, mean_returns_annualized, stdev_annualized
                )
            ],
            performance=SecurityPerformance(
                sharpe=self._calculate_sharpe_ratio(optimal_return, optimal_risk, risk_free_rate),
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
