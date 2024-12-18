from typing import Sequence
from pytest import fixture

from portfolio_optimizer import PortfolioOptimizer
from portfolio_optimizer import MockStockFetcher


@fixture
def portfolio_optimizer(
    mock_stock_fetcher: MockStockFetcher
) -> PortfolioOptimizer:
    return PortfolioOptimizer(mock_stock_fetcher)


def test_portfolio_optimizer_works(
    portfolio_optimizer: PortfolioOptimizer,
    example_stock_ticker_symbols: Sequence[str],
    date_range: tuple[str, str]
) -> None:
    portfolio = portfolio_optimizer.find_optimal_portfolio(
        example_stock_ticker_symbols, date_range[0], date_range[1]
    )

    assert portfolio.security('BULL').weight > portfolio.security('BEAR').weight
    assert portfolio.security('WOLF').performance.annual_return > portfolio.performance.annual_return
    assert all(
        portfolio.performance.sharpe > security.performance.sharpe for security in portfolio.securities
    )


def test_portfolio_optimizer_works_for_weighing_return_and_risk(
    portfolio_optimizer: PortfolioOptimizer,
    example_stock_ticker_symbols: Sequence[str],
    date_range: tuple[str, str]
) -> None:
    portfolio_high_return = portfolio_optimizer.find_optimal_portfolio(
        example_stock_ticker_symbols, date_range[0], date_range[1], weight_return=1
    )
    portfolio_low_risk = portfolio_optimizer.find_optimal_portfolio(
        example_stock_ticker_symbols, date_range[0], date_range[1], weight_return=0
    )

    assert portfolio_high_return.performance.annual_return > portfolio_low_risk.performance.annual_return
    assert portfolio_high_return.performance.annual_risk > portfolio_low_risk.performance.annual_risk
