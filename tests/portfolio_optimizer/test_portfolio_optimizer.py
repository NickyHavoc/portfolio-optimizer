from typing import Sequence
import pandas as pd
from pytest import fixture

from portfolio_optimizer import PortfolioOptimizer
from stock_getter import CsvStockGetter


@fixture
def portfolio_optimizer(
    csv_stock_getter: CsvStockGetter
) -> PortfolioOptimizer:
    return PortfolioOptimizer(csv_stock_getter)


def test_portfolio_optimizer_works(
    portfolio_optimizer: PortfolioOptimizer,
    example_stock_ticker_symbols: Sequence[str],
    date_range: tuple[str, str]
) -> None:
    portfolio = portfolio_optimizer.find_optimal_portfolio(
        example_stock_ticker_symbols, date_range[0], date_range[1]
    )

    assert portfolio.security('BULL').weight > portfolio.security('BEAR').weight
    assert portfolio.security('BULL').annual_return > portfolio.performance.annual_return
