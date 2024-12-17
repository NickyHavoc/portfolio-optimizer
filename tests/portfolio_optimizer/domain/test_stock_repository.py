from typing import Sequence

import pandas as pd
from portfolio_optimizer import StockRepository


def test_stock_repository_can_get_stocks(
    stock_repository: StockRepository,
    example_stock_ticker_symbols: Sequence[str]
) -> None:
    example_stocks_df = stock_repository.get_stocks(
        ticker_symbols=example_stock_ticker_symbols,
        start_date='2020-01-01',
        end_date='2024-12-31',
    )

    assert len(example_stocks_df.columns) == len(example_stock_ticker_symbols)
    assert len(example_stocks_df) == 1258


def test_stock_repository_can_get_ticker_symbols_for_date_range(
    stock_repository: StockRepository,
    date_range: tuple[str, str],
) -> None:

    ticker_symbols = stock_repository.get_available_ticker_symbols(
        date_range[0], date_range[1]
    )

    assert len(ticker_symbols) == 5
