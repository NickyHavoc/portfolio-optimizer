from pathlib import Path
from typing import Sequence
import pandas as pd
from pytest import fixture
from stock_getter import CsvStockGetter, MockStockGetter


@fixture
def mock_stock_getter() -> MockStockGetter:
    return MockStockGetter()


@fixture
def csv_path() -> Path:
    return Path(__file__).parent / 'data/example_stocks.csv'


@fixture
def csv_stock_getter(csv_path: Path) -> CsvStockGetter:
    return CsvStockGetter(csv_path)


@fixture
def example_stock_ticker_symbols() -> Sequence[str]:
    return ['BULL', 'BEAR', 'WOLF', 'DOG', 'LION']


@fixture
def date_range() -> tuple[str, str]:
    return ('2020-01-01', '2024-12-31')


@fixture
def example_stocks(
    csv_stock_getter: CsvStockGetter,
    example_stock_ticker_symbols: Sequence[str],
    date_range: tuple[str, str]
) -> pd.DataFrame:
    return csv_stock_getter.get_stocks(
        example_stock_ticker_symbols,
        date_range[0],
        date_range[1]
    )
