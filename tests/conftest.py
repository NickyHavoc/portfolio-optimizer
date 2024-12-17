from pathlib import Path
from typing import Sequence
import pandas as pd
from pytest import fixture
from portfolio_optimizer import StockRepository, MockStockFetcher


@fixture(scope='function')
def mock_stock_fetcher() -> MockStockFetcher:
    return MockStockFetcher()


@fixture
def csv_dir() -> Path:
    return Path(__file__).parent / 'data'


@fixture(scope='function')
def stock_repository(csv_dir: Path) -> StockRepository:
    return StockRepository(csv_dir)


@fixture
def example_stock_ticker_symbols() -> Sequence[str]:
    return ['BULL', 'BEAR', 'WOLF', 'DOG', 'LION']


@fixture
def date_range() -> tuple[str, str]:
    return ('2020-01-01', '2024-12-31')


@fixture
def example_stocks(
    stock_repository: StockRepository,
    example_stock_ticker_symbols: Sequence[str],
    date_range: tuple[str, str]
) -> pd.DataFrame:
    return stock_repository.get_stocks(
        example_stock_ticker_symbols,
        date_range[0],
        date_range[1]
    )
