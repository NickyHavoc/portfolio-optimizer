from abc import abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yfinance as yf

from .base_stock_fetcher import BaseStockFetcher, DataFrequency
from .stock_repository import StockRepository


class YFinanceStockFetcher(BaseStockFetcher):
    def __init__(
        self,
        stock_repository: StockRepository | None = None,
        data_frequency: DataFrequency = DataFrequency.DAILY
    ) -> None:
        self.stock_repository = stock_repository or StockRepository(Path.cwd() / "stock_repository")
        self.data_frequency = data_frequency

    def create_stocks(
        self,
        ticker_symbols: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        data: pd.DataFrame = yf.download(ticker_symbols, start=start_date, end=self._add_one_day_to_date(end_date))['Adj Close']
        self.store_stock_df(data)
        return data

    @staticmethod
    def _add_one_day_to_date(date_str: str) -> str:
        """Yahoo finance won't include end_date"""
        end_date_dt = datetime.strptime(date_str, '%Y-%m-%d')
        new_end_date_dt = end_date_dt + timedelta(days=1)
        return new_end_date_dt.strftime('%Y-%m-%d')
