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
        data_frequency: DataFrequency = DataFrequency.DAILY,
        include_dividend: bool = True
    ) -> None:
        self.stock_repository = stock_repository or StockRepository(Path.cwd() / "stock_repository")
        self.data_frequency = data_frequency
        self.include_dividend = include_dividend

    def create_stocks(
        self,
        ticker_symbols: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if self.include_dividend:
            stock_data = self._create_stocks_with_dividends(
                ticker_symbols, start_date, end_date
            )
        else:
            stock_data = self._create_stocks_without_dividends(
                ticker_symbols, start_date, end_date
            )
        self.store_stock_df(stock_data)
        return stock_data
    
    def _create_stocks_with_dividends(
        self,
        ticker_symbols: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        data = yf.download(
            ticker_symbols, 
            start=start_date, 
            end=self._add_one_day_to_date(end_date),
            actions=True
        )

        adj_close = data['Adj Close']
        dividends = data.get('Dividends', pd.DataFrame(0, index=adj_close.index, columns=ticker_symbols))

        adj_close_with_dividends = adj_close.copy()
        for ticker in adj_close.columns:
            if not ticker in dividends.columns:
                continue

            cumulative_return = 1.0
            for i in range(len(adj_close)):
                if dividends[ticker].iloc[i] > 0:  # Account for dividends
                    cumulative_return += dividends[ticker].iloc[i] / adj_close[ticker].iloc[i]
                adj_close_with_dividends[ticker].iloc[i] *= cumulative_return

        return adj_close_with_dividends
    
    def _create_stocks_without_dividends(
        self,
        ticker_symbols: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        return yf.download(
            ticker_symbols, start=start_date, end=self._add_one_day_to_date(end_date)
        )['Adj Close']

    @staticmethod
    def _add_one_day_to_date(date_str: str) -> str:
        """Yahoo finance won't include end_date"""
        end_date_dt = datetime.strptime(date_str, '%Y-%m-%d')
        new_end_date_dt = end_date_dt + timedelta(days=1)
        return new_end_date_dt.strftime('%Y-%m-%d')
