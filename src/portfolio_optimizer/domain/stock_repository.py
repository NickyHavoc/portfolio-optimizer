from abc import abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


@lru_cache(maxsize=1)
def base_dataframe() -> pd.DataFrame:
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date='1900-01-01', end_date='2100-12-31')
    trading_days = mcal.date_range(schedule, frequency='1D')
    trading_days = pd.to_datetime(trading_days.date)
    df = pd.DataFrame(index=trading_days)
    df.index.name = 'Date'
    return df


class StockRepository:
    def __init__(self, csv_directory: Path) -> None:
        self.csv_directory = csv_directory
        self.stocks = self._concatenate_csv_files(csv_directory)

    def _concatenate_csv_files(self, directory: Path) -> pd.DataFrame:
        csv_files = directory.glob('*.csv')

        base_df = base_dataframe()

        for file in csv_files:
            df = pd.read_csv(file)
            
            if 'Unnamed: 0' in df.columns:
                df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
                df.set_index('Unnamed: 0', inplace=True)
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df.set_index('Date', inplace=True)

            df = self._validate_stock_df(df)

            base_df = base_df.merge(df, how='left', left_index=True, right_index=True)
        
        return base_df

    @staticmethod
    def _validate_stock_df(stock_df: pd.DataFrame) -> pd.DataFrame:
        if not stock_df.apply(lambda col: all(isinstance(x, float) for x in col)).all():
            raise ValueError('All values in stocks dataframe must be floats.')
        
        if not all(isinstance(col, str) for col in stock_df.columns):
            raise ValueError('All column names must be strings.')
            
        base_df = base_dataframe()
        stock_df.index.name = 'Date'
        stock_df.index = pd.to_datetime(stock_df.index.date)
        stock_df = stock_df.join(base_df, how='inner')
        return stock_df

    def add_stocks(self, stocks: pd.DataFrame) -> None:
        validated_stocks = self._validate_stock_df(stocks)
        
        for stock_symbol in validated_stocks.columns:
            stock = validated_stocks.loc[:, [stock_symbol]]
            self._update_stock(stock)

    def _update_stock(self, stock: pd.DataFrame) -> None:
        assert len(stock.columns) == 1

        stock_symbol = stock.columns[0]
        stock_path = self._stock_path(stock_symbol)

        if stock_symbol in self.stocks.columns:
            in_memory_stock = self.stocks.loc[:, [stock_symbol]]
            new_stock = in_memory_stock.combine_first(stock).dropna()

        else:
            new_stock = stock

        self._store_stock_file(new_stock, stock_path)
        self._store_stock_memory(new_stock, stock_symbol)

    def _stock_path(self, stock_symbol: str) -> Path:
        return self.csv_directory / f"{stock_symbol}.csv"

    def _store_stock_memory(self, stock: pd.DataFrame, stock_symbol: str | None = None) -> None:
        assert len(stock.columns) == 1
        if stock_symbol is None:
            
            stock_symbol = stock.columns[0]

        if not stock.index.isin(self.stocks.index).all():
            raise ValueError("The indices of the smaller dataframe must be a subset of the larger dataframe.")

        self.stocks[stock_symbol] = stock[stock_symbol]

    def _store_stock_file(self, stock: pd.DataFrame, stock_path: Path | None = None) -> None:
        assert len(stock.columns) == 1
        if stock_path is None:
            
            stock_symbol = stock.columns[0]
            stock_path = self._stock_path(stock_symbol)

        assert isinstance(stock_path, Path)

        stock_path.parent.mkdir(parents=True, exist_ok=True)
        stock.to_csv(stock_path)
    
    def get_stocks(
        self,
        ticker_symbols: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        return self._get_columns_for_date_range(
            self.stocks, ticker_symbols, start_date, end_date
        )

    @staticmethod
    def _get_columns_for_date_range(
        df: pd.DataFrame,
        columns: Sequence[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        return df.loc[start_date:end_date, columns]

    def get_stocks_without_nan(
        self,
        ticker_symbols: Sequence[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        stock_df = self.get_stocks(ticker_symbols, start_date, end_date)
        return self._get_all_full_columns_in_date_range(stock_df, start_date, end_date)

    @staticmethod
    def _get_all_full_columns_in_date_range(
        df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        filtered_df = df.loc[start_date:end_date]
        return filtered_df.dropna(axis=1)

    def get_available_ticker_symbols(self, start_date: str, end_date: str) -> Sequence[str]:
        return self._get_all_full_columns_in_date_range(self.stocks, start_date, end_date).columns.to_list()
