from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


class StockRepository:
    def __init__(self, csv_directory: Path) -> None:
        self.csv_directory = csv_directory
        self.stocks = self._concatenate_csv_files(csv_directory)

    @staticmethod
    def _concatenate_csv_files(directory: Path) -> pd.DataFrame:
        csv_files = directory.glob('*.csv')

        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            
            if 'Unnamed: 0' in df.columns:
                df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
                df.set_index('Unnamed: 0', inplace=True)
            
            dfs.append(df)
        
        joined_df = pd.concat(dfs, axis=1, ignore_index=False)
        return joined_df

    def add_stocks(self, stocks: pd.DataFrame) -> None:
        self._validate_stock_df(stocks)
        
        for stock_symbol in stocks.columns:
            stock = stocks.loc[:, [stock_symbol]]
            self._update_stock(stock)

    @staticmethod
    def _validate_stock_df(stock_df: pd.DataFrame) -> None:
        if not stock_df.apply(lambda col: all(isinstance(x, float) for x in col)).all():
            raise ValueError('All values in stocks dataframe must be floats.')
        
        if not all(isinstance(col, str) for col in stock_df.columns):
            raise ValueError('All column names must be strings.')
        
        if not pd.api.types.is_datetime64_any_dtype(stock_df.index):
            raise ValueError('The dataframe\'s index column must hold dates.')

    def _update_stock(self, stock: pd.DataFrame) -> None:
        assert len(stock.columns) == 1

        stock_symbol = stock.columns[0]
        stock_path = self._stock_path(stock_symbol)
        read_stock = pd.read_csv(stock_path, index_col=0)

        assert stock_symbol in read_stock.columns
        
        for iter_index, value in stock[stock_symbol].items():
            if iter_index in stock.index:
                read_stock.at[iter_index, stock_symbol] = value
            else:
                new_row = pd.DataFrame({stock_symbol: [value]}, index=[iter_index])
                read_stock = pd.concat([read_stock, new_row])

        self._store_stock_file(read_stock, stock_path)
        self._store_stock_memory(read_stock, stock_symbol)

    def _stock_path(self, stock_symbol: str) -> Path:
        return self.csv_directory / f"{stock_symbol}.csv"

    def _store_stock_memory(self, stock: pd.DataFrame, stock_symbol: str | None = None) -> None:
        assert len(stock.columns) == 1
        if stock_symbol is None:
            
            stock_symbol = stock.columns[0]

        combined_index = self.stocks.index.union(stock.index)
        self.stocks = self.stocks.reindex(combined_index, fill_value=np.nan)
        
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
        earliest_timestamp: pd.Timestamp = df.index.min()
        latest_timestamp: pd.Timestamp = df.index.max()
        assert isinstance(earliest_timestamp, pd.Timestamp) and isinstance(latest_timestamp, pd.Timestamp)

        if (
            pd.to_datetime(start_date).tz_localize('UTC').normalize() < earliest_timestamp.normalize()
        ) or (
            pd.to_datetime(end_date).tz_localize('UTC').normalize() > latest_timestamp.normalize()
        ) :
            raise ValueError(
                f"Earliest value: '{str(earliest_timestamp)}', latest value: '{str(latest_timestamp)}'. Please specify different dates."
            )

        filtered_df = df.loc[start_date:end_date]
        return filtered_df.dropna(axis=1)

    @abstractmethod
    def _create_stocks(
        self,
        ticker_symbols: Sequence[str],
        start_date: str,
        end_date: str,
        config: Any = None
    ) -> pd.DataFrame:
        ...

    def create_stocks(
        self,
        ticker_symbols: Sequence[str],
        start_date: str,
        end_date: str,
        config: Any = None
    ) -> None:
        stock_df = self._create_stocks(ticker_symbols, start_date, end_date, config)
        self.add_stocks(stock_df)

    def get_available_ticker_symbols(self, start_date: str, end_date: str) -> Sequence[str]:
        return self._get_all_full_columns_in_date_range(self.stocks, start_date, end_date).columns.to_list()
