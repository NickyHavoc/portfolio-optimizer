from pathlib import Path
from typing import Sequence

import pandas as pd

from .stock_getter import StockGetter
from .utils import load_df_from_csv


class CsvStockGetter(StockGetter):
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path

        df = load_df_from_csv(csv_path)
        df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
        df.set_index('Unnamed: 0', inplace=True)

        self.table = df

    def _get_columns(self, columns: Sequence[str], start_date: str, end_date: str) -> pd.DataFrame:
        return self.table.loc[start_date:end_date, columns]

    def get_stock(self, ticker_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._get_columns([ticker_symbol], start_date, end_date) 

    def get_stocks(self, ticker_symbols: Sequence[str], start_date: str, end_date: str) -> pd.DataFrame:
        return self._get_columns(ticker_symbols, start_date, end_date) 

