from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence

import pandas as pd
from pydantic import BaseModel

from .stock_repository import StockRepository


class DataFrequencyInformation(BaseModel, frozen = True):
    mcal_denotation_: str
    approx_frequency_factor_: int


class DataFrequency(Enum):
    DAILY = DataFrequencyInformation(
        mcal_denotation_='1D',
        approx_frequency_factor_=252
    )

    @property
    def mcal_denotation(self) -> str:
        return self.value.mcal_denotation_
    
    @property
    def approx_frequency_factor(self) -> int:
        return self.value.approx_frequency_factor_


class BaseStockFetcher(ABC):
    stock_repository: StockRepository
    data_frequency: DataFrequency

    @abstractmethod
    def create_stocks(
        self,
        ticker_symbols: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Should take care of storing"""
        ...

    def fetch(
        self,
        ticker_symbols: Sequence[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        available_ticker_symbols = set(ticker_symbols) & set(
            self.stock_repository.get_available_ticker_symbols(
                start_date, end_date
            )
        )

        available_stocks = self.stock_repository.get_stocks_without_nan(
            list(available_ticker_symbols),
            start_date,
            end_date
        ) if len(available_ticker_symbols) else pd.DataFrame()
        
        if len(available_stocks.columns) == len(ticker_symbols):
            return available_stocks.reindex(columns=ticker_symbols)
        
        created_stocks = self.create_stocks(
            [s for s in ticker_symbols if s not in available_ticker_symbols], start_date, end_date
        )
        combined_stocks = pd.concat([available_stocks, created_stocks], axis=1)

        return combined_stocks.reindex(columns=ticker_symbols)

    def store_stock_df(self, stock_dataframe: pd.DataFrame) -> None:
        self.stock_repository.add_stocks(stock_dataframe)

    def get_available_ticker_symbols(self, start_date: str, end_date: str) -> Sequence[str]:
        return self.stock_repository.get_available_ticker_symbols(start_date, end_date)
