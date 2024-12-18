from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence

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
    def fetch(
        self,
        ticker_symbols: Sequence[str],
        start_date: str,
        end_date: str,
    ):
        ...
