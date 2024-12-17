from abc import ABC, abstractmethod
from typing import Sequence


class BaseStockFetcher(ABC):
    
    @abstractmethod
    def fetch(
        self,
        ticker_symbols: Sequence[str],
        start_date: str,
        end_date: str,
    ):
        ...
