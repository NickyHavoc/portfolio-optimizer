from abc import ABC, abstractmethod
from typing import Sequence

import pandas as pd


class StockGetter(ABC):
    @abstractmethod
    def get_stock(
        self,
        ticker_symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_stocks(
        self,
        ticker_symbols: Sequence[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        ...
