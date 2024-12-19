from abc import abstractmethod
from pathlib import Path
from typing import Any, Sequence
import numpy as np

import pandas as pd
import pandas_market_calendars as mcal
from pydantic import BaseModel

from .base_stock_fetcher import BaseStockFetcher, DataFrequency
from .stock_repository import StockRepository


class SamplingRange(BaseModel):
    min: float
    max: float

    @abstractmethod
    def sample(self) -> float:
        pass


class PriceSamplingRange(SamplingRange):
    def sample(self) -> float:
        log_sample = np.random.uniform(np.log(self.min), np.log(self.max))
        return np.exp(log_sample)


class StatisticSamplingRange(SamplingRange):
    def sample(self) -> float:
        return np.random.uniform(self.min, self.max)


class MockStockFetcher(BaseStockFetcher):
    def __init__(
        self,
        stock_repository: StockRepository | None = None,
        data_frequency: DataFrequency = DataFrequency.DAILY,
        initial_price_range: PriceSamplingRange = PriceSamplingRange(
            min=0.1, max=1000
        ),
        annual_return_range: StatisticSamplingRange = StatisticSamplingRange(
            min=-0.2, max=0.5
        ),
        annual_stdev_range: StatisticSamplingRange = StatisticSamplingRange(
            min=0.1, max=0.6
        )
    ) -> None:
        self.stock_repository = stock_repository or StockRepository(Path.cwd() / "stock_repository")
        self.data_frequency = data_frequency
        self.initial_price_range = initial_price_range
        self.annual_return_range = annual_return_range
        self.annual_stdev_range = annual_stdev_range

    def create_stocks(
        self,
        ticker_symbols: Sequence[str],
        start_date: str,
        end_date: str,
        initial_prices: Sequence[float] | None = None,
        annual_returns: Sequence[float] | None = None,
        annual_stdevs: Sequence[float] | None = None
    ) -> pd.DataFrame:

        def get_safe_element(something: Any, index: int) -> Any:
            try:
                return something[index]
            except:
                return None
        
        stocks = [
            self._create_stock(
                ticker_symbol,
                start_date,
                end_date,
                initial_price=get_safe_element(initial_prices, index),
                annual_return=get_safe_element(annual_returns, index),
                annual_stdev=get_safe_element(annual_stdevs, index)
            ) for index, ticker_symbol in enumerate(ticker_symbols)
        ]
        return pd.concat(stocks, axis=1)

    def _create_stock(
        self,
        ticker_symbol: str,
        start_date: str,
        end_date: str,
        initial_price: float | None = None,
        annual_return: float | None = None,
        annual_stdev: float | None = None
    ) -> pd.DataFrame:
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        trading_days = mcal.date_range(schedule, frequency=self.data_frequency.mcal_denotation)
        df = pd.DataFrame(index=trading_days)

        _annual_return, _annual_stdev, _initial_price = (
            annual_return if annual_return is not None else self._sample_annual_return(),
            annual_stdev if annual_stdev is not None else self._sample_annual_stdev(),
            initial_price if initial_price is not None else self._sample_initial_price()
        )

        daily_return, daily_stdev = self._annual_to_daily(_annual_return, _annual_stdev)
        stock_values = self._simulate_stock(_initial_price, daily_return, daily_stdev, len(df))
        df[ticker_symbol] = stock_values
        
        self.store_stock_df(df)
        return df

    def _sample_initial_price(self) -> float:
        return self.initial_price_range.sample()

    def _sample_annual_return(self) -> float:
        return self.annual_return_range.sample()

    def _sample_annual_stdev(self) -> float:
        return self.annual_stdev_range.sample()
    
    def _annual_to_daily(
        self,
        annual_return: float,
        annual_stdev: float,
        trading_days: int = 252
    ) -> tuple[float, float]:
        daily_return = (1 + annual_return) ** (1 / trading_days) - 1
        daily_stdev = annual_stdev / np.sqrt(trading_days)
        return daily_return, daily_stdev

    def _simulate_stock(
        self,
        initial_price: float,
        daily_return: float,
        daily_stdev: float,
        n_days: int,
    ) -> pd.DataFrame:
        prices = [initial_price]
        for _ in range(n_days - 1):
            # Generate a random shock (Z_t)
            random_shock = np.random.normal(0, 1)
            # Apply the Geometric Brownian Motion formula
            daily_change = np.exp((daily_return - (daily_stdev ** 2) / 2) + daily_stdev * random_shock)
            new_price = round(prices[-1] * daily_change, 3)
            prices.append(new_price)
        return prices
