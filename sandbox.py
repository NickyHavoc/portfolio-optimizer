
from pathlib import Path
from matplotlib import pyplot as plt
from stock_getter.mock_stock_getter import MockStockGetter
from stock_getter.utils import save_df_to_csv


mock_stock_getter = MockStockGetter()

example_stocks = mock_stock_getter.get_stocks(
    ticker_symbols=['BULL', 'BEAR', 'WOLF', 'DOG', 'LION'],
    start_date='2020-01-01',
    end_date='2024-12-31',
    initial_prices=[15.931, 7.465, 102.410, 88.058, 55.678],
    annual_returns=[0.2, 0.0, 0.3, -0.05, 0.25],
    annual_stdevs=[0.2, 0.10, 0.45, 0.3, 0.15]
)

save_df_to_csv(example_stocks, Path(__file__).parent / 'tests/data' / 'example_stocks.csv')

plt.figure(figsize=(10, 6))
for column in example_stocks.columns:
    plt.plot(example_stocks.index, example_stocks[column], label=column)

# Add labels, title, and legend
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Stock Price Trends")
plt.legend()
plt.grid()
plt.show()
