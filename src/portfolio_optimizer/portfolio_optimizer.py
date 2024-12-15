import pandas as pd


class PortfolioOptimizer:
    def __init__(self, stock_table: pd.DataFrame) -> None:
        self.stock_table = stock_table
