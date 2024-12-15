import pandas as pd
from pytest import fixture

from portfolio_optimizer import PortfolioOptimizer


@fixture
def portfolio_optimizer(
    example_stocks: pd.DataFrame
) -> PortfolioOptimizer:
    pass


def test_portfolio_optimizer_works(portfolio_optimizer: PortfolioOptimizer) -> None:
    pass
