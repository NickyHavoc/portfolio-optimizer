from portfolio_optimizer import MockStockFetcher


def test_mock_stock_fetcher_can_fetch(temp_mock_stock_fetcher: MockStockFetcher, date_range: tuple[str, str]) -> None:
    example_symbol = 'EXA1'
    example_stock_df = temp_mock_stock_fetcher.fetch(
        ticker_symbols=[example_symbol],
        start_date=date_range[0],
        end_date=date_range[1],
    )

    assert len(example_stock_df.columns) == 1
    assert len(example_stock_df) == 1258
    assert len(set(example_stock_df[example_symbol].values)) > 1


def test_mock_stock_getter_can_get_stocks(temp_mock_stock_fetcher: MockStockFetcher, date_range: tuple[str, str]) -> None:
    example_symbols = ['EXA1', 'EXA2', 'EXA3']
    example_stocks_df = temp_mock_stock_fetcher.fetch(
        ticker_symbols=example_symbols,
        start_date=date_range[0],
        end_date=date_range[1],
    )

    assert len(example_stocks_df.columns) == len(example_symbols)
    assert len(example_stocks_df) == 1258
    assert all(
        list(example_stocks_df[s]) != list(example_stocks_df[example_symbols[0]]) for s in example_symbols[1:]
    )
