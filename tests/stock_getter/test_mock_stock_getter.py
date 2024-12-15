from stock_getter import MockStockGetter


def test_mock_stock_getter_can_get_stock(mock_stock_getter: MockStockGetter) -> None:
    example_symbol = 'EXA1'
    example_stock_df =  mock_stock_getter.get_stock(
        ticker_symbol=example_symbol,
        start_date='2020-01-01',
        end_date='2024-12-31',
    )

    assert len(example_stock_df.columns) == 1
    assert len(example_stock_df) == 1258
    assert len(set(example_stock_df[example_symbol].values)) > 1


def test_mock_stock_getter_can_get_stocks(mock_stock_getter: MockStockGetter) -> None:
    example_symbols = ['EXA1', 'EXA2', 'EXA3']
    example_stocks_df = mock_stock_getter.get_stocks(
        ticker_symbols=example_symbols,
        start_date='2020-01-01',
        end_date='2024-12-31',
    )

    assert len(example_stocks_df.columns) == len(example_symbols)
    assert len(example_stocks_df) == 1258
    assert all(
        list(example_stocks_df[s]) != list(example_stocks_df[example_symbols[0]]) for s in example_symbols[1:]
    )
