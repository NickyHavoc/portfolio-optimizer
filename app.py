from pathlib import Path
import plotly.express as px
import pandas as pd
import streamlit as st
from datetime import datetime
from portfolio_optimizer import PortfolioOptimizer, Portfolio, StockRepository, YFinanceStockFetcher
from portfolio_optimizer.portfolio_optimizer import PortfolioSecurity, YearlyReturn


# Initialize session state for persistent objects
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    STOCK_REPOSITORY = StockRepository(Path(__file__).parent / "stock_repository")
    st.session_state['STOCK_REPOSITORY'] = STOCK_REPOSITORY
    STOCK_FETCHER = YFinanceStockFetcher(STOCK_REPOSITORY)
    st.session_state['STOCK_FETCHER'] = STOCK_FETCHER
    OPTIMIZER = PortfolioOptimizer(STOCK_FETCHER)
    st.session_state['OPTIMIZER'] = OPTIMIZER

st.title("Optimal Portfolio Calculator")
st.markdown("Enter the parameters to compute the optimal portfolio.")

# Access session state objects
STOCK_FETCHER: YFinanceStockFetcher = st.session_state['STOCK_FETCHER']
OPTIMIZER: PortfolioOptimizer = st.session_state['OPTIMIZER']

if "portfolio" not in st.session_state:
    st.session_state["portfolio"] = None

# Date Input
start_date = st.date_input(
    "Start Date:", value=datetime(2015, 1, 1), min_value=datetime(2000, 1, 1)
).strftime("%Y-%m-%d")

end_date = st.date_input(
    "End Date:", value=datetime(2024, 12, 17), min_value=datetime(2000, 1, 1)
).strftime("%Y-%m-%d")

if start_date >= end_date:
    st.error("Start date must be earlier than end date.")

# Section 1: Define Fixed Securities
st.subheader("1. Enter Fixed Securities and Weights")
fixed_securities_df = pd.DataFrame({'Ticker Symbol': [''], 'Weight': [0.0]})
edited_fixed_securities = st.data_editor(
    fixed_securities_df, use_container_width=True, num_rows="dynamic"
)

fixed_securities = []
for _, row in edited_fixed_securities.iterrows():
    try:
        ticker = str(row["Ticker Symbol"]).strip().upper()
        weight = float(row["Weight"])
        if ticker and 0 <= weight <= 1:
            fixed_securities.append(PortfolioSecurity(ticker_symbol=ticker, weight=weight))
        else:
            st.warning(f"Invalid entry: {ticker} with weight {weight}")
    except ValueError:
        st.warning(f"Invalid data format in row: {row}")

if fixed_securities:
    st.write("### Fixed Securities")
    st.table(pd.DataFrame([sec.dict() for sec in fixed_securities]))

# Section 2: Select Additional Tickers
st.subheader("2. Select Additional Tickers")
if st.button("Fetch Available Tickers"):
    try:
        available_tickers = STOCK_FETCHER.get_available_ticker_symbols(
            start_date=start_date, end_date=end_date
        )
        st.session_state["available_tickers"] = available_tickers
    except Exception as e:
        st.error(f"Error fetching tickers: {e}")

if "available_tickers" in st.session_state:
    available_tickers = st.session_state["available_tickers"]
    selected_tickers = st.multiselect(
        "Select Tickers:", available_tickers, default=set(available_tickers)
    )
    custom_tickers_input = st.text_area(
        "Add Custom Tickers (comma-separated, e.g., TSLA,AAPL,GOOG):"
    )
    if custom_tickers_input:
        custom_tickers = [ticker.strip().upper() for ticker in custom_tickers_input.split(",") if ticker.strip()]
        selected_tickers = list(set(selected_tickers + custom_tickers))
    st.write("Selected Tickers:", selected_tickers)
else:
    selected_tickers = []

# Section 3: Optimization Parameters
st.subheader("3. Set Optimization Parameters")

yearly_return_method = st.selectbox(
    "Select the method for yearly return calculation:",
    options=[YearlyReturn[m].value for m in YearlyReturn.__members__],
    index=0
)
selected_yearly_return_enum = YearlyReturn.from_string(yearly_return_method)

weight_return = st.slider(
    "Weight for Return in Sharpe Ratio (0 to 1):", min_value=0.0, max_value=1.0, value=0.5, step=0.01
)

risk_free_rate = st.slider(
    "Risk-Free Rate (0.0 to 0.1):", min_value=0.0, max_value=0.1, value=0.02, step=0.001
)

max_weight = st.slider(
    "Maximum Weight per Security (0 to 1):", min_value=0.0, max_value=1.0, value=0.2, step=0.01
)

# Section 4: Optimize Portfolio
st.subheader("4. Optimize Portfolio")
if st.button("Calculate Optimal Portfolio"):
    if not selected_tickers and not fixed_securities:
        st.error("Please provide fixed securities or select at least one additional ticker.")
    elif start_date >= end_date:
        st.error("Start date must be earlier than end date.")
    else:
        try:
            portfolio: Portfolio = OPTIMIZER.find_optimal_portfolio(
                ticker_symbols=selected_tickers,
                start_date=start_date,
                end_date=end_date,
                weight_return=weight_return,
                risk_free_rate=risk_free_rate,
                fixed_securities=fixed_securities,
                max_weight=max_weight,
                yearly_return_method=selected_yearly_return_enum  # Pass the enum here
            )
            st.session_state["portfolio"] = portfolio
        except Exception as e:
            st.error(f"Error during optimization: {e}")

# Display Results
if st.session_state["portfolio"] is not None:
    portfolio = st.session_state["portfolio"]
    st.subheader("Portfolio Performance and Securities")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("**Portfolio Summary**")
        st.write({
            "Sharpe Ratio": portfolio.performance.sharpe,
            "Annual Return (%)": portfolio.performance.annual_return * 100,
            "Annual Risk (%)": portfolio.performance.annual_risk * 100,
        })

    with col2:
        securities_df = portfolio.securities_to_dataframe()
        st.write("**Individual Securities**")
        st.dataframe(securities_df)

    securities_df["Type"] = "Security"
    portfolio_point = {
        "Ticker": "Optimal Portfolio",
        "Weight": None,
        "Sharpe": portfolio.performance.sharpe,
        "Annual Return (%)": portfolio.performance.annual_return * 100,
        "Annual Risk (%)": portfolio.performance.annual_risk * 100,
        "Type": "Portfolio",
    }
    combined_df = pd.concat(
        [securities_df, pd.DataFrame([portfolio_point])], ignore_index=True
    )

    fig = px.scatter(
        combined_df,
        x="Annual Risk (%)",
        y="Annual Return (%)",
        color="Type",
        hover_data=["Ticker", "Annual Risk (%)", "Annual Return (%)", "Sharpe"],
        size=combined_df["Type"].apply(lambda x: 15 if x == "Portfolio" else 5),
        size_max=20,
        title="Portfolio and Securities Performance",
        labels={"Annual Risk (%)": "Risk (%)", "Annual Return (%)": "Return (%)"},
    )

    fig.update_traces(marker=dict(line=dict(width=1, color="Black")))
    fig.update_layout(
        title_font_size=16,
        legend_title_text="Type",
        xaxis=dict(showgrid=True, gridcolor="LightGrey"),
        yaxis=dict(showgrid=True, gridcolor="LightGrey"),
    )

    st.plotly_chart(fig, use_container_width=True)
