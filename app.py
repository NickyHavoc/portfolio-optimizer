from pathlib import Path
import plotly.express as px
import pandas as pd
import streamlit as st
from datetime import datetime
from portfolio_optimizer import PortfolioOptimizer, Portfolio, StockRepository

STOCK_REPO = StockRepository(Path(__file__).parent / "data_example")
OPTIMIZER = PortfolioOptimizer(STOCK_REPO)

st.title("Optimal Portfolio Calculator")
st.markdown("Enter the parameters to compute the optimal portfolio.")

if "portfolio" not in st.session_state:
    st.session_state["portfolio"] = None

if "start_and_end_date" not in st.session_state:
    st.session_state["start_and_end_date"] = None

start_date = st.date_input(
    "Start Date:", value=datetime(2020, 1, 1), min_value=datetime(2000, 1, 1)
).strftime("%Y-%m-%d")

end_date = st.date_input(
    "End Date:", value=datetime(2024, 12, 31), min_value=datetime(2000, 1, 1)
).strftime("%Y-%m-%d")

available_tickers = []
if start_date and end_date:
    st.session_state["start_and_end_date"] = (start_date, end_date)

if (
    st.session_state["start_and_end_date"] is not None
):
    start_and_end_date = st.session_state["start_and_end_date"][0], st.session_state["start_and_end_date"][1]
    
    try:
        available_tickers = STOCK_REPO.get_available_ticker_symbols(start_date=start_date, end_date=end_date)

    except Exception as e:
        st.error(f"Error retrieving available tickers: {e}")

    # Default the selected tickers to the available ones
    selected_tickers = st.multiselect(
        "Select Tickers:", available_tickers, default=available_tickers
    )

    # Save the selected tickers in session state
    if selected_tickers:
        st.session_state["start_and_end_date"] = selected_tickers

weight_return = st.slider(
    "Weight for Return in Sharpe Ratio (0 to 1):", min_value=0.0, max_value=1.0, value=0.5, step=0.01
)

risk_free_rate = st.slider(
    "Risk-Free Rate (0.0 to 0.1):", min_value=0.0, max_value=0.1, value=0.02, step=0.001
)

if st.button("Calculate Optimal Portfolio"):
    if not selected_tickers:
        st.error("Please select at least one ticker symbol.")
    elif start_date >= end_date:
        st.error("Start date must be earlier than end date.")
    else:
        try:
            portfolio: Portfolio = OPTIMIZER.find_optimal_portfolio(
                ticker_symbols=selected_tickers,
                start_date=start_date,
                end_date=end_date,
                weight_return=weight_return,
                risk_free_rate=risk_free_rate
            )

            st.session_state["portfolio"] = portfolio

        except Exception as e:
            st.error(f"An error occurred: {e}")


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
