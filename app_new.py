from mt5_manager import MT5Manager
import streamlit as st
from streamlit_auth0 import login_button
from datetime import datetime
from plotly import graph_objects as go

# use full width of the screen
st.set_page_config(layout="wide")

# # SSO Authentification

# # Load Auth0 secrets
# auth0_client_id = st.secrets["auth0"]["client_id"]
# auth0_domain = st.secrets["auth0"]["domain"]
# auth0_client_secret = st.secrets["auth0"]["client_secret"]

# # Render the login button
# auth_button = login_button(
#     client_id=auth0_client_id,
#     domain=auth0_domain,
#     key=auth0_client_secret,
# )

# if auth_button:
#     st.success(f"Welcome, {auth_button['name']}!")
# else:
#     st.info("Please log in to continue.")
    
def get_balance_evolution(trades, aggregate_by="Day"):
    trades = trades.copy()
    if aggregate_by == "Day":
        trades['time'] = trades['time'].dt.date
        trades['time'].asfreq('D')
    elif aggregate_by == "Week":
        trades['time'] = trades['time'].dt.to_period('W').apply(lambda r: r.start_time)
        trades['time'].asfreq('W-MON')
    elif aggregate_by == "Month":
        trades['time'] = trades['time'].dt.to_period('M').apply(lambda r: r.start_time)
        trades['time'].asfreq('M')
    elif aggregate_by == "Year":
        trades['time'] = trades['time'].dt.year
        trades['time'].asfreq('Y')
        
    aggregated_pnl = trades[['time', 'profit']].groupby('time').sum()
    balance_evolution = aggregated_pnl.cumsum()
    return balance_evolution, aggregated_pnl
    
def sharpe_ratio(trades):
    trades = trades.copy()
    # Compute the Sharpe ratio
    # 1. Compute the daily returns
    # 2. Compute the mean and standard deviation of the daily returns
    # 3. Compute the Sharpe ratio using the formula: Sharpe Ratio = (mean(daily returns) - risk-free rate) / std(daily returns)
    # 4. Return the Sharpe ratio
    trades['time'] = trades['time'].dt.date
    daily_trades = trades[['time', 'profit']].groupby('time').sum()
    daily_returns = daily_trades['profit'].diff() / daily_trades['profit'].shift(1)
    mean_daily_returns = daily_returns.mean()
    std_daily_returns = daily_returns.std()
    risk_free_rate = 0.0
    sharpe_ratio = (mean_daily_returns - risk_free_rate) / std_daily_returns
    print("mean_daily_returns", mean_daily_returns)
    print("std_daily_returns", std_daily_returns)
    return sharpe_ratio
    
mt5_manager = st.session_state.get('mt5_manager', None)

mt5_manager = MT5Manager() if mt5_manager is None else mt5_manager

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    # Prompt user for login and password
    st.subheader("Login to MetaTrader 5")
    login = st.text_input("MetaTrader5 Login", value="51648347") # 51648347
    password = st.text_input("Password", type="password", value="7^bi&jH#") # 7^bi&jH#
    
    # List of servers
    server_list = ("VantageInternational-Live 4",)
    server = st.selectbox("Server", server_list)

    # Display a login button
    if st.button("Login"):
        try:
            print(int(login), password, server)
            logged_in = mt5_manager.login(int(login), password, server)
            if logged_in:
                st.success("Login successful!")
                st.session_state.logged_in = True
                st.session_state.mt5_manager = mt5_manager
                st.rerun()  # Refresh the app to continue execution
            else:
                st.error("Invalid login credentials. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Stop further execution until login succeeds
    st.stop()

# Code beyond this point only runs if logged_in is True
st.success("You are logged in!")
mt5_manager = st.session_state.mt5_manager

trades = mt5_manager.get_all_trades_since_to(date_from=datetime(2024, 12, 1), date_to=datetime.now())


# display key metrics
metric_tabs = st.columns(2)

st.title("Key Metrics")
sharpe = sharpe_ratio(trades[trades['order'] != 0])
metric_tabs[0].metric(label="Sharpe Ratio", value=f"{sharpe:.2f}", delta=f"{(sharpe-1):.2f}", delta_color="normal")
win_rate = (trades['profit'] > 0).mean()
metric_tabs[1].metric(label="Win Rate", value=f"{win_rate:.2%}", delta=f"{win_rate-0.33:.2%}", delta_color="normal")

# list of other metrics to display
# - total profit
# - total number of trades
# - average profit per trade
# - average profit per day
# - average profit per week
# - average profit per month
# - average profit per year



# display balance evolution as cumulative sum of profits and add a toggle to display the daily PnL

st.title("Balance and Profit Evolution")

balance_profit_aggregate_by = st.selectbox("Aggregate by", ["Trade", "Day", "Week", "Month", "Year"], index=1)

if balance_profit_aggregate_by == "Trade":
    raise Exception("Aggregate by Trade not implemented yet")


balance_evolution, aggregated_pnl = get_balance_evolution(trades, balance_profit_aggregate_by)

show_cumulative = st.toggle("Show cumulative profits", value=False)

if show_cumulative:
    fig = go.Figure(
        data=[
            go.Bar(
                x=balance_evolution.index,
                y=balance_evolution['profit'],
            )
        ]
    )
    fig.update_layout(
        xaxis=dict(
            tickvals=balance_evolution.index
        )
    )
    # st.bar_chart(balance_evolution, use_container_width=True)
    st.plotly_chart(fig, use_container_width=True)
    
    
else:
    fig = go.Figure(
        data=[
            go.Bar(
                x=aggregated_pnl.index,
                y=aggregated_pnl['profit'],
            )
        ]
    )
    fig.update_layout(
        xaxis=dict(
            tickvals=aggregated_pnl.index
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    # st.bar_chart(aggregated_pnl, use_container_width=True)
    

st.title("Win Rate Evolution")


st.title("Win Rate Breakdown")
# breakdown of win rate by day of the week, hour of the day, month of the year, etc.

def get_win_rate_breakdown(trades, by="Day"):
    trades = trades.copy()
    if by == "Day":
        trades['time'] = trades['time'].dt.day_name()
    elif by == "Hour":
        trades['time'] = trades['time'].dt.hour
    elif by == "Month":
        trades['time'] = trades['time'].dt.month_name()
    elif by == "Year":
        trades['time'] = trades['time'].dt.year
    win_rate_breakdown = trades.groupby('time')['profit'].apply(lambda x: (x > 0).mean())
    win_rate_breakdown *= 100
    return win_rate_breakdown

win_rate_breakdown_by = st.selectbox("Breakdown by", ["Day", "Hour", "Month", "Year"], index=0)

win_rate_breakdown = get_win_rate_breakdown(trades[trades['order'] != 0], win_rate_breakdown_by)

if win_rate_breakdown_by == "Hour":
    win_rate_breakdown.index = [f"{i}:00" for i in win_rate_breakdown.index]

if win_rate_breakdown_by == "Day":
    idx = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
elif win_rate_breakdown_by == "Month":
    idx = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
else:
    idx = win_rate_breakdown.index

fig = go.Figure(
    data=[
        go.Bar(
            x=win_rate_breakdown.index,
            y=win_rate_breakdown,
        )
    ]
)
fig.update_layout(
    xaxis=dict(
        tickvals=idx
    ),
    yaxis=dict(
        title="Win Rate (%)",
        tickvals=[i*10 for i in range(11)],
    )
)

if win_rate_breakdown_by in ["Day", "Month"]:
    fig.update_layout(
        xaxis=dict(
            categoryorder='array',
            categoryarray=idx
        )
    )

st.plotly_chart(fig, use_container_width=True)

# st.session_state.logged_in = False

# mt5_manager.disconnect()
        
