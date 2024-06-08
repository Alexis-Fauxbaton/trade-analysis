import streamlit as st
import pandas as pd
# import random
import numpy as np
import plotly.graph_objects as go
from st_utils import get_all_trades

initial_balance = 100000

st.set_page_config(layout='wide')

st.markdown('## Analyse des Trades')

agg_period = st.sidebar.radio('Période d\'aggrégation', ['Trade', 'Jour', 'Semaine', 'Mois'])

trades = get_all_trades().sort_index(ascending=False)

def aggregate_trades(trades, period):
    if period == 'Trade':
        return trades
    elif period == 'Jour':
        return trades.groupby(pd.to_datetime(trades['time']).dt.date).agg({'Profit': 'sum', 'Commission': 'sum'}).reset_index()
    elif period == 'Semaine':
        return trades.groupby(pd.to_datetime(trades['time']).dt.isocalendar().week).agg({'Profit': 'sum', 'Commission': 'sum'}).reset_index()
    elif period == 'Mois':
        return trades.groupby(pd.to_datetime(trades['time']).dt.to_period('M')).agg({'Profit': 'sum', 'Commission': 'sum'}).reset_index()

trades_agg = aggregate_trades(trades, agg_period)

#aggregate trades by day
# trades_by_day = trades.groupby(pd.to_datetime(trades['time']).dt.date).agg({'Profit': 'sum', 'Commission': 'sum'}).reset_index()

trades_agg['time'] = pd.to_datetime(trades_agg['time'])

fig = go.Figure(data=[go.Scatter(x=(trades_agg['time'].dt.date if agg_period != 'Trade' else trades_agg['time']), y=trades_agg['Profit'].cumsum(), mode='lines', name='Profit'),])
fig.add_trace(go.Scatter(x=(trades_agg['time'].dt.date if agg_period != 'Trade' else trades_agg['time']), y=-trades_agg['Commission'].cumsum(), mode='lines', name='Commission'))

st.plotly_chart(fig, use_container_width=True)

monte_carlo = st.toggle('Monte Carlo Simulation')

def monte_carlo_simulation(trades, n=1000, horizon=100, threshold=0.9*initial_balance):

    true_profits = trades['Profit'] + trades['Commission']
    
    # evaluate the mean and standard deviation of the profit
    mean = true_profits.mean()
    std = true_profits.std()
    
    
    results = []
    results_raw = []
    
    for i in range(n):
        # simulate the profit for the next horizon trades and truncate the cumsum if the profit goes below the threshold
        simulated_profits = np.random.normal(mean, std, horizon)
        simulated_profits_raw = simulated_profits
        simulated_profits = np.minimum(np.cumsum(simulated_profits), threshold)
        
        results.append(simulated_profits)
        results_raw.append(simulated_profits_raw)
    
    results_matrix = np.array(results)
    results_matrix_raw = np.array(results_raw)
    
    final_results = [result[-1] for result in results]
    
    # create buckets for the histogram
    
    bins_matrix = np.zeros((100, horizon))
    
    # create 100 bins for each horizon step
    for i in range(horizon):
        bins = np.linspace(np.min(results_matrix[:, i]), np.max(results_matrix[:, i]), 100)
        bins_matrix[:, i] = bins
    

    return final_results, results_matrix, results_matrix_raw, bins_matrix

if monte_carlo:
    n = st.number_input('Nombre de simulations', min_value=1, max_value=1000000, value=1000)
    horizon = st.number_input('Horizon de simulation', min_value=1, max_value=1000, value=100)
    threshold = st.number_input('Seuil de perte', min_value=0, max_value=initial_balance, value=int(0.9*initial_balance))

    with st.spinner('Simulation en cours...'):
        final_results, results_matrix, results_matrix_raw, bins_matrix = monte_carlo_simulation(trades, n, horizon, threshold)
    
    # treat raw results as the continuation of the profit trajectory, create new matrix with previous results
    # copy previous results n times and concatenate with the new results
    
    previous_results_matrix_raw = np.tile(trades['Profit'] + trades['Commission'], (n, 1))
    
    results_matrix_raw = np.concatenate((previous_results_matrix_raw, results_matrix_raw), axis=1)
    
    print(results_matrix_raw.shape)
    print(results_matrix_raw[0, :])
    
    
    results_matrix_raw = np.cumsum(results_matrix_raw, axis=1) + initial_balance
    
    # plot the profit trajectory for 100 random simulations
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=np.arange(trades.shape[0]), y=(trades['Profit'] + trades['Commission']).cumsum() + initial_balance, mode='lines', name='Profit réel'))
    
    for i in range(np.minimum(100, n)):
        fig.add_trace(go.Scatter(x=np.arange(trades.shape[0], trades.shape[0] + horizon), y=results_matrix_raw[i, trades.shape[0]:], mode='lines', name='Simulation {}'.format(i)))
    

    fig.update_layout(yaxis_range=[int(0.9*np.min(results_matrix_raw)), int(1.1*np.max(results_matrix_raw))])
    #plot the threshold
    fig.add_shape(
        dict(
            type="line",
            x0=0,
            y0=threshold,
            x1=trades.shape[0] + horizon,
            y1=threshold,
            line=dict(color="red", width=2)
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # st metric the risk of ruin (proportion of simulations that go below the threshold at at least one point)
    risk_of_ruin = np.mean([np.any(result < threshold) for result in results_matrix_raw])
    
    st.metric('Risque de ruine', f'{risk_of_ruin:.2%}')    