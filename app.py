import streamlit as st
import pandas as pd
from pandas.tseries.offsets import MonthEnd
# import random
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.mixture import GaussianMixture
from st_utils import get_all_trades

initial_balance = 100000

st.set_page_config(layout='wide')

random_seed = st.number_input('Graine aléatoire', min_value=0, max_value=1000000, value=42)

np.random.seed(random_seed)

st.markdown('## Analyse des Trades')

agg_period = st.sidebar.radio('Période d\'aggrégation', ['Trade', 'Jour', 'Semaine', 'Mois'])

trades = get_all_trades('./data/11154302-trading-data.csv').sort_index(ascending=False).query("Symbol == 'EURUSD'")

reference_trades = pd.concat([trades, get_all_trades()]).sort_index(ascending=False)

def aggregate_trades(trades, period):
    if period == 'Trade':
        trades['time'] = pd.to_datetime(trades['time'])
        return trades
    elif period == 'Jour':
        return trades.groupby(pd.to_datetime(trades['time']).dt.date).agg({'Profit': 'sum', 'Commission': 'sum'}).reset_index()
    elif period == 'Semaine':
        return trades.groupby(pd.to_datetime(trades['time']).dt.isocalendar().week).agg({'Profit': 'sum', 'Commission': 'sum'}).reset_index()
    elif period == 'Mois':
        return trades.groupby(pd.to_datetime(trades['time']).dt.to_period('M')).agg({'Profit': 'sum', 'Commission': 'sum'}).reset_index()

trades_agg = aggregate_trades(trades, agg_period)
time_offsets = {'Trades': pd.Timedelta(days=-1), 'Jour': pd.Timedelta(days=-1), 'Semaine': pd.Timedelta(weeks=-1), 'Mois': MonthEnd(-1)}
min_date = trades_agg['time'].min() + time_offsets.get(agg_period, pd.Timedelta(days=-1))
new_row = pd.DataFrame([{'Profit': 0, 'Commission': 0, 'time': min_date}])
trades_agg = pd.concat([trades_agg, new_row], ignore_index=True).sort_values('time').reset_index(drop=True)
# print(trades_agg)

reference_trades_agg = aggregate_trades(reference_trades, agg_period)
min_date = reference_trades_agg['time'].min() + time_offsets.get(agg_period, pd.Timedelta(days=-1))
new_row = pd.DataFrame([{'Profit': 0, 'Commission': 0, 'time': min_date}])
reference_trades_agg = pd.concat([reference_trades_agg, new_row], ignore_index=True).sort_values('time').reset_index(drop=True)
# print(reference_trades_agg)


trades_agg['time'] = pd.to_datetime(trades_agg['time'])

true_profits_toggle = st.toggle('Combiner Profits et commissions', False)

if not true_profits_toggle:
    fig = go.Figure(data=[go.Scatter(x=(trades_agg['time'].dt.date if agg_period != 'Trade' else trades_agg['time']), y=trades_agg['Profit'].cumsum(), mode='lines', name='Profit'),])
    fig.add_trace(go.Scatter(x=(trades_agg['time'].dt.date if agg_period != 'Trade' else trades_agg['time']), y=-trades_agg['Commission'].cumsum(), mode='lines', name='Commission'))
else:
    fig = go.Figure(data=[go.Scatter(x=(trades_agg['time'].dt.date if agg_period != 'Trade' else trades_agg['time']), y=(trades_agg['Profit'] + trades_agg['Commission']).cumsum(), mode='lines', name='Profit')])
st.plotly_chart(fig, use_container_width=True)


st.markdown('## Analyse des Profits')

use_all_trades = st.toggle('Utiliser la totalité de l\'historique des trades', False)

if use_all_trades:
    true_profits = reference_trades['Profit'] + reference_trades['Commission']
else:
    true_profits = trades['Profit'] + trades['Commission']

fig = ff.create_distplot([true_profits[true_profits > 0], true_profits[true_profits < 0]], group_labels=['Profit', 'Perte'], colors=['green', 'red'], bin_size=200)

fig.update_layout(title='Distribution des profits', xaxis_title='Profit', yaxis_title='Densité')

st.plotly_chart(fig, use_container_width=True)


gmm_all_trades = GaussianMixture(n_components=2)
gmm_all_trades.fit((reference_trades['Profit'] + reference_trades['Commission']).values.reshape(-1, 1))

gmm_current_trades = GaussianMixture(n_components=2)
gmm_current_trades.fit((trades['Profit'] + trades['Commission']).values.reshape(-1, 1))

gmm = gmm_all_trades if use_all_trades else gmm_current_trades

# display means and standard deviations of the two components in streamlit

st.write(f"Mean 1: {gmm.means_[0][0]:.2f}, Mean 2: {gmm.means_[1][0]:.2f}")
st.write(f"Standard Deviation 1: {np.sqrt(gmm.covariances_[0][0][0]):.2f}, Standard Deviation 2: {np.sqrt(gmm.covariances_[1][0][0]):.2f}")

# sample from the gaussian mixture model and plot the distribution

samples = gmm.sample(1000)

fig = ff.create_distplot([samples[0].flatten()], group_labels=['Sampled Profits'], bin_size=200)

fig.update_layout(title='Distribution des profits échantillonnés', xaxis_title='Profit', yaxis_title='Densité')

st.plotly_chart(fig, use_container_width=True)

# st.plotly_chart(fig, use_container_width=True)

st.markdown('## Simulation de Monte Carlo')

monte_carlo = st.toggle('Monte Carlo Simulation')

def monte_carlo_simulation(trades, n=1000, horizon=100, threshold=0.9*initial_balance, use_gmm=False, use_all_trades=False):

    true_profits = trades['Profit'] + trades['Commission']
    
    # evaluate the mean and standard deviation of the profit
    mean = true_profits.mean()
    std = true_profits.std()
    
    
    results = []
    results_raw = []
    
    gmm = gmm_all_trades if use_all_trades else gmm_current_trades
    
    for i in range(n):
        # simulate the profit for the next horizon trades and truncate the cumsum if the profit goes below the threshold
        simulated_profits = np.random.normal(mean, std, horizon) if not use_gmm else gmm.sample(horizon)[0].flatten()
        if use_gmm:
            np.random.shuffle(simulated_profits)
        simulated_profits_raw = simulated_profits
        simulated_profits = np.minimum(np.cumsum(simulated_profits), threshold)
        
        results.append(simulated_profits)
        results_raw.append(simulated_profits_raw)
    
    fig = ff.create_distplot([simulated_profits_raw], group_labels=['Final Profits'], bin_size=200)
    fig.update_layout(title='Distribution des profits finaux simulés', xaxis_title='Profit', yaxis_title='Densité')
    
    st.plotly_chart(fig, use_container_width=True)
    
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
    target = st.number_input('Objectif de profit', min_value=0, value=int(1.1*initial_balance))
    use_all_trades = st.toggle('Utiliser la totalité de l\'historique des trades comme référence', False)
    use_gmm = st.toggle('Utiliser un modèle de mélange gaussien pour simuler les profits')

    with st.spinner('Simulation en cours...'):
        final_results, results_matrix, results_matrix_raw, bins_matrix = monte_carlo_simulation(reference_trades if use_all_trades else trades, n, horizon, threshold, use_gmm, use_all_trades)
    
    previous_results_matrix_raw = np.tile(trades['Profit'] + trades['Commission'], (n, 1))
    
    results_matrix_raw = np.concatenate((previous_results_matrix_raw, results_matrix_raw), axis=1)
    
    # print(results_matrix_raw.shape)
    # print(results_matrix_raw[0, :])
    
    
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
    risk_of_ruin = np.mean([np.any(result < threshold) for result in results_matrix_raw[:, trades.shape[0]:]])
    
    chance_of_success = np.mean([np.any(result > target) for result in results_matrix_raw]) # even if hitting the threshold
    
    # chance of success without hitting the threshold before
    chance_of_success_without_threshold = np.mean([np.all(result > threshold) and np.any(result > target) for result in results_matrix_raw[:, trades.shape[0]:]])
    
    # print(results_matrix_raw, results_matrix_raw.shape)
    
    col1, col2, col3 = st.columns(3)
    
    # print(risk_of_ruin, chance_of_success)
    
    col1.metric('Risque de ruine', f'{risk_of_ruin:.2%}')
    col2.metric('Probabilité d\'atteindre l\'objectif de profit (même si seuil limite touché)', f'{chance_of_success:.2%}')
    col3.metric('Probabilité d\'atteindre l\'objectif de profit (sans toucher le seuil limite)', f'{chance_of_success_without_threshold:.2%}')
    
    # plot histogram of final results
    bin_size = st.number_input('Taille des bins', min_value=1, value=int(0.03 * initial_balance))
    fig = ff.create_distplot([final_results], group_labels=['Final Profits'], bin_size=bin_size)
    fig.update_layout(title='Distribution des profits finaux simulés', xaxis_title='Profit', yaxis_title='Densité')
    st.plotly_chart(fig, use_container_width=True)
    
    # calculate probability of being between two final profit levels
    lower_bound = st.number_input('Seuil inférieur de balance', min_value=0, value=int(0.9*initial_balance))
    upper_bound = st.number_input('Seuil supérieur de balance', min_value=0, value=int(1.1*initial_balance))
    
    probability = np.mean([lower_bound - initial_balance < result < upper_bound - initial_balance for result in final_results])
    
    st.metric(f'Probabilité d\'être entre les niveaux de balance {lower_bound} et {upper_bound}', f'{probability:.2%}')