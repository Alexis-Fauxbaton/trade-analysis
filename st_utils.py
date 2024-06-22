# import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

def get_all_trades(path='./data/11083245-trading-data.csv'):
    # Récupérer les trades
    try:
        # trades = mt5.history_orders_get(date_from=datetime(2024, 1, 27), date_to=datetime.now())
        
        # print(trades)
        
        # # Créer un DataFrame à partir des trades
        # df = pd.DataFrame(trades)
        
        # # Convertir le temps en format datetime
        # df['time'] = pd.to_datetime(df['time'], unit='s')
        raise Exception('Not implemented')
    except Exception as e:
        df = pd.read_csv(path)
        df.rename({'Open Time': 'time'}, inplace=True, axis=1)
    return df