import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

class MT5Manager:
    def __init__(self):
        self.connected = False
        self.connected = self.connect()

    def connect(self, **kwargs):
        if not self.connected:
            self.connected = mt5.initialize(**kwargs)
            return self.connected
        return True

    def disconnect(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
    
    def login(self, account, password, server):
        if not self.connected:
            self.connect()
        print("Here")
        return mt5.login(account, password, server)

    def __del__(self):
        self.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

    def get_connected(self):
        return self.connected

    def get_last_error(self):
        return mt5.last_error()

    def get_last_error_message(self):
        return mt5.last_error_message()

    def get_last_error_code(self):
        return mt5.last_error_code
    
    def get_all_trades_since_to(self, date_from, date_to):
        # trades = mt5.history_orders_get(date_from=datetime(2024, 1, 27), date_to=datetime.now())
        if not isinstance(date_from, datetime):
            raise ValueError('date_from should be a datetime object')
        
        if not isinstance(date_to, datetime):
            raise ValueError('date_to should be a datetime object')

        trades = mt5.history_deals_get(date_from, date_to)
        
        print("Trades")
        print(trades)
        
        # Créer un DataFrame à partir des trades
        df = pd.DataFrame(list(trades), columns=trades[0]._asdict().keys())
        
        print("DataFrame")
        print(df.columns)
        
        print(df.head())
        
        # Convertir le temps en format datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        return df
    
    def reconnect(self, **kwargs):
        self.disconnect()
        # connect using the new parameters
        self.connected = self.connect(**kwargs)
        return self.connected