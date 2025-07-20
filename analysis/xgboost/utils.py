import pandas as pd

def load_and_engineer_features(filepath: str) -> pd.DataFrame:
    """
    Loads stock data from a CSV and applies feature engineering for modeling.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with engineered features and no NaNs.
    """
    # Load data
    df = pd.read_csv(filepath)

    # Parse datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by time
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Feature engineering
    df['returns_1'] = df['close'].pct_change()
    df['returns_3'] = df['close'].pct_change(3)
    df['returns_5'] = df['close'].pct_change(5)
    df['ma_3'] = df['close'].rolling(3).mean()
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_10'] = df['close'].rolling(10).mean()
    df['std_5'] = df['close'].rolling(5).std()
    df['std_10'] = df['close'].rolling(10).std()
    df['high_low_range'] = df['high'] - df['low']
    df['open_close_diff'] = df['open'] - df['close']
    df['volume_change'] = df['volume'].pct_change()

    # Drop rows with NaNs
    df.dropna(inplace=True)

    return df