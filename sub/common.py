import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# honour .env or default to "false"
USE_META_LABEL = os.getenv("USE_META_LABEL", "false").lower() == "true"

def compute_meta_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary label: 1 if next close > current close, else 0.
    Feature columns are *not* checked here; that’s up to caller.
    """
    df = df.copy()
    df["meta_label"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df
