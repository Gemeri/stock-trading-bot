import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

USE_META_LABEL = os.getenv("USE_META_LABEL", "false").lower() == "true"

def compute_meta_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["meta_label"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df
