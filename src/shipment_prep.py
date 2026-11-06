import pandas as pd
import numpy as np

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

def build_features(clean, hubs):
    """
    Merges hub data and builds engineered features.
    """
    # Left merge on hub_id
    df = clean.merge(hubs, on="hub_id", how="left")
    
    # 1. promised_hours
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    
    # 2. is_cod
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    
    # 3. is_weekend
    dispatch_dt = pd.to_datetime(df["dispatch_time"])
    df["is_weekend"] = (dispatch_dt.dt.dayofweek >= 5).astype(int)
    
    # 4. dispatch_hour
    df["dispatch_hour"] = dispatch_dt.dt.hour.astype(int)
    
    # 5. log_declared_value
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    
    return df
