import pandas as pd
import numpy as np
import os
from src.shipment_prep import build_features

def clean_shipments_and_get_metrics(raw_df):
    metrics = {}
    
    # Ensure one row per unique shipment (drop duplicates based on ID)
    df = raw_df.drop_duplicates(subset=['shipment_id']).copy()

    # Record "before" metrics
    metrics['missing_distance_before'] = int(df['distance_km'].isna().sum())
    metrics['missing_parcel_weight_before'] = int(df['parcel_weight_kg'].isna().sum())
    metrics['service_type_spellings_before'] = int(df['service_type'].nunique())

    # Normalize service_type
    df['service_type'] = df['service_type'].astype(str).str.strip().str.title()
    metrics['service_type_spellings_after'] = int(df['service_type'].nunique())

    # Handle missing payment_mode
    df['payment_mode'] = df['payment_mode'].fillna('Unknown')

    # Clean and convert declared_value
    cleaned_values = df['declared_value'].astype(str).str.replace('₹', '', regex=False).str.replace(',', '', regex=False)
    df['declared_value_inr'] = pd.to_numeric(cleaned_values, errors='coerce')
    df = df.drop(columns=['declared_value'])

    # Convert dispatch_time
    df['dispatch_time'] = pd.to_datetime(df['dispatch_time'])

    # Remove sentinel/extreme values in weight
    df.loc[df['parcel_weight_kg'] > 100, 'parcel_weight_kg'] = np.nan

    # Fill missing values with medians
    medians = {
        'distance_km': df['distance_km'].median(),
        'parcel_weight_kg': df['parcel_weight_kg'].median(),
        'declared_value_inr': df['declared_value_inr'].median()
    }
    for col, val in medians.items():
        df[col] = df[col].fillna(val)

    return df, metrics

def main():
    # Read data
    raw_ship_df = pd.read_csv("data/raw_shipments.csv")
    hubs_df = pd.read_csv("data/hubs.csv")

    # Clean data and get pre-cleaning metrics
    clean_df, metrics = clean_shipments_and_get_metrics(raw_ship_df)
    
    # Build features
    model_df = build_features(clean_df, hubs_df)

    # Add the target is_delayed (1 when delivery_hours > promised_hours, else 0)
    model_df['is_delayed'] = (model_df['delivery_hours'] > model_df['promised_hours']).astype(int)

    # Calculate remaining metrics
    FEATURES = [
        "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value",
        "n_stops", "rain_flag", "sale_event_flag", "is_cod", "is_weekend", 
        "dispatch_hour", "hub_tier", "hub_daily_capacity", "label_print_batch", 
        "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"
    ]
    cols_to_check = FEATURES + ["delivery_hours", "is_delayed"]
    metrics['nulls_in_features_and_target'] = int(model_df[cols_to_check].isna().sum().sum())
    
    metrics['declared_value_mean_inr'] = round(model_df['declared_value_inr'].mean(), 2)
    metrics['declared_value_median_inr'] = round(model_df['declared_value_inr'].median(), 2)
    metrics['log_declared_value_mean'] = round(model_df['log_declared_value'].mean(), 2)
    metrics['log_declared_value_median'] = round(model_df['log_declared_value'].median(), 2)
    
    q1 = model_df['delivery_hours'].quantile(0.25)
    q3 = model_df['delivery_hours'].quantile(0.75)
    iqr = q3 - q1
    metrics['hours_q1'] = round(q1, 2)
    metrics['hours_q3'] = round(q3, 2)
    metrics['hours_iqr'] = round(iqr, 2)
    metrics['hours_lower_bound'] = round(q1 - 1.5 * iqr, 2)
    metrics['hours_upper_bound'] = round(q3 + 1.5 * iqr, 2)
    
    # Save metrics to outputs/part_a_checks.csv
    os.makedirs('outputs', exist_ok=True)
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['metric', 'value'])
    metrics_df.to_csv('outputs/part_a_checks.csv', index=False)

    # Write the result to data/modelling_table.csv
    model_df.to_csv("data/modelling_table.csv", index=False)
    print(f"Created modelling_table.csv successfully with shape {model_df.shape}")
    print(f"Created outputs/part_a_checks.csv with {len(metrics_df)} metrics")

if __name__ == "__main__":
    main()
