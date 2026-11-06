import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # 1. Read config and data
    config = pd.read_csv('outputs/part_c_final_config.csv').iloc[0]
    df = pd.read_csv('data/modelling_table.csv')
    raw_df = pd.read_csv('data/raw_shipments.csv')
    
    FEATURES = [
        "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value",
        "n_stops", "rain_flag", "sale_event_flag", "is_cod", "is_weekend",
        "dispatch_hour", "hub_tier", "hub_daily_capacity", "label_print_batch",
        "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"
    ]
    
    X = df[FEATURES]
    y = df["is_delayed"]
    
    # 2. Rebuild split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # 3. Fit final pipeline
    n_estimators = int(config['n_estimators'])
    max_depth = None if str(config['max_depth']) == 'unlimited' else int(config['max_depth'])
    min_samples_leaf = int(config['min_samples_leaf'])
    k_features = int(config['selector_k'])
    threshold = float(config['threshold'])
    
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
    pipeline = Pipeline([
        ('selector', SelectKBest(f_classif, k=k_features)),
        ('model', rf)
    ])
    pipeline.fit(X_train, y_train)
    
    # 4. Save artifacts
    joblib.dump(pipeline, 'models/delay_model.pkl')
    pd.DataFrame([{'threshold': threshold}]).to_csv('models/delay_threshold.csv', index=False)
    pd.DataFrame({'feature': FEATURES}).to_csv('models/feature_order.csv', index=False)
    
    # Medians from clean_shipments (which was computed from raw_shipments before extreme value removal of >100 weight)
    # Wait, the prompt says "the three medians from clean_shipments". We can recompute or load from part A.
    # Let's compute directly from raw exactly as Part A did.
    raw_df['declared_value_inr'] = pd.to_numeric(raw_df['declared_value'].astype(str).str.replace('₹', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
    raw_df.loc[raw_df['parcel_weight_kg'] > 100, 'parcel_weight_kg'] = np.nan
    medians = {
        'distance_km': raw_df['distance_km'].median(),
        'parcel_weight_kg': raw_df['parcel_weight_kg'].median(),
        'declared_value_inr': raw_df['declared_value_inr'].median()
    }
    pd.DataFrame(list(medians.items()), columns=['column', 'median']).to_csv('models/impute_values.csv', index=False)
    
    # 5. Reload and check
    loaded_pipeline = joblib.load('models/delay_model.pkl')
    
    probs_memory = pipeline.predict_proba(X_test)[:, 1]
    probs_loaded = loaded_pipeline.predict_proba(X_test)[:, 1]
    
    diff = np.max(np.abs(probs_memory - probs_loaded))
    
    check_records = [
        {'metric': 'test_rows_scored', 'value': len(X_test)},
        {'metric': 'reload_max_abs_probability_diff', 'value': diff},
        {'metric': 'threshold', 'value': threshold}
    ]
    pd.DataFrame(check_records).to_csv('outputs/part_d_reload_check.csv', index=False)
    print("Model packaging completed successfully!")

if __name__ == "__main__":
    main()
