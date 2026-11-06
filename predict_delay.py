import pandas as pd
import numpy as np
import joblib
import os
from src.shipment_prep import build_features

def clean_new_shipments(df, impute_df):
    # Replicate cleaning logic
    df['service_type'] = df['service_type'].astype(str).str.strip().str.title()
    df['payment_mode'] = df['payment_mode'].fillna('Unknown')
    df['dispatch_time'] = pd.to_datetime(df['dispatch_time'])
    
    cleaned_values = df['declared_value'].astype(str).str.replace('₹', '', regex=False).str.replace(',', '', regex=False)
    df['declared_value_inr'] = pd.to_numeric(cleaned_values, errors='coerce')
    df = df.drop(columns=['declared_value'])
    
    df.loc[df['parcel_weight_kg'] > 100, 'parcel_weight_kg'] = np.nan
    
    # Apply stored medians
    medians = dict(zip(impute_df['column'], impute_df['median']))
    for col in ['distance_km', 'parcel_weight_kg', 'declared_value_inr']:
        if col in medians:
            df[col] = df[col].fillna(medians[col])
    
    return df

def main():
    os.makedirs('outputs', exist_ok=True)
    
    new_df = pd.read_csv('data/new_shipments.csv')
    hubs_df = pd.read_csv('data/hubs.csv')
    impute_df = pd.read_csv('models/impute_values.csv')
    feature_order = pd.read_csv('models/feature_order.csv')['feature'].tolist()
    threshold = pd.read_csv('models/delay_threshold.csv')['threshold'].iloc[0]
    hub_segments = pd.read_csv('outputs/hub_segments.csv')
    model = joblib.load('models/delay_model.pkl')
    
    # 1. Clean
    clean_df = clean_new_shipments(new_df, impute_df)
    
    # 2. Build features
    features_df = build_features(clean_df, hubs_df)
    
    # 3. Arrange columns
    X = features_df[feature_order]
    
    # 4. Predict
    probs = model.predict_proba(X)[:, 1]
    probs_rounded = np.round(probs, 4)
    flagged = (probs_rounded >= threshold).astype(int)
    
    # 5. Join segments and set action
    results = clean_df[['shipment_id', 'hub_id']].copy()
    results['delay_probability'] = probs_rounded
    results['flagged'] = flagged
    
    results = results.merge(hub_segments[['hub_id', 'segment']], on='hub_id', how='left')
    
    def get_action(row):
        if row['flagged'] == 1:
            if row['segment'] == 'Escalate':
                return 'Escalate to hub manager'
            else:
                return 'Proactive customer call'
        else:
            return 'No action'
            
    results['action'] = results.apply(get_action, axis=1)
    
    # Arrange final columns
    final_cols = ['shipment_id', 'hub_id', 'segment', 'delay_probability', 'flagged', 'action']
    results = results[final_cols]
    
    results.to_csv('outputs/new_shipment_actions.csv', index=False)
    print("Scoring script completed successfully! Wrote outputs/new_shipment_actions.csv")

if __name__ == "__main__":
    main()
