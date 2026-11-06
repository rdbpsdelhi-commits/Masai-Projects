import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from itertools import combinations

def main():
    os.makedirs('outputs', exist_ok=True)
    df = pd.read_csv('data/modelling_table.csv')
    hubs = pd.read_csv('data/hubs.csv')
    
    # Task 1: Hub profile
    profile = df.groupby('hub_id').agg(
        shipments=('shipment_id', 'count'),
        mean_delivery_hours=('delivery_hours', 'mean'),
        delay_rate=('is_delayed', 'mean'),
        mean_distance_km=('distance_km', 'mean'),
        mean_n_stops=('n_stops', 'mean')
    ).reset_index().sort_values('hub_id')
    profile.to_csv('outputs/hub_profile.csv', index=False)
    
    # Task 2 & 3: K-Means segmentation & Separation evidence
    cols_to_scale = ['mean_delivery_hours', 'delay_rate', 'mean_distance_km', 'mean_n_stops']
    X = profile[cols_to_scale].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    sep_records = []
    for k in [2, 3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # calculate distances
        n_hubs = len(X_scaled)
        same_d, diff_d = [], []
        for i, j in combinations(range(n_hubs), 2):
            d = np.linalg.norm(X_scaled[i] - X_scaled[j])
            if labels[i] == labels[j]:
                same_d.append(d)
            else:
                diff_d.append(d)
                
        across_over_within = np.mean(diff_d) / np.mean(same_d) if same_d else np.nan
        sizes = '-'.join(map(str, sorted(pd.Series(labels).value_counts().values, reverse=True)))
        sep_records.append({'K': k, 'across_over_within': across_over_within, 'cluster_sizes': sizes})
        
        if k == 3:
            final_labels = labels
            
    pd.DataFrame(sep_records).to_csv('outputs/part_d_separation.csv', index=False)
    
    # Task 4: Segment names and tier check
    profile['cluster'] = final_labels
    cluster_delays = profile.groupby('cluster')['delay_rate'].mean().sort_values()
    
    # Standard (lowest), Watch (middle), Escalate (highest)
    segment_map = {
        cluster_delays.index[0]: 'Standard',
        cluster_delays.index[1]: 'Watch',
        cluster_delays.index[2]: 'Escalate'
    }
    profile['segment'] = profile['cluster'].map(segment_map)
    
    hub_segments = profile[['hub_id', 'segment']].merge(hubs[['hub_id', 'hub_city', 'hub_tier']], on='hub_id')
    hub_segments.to_csv('outputs/hub_segments.csv', index=False)
    
    segment_profile = profile.groupby('segment').agg(
        hubs=('hub_id', 'count'),
        mean_delivery_hours=('mean_delivery_hours', 'mean'),
        delay_rate=('delay_rate', 'mean'),
        mean_distance_km=('mean_distance_km', 'mean'),
        mean_n_stops=('mean_n_stops', 'mean')
    ).reset_index()
    segment_profile = segment_profile.sort_values('delay_rate')
    segment_profile.to_csv('outputs/segment_profile.csv', index=False)
    
    tier_implied = {1: 'Standard', 2: 'Watch', 3: 'Escalate'}
    hub_segments['implied'] = hub_segments['hub_tier'].map(tier_implied)
    mismatches = hub_segments[hub_segments['segment'] != hub_segments['implied']].drop(columns=['implied'])
    mismatches.to_csv('outputs/tier_vs_segment_mismatch.csv', index=False)
    
    print("Part D Segments completed. Created part_d_separation.csv, hub_segments.csv, segment_profile.csv, tier_vs_segment_mismatch.csv")

if __name__ == "__main__":
    main()
