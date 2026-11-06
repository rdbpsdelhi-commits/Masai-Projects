import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

def main():
    os.makedirs('outputs', exist_ok=True)
    df = pd.read_csv("data/modelling_table.csv")
    
    FEATURES = [
        "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value",
        "n_stops", "rain_flag", "sale_event_flag", "is_cod", "is_weekend",
        "dispatch_hour", "hub_tier", "hub_daily_capacity", "label_print_batch",
        "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"
    ]
    
    X = df[FEATURES]
    y = df["delivery_hours"]
    
    # 1. Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 2. Scaling without leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_max_abs_mean = np.max(np.abs(X_train_scaled.mean(axis=0)))
    test_max_abs_mean = np.max(np.abs(X_test_scaled.mean(axis=0)))
    
    checks_df = pd.DataFrame({
        'metric': ['train_rows', 'val_rows', 'test_rows', 'train_scaled_max_abs_mean', 'test_scaled_max_abs_mean'],
        'value': [len(X_train), len(X_val), len(X_test), train_max_abs_mean, test_max_abs_mean]
    })
    checks_df.to_csv("outputs/part_b_checks.csv", index=False)
    
    # 3. Baseline
    mean_baseline = y_train.mean()
    y_pred_base = np.full_like(y_val, mean_baseline, dtype=float)
    base_mae, base_mse, base_rmse, base_r2 = get_metrics(y_val, y_pred_base)
    
    metrics_records = []
    metrics_records.append({
        'model': 'mean_baseline', 'split': 'val',
        'MAE': base_mae, 'MSE': base_mse, 'RMSE': base_rmse, 'R2': base_r2
    })
    
    # 4. Tree-depth diagnostic
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    tree_records = []
    
    best_depth = None
    min_val_rmse = float('inf')
    
    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        
        y_pred_train = dt.predict(X_train)
        y_pred_val = dt.predict(X_val)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        
        depth_label = 'unlimited' if d is None else d
        tree_records.append({'max_depth': depth_label, 'train_rmse': train_rmse, 'val_rmse': val_rmse})
        
        if val_rmse < min_val_rmse:
            min_val_rmse = val_rmse
            best_depth = d
            
    pd.DataFrame(tree_records).to_csv("outputs/part_b_tree_depth.csv", index=False)
    
    # 5. Four candidate models
    models = {
        'linear_regression': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
        'decision_tree': DecisionTreeRegressor(max_depth=best_depth, random_state=42),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    model_objects = {}
    val_rmses = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        mae, mse, rmse, r2 = get_metrics(y_val, y_pred_val)
        metrics_records.append({
            'model': name, 'split': 'val',
            'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2
        })
        model_objects[name] = model
        val_rmses[name] = rmse
        
    # 6. Learning-rate experiment
    lr_records = []
    for lr in [0.01, 0.1, 1.0]:
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        gb.fit(X_train, y_train)
        y_pred_val = gb.predict(X_val)
        _, _, rmse, r2 = get_metrics(y_val, y_pred_val)
        lr_records.append({'learning_rate': lr, 'val_rmse': rmse, 'val_r2': r2})
        
    pd.DataFrame(lr_records).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
    
    # 7. Cross-validated comparison
    sorted_candidates = sorted(val_rmses.items(), key=lambda x: x[1])
    top_2_names = [sorted_candidates[0][0], sorted_candidates[1][0]]
    
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_records = []
    
    for name in top_2_names:
        model = models[name]
        scores = cross_val_score(model, X_train_val, y_train_val, cv=kf)
        cv_records.append({'model': name, 'mean_r2': scores.mean(), 'std_r2': scores.std()})
        
    pd.DataFrame(cv_records).to_csv("outputs/part_b_cv.csv", index=False)
    
    # Final model on test split
    final_model_name = sorted_candidates[0][0]
    final_model = model_objects[final_model_name]
    
    y_pred_test = final_model.predict(X_test)
    test_mae, test_mse, test_rmse, test_r2 = get_metrics(y_test, y_pred_test)
    metrics_records.append({
        'model': final_model_name, 'split': 'test',
        'MAE': test_mae, 'MSE': test_mse, 'RMSE': test_rmse, 'R2': test_r2
    })
    
    pd.DataFrame(metrics_records).to_csv("outputs/part_b_metrics.csv", index=False)
    print("Part B regression script executed successfully!")

if __name__ == "__main__":
    main()
