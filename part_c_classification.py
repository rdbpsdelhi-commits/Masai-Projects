import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

def get_metrics_from_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return TN, FP, FN, TP, accuracy, precision, recall, f1

def main():
    os.makedirs('outputs', exist_ok=True)
    df = pd.read_csv("data/modelling_table.csv")
    
    FEATURES = [
        "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value",
        "n_stops", "rain_flag", "sale_event_flag", "is_cod", "is_weekend",
        "dispatch_hour", "hub_tier", "hub_daily_capacity", "label_print_batch",
        "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"
    ]
    NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]
    
    X = df[FEATURES]
    y = df["is_delayed"]
    
    # 1. Stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # 2. Baseline and classifiers
    models_records = []
    
    # Blind baseline
    y_pred_blind = np.zeros_like(y_val)
    TN, FP, FN, TP, acc, prec, rec, f1 = get_metrics_from_cm(y_val, y_pred_blind)
    models_records.append({'model': 'blind_all_on_time', 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})
    
    # Logistic Regression
    lr_pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(random_state=42))])
    lr_pipe.fit(X_train, y_train)
    TN, FP, FN, TP, acc, prec, rec, f1 = get_metrics_from_cm(y_val, lr_pipe.predict(X_val))
    models_records.append({'model': 'logistic_regression', 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})
    
    # Decision tree depths
    best_dt_f1 = -1
    best_dt_d = 2
    for d in [2, 3, 4, 5, 6, 8, 10]:
        dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        _, _, _, _, _, _, _, dt_f1 = get_metrics_from_cm(y_val, dt.predict(X_val))
        if dt_f1 > best_dt_f1:
            best_dt_f1 = dt_f1
            best_dt_d = d
            
    dt_best = DecisionTreeClassifier(criterion="gini", max_depth=best_dt_d, random_state=42)
    dt_best.fit(X_train, y_train)
    TN, FP, FN, TP, acc, prec, rec, f1 = get_metrics_from_cm(y_val, dt_best.predict(X_val))
    models_records.append({'model': f'decision_tree_depth_{best_dt_d}', 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})
    
    # Random Forest Default
    rf_def = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def.fit(X_train, y_train)
    TN, FP, FN, TP, acc, prec, rec, f1 = get_metrics_from_cm(y_val, rf_def.predict(X_val))
    models_records.append({'model': 'random_forest_default', 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})
    
    pd.DataFrame(models_records).to_csv("outputs/part_c_models.csv", index=False)
    
    # 3. Feature importance and SelectKBest
    rf_feat = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_feat.fit(X_train, y_train)
    importances = rf_feat.feature_importances_
    
    skb_all = SelectKBest(f_classif, k="all")
    skb_all.fit(X_train, y_train)
    f_scores = skb_all.scores_
    
    ranks = pd.Series(f_scores).rank(ascending=False, method='min').astype(int).values
    
    feature_records = []
    for i, f in enumerate(FEATURES):
        feature_records.append({
            'feature': f,
            'f_score': f_scores[i],
            'f_rank': ranks[i],
            'rf_importance': importances[i],
            'is_noise': f in NOISE_COLUMNS
        })
    pd.DataFrame(feature_records).to_csv("outputs/part_c_feature_scores.csv", index=False)
    
    k_records = []
    for k in [5, 7, 10, 12, 17]:
        pipe = Pipeline([('skb', SelectKBest(f_classif, k=k)), ('rf', RandomForestClassifier(n_estimators=100, random_state=42))])
        pipe.fit(X_train, y_train)
        _, _, _, _, _, _, _, k_f1 = get_metrics_from_cm(y_val, pipe.predict(X_val))
        k_records.append({'k': k, 'val_f1': k_f1})
    pd.DataFrame(k_records).to_csv("outputs/part_c_selectk.csv", index=False)
    
    # 4. Tuning
    skb7 = SelectKBest(f_classif, k=7)
    X_train_7 = skb7.fit_transform(X_train, y_train)
    X_val_7 = skb7.transform(X_val)
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [4, 6, 8],
        'min_samples_leaf': [1, 5, 10]
    }
    
    gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    gs.fit(X_train_7, y_train)
    _, _, _, _, _, _, _, gs_f1 = get_metrics_from_cm(y_val, gs.predict(X_val_7))
    
    rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=9, cv=5, random_state=42)
    rs.fit(X_train_7, y_train)
    _, _, _, _, _, _, _, rs_f1 = get_metrics_from_cm(y_val, rs.predict(X_val_7))
    
    rf_def_7 = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def_7.fit(X_train_7, y_train)
    _, _, _, _, _, _, _, def_f1 = get_metrics_from_cm(y_val, rf_def_7.predict(X_val_7))
    
    tuning_records = [
        {'search': 'grid_search', 'candidates': len(gs.cv_results_['params']), 'cv_folds': 5, 'total_fits': len(gs.cv_results_['params'])*5, 'best_params': json.dumps(gs.best_params_), 'best_cv_accuracy': gs.best_score_, 'val_f1': gs_f1},
        {'search': 'randomized_search', 'candidates': len(rs.cv_results_['params']), 'cv_folds': 5, 'total_fits': len(rs.cv_results_['params'])*5, 'best_params': json.dumps(rs.best_params_), 'best_cv_accuracy': rs.best_score_, 'val_f1': rs_f1},
        {'search': 'default_forest_same_7_features', 'candidates': 1, 'cv_folds': 0, 'total_fits': 1, 'best_params': json.dumps({'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1}), 'best_cv_accuracy': '', 'val_f1': def_f1}
    ]
    pd.DataFrame(tuning_records).to_csv("outputs/part_c_tuning.csv", index=False)
    
    # 5. Final config
    best_search = max(tuning_records, key=lambda x: (x['val_f1'], x['search'] == 'default_forest_same_7_features', x['search'] == 'grid_search', x['search'] == 'randomized_search'))
    chosen_from = best_search['search']
    best_p = json.loads(best_search['best_params'])
    
    final_rf = RandomForestClassifier(random_state=42, n_estimators=best_p.get('n_estimators', 100), max_depth=best_p.get('max_depth'), min_samples_leaf=best_p.get('min_samples_leaf', 1))
    final_pipe = Pipeline([('selector', SelectKBest(f_classif, k=7)), ('model', final_rf)])
    final_pipe.fit(X_train, y_train)
    
    # 6. Alert threshold
    y_val_probs = final_pipe.predict_proba(X_val)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_records = []
    
    alert_t = 0.1
    for t in thresholds:
        y_pred_t = (y_val_probs >= t).astype(int)
        TN, FP, FN, TP, acc, prec, rec, f1 = get_metrics_from_cm(y_val, y_pred_t)
        thresh_records.append({'threshold': t, 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})
        if rec >= 0.80:
            alert_t = max(alert_t, t)
            
    pd.DataFrame(thresh_records).to_csv("outputs/part_c_thresholds.csv", index=False)
    
    # 7. Final Test
    y_test_probs = final_pipe.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_probs >= alert_t).astype(int)
    TN, FP, FN, TP, acc, prec, rec, f1 = get_metrics_from_cm(y_test, y_test_pred)
    pd.DataFrame([{'threshold': alert_t, 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}]).to_csv("outputs/part_c_final_test.csv", index=False)
    
    pd.DataFrame([{
        'selector_k': 7, 
        'n_estimators': best_p.get('n_estimators', 100), 
        'max_depth': 'unlimited' if best_p.get('max_depth') is None else best_p.get('max_depth'), 
        'min_samples_leaf': best_p.get('min_samples_leaf', 1), 
        'threshold': alert_t, 
        'chosen_from': chosen_from
    }]).to_csv("outputs/part_c_final_config.csv", index=False)
    
    print("Part C classification script executed successfully!")

if __name__ == "__main__":
    main()
