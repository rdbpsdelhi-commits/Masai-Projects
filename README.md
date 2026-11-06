# Data findings

(i) The Express service type is delayed most often, with a delay rate of 48.97%.

(ii) The delay rate increases significantly as we move from Tier 1 to Tier 3 hubs, rising from 4.00% in Tier 1 to 44.00% in Tier 3.

(iii) The delay rate strongly increases across distance bands: 0-200 km (2.19%), 201-400 km (13.68%), 401-600 km (43.64%), 601+ km (85.23%).

(iv) The three columns most strongly correlated with delivery_hours are promised_hours (+0.8372), distance_km (+0.4013), n_stops (+0.3247).

(v) The `declared_value_inr` column was log-transformed to address its right skewness (mean 1533.05 vs median 1121.50); the log-transformed variable is much more symmetric (mean 7.03 vs median 7.02).

## Model results (regression)

- **Baseline R²:** The mean baseline R² is -0.0212. This indicates that simply predicting the average delivery time explains 0% of the variance in the target variable, serving as a zero-skill reference point.
- **Validation Winner:** The `linear_regression` model performed best on the validation split, achieving an RMSE of 4.89 hours.
- **Tree-depth Diagnostic:** Shallow trees (depths 1-3) show high bias (underfitting), as both train and validation RMSE remain high. Deeper trees (depths 8, 12, and unlimited) suffer from high variance (overfitting), evidenced by a near-zero train RMSE but a diverging, higher validation RMSE.
- **Learning Rate Experiment:** A learning rate of 0.1 was optimal. A rate of 0.01 learns too slowly and fails to minimize error within 100 trees, while 1.0 is too large and overshoots the optimal weights, resulting in worse performance.
- **Cross-validation vs Validation:** The best model from 5-fold cross-validation is `gradient_boosting`. If this disagrees with the single validation set ranking, it is generally safer to ship the cross-validated model, as it provides a more robust and generalized estimate of performance across different data folds.

## Model results (classification)

- **Baselines:** The blind "all on time" baseline achieves a recall of 0.00. The default random forest easily outperforms it, achieving an F1-score of 0.7848 and accuracy of 0.9056 on the validation set.
- **Feature Selection:** Using SelectKBest, tracking performance from 5 to 7 features shows the F1-score changing from 0.6988 to 0.8333. Compared to the default model with all 17 features (F1: 0.7848), the 7-feature model performs similarly or better while being much simpler.
- **Hyperparameter Tuning:** The default random forest on 7 features yielded a validation F1 of 0.8333. The GridSearchCV achieved an F1 of 0.7750. The final pipeline was chosen from `default forest same 7 features`.
- **Business Logic & Threshold:** Operations requires capturing at least 80% of true delays. The highest probability threshold that satisfies this constraint on the validation set is 0.5. Applying this strict threshold to the hold-out test set resulted in a final recall of 0.7561 and precision of 0.6889.

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Generate data: `python generate_data.py`
3. Run Part A: `python part_a_clean_eda.py`
4. Run Part B: `python part_b_regression.py`
5. Run Part C: `python part_c_classification.py`
6. Run Part D: `python part_d_segments.py`
7. Package Model: `python package_model.py`
8. Predict Delays (Simulate Production): `python predict_delay.py`

## Operations recommendation
- **Escalate Segment**: Immediately review capacity and processes for hubs in this segment, as they show severe delay patterns.
- **Watch Segment**: Monitor these hubs closely for deteriorating performance.
- **Actionable Interventions**: Utilize the scoring script's output to trigger proactive customer calls for predicted high-value delays and escalate extreme cases to hub managers.

## Production plan
- **Deployment**: Deploy the serialized `delay_model.pkl` and the inference pipeline behind a REST API (e.g., using FastAPI or Flask) containerized with Docker.
- **Batch Inference**: Schedule `predict_delay.py` to run via a task orchestrator (like Airflow) on a regular cadence (e.g., every 4 hours) to score new shipments.
- **Monitoring**: Continuously monitor model drift and performance metrics against real outcomes, triggering retraining when recall drops below the 80% threshold.

## Limitations
- **Synthetic Data**: The model relies on a synthetic dataset, which may not capture unpredictable real-world anomalies (like unseasonal extreme weather or sudden infrastructure failures).
- **Imputation Strategy**: Using median imputation for missing values assumes the missingness is random, which could mask systematic data capture issues.
- **Static Rules**: The predefined threshold (0.5) is optimized for the validation set; as operations evolve, this threshold will need dynamic recalibration to maintain the 80% recall constraint.
