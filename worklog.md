# Competition Work Log

## 2026-05-06 Session

### Task: Improve paint aging prediction model (v13 score=61.9 → v14 score=63.02, first place=73)

### Previous Session Summary
- v13=61.9 (only submitted), v14 never submitted, v17=57.55 (submitted), v18 never submitted
- v14 has best local metrics (R²=0.9879, MAE=0.1975) but LOLO eval shows it's not that good
- User submitted v14 → scored 63.02 (improvement of +1.12)

### This Session - Key Analysis
- Baseline uses RandomForest with 4 features (aging_time_day, L0, a0, b0)
- First place is 73, second place is 70+
- Training data: 181 rows, 37 test rows
- Test set requires extrapolation (e.g., jade_green trains to t=24, test needs t=30)

### True LOLO (Leave-Last-Out) Evaluation
- **v14 true LOLO**: R²=0.846, MAE=0.859
  - dye: MAE=0.489 (good)
  - paper: MAE=0.663 (ok)
  - shu_red: MAE=1.179 (BAD - underestimates)
  - jade_green: MAE=0.317 (ok)
  - cobalt_blue: MAE=1.327 (BAD - very unstable)

### Versions Created This Session
1. **v26.py**: ML-based (XGBoost/LightGBM/CatBoost) + feature engineering + v14 ensemble
   - Local eval MAE=0.44 (but uses in-sample evaluation, misleading)
   - Too few data (181 rows) for ML to generalize well

2. **v27.py**: Color channel time series prediction (predict L/a/b then compute dE)
   - In-sample MAE=0.113 (but uses full training data, misleading)
   - True LOLO: R²=-91.6, MAE=9.44 (COMPLETE FAILURE)
   - Small L/a/b prediction errors get amplified when computing dE

3. **v28.py**: Robust group model with improved weak groups
   - True LOLO: R²=0.918, MAE=0.626 (BIG improvement over v14's 0.859!)
   - cobalt_blue: MAE 1.327→0.277 (massive improvement)
   - shu_red: MAE 1.179→0.781 (significant improvement)
   - jade_green: MAE 0.317→0.328 (slightly worse)
   - paper: MAE 0.663→0.671 (slightly worse)
   - dye: MAE 0.489→0.665 (worse)
   - Key insight: v28 uses robust median pooling for noisy groups (cobalt_blue)

### Prediction Files Ready
- `/home/z/my-project/upload/predict_out_v28.csv` → v28 pure predictions
- `/home/z/my-project/upload/predict_out_v14_v28_w0.3.csv` → 30% v14 + 70% v28
- `/home/z/my-project/upload/predict_out_v14_v28_w0.5.csv` → 50/50 ensemble
- `/home/z/my-project/upload/predict_out_v14_v28_w0.7.csv` → 70% v14 + 30% v28

### Recommendation
Try v28 first (best LOLO metrics), then try 50/50 ensemble if v28 underperforms.
The main improvement is in cobalt_blue and shu_red groups which were v14's biggest weaknesses.
