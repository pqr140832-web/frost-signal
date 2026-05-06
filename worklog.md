# Competition Work Log

## 2026-05-06 Session 2

### Current Best Scores
- v13: 61.9 (submitted)
- v14: 63.02 (submitted) 
- v28: not yet submitted

### v28 Key Results
- **True LOLO eval**: R²=0.9179, MAE=0.6228 (v14: R²=0.846, MAE=0.859)
- Test-relevant groups LOLO (only groups in test set):
  - dye: MAE=0.639 (v14=0.489)
  - paper: MAE=0.674 (v14=0.663)
  - shu_red: MAE=0.781 (v14=1.179) ← big improvement
  - jade_green: MAE=0.328 (v14=0.317) ← similar
  - cobalt_blue: MAE=0.277 (v14=1.327) ← massive improvement
- **Test set predictions saved**: baseline_and_data/predict_out.csv

### Key Discovery: "other" group NOT in test set
The "other" group (中国画-大红, 孔雀蓝, 柠檬黄, 矿物颜料-*, 颜彩-*) has LOLO MAE=0.97 but these samples are NOT in the test set. So this doesn't affect scoring.

### Files Available for Submission
1. **predict_out_v28.csv** - v28 pure (best LOLO overall)
2. **predict_out_ensemble_lolo.csv** - LOLO-weighted v14+v28 ensemble
   - dye: 60% v14 + 40% v28 (v14 better for dye)
   - paper: 50/50
   - jade_green: 45% v14 + 55% v28
   - cobalt_blue: 15% v14 + 85% v28 (v28 much better)
   - shu_red: 40% v14 + 60% v28 (v28 better)
3. **predict_out_v14_v28_w0.5.csv** - simple 50/50 ensemble

### v28 Technical Details
- RobustGroupModel: uses median pooling for group channel statistics
- Michaelis-Menten saturation curve added to model library
- Per-group strategy selection in _select_strategy()
- Added NaN/Inf safety for predictions

### Versions Tried This Session
- v26: XGBoost/LightGBM ML approach → worse than v14 (too few data)
- v27: Color channel time series → R²=-91.6 in true LOLO (complete failure)
- v28: Robust group model → best LOLO (MAE=0.6228)
- v28+MM: Added Michaelis-Menten → slight improvement (MAE=0.6228→same)
