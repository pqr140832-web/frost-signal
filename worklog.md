# Competition Work Log

## 2026-05-06 Session

### Task: Improve paint aging prediction model (v13 score=61.9, first place=73)

### Analysis Summary
- Reviewed all existing code: v13 (best, 61.9), v14 (never submitted), v15 (never submitted)
- v13 uses hierarchical group model + v12 power-law hybrid
- v14 uses color channel decomposition (ΔL, Δa, Δb) + multi-strategy ensemble

### Versions Created This Session
1. **v24.py**: Replaced v13's fixed params with auto-optimized ones + 远距离保护 → MAE=0.305 (worse than v13)
2. **v24b.py**: v13 base + 曙红 enhanced model → MAE=0.308 (worse, 曙红 model overpredicts)
3. **v24c.py**: v13 base + weight re-search → MAE=0.287 (marginal improvement, 0.0034 better)
4. **v25.py**: Full parameter search (ws, n, w_hier) → MAE=1.08 (broken for 曙红 and other groups)

### Key Discovery
**v14 was NEVER submitted!** v14's forward eval:
- R²=0.9879 (v13: 0.9784)
- MAE=0.1975 (v13: 0.2906)
- Better in EVERY group: dye 0.070 vs 0.390, paper 0.060 vs 0.080, shu_red 0.224 vs 0.409, jade_green 0.092 vs 0.128, cobalt_blue 0.240 vs 0.268

If competition scoring scales linearly with MAE, v14's MAE=0.198 would correspond to ~73 points (first place level).

### Files Prepared
- `/home/z/my-project/upload/predict_out.csv` → v14 predictions
- `/home/z/my-project/download/predict_out_v13.csv` → v13 backup
- `/home/z/my-project/download/predict_out_v14.csv` → v14 predictions
- `/home/z/my-project/download/predict_out_ensemble.csv` → v13+v14 average

### Conclusion
v13's parameters are near-optimal. The biggest opportunity is submitting v14 which was previously overlooked. Recommend submitting v14 predictions first, fall back to ensemble if v14 underperforms.
