# Experiment 4 — Strategy development (BTC, instrument BTCUSDT)

All figures are R per trade AFTER costs (exact fractional sizing, risk governor off).
`score` = mean - 1.96 day-clustered SE (lower edge of a 95% interval).

## 1. Grid overview (share of configurations positive after costs)

| family | tf | configs | train_pos | val_pos | both_pos | eligible | best_val_score | median_val_mean | median_cost_r | median_trades_day |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 108 | 0.000 | 0.000 | 0.000 | 0 | -0.526 | -0.620 | 0.622 | 13.225 |
| A | 5 | 108 | 0.000 | 0.000 | 0.000 | 0 | -0.278 | -0.349 | 0.347 | 2.523 |
| A | 15 | 108 | 0.306 | 0.093 | 0.093 | 0 | -0.042 | -0.131 | 0.212 | 0.928 |
| A | 60 | 108 | 0.694 | 0.306 | 0.241 | 8 | -0.044 | -0.072 | 0.106 | 0.255 |
| B | 1 | 108 | 0.000 | 0.000 | 0.000 | 0 | -0.551 | -0.642 | 0.662 | 27.213 |
| B | 5 | 108 | 0.000 | 0.000 | 0.000 | 0 | -0.287 | -0.363 | 0.399 | 6.051 |
| B | 15 | 108 | 0.000 | 0.000 | 0.000 | 0 | -0.189 | -0.247 | 0.257 | 2.117 |
| B | 60 | 108 | 0.000 | 0.000 | 0.000 | 0 | -0.200 | -0.209 | 0.142 | 0.527 |

### Top 8 per family by validation score

| key | train_n | train_mean | val_n | val_mean | val_score | val_pf | plateau | eligible |
|---|---|---|---|---|---|---|---|---|
| 15m A w=40 tau=2.0 k=2.0 rr=3.0 | 443 | 0.049 | 182 | 0.171 | -0.042 | 1.306 | 0.400 | False |
| 60m A w=40 tau=2.0 k=1.5 rr=1.5 | 150 | 0.070 | 52 | 0.223 | -0.044 | 1.627 | 0.500 | False |
| 15m A w=20 tau=2.5 k=2.0 rr=2.0 | 407 | 0.150 | 173 | 0.106 | -0.065 | 1.220 | 0.333 | False |
| 60m A w=10 tau=1.5 k=2.0 rr=2.0 | 564 | -0.018 | 209 | 0.056 | -0.086 | 1.137 | 0.800 | False |
| 60m A w=10 tau=1.5 k=2.0 rr=2.5 | 554 | -0.012 | 203 | 0.039 | -0.110 | 1.091 | 0.800 | False |
| 60m A w=10 tau=1.5 k=2.0 rr=1.5 | 575 | -0.027 | 212 | 0.013 | -0.113 | 1.032 | 0.500 | False |
| 60m A w=10 tau=2.0 k=2.0 rr=2.5 | 305 | 0.123 | 129 | 0.096 | -0.113 | 1.223 | 0.833 | True |
| 15m A w=20 tau=2.5 k=2.0 rr=1.5 | 426 | 0.088 | 186 | 0.030 | -0.115 | 1.065 | 0.200 | False |

| key | train_n | train_mean | val_n | val_mean | val_score | val_pf | plateau | eligible |
|---|---|---|---|---|---|---|---|---|
| 15m B w=20 z=2.0 k=2.0 rr=3.0 | 3498 | -0.102 | 1439 | -0.134 | -0.189 | 0.739 | 0.000 | False |
| 15m B w=20 z=2.0 k=2.0 rr=2.5 | 3560 | -0.096 | 1470 | -0.143 | -0.194 | 0.720 | 0.000 | False |
| 60m B w=20 z=2.0 k=2.0 rr=2.5 | 984 | -0.058 | 418 | -0.103 | -0.200 | 0.782 | 0.000 | False |
| 15m B w=20 z=2.0 k=2.0 rr=2.0 | 3675 | -0.093 | 1525 | -0.153 | -0.201 | 0.698 | 0.000 | False |
| 60m B w=40 z=2.5 k=2.0 rr=3.0 | 453 | -0.157 | 213 | -0.069 | -0.212 | 0.860 | 0.000 | False |
| 60m B w=20 z=2.0 k=2.0 rr=2.0 | 1011 | -0.053 | 428 | -0.124 | -0.212 | 0.737 | 0.000 | False |
| 15m B w=20 z=2.0 k=2.0 rr=1.5 | 3850 | -0.108 | 1589 | -0.170 | -0.212 | 0.656 | 0.000 | False |
| 60m B w=80 z=2.0 k=2.0 rr=3.0 | 470 | -0.123 | 174 | -0.053 | -0.219 | 0.891 | 0.000 | False |

## 2. Strategy C (regime-adaptive)

| key | regime | train_n | train_mean | val_n | val_mean | val_score | eligible |
|---|---|---|---|---|---|---|---|
| 1m C1 [A w=20 tau=2.5 k=2.0 rr=3.0 | B w=80 z=3.0 k=2.0 rr=1.5] | variance-ratio rule, CUSUM+extreme-vol block | 3491 | -0.443 | 973 | -0.502 | -0.556 | False |
| 1m C2 [A w=20 tau=2.5 k=2.0 rr=3.0 | B w=80 z=3.0 k=2.0 rr=1.5] | session router RTH->NONE ETH->NONE | 0 |  | 0 |  |  | False |
| 5m C1 [A w=20 tau=2.5 k=2.0 rr=2.0 | B w=80 z=3.0 k=2.0 rr=3.0] | variance-ratio rule, CUSUM+extreme-vol block | 827 | -0.187 | 338 | -0.306 | -0.431 | False |
| 5m C2 [A w=20 tau=2.5 k=2.0 rr=2.0 | B w=80 z=3.0 k=2.0 rr=3.0] | session router RTH->NONE ETH->NONE | 0 |  | 0 |  |  | False |
| 15m C1 [A w=40 tau=2.0 k=2.0 rr=3.0 | B w=20 z=2.0 k=2.0 rr=3.0] | variance-ratio rule, CUSUM+extreme-vol block | 1242 | -0.112 | 506 | -0.170 | -0.271 | False |
| 15m C2 [A w=40 tau=2.0 k=2.0 rr=3.0 | B w=20 z=2.0 k=2.0 rr=3.0] | session router RTH->A ETH->A | 443 | 0.049 | 182 | 0.171 | -0.042 | True |
| 60m C1 [A w=40 tau=2.0 k=1.5 rr=1.5 | B w=20 z=2.0 k=2.0 rr=2.5] | variance-ratio rule, CUSUM+extreme-vol block | 406 | -0.144 | 159 | -0.163 | -0.327 | False |
| 60m C2 [A w=40 tau=2.0 k=1.5 rr=1.5 | B w=20 z=2.0 k=2.0 rr=2.5] | session router RTH->A ETH->A | 150 | 0.070 | 52 | 0.223 | -0.044 | True |

## 3. Entry methods in isolation (identical exits: 1.5 ATR stop, 2R target, 20-bar cap)

Order-flow imbalance is not in this table: no source provides it for the development period.

| tf | method | train_n | train_mean | train_t | val_n | val_mean | val_t |
|---|---|---|---|---|---|---|---|
| 1 | deviation_fade (z-score) | 63307 | -0.520 | -89.494 | 25084 | -0.655 | -87.690 |
| 1 | deviation_fade (causal Nadaraya-Watson) | 61520 | -0.517 | -88.158 | 24923 | -0.650 | -83.326 |
| 1 | deviation_fade (session VWAP) | 19922 | -0.528 | -76.153 | 8084 | -0.661 | -69.371 |
| 1 | regression_trend_change (OLS slope z) | 6902 | -0.444 | -39.831 | 2938 | -0.614 | -39.420 |
| 1 | regression_trend_change (Kalman slope z) | 37306 | -0.524 | -79.702 | 16040 | -0.670 | -78.564 |
| 1 | momentum_shift (z_mom) | 11097 | -0.450 | -47.991 | 4902 | -0.613 | -50.066 |
| 1 | volatility_expansion (range shock, follow) | 23353 | -0.439 | -56.362 | 7238 | -0.598 | -49.121 |
| 1 | breakout (N-bar high/low + z_mom>1) | 40132 | -0.523 | -81.707 | 17917 | -0.656 | -73.602 |
| 5 | deviation_fade (z-score) | 12266 | -0.270 | -30.460 | 4887 | -0.398 | -30.484 |
| 5 | deviation_fade (causal Nadaraya-Watson) | 11805 | -0.265 | -29.428 | 4742 | -0.401 | -30.534 |
| 5 | deviation_fade (session VWAP) | 5411 | -0.304 | -23.634 | 2232 | -0.401 | -22.250 |
| 5 | regression_trend_change (OLS slope z) | 1396 | -0.175 | -5.904 | 560 | -0.299 | -7.118 |
| 5 | regression_trend_change (Kalman slope z) | 6765 | -0.293 | -23.490 | 2813 | -0.402 | -22.452 |
| 5 | momentum_shift (z_mom) | 2244 | -0.239 | -10.285 | 933 | -0.400 | -13.040 |
| 5 | volatility_expansion (range shock, follow) | 4756 | -0.237 | -15.018 | 1605 | -0.361 | -13.952 |
| 5 | breakout (N-bar high/low + z_mom>1) | 6984 | -0.309 | -25.203 | 2946 | -0.394 | -21.637 |
| 15 | deviation_fade (z-score) | 4096 | -0.137 | -8.432 | 1679 | -0.223 | -9.405 |
| 15 | deviation_fade (causal Nadaraya-Watson) | 3912 | -0.144 | -8.162 | 1594 | -0.230 | -9.012 |
| 15 | deviation_fade (session VWAP) | 1712 | -0.203 | -8.582 | 681 | -0.289 | -8.380 |
| 15 | regression_trend_change (OLS slope z) | 527 | 0.062 | 1.245 | 193 | -0.001 | -0.008 |
| 15 | regression_trend_change (Kalman slope z) | 2205 | -0.153 | -6.590 | 854 | -0.213 | -5.778 |
| 15 | momentum_shift (z_mom) | 810 | -0.059 | -1.437 | 343 | -0.054 | -0.887 |
| 15 | volatility_expansion (range shock, follow) | 1608 | -0.104 | -3.560 | 617 | -0.208 | -4.712 |
| 15 | breakout (N-bar high/low + z_mom>1) | 2214 | -0.134 | -5.432 | 878 | -0.213 | -5.883 |
| 60 | deviation_fade (z-score) | 1089 | -0.076 | -2.429 | 463 | -0.203 | -4.494 |
| 60 | deviation_fade (causal Nadaraya-Watson) | 1007 | -0.075 | -2.266 | 431 | -0.170 | -3.653 |
| 60 | deviation_fade (session VWAP) | 513 | -0.164 | -3.972 | 227 | -0.260 | -4.666 |
| 60 | regression_trend_change (OLS slope z) | 153 | 0.118 | 1.330 | 64 | 0.007 | 0.051 |
| 60 | regression_trend_change (Kalman slope z) | 570 | -0.037 | -0.805 | 202 | -0.053 | -0.705 |
| 60 | momentum_shift (z_mom) | 224 | 0.087 | 1.116 | 93 | -0.055 | -0.490 |
| 60 | volatility_expansion (range shock, follow) | 413 | 0.055 | 1.004 | 183 | 0.098 | 1.195 |
| 60 | breakout (N-bar high/low + z_mom>1) | 592 | -0.050 | -1.087 | 214 | 0.073 | 0.940 |

## 4. Walk-forward — family A (12m in-sample -> next quarter)

| fold | choice | is_score | oos_n | oos_mean |
|---|---|---|---|---|
| 2022Q1 | 15m A w=20 tau=2.5 k=2.0 rr=3.0 | 0.018 | 37 | 0.215 |
| 2022Q2 | 15m A w=40 tau=2.5 k=2.0 rr=3.0 | 0.115 | 25 | 0.351 |
| 2022Q3 | 15m A w=40 tau=2.5 k=2.0 rr=3.0 | 0.194 | 22 | -0.246 |
| 2022Q4 | 15m A w=40 tau=2.5 k=1.5 rr=3.0 | 0.093 | 19 | -0.575 |
| 2023Q1 | 60m A w=20 tau=1.5 k=1.5 rr=1.5 | -0.005 | 44 | -0.186 |
| 2023Q2 | 15m A w=20 tau=2.5 k=1.5 rr=2.5 | -0.022 | 49 | 0.256 |
| 2023Q3 | 15m A w=20 tau=2.5 k=1.5 rr=3.0 | -0.021 | 48 | -0.297 |
| 2023Q4 | 60m A w=10 tau=2.0 k=2.0 rr=2.0 | -0.007 | 29 | 0.079 |
| 2024Q1 | 60m A w=40 tau=2.0 k=1.5 rr=2.5 | -0.022 | 16 | -0.063 |
| 2024Q2 | 15m A w=40 tau=2.0 k=2.0 rr=3.0 | 0.003 | 53 | 0.057 |

OOS: 342 trades, mean -0.0154 R, 95% CI [-0.1521, 0.1204], p(mean<=0) = 0.587, total -5.3 R

## 4. Walk-forward — family B (12m in-sample -> next quarter)

| fold | choice | is_score | oos_n | oos_mean |
|---|---|---|---|---|
| 2022Q1 | 60m B w=20 z=3.0 k=2.0 rr=2.5 | -0.197 | 26 | -0.273 |
| 2022Q2 | 60m B w=20 z=2.0 k=2.0 rr=2.0 | -0.078 | 100 | -0.196 |
| 2022Q3 | NO TRADE (nothing positive in-sample) |  | 0 |  |
| 2022Q4 | 60m B w=20 z=2.0 k=2.0 rr=2.5 | -0.075 | 94 | -0.233 |
| 2023Q1 | NO TRADE (nothing positive in-sample) |  | 0 |  |
| 2023Q2 | NO TRADE (nothing positive in-sample) |  | 0 |  |
| 2023Q3 | 60m B w=20 z=3.0 k=1.0 rr=2.5 | -0.191 | 34 | -0.311 |
| 2023Q4 | NO TRADE (nothing positive in-sample) |  | 0 |  |
| 2024Q1 | NO TRADE (nothing positive in-sample) |  | 0 |  |
| 2024Q2 | NO TRADE (nothing positive in-sample) |  | 0 |  |

OOS: 254 trades, mean -0.2329 R, 95% CI [-0.3381, -0.1210], p(mean<=0) = 1.000, total -59.1 R

## 4. Walk-forward — family A or B (12m in-sample -> next quarter)

| fold | choice | is_score | oos_n | oos_mean |
|---|---|---|---|---|
| 2022Q1 | 15m A w=20 tau=2.5 k=2.0 rr=3.0 | 0.018 | 37 | 0.215 |
| 2022Q2 | 15m A w=40 tau=2.5 k=2.0 rr=3.0 | 0.115 | 25 | 0.351 |
| 2022Q3 | 15m A w=40 tau=2.5 k=2.0 rr=3.0 | 0.194 | 22 | -0.246 |
| 2022Q4 | 15m A w=40 tau=2.5 k=1.5 rr=3.0 | 0.093 | 19 | -0.575 |
| 2023Q1 | 60m A w=20 tau=1.5 k=1.5 rr=1.5 | -0.005 | 44 | -0.186 |
| 2023Q2 | 15m A w=20 tau=2.5 k=1.5 rr=2.5 | -0.022 | 49 | 0.256 |
| 2023Q3 | 15m A w=20 tau=2.5 k=1.5 rr=3.0 | -0.021 | 48 | -0.297 |
| 2023Q4 | 60m A w=10 tau=2.0 k=2.0 rr=2.0 | -0.007 | 29 | 0.079 |
| 2024Q1 | 60m A w=40 tau=2.0 k=1.5 rr=2.5 | -0.022 | 16 | -0.063 |
| 2024Q2 | 15m A w=40 tau=2.0 k=2.0 rr=3.0 | 0.003 | 53 | 0.057 |

OOS: 342 trades, mean -0.0154 R, 95% CI [-0.1521, 0.1204], p(mean<=0) = 0.587, total -5.3 R

## 5. Finalists (validation)

Deflated Sharpe uses 904 trials and the cross-config variance of daily Sharpe (1.32498).

| index | A | B | C |
|---|---|---|---|
| key | 60m A w=10 tau=2.0 k=2.0 rr=2.5 | 15m B w=20 z=2.0 k=2.0 rr=3.0 | 15m C2 [A w=40 tau=2.0 k=2.0 rr=3.0 | B w=20 z=2.0 k=2.0 rr=3.0] |
| tf | 60 | 15 | 15 |
| val_n | 129 | 1439 | 182 |
| val_score | -0.1130 | -0.1886 | -0.0423 |
| val_mean | 0.0960 | -0.1343 | 0.1706 |
| ci_lo | -0.1058 | -0.1878 | -0.0414 |
| ci_hi | 0.3068 | -0.0793 | 0.3845 |
| p_boot | 0.1814 | 1.0000 | 0.0565 |
| train_mean | 0.1234 | -0.1017 | 0.0486 |
| plateau | 0.8333 | 0.0000 |  |
| null_mean | -0.0966 | -0.2020 | -0.1999 |
| null_p | 0.0000 | 0.0233 | 0.0000 |
| dsr | 0.0000 | 0.0000 | 0.0000 |
| sr_ann | 0.7419 | -3.9175 | 1.2850 |
| sr0_ann | 58.9537 | 58.9537 | 58.9537 |
| dsr_null | 0.0069 | 0.0000 | 0.0356 |
| sr0_null_ann | 2.6771 | 2.6771 | 2.6771 |
| D1 validation mean > 0 | True | False | True |
| D2 train mean > 0 | True | False | True |
| D3 plateau >= 60% | True | False | None |
| D4 walk-forward OOS mean > 0 | False | False | False |
| D5 beats random entries (p<0.05) | True | True | True |
| passes_dev | False | False | False |

### Finalist A breakdown: 60m A w=10 tau=2.0 k=2.0 rr=2.5

| split | subset | n | mean_r | total_r |
|---|---|---|---|---|
| train | long | 154 | 0.154 | 23.668 |
| train | short | 151 | 0.093 | 13.984 |
| train | RTH entry | 77 | 0.225 | 17.335 |
| train | ETH entry | 228 | 0.089 | 20.317 |
| validation | long | 73 | 0.185 | 13.508 |
| validation | short | 56 | -0.020 | -1.128 |
| validation | RTH entry | 40 | 0.002 | 0.095 |
| validation | ETH entry | 89 | 0.138 | 12.285 |

### Finalist A: 60m A w=10 tau=2.0 k=2.0 rr=2.5

Exits (Module 5):

| exit | train_mean | val_n | val_mean | val_score |
|---|---|---|---|---|
| fixed RR (selected) | 0.1234 | 129 | 0.0960 | -0.1130 |
| time exit only (no target) | 0.1080 | 119 | 0.0694 | -0.1827 |
| stop from seasonal vol forecast | 0.1244 | 129 | 0.0869 | -0.1123 |

Cost sensitivity (Module 9):

| costs | val_mean | val_score | val_total_r |
|---|---|---|---|
| as modelled | 0.0960 | -0.1130 | 12.3794 |
| zero slippage | 0.1150 | -0.0970 | 14.8331 |
| slippage x2 | 0.0457 | -0.1506 | 5.8945 |
| commission/fees x2 | 0.0213 | -0.1724 | 2.7432 |
| target needs 0 ticks through | 0.0960 | -0.1130 | 12.3794 |
| target needs 2x through | 0.0960 | -0.1130 | 12.3794 |
| frictionless | 0.2056 | -0.0254 | 26.5282 |

Component ablation (a component earns its place only if it improves train AND validation):

| component | train_mean | val_n | val_mean | val_score | helps_both |
|---|---|---|---|---|---|
| none (base) | 0.1234 | 129 | 0.0960 | -0.1130 | False |
| RTH only | 0.1186 | 48 | 0.0260 | -0.2518 | False |
| block CUSUM transitions | 0.0746 | 103 | 0.1772 | -0.0679 | False |
| block extreme volatility | 0.1444 | 113 | 0.1046 | -0.1234 | True |
| probability filter (logistic, p >= break-even+2pp) | 0.1234 | 129 | 0.0960 | -0.1130 | False |

### Finalist B breakdown: 15m B w=20 z=2.0 k=2.0 rr=3.0

| split | subset | n | mean_r | total_r |
|---|---|---|---|---|
| train | long | 1842 | -0.111 | -204.440 |
| train | short | 1656 | -0.091 | -151.190 |
| train | RTH entry | 1014 | -0.123 | -124.272 |
| train | ETH entry | 2484 | -0.093 | -231.358 |
| validation | long | 688 | -0.150 | -103.066 |
| validation | short | 751 | -0.120 | -90.208 |
| validation | RTH entry | 428 | -0.177 | -75.574 |
| validation | ETH entry | 1011 | -0.116 | -117.700 |

### Finalist B: 15m B w=20 z=2.0 k=2.0 rr=3.0

Exits (Module 5):

| exit | train_mean | val_n | val_mean | val_score |
|---|---|---|---|---|
| fixed RR (selected) | -0.1017 | 1439 | -0.1343 | -0.1886 |
| time exit only (no target) | -0.1171 | 1388 | -0.1270 | -0.1885 |
| target = distance to rolling mean | -0.1183 | 1668 | -0.2040 | -0.2407 |
| stop from seasonal vol forecast | -0.0987 | 1376 | -0.1009 | -0.1538 |

Cost sensitivity (Module 9):

| costs | val_mean | val_score | val_total_r |
|---|---|---|---|
| as modelled | -0.1343 | -0.1886 | -193.2744 |
| zero slippage | -0.0980 | -0.1527 | -140.2040 |
| slippage x2 | -0.1699 | -0.2230 | -245.7984 |
| commission/fees x2 | -0.2616 | -0.3087 | -376.4646 |
| target needs 0 ticks through | -0.1329 | -0.1872 | -191.3327 |
| target needs 2x through | -0.1343 | -0.1886 | -193.2744 |
| frictionless | 0.1232 | 0.0549 | 176.3508 |

Component ablation (a component earns its place only if it improves train AND validation):

| component | train_mean | val_n | val_mean | val_score | helps_both |
|---|---|---|---|---|---|
| none (base) | -0.1017 | 1439 | -0.1343 | -0.1886 | False |
| RTH only | -0.1212 | 488 | -0.1696 | -0.2660 | False |
| block CUSUM transitions | -0.1001 | 1235 | -0.1527 | -0.2116 | False |
| block extreme volatility | -0.1089 | 1374 | -0.1338 | -0.1894 | False |
| probability filter (logistic, p >= break-even+2pp) | -0.0418 | 391 | -0.2128 | -0.3313 | False |

### Finalist C breakdown: 15m C2 [A w=40 tau=2.0 k=2.0 rr=3.0 | B w=20 z=2.0 k=2.0 rr=3.0]

| split | subset | n | mean_r | total_r |
|---|---|---|---|---|
| train | long | 219 | 0.001 | 0.319 |
| train | short | 224 | 0.095 | 21.213 |
| train | RTH entry | 140 | 0.014 | 1.899 |
| train | ETH entry | 303 | 0.065 | 19.633 |
| validation | long | 110 | 0.193 | 21.270 |
| validation | short | 72 | 0.136 | 9.787 |
| validation | RTH entry | 71 | 0.052 | 3.706 |
| validation | ETH entry | 111 | 0.246 | 27.351 |

## 6. Selection

**NO strategy passed the development gates -> system outputs NO TRADE**

```json
{
  "A": {
    "key": "60m A w=10 tau=2.0 k=2.0 rr=2.5",
    "tf": 60,
    "passes_dev": false,
    "params": {
      "w": 10,
      "tau": 2.0,
      "k": 2.0,
      "rr": 2.5
    }
  },
  "B": {
    "key": "15m B w=20 z=2.0 k=2.0 rr=3.0",
    "tf": 15,
    "passes_dev": false,
    "params": {
      "w": 20,
      "z": 2.0,
      "k": 2.0,
      "rr": 3.0
    }
  },
  "C": {
    "key": "15m C2 [A w=40 tau=2.0 k=2.0 rr=3.0 | B w=20 z=2.0 k=2.0 rr=3.0]",
    "tf": 15,
    "passes_dev": false,
    "variant": "C2",
    "A": {
      "w": 40,
      "tau": 2.0,
      "k": 2.0,
      "rr": 3.0,
      "hold_mult": 2.0
    },
    "B": {
      "w": 20,
      "z": 2.0,
      "k": 2.0,
      "rr": 3.0,
      "hold_mult": 1.0
    },
    "route": {
      "RTH": "A",
      "RTH_train_exp": {
        "A": 0.013565796183361059,
        "B": -0.12255583117409642
      },
      "ETH": "A",
      "ETH_train_exp": {
        "A": 0.06479511303489341,
        "B": -0.09313931697452509
      }
    }
  }
}
```