# Experiment 4 — Strategy development (NQ, instrument MNQ)

All figures are R per trade AFTER costs (exact fractional sizing, risk governor off).
`score` = mean - 1.96 day-clustered SE (lower edge of a 95% interval).

## 1. Grid overview (share of configurations positive after costs)

| family | tf | configs | train_pos | val_pos | both_pos | eligible | best_val_score | median_val_mean | median_cost_r | median_trades_day |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 108 | 0.000 | 0.000 | 0.000 | 0 | -0.119 | -0.116 | 0.134 | 11.446 |
| A | 5 | 108 | 0.157 | 0.361 | 0.102 | 7 | 0.046 | -0.032 | 0.058 | 2.351 |
| A | 15 | 108 | 0.528 | 0.593 | 0.343 | 26 | 0.153 | 0.025 | 0.033 | 0.765 |
| A | 60 | 108 | 0.648 | 0.926 | 0.583 | 46 | 0.287 | 0.259 | 0.015 | 0.253 |
| B | 1 | 108 | 0.000 | 0.000 | 0.000 | 0 | -0.148 | -0.173 | 0.152 | 24.716 |
| B | 5 | 108 | 0.000 | 0.065 | 0.000 | 0 | -0.073 | -0.072 | 0.064 | 5.343 |
| B | 15 | 108 | 0.130 | 0.361 | 0.065 | 6 | -0.024 | -0.017 | 0.032 | 2.054 |
| B | 60 | 108 | 0.065 | 0.139 | 0.009 | 0 | -0.120 | -0.108 | 0.014 | 0.423 |

### Top 8 per family by validation score

| key | train_n | train_mean | val_n | val_mean | val_score | val_pf | plateau | eligible |
|---|---|---|---|---|---|---|---|---|
| 60m A w=10 tau=2.5 k=1.0 rr=2.5 | 77 | -0.138 | 30 | 0.895 | 0.287 | 3.066 | 1.000 | False |
| 60m A w=10 tau=2.5 k=1.0 rr=3.0 | 77 | -0.242 | 30 | 0.935 | 0.260 | 3.004 | 1.000 | False |
| 15m A w=40 tau=2.0 k=2.0 rr=3.0 | 232 | 0.020 | 85 | 0.539 | 0.153 | 2.125 | 0.800 | True |
| 15m A w=40 tau=2.0 k=1.5 rr=3.0 | 250 | 0.070 | 91 | 0.514 | 0.150 | 1.968 | 1.000 | True |
| 15m A w=40 tau=2.0 k=2.0 rr=2.0 | 250 | 0.038 | 89 | 0.418 | 0.142 | 1.960 | 0.833 | True |
| 60m A w=10 tau=2.0 k=1.5 rr=2.5 | 170 | 0.038 | 66 | 0.509 | 0.141 | 2.245 | 1.000 | True |
| 60m A w=10 tau=2.0 k=1.0 rr=3.0 | 175 | -0.024 | 69 | 0.579 | 0.141 | 2.173 | 1.000 | False |
| 15m A w=40 tau=2.0 k=2.0 rr=2.5 | 239 | 0.022 | 88 | 0.468 | 0.139 | 1.988 | 0.833 | True |

| key | train_n | train_mean | val_n | val_mean | val_score | val_pf | plateau | eligible |
|---|---|---|---|---|---|---|---|---|
| 15m B w=20 z=3.0 k=2.0 rr=3.0 | 360 | -0.049 | 180 | 0.200 | -0.024 | 1.351 | 1.000 | False |
| 15m B w=20 z=3.0 k=2.0 rr=2.5 | 363 | 0.010 | 182 | 0.153 | -0.052 | 1.270 | 1.000 | True |
| 15m B w=20 z=2.5 k=2.0 rr=3.0 | 892 | -0.021 | 426 | 0.077 | -0.058 | 1.136 | 0.600 | False |
| 15m B w=20 z=2.5 k=1.5 rr=3.0 | 964 | -0.034 | 450 | 0.074 | -0.065 | 1.115 | 0.667 | False |
| 15m B w=20 z=3.0 k=1.5 rr=3.0 | 365 | -0.067 | 185 | 0.168 | -0.068 | 1.260 | 1.000 | False |
| 15m B w=20 z=2.5 k=2.0 rr=2.5 | 910 | -0.011 | 434 | 0.059 | -0.068 | 1.105 | 0.667 | False |
| 5m B w=20 z=2.0 k=2.0 rr=1.5 | 4461 | -0.087 | 2237 | -0.027 | -0.073 | 0.946 | 0.000 | False |
| 15m B w=20 z=3.0 k=2.0 rr=2.0 | 363 | -0.005 | 183 | 0.122 | -0.073 | 1.220 | 1.000 | False |

## 2. Strategy C (regime-adaptive)

| key | regime | train_n | train_mean | val_n | val_mean | val_score | eligible |
|---|---|---|---|---|---|---|---|
| 1m C1 [A w=20 tau=2.0 k=2.0 rr=3.0 | B w=80 z=2.5 k=2.0 rr=1.5] | variance-ratio rule, CUSUM+extreme-vol block | 2636 | -0.151 | 1192 | -0.107 | -0.176 | False |
| 1m C2 [A w=20 tau=2.0 k=2.0 rr=3.0 | B w=80 z=2.5 k=2.0 rr=1.5] | session router RTH->NONE ETH->NONE | 0 |  | 0 |  |  | False |
| 5m C1 [A w=40 tau=2.5 k=2.0 rr=1.5 | B w=20 z=2.0 k=2.0 rr=1.5] | variance-ratio rule, CUSUM+extreme-vol block | 1278 | -0.130 | 693 | 0.007 | -0.074 | False |
| 5m C2 [A w=40 tau=2.5 k=2.0 rr=1.5 | B w=20 z=2.0 k=2.0 rr=1.5] | session router RTH->NONE ETH->NONE | 0 |  | 0 |  |  | False |
| 15m C1 [A w=40 tau=2.0 k=2.0 rr=3.0 | B w=20 z=3.0 k=2.0 rr=3.0] | variance-ratio rule, CUSUM+extreme-vol block | 178 | -0.082 | 69 | 0.205 | -0.159 | False |
| 15m C2 [A w=40 tau=2.0 k=2.0 rr=3.0 | B w=20 z=3.0 k=2.0 rr=3.0] | session router RTH->A ETH->NONE | 70 | 0.154 | 31 | 0.209 | -0.263 | True |
| 60m C1 [A w=10 tau=2.5 k=1.0 rr=2.5 | B w=20 z=2.5 k=1.5 rr=1.5] | variance-ratio rule, CUSUM+extreme-vol block | 126 | -0.116 | 49 | 0.094 | -0.226 | False |
| 60m C2 [A w=10 tau=2.5 k=1.0 rr=2.5 | B w=20 z=2.5 k=1.5 rr=1.5] | session router RTH->NONE ETH->NONE | 0 |  | 0 |  |  | False |

## 3. Entry methods in isolation (identical exits: 1.5 ATR stop, 2R target, 20-bar cap)

Order-flow imbalance is not in this table: no source provides it for the development period.

| tf | method | train_n | train_mean | train_t | val_n | val_mean | val_t |
|---|---|---|---|---|---|---|---|
| 1 | deviation_fade (z-score) | 23668 | -0.232 | -32.427 | 11785 | -0.159 | -15.613 |
| 1 | deviation_fade (causal Nadaraya-Watson) | 23173 | -0.227 | -30.282 | 11628 | -0.158 | -14.340 |
| 1 | deviation_fade (session VWAP) | 8141 | -0.227 | -17.550 | 3720 | -0.150 | -7.808 |
| 1 | regression_trend_change (OLS slope z) | 3025 | -0.189 | -9.092 | 1523 | -0.093 | -3.072 |
| 1 | regression_trend_change (Kalman slope z) | 16018 | -0.232 | -27.382 | 7923 | -0.144 | -11.196 |
| 1 | momentum_shift (z_mom) | 4570 | -0.209 | -12.538 | 2255 | -0.115 | -4.244 |
| 1 | volatility_expansion (range shock, follow) | 8727 | -0.245 | -18.953 | 4715 | -0.131 | -7.270 |
| 1 | breakout (N-bar high/low + z_mom>1) | 15105 | -0.229 | -25.062 | 7387 | -0.135 | -9.144 |
| 5 | deviation_fade (z-score) | 4792 | -0.112 | -6.613 | 2400 | -0.056 | -2.205 |
| 5 | deviation_fade (causal Nadaraya-Watson) | 4814 | -0.126 | -7.345 | 2345 | -0.090 | -3.643 |
| 5 | deviation_fade (session VWAP) | 2436 | -0.085 | -3.417 | 1152 | -0.092 | -2.472 |
| 5 | regression_trend_change (OLS slope z) | 605 | 0.001 | 0.030 | 323 | -0.093 | -1.262 |
| 5 | regression_trend_change (Kalman slope z) | 3199 | -0.080 | -3.811 | 1567 | -0.072 | -2.414 |
| 5 | momentum_shift (z_mom) | 953 | -0.045 | -1.164 | 464 | 0.002 | 0.037 |
| 5 | volatility_expansion (range shock, follow) | 1718 | -0.107 | -3.538 | 900 | -0.042 | -1.025 |
| 5 | breakout (N-bar high/low + z_mom>1) | 3105 | -0.062 | -2.872 | 1495 | 0.008 | 0.233 |
| 15 | deviation_fade (z-score) | 1698 | -0.047 | -1.499 | 840 | -0.033 | -0.707 |
| 15 | deviation_fade (causal Nadaraya-Watson) | 1740 | -0.080 | -2.752 | 822 | -0.034 | -0.721 |
| 15 | deviation_fade (session VWAP) | 651 | -0.097 | -2.109 | 316 | -0.039 | -0.554 |
| 15 | regression_trend_change (OLS slope z) | 223 | -0.021 | -0.277 | 104 | 0.020 | 0.166 |
| 15 | regression_trend_change (Kalman slope z) | 1085 | 0.010 | 0.289 | 524 | -0.066 | -1.288 |
| 15 | momentum_shift (z_mom) | 351 | 0.101 | 1.586 | 151 | 0.006 | 0.063 |
| 15 | volatility_expansion (range shock, follow) | 566 | -0.056 | -1.197 | 299 | 0.028 | 0.419 |
| 15 | breakout (N-bar high/low + z_mom>1) | 1123 | -0.039 | -1.182 | 518 | 0.015 | 0.254 |
| 60 | deviation_fade (z-score) | 536 | -0.042 | -0.822 | 267 | -0.074 | -1.046 |
| 60 | deviation_fade (causal Nadaraya-Watson) | 479 | -0.076 | -1.495 | 234 | -0.062 | -0.796 |
| 60 | deviation_fade (session VWAP) | 274 | -0.028 | -0.447 | 137 | -0.053 | -0.592 |
| 60 | regression_trend_change (OLS slope z) | 54 | 0.101 | 0.635 | 31 | -0.199 | -1.199 |
| 60 | regression_trend_change (Kalman slope z) | 276 | 0.024 | 0.338 | 138 | -0.004 | -0.035 |
| 60 | momentum_shift (z_mom) | 104 | -0.013 | -0.112 | 49 | 0.231 | 1.314 |
| 60 | volatility_expansion (range shock, follow) | 155 | -0.030 | -0.310 | 83 | 0.050 | 0.337 |
| 60 | breakout (N-bar high/low + z_mom>1) | 359 | -0.009 | -0.140 | 143 | 0.268 | 2.436 |

## 4. Walk-forward — family A (12m in-sample -> next quarter)

| fold | choice | is_score | oos_n | oos_mean |
|---|---|---|---|---|
| 2024Q1 | 60m A w=20 tau=1.5 k=1.0 rr=2.0 | -0.040 | 39 | -0.011 |
| 2024Q2 | 60m A w=10 tau=1.5 k=2.0 rr=2.0 | -0.008 | 55 | -0.359 |
| 2024Q3 | 15m A w=20 tau=2.5 k=1.0 rr=2.5 | -0.007 | 27 | -0.044 |
| 2024Q4 | 60m A w=10 tau=2.0 k=1.5 rr=1.5 | -0.000 | 24 | 0.261 |
| 2025Q1 | 15m A w=40 tau=2.0 k=1.5 rr=3.0 | 0.094 | 21 | 0.435 |

OOS: 166 trades, mean -0.0357 R, 95% CI [-0.2333, 0.1716], p(mean<=0) = 0.637, total -5.9 R

## 4. Walk-forward — family B (12m in-sample -> next quarter)

| fold | choice | is_score | oos_n | oos_mean |
|---|---|---|---|---|
| 2024Q1 | 15m B w=20 z=2.5 k=2.0 rr=1.5 | -0.068 | 153 | 0.080 |
| 2024Q2 | 15m B w=20 z=2.5 k=2.0 rr=1.5 | -0.045 | 158 | -0.085 |
| 2024Q3 | 15m B w=20 z=3.0 k=2.0 rr=1.5 | -0.056 | 55 | 0.510 |
| 2024Q4 | 15m B w=20 z=3.0 k=2.0 rr=2.5 | 0.050 | 66 | -0.031 |
| 2025Q1 | 15m B w=20 z=3.0 k=2.0 rr=2.5 | 0.019 | 61 | -0.102 |

OOS: 493 trades, mean 0.0377 R, 95% CI [-0.0724, 0.1462], p(mean<=0) = 0.251, total 18.6 R

## 4. Walk-forward — family A or B (12m in-sample -> next quarter)

| fold | choice | is_score | oos_n | oos_mean |
|---|---|---|---|---|
| 2024Q1 | 60m A w=20 tau=1.5 k=1.0 rr=2.0 | -0.040 | 39 | -0.011 |
| 2024Q2 | 60m A w=10 tau=1.5 k=2.0 rr=2.0 | -0.008 | 55 | -0.359 |
| 2024Q3 | 15m A w=20 tau=2.5 k=1.0 rr=2.5 | -0.007 | 27 | -0.044 |
| 2024Q4 | 15m B w=20 z=3.0 k=2.0 rr=2.5 | 0.050 | 66 | -0.031 |
| 2025Q1 | 15m A w=40 tau=2.0 k=1.5 rr=3.0 | 0.094 | 21 | 0.435 |

OOS: 208 trades, mean -0.0685 R, 95% CI [-0.2537, 0.1227], p(mean<=0) = 0.764, total -14.2 R

## 5. Finalists (validation)

Deflated Sharpe uses 904 trials and the cross-config variance of daily Sharpe (0.08525).

| index | A | B | C |
|---|---|---|---|
| key | 15m A w=40 tau=2.0 k=2.0 rr=3.0 | 15m B w=20 z=3.0 k=2.0 rr=2.5 | 15m C2 [A w=40 tau=2.0 k=2.0 rr=3.0 | B w=20 z=3.0 k=2.0 rr=3.0] |
| tf | 15 | 15 | 15 |
| val_n | 85 | 182 | 31 |
| val_score | 0.1531 | -0.0519 | -0.2631 |
| val_mean | 0.5386 | 0.1534 | 0.2087 |
| ci_lo | 0.1635 | -0.0492 | -0.2294 |
| ci_hi | 0.9376 | 0.3615 | 0.6985 |
| p_boot | 0.0018 | 0.0704 | 0.1908 |
| train_mean | 0.0203 | 0.0098 | 0.1545 |
| plateau | 0.8000 | 1.0000 |  |
| null_mean | 0.0258 | -0.0118 | 0.0251 |
| null_p | 0.0000 | 0.0567 | 0.2900 |
| dsr | 0.0000 | 0.0000 | 0.0000 |
| sr_ann | 3.1187 | 1.6686 | 1.0071 |
| sr0_ann | 14.9542 | 14.9542 | 14.9542 |
| dsr_null | 0.2852 | 0.0348 | 0.0058 |
| sr0_null_ann | 3.6771 | 3.6771 | 3.6771 |
| D1 validation mean > 0 | True | True | True |
| D2 train mean > 0 | True | True | True |
| D3 plateau >= 60% | True | True | None |
| D4 walk-forward OOS mean > 0 | False | True | False |
| D5 beats random entries (p<0.05) | True | False | False |
| passes_dev | False | False | False |

### Finalist A breakdown: 15m A w=40 tau=2.0 k=2.0 rr=3.0

| split | subset | n | mean_r | total_r |
|---|---|---|---|---|
| train | long | 127 | 0.213 | 27.063 |
| train | short | 105 | -0.213 | -22.364 |
| train | RTH entry | 66 | 0.171 | 11.271 |
| train | ETH entry | 166 | -0.040 | -6.572 |
| validation | long | 42 | 0.580 | 24.355 |
| validation | short | 43 | 0.498 | 21.425 |
| validation | RTH entry | 30 | 0.249 | 7.471 |
| validation | ETH entry | 55 | 0.697 | 38.309 |

### Finalist A: 15m A w=40 tau=2.0 k=2.0 rr=3.0

Exits (Module 5):

| exit | train_mean | val_n | val_mean | val_score |
|---|---|---|---|---|
| fixed RR (selected) | 0.0203 | 85 | 0.5386 | 0.1531 |
| time exit only (no target) | 0.0660 | 82 | 0.6643 | 0.1062 |
| stop from seasonal vol forecast | 0.0518 | 85 | 0.2632 | -0.0733 |

Cost sensitivity (Module 9):

| costs | val_mean | val_score | val_total_r |
|---|---|---|---|
| as modelled | 0.5386 | 0.1531 | 45.7803 |
| zero slippage | 0.5493 | 0.1609 | 46.6911 |
| slippage x2 | 0.5281 | 0.1454 | 44.8878 |
| commission/fees x2 | 0.5210 | 0.1402 | 44.2818 |
| target needs 0 ticks through | 0.5386 | 0.1531 | 45.7803 |
| target needs 2x through | 0.5386 | 0.1531 | 45.7803 |
| frictionless | 0.5678 | 0.1743 | 48.2626 |

Component ablation (a component earns its place only if it improves train AND validation):

| component | train_mean | val_n | val_mean | val_score | helps_both |
|---|---|---|---|---|---|
| none (base) | 0.0203 | 85 | 0.5386 | 0.1531 | False |
| RTH only | 0.1545 | 31 | 0.2087 | -0.2631 | False |
| block CUSUM transitions | 0.0384 | 65 | 0.4950 | 0.0700 | False |
| block extreme volatility | 0.0297 | 70 | 0.3842 | -0.0264 | False |
| probability filter (logistic, p >= break-even+2pp) | 0.3915 | 19 | 0.3175 | -0.4786 | False |

### Finalist B breakdown: 15m B w=20 z=3.0 k=2.0 rr=2.5

| split | subset | n | mean_r | total_r |
|---|---|---|---|---|
| train | long | 192 | 0.095 | 18.156 |
| train | short | 171 | -0.085 | -14.615 |
| train | RTH entry | 146 | -0.033 | -4.760 |
| train | ETH entry | 217 | 0.038 | 8.301 |
| validation | long | 110 | 0.119 | 13.066 |
| validation | short | 72 | 0.206 | 14.848 |
| validation | RTH entry | 79 | -0.087 | -6.847 |
| validation | ETH entry | 103 | 0.337 | 34.760 |

### Finalist B: 15m B w=20 z=3.0 k=2.0 rr=2.5

Exits (Module 5):

| exit | train_mean | val_n | val_mean | val_score |
|---|---|---|---|---|
| fixed RR (selected) | 0.0098 | 182 | 0.1534 | -0.0519 |
| time exit only (no target) | -0.0216 | 176 | 0.2139 | -0.0824 |
| target = distance to rolling mean | 0.0354 | 184 | 0.0214 | -0.1506 |
| stop from seasonal vol forecast | -0.0394 | 174 | 0.0970 | -0.0734 |

Cost sensitivity (Module 9):

| costs | val_mean | val_score | val_total_r |
|---|---|---|---|
| as modelled | 0.1534 | -0.0519 | 27.9133 |
| zero slippage | 0.1639 | -0.0431 | 29.8215 |
| slippage x2 | 0.1346 | -0.0671 | 24.6395 |
| commission/fees x2 | 0.1384 | -0.0643 | 25.1805 |
| target needs 0 ticks through | 0.1534 | -0.0519 | 27.9133 |
| target needs 2x through | 0.1534 | -0.0519 | 27.9133 |
| frictionless | 0.1797 | -0.0300 | 32.7072 |

Component ablation (a component earns its place only if it improves train AND validation):

| component | train_mean | val_n | val_mean | val_score | helps_both |
|---|---|---|---|---|---|
| none (base) | 0.0098 | 182 | 0.1534 | -0.0519 | False |
| RTH only | -0.0203 | 79 | -0.1101 | -0.4083 | False |
| block CUSUM transitions | -0.0594 | 136 | 0.1761 | -0.0680 | False |
| block extreme volatility | 0.0275 | 150 | 0.1820 | -0.0534 | True |
| probability filter (logistic, p >= break-even+2pp) | 0.2704 | 93 | 0.1688 | -0.1280 | True |

### Finalist C breakdown: 15m C2 [A w=40 tau=2.0 k=2.0 rr=3.0 | B w=20 z=3.0 k=2.0 rr=3.0]

| split | subset | n | mean_r | total_r |
|---|---|---|---|---|
| train | long | 40 | 0.202 | 8.094 |
| train | short | 30 | 0.091 | 2.720 |
| train | RTH entry | 69 | 0.161 | 11.099 |
| train | ETH entry | 1 | -0.285 | -0.285 |
| validation | long | 10 | 0.084 | 0.838 |
| validation | short | 21 | 0.268 | 5.632 |
| validation | RTH entry | 31 | 0.209 | 6.471 |
| validation | ETH entry | 0 |  | 0.000 |

## 6. Selection

**NO strategy passed the development gates -> system outputs NO TRADE**

```json
{
  "A": {
    "key": "15m A w=40 tau=2.0 k=2.0 rr=3.0",
    "tf": 15,
    "passes_dev": false,
    "params": {
      "w": 40,
      "tau": 2.0,
      "k": 2.0,
      "rr": 3.0
    }
  },
  "B": {
    "key": "15m B w=20 z=3.0 k=2.0 rr=2.5",
    "tf": 15,
    "passes_dev": false,
    "params": {
      "w": 20,
      "z": 3.0,
      "k": 2.0,
      "rr": 2.5
    }
  },
  "C": {
    "key": "15m C2 [A w=40 tau=2.0 k=2.0 rr=3.0 | B w=20 z=3.0 k=2.0 rr=3.0]",
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
      "z": 3.0,
      "k": 2.0,
      "rr": 3.0,
      "hold_mult": 1.0
    },
    "route": {
      "RTH": "A",
      "RTH_train_exp": {
        "A": 0.17076604128226874,
        "B": -0.11479052691564194
      },
      "ETH": "NONE",
      "ETH_train_exp": {
        "A": -0.03958835309515855,
        "B": -0.0055942706324358035
      }
    }
  }
}
```