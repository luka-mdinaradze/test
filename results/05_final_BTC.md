# Experiment 5 — Frozen candidates on unseen data (BTC)

Frozen configuration committed in `cd073446be`; development verdict: **NO strategy passed the development gates -> system outputs NO TRADE**.

R per trade after costs, exact fractional sizing, risk governor off, unless stated.

## 1. Identical-period comparison

| candidate | period | trades | exp_r | ci_lo | ci_hi | p_le0 | win_rate | pf | total_r | sharpe | max_dd_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | train | 305 | 0.123 | 0.004 | 0.249 | 0.021 | 0.449 | 1.328 | 37.652 | 1.036 | 4.227 |
| A | validation | 129 | 0.096 | -0.106 | 0.307 | 0.181 | 0.403 | 1.223 | 12.379 | 0.742 | 3.437 |
| A | test | 123 | 0.077 | -0.107 | 0.268 | 0.217 | 0.439 | 1.200 | 9.520 | 0.672 | 4.711 |
| A | holdout | 176 | -0.063 | -0.200 | 0.079 | 0.808 | 0.392 | 0.850 | -11.070 | -0.648 | 10.290 |
| B | train | 3498 | -0.102 | -0.137 | -0.065 | 1.000 | 0.414 | 0.799 | -355.630 | -2.918 | 177.594 |
| B | validation | 1439 | -0.134 | -0.188 | -0.079 | 1.000 | 0.399 | 0.739 | -193.274 | -3.918 | 96.742 |
| B | test | 1395 | -0.145 | -0.201 | -0.087 | 1.000 | 0.374 | 0.728 | -202.416 | -4.098 | 103.103 |
| B | holdout | 1819 | -0.191 | -0.239 | -0.143 | 1.000 | 0.366 | 0.651 | -348.005 | -5.558 | 178.206 |
| C | train | 443 | 0.049 | -0.080 | 0.176 | 0.225 | 0.357 | 1.085 | 21.532 | 0.386 | 9.309 |
| C | validation | 182 | 0.171 | -0.041 | 0.384 | 0.057 | 0.385 | 1.306 | 31.057 | 1.285 | 4.867 |
| C | test | 200 | -0.032 | -0.228 | 0.164 | 0.633 | 0.335 | 0.946 | -6.431 | -0.267 | 9.074 |
| C | holdout | 255 | -0.158 | -0.309 | -0.003 | 0.977 | 0.314 | 0.750 | -40.300 | -1.432 | 21.742 |

## 2. Pre-registered gates

| index | A | B | C |
|---|---|---|---|
| T1 test mean>0, PF>1, p<0.10 | False | False | False |
| H1 holdout mean>0 | False | False | False |
| passed development | False | False | False |
| would be VALIDATED | False | False | False |

**Final verdict: NOT VALIDATED — the system outputs NO TRADE.**

## 3. A on test — full metrics

| index | value |
|---|---|
| trades | 123.000 |
| exp_r | 0.077 |
| win_rate | 0.439 |
| avg_win_r | 1.059 |
| avg_loss_r | -0.691 |
| pf | 1.200 |
| total_r | 9.520 |
| gross_r | 0.156 |
| cost_r | 0.079 |
| net_usd | 237.999 |
| return_pct | 4.760 |
| max_dd_pct | 4.711 |
| sharpe | 0.672 |
| sortino | 1.162 |
| calmar | 0.693 |
| avg_minutes | 490.317 |
| max_consec_wins | 8.000 |
| max_consec_losses | 7.000 |
| n_overlap | 40.000 |
| n_session | 6.000 |
| days | 365.000 |

Long vs short:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| short | 61 | -0.119 | 0.328 | -7.258 | -181.452 |
| long | 62 | 0.271 | 0.548 | 16.778 | 419.452 |

By entry session:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| ASIA | 42 | 0.297 | 0.548 | 12.470 | 311.745 |
| EUROPE | 19 | -0.141 | 0.316 | -2.682 | -67.056 |
| US | 42 | 0.001 | 0.405 | 0.025 | 0.625 |
| LATE | 20 | -0.015 | 0.400 | -0.293 | -7.314 |

By volatility regime at entry (thresholds from train):

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| low | 24 | 0.109 | 0.375 | 2.618 | 65.453 |
| normal | 28 | 0.280 | 0.500 | 7.826 | 195.653 |
| high | 54 | 0.062 | 0.463 | 3.374 | 84.362 |
| extreme | 17 | -0.253 | 0.353 | -4.299 | -107.468 |

## 3. A on holdout — full metrics

| index | value |
|---|---|
| trades | 176.000 |
| exp_r | -0.063 |
| win_rate | 0.392 |
| avg_win_r | 0.912 |
| avg_loss_r | -0.692 |
| pf | 0.850 |
| total_r | -11.070 |
| gross_r | 0.030 |
| cost_r | 0.093 |
| net_usd | -276.742 |
| return_pct | -5.535 |
| max_dd_pct | 10.290 |
| sharpe | -0.648 |
| sortino | -0.998 |
| calmar | -0.290 |
| avg_minutes | 473.670 |
| max_consec_wins | 6.000 |
| max_consec_losses | 9.000 |
| n_overlap | 35.000 |
| n_session | 7.000 |
| days | 473.000 |

Long vs short:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| short | 84 | 0.077 | 0.452 | 6.505 | 162.629 |
| long | 92 | -0.191 | 0.337 | -17.575 | -439.371 |

By entry session:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| ASIA | 55 | -0.177 | 0.309 | -9.710 | -242.752 |
| EUROPE | 37 | 0.118 | 0.486 | 4.354 | 108.838 |
| US | 59 | -0.047 | 0.390 | -2.780 | -69.512 |
| LATE | 25 | -0.117 | 0.440 | -2.933 | -73.315 |

By volatility regime at entry (thresholds from train):

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| low | 31 | -0.233 | 0.323 | -7.209 | -180.225 |
| normal | 45 | -0.060 | 0.422 | -2.686 | -67.162 |
| high | 94 | -0.026 | 0.404 | -2.412 | -60.310 |
| extreme | 6 | 0.206 | 0.333 | 1.238 | 30.955 |

## 3. B on test — full metrics

| index | value |
|---|---|
| trades | 1395.000 |
| exp_r | -0.145 |
| win_rate | 0.374 |
| avg_win_r | 1.039 |
| avg_loss_r | -0.853 |
| pf | 0.728 |
| total_r | -202.416 |
| gross_r | 0.041 |
| cost_r | 0.186 |
| net_usd | -5060.388 |
| return_pct | -101.208 |
| max_dd_pct | 103.103 |
| sharpe | -4.098 |
| sortino | -4.955 |
| calmar |  |
| avg_minutes | 164.335 |
| max_consec_wins | 7.000 |
| max_consec_losses | 12.000 |
| n_overlap | 615.000 |
| n_session | 14.000 |
| days | 365.000 |

Long vs short:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| short | 699 | -0.151 | 0.363 | -105.300 | -2632.502 |
| long | 696 | -0.140 | 0.385 | -97.115 | -2427.885 |

By entry session:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| ASIA | 438 | -0.119 | 0.404 | -52.231 | -1305.771 |
| EUROPE | 330 | -0.187 | 0.330 | -61.629 | -1540.713 |
| US | 421 | -0.185 | 0.347 | -78.074 | -1951.860 |
| LATE | 206 | -0.051 | 0.437 | -10.482 | -262.043 |

By volatility regime at entry (thresholds from train):

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| low | 463 | -0.220 | 0.341 | -101.993 | -2549.828 |
| normal | 468 | -0.113 | 0.400 | -52.699 | -1317.465 |
| high | 414 | -0.095 | 0.379 | -39.346 | -983.651 |
| extreme | 50 | -0.168 | 0.400 | -8.378 | -209.442 |

## 3. B on holdout — full metrics

| index | value |
|---|---|
| trades | 1819.000 |
| exp_r | -0.191 |
| win_rate | 0.366 |
| avg_win_r | 0.975 |
| avg_loss_r | -0.864 |
| pf | 0.651 |
| total_r | -348.005 |
| gross_r | 0.032 |
| cost_r | 0.223 |
| net_usd | -8700.129 |
| return_pct | -174.003 |
| max_dd_pct | 178.206 |
| sharpe | -5.558 |
| sortino | -6.197 |
| calmar |  |
| avg_minutes | 162.235 |
| max_consec_wins | 6.000 |
| max_consec_losses | 13.000 |
| n_overlap | 761.000 |
| n_session | 32.000 |
| days | 473.000 |

Long vs short:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| short | 894 | -0.184 | 0.364 | -164.739 | -4118.463 |
| long | 925 | -0.198 | 0.368 | -183.267 | -4581.666 |

By entry session:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| ASIA | 600 | -0.198 | 0.375 | -118.553 | -2963.836 |
| EUROPE | 435 | -0.198 | 0.320 | -86.061 | -2151.526 |
| US | 537 | -0.190 | 0.374 | -101.863 | -2546.565 |
| LATE | 247 | -0.168 | 0.405 | -41.528 | -1038.202 |

By volatility regime at entry (thresholds from train):

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| low | 608 | -0.280 | 0.332 | -170.153 | -4253.828 |
| normal | 667 | -0.158 | 0.376 | -105.204 | -2630.098 |
| high | 502 | -0.151 | 0.386 | -75.590 | -1889.750 |
| extreme | 42 | 0.070 | 0.429 | 2.942 | 73.547 |

## 3. C on test — full metrics

| index | value |
|---|---|
| trades | 200.000 |
| exp_r | -0.032 |
| win_rate | 0.335 |
| avg_win_r | 1.675 |
| avg_loss_r | -0.892 |
| pf | 0.946 |
| total_r | -6.431 |
| gross_r | 0.113 |
| cost_r | 0.146 |
| net_usd | -160.765 |
| return_pct | -3.215 |
| max_dd_pct | 9.074 |
| sharpe | -0.267 |
| sortino | -0.440 |
| calmar | -0.246 |
| avg_minutes | 275.465 |
| max_consec_wins | 5.000 |
| max_consec_losses | 8.000 |
| n_overlap | 144.000 |
| n_session | 0.000 |
| days | 365.000 |

Long vs short:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| short | 99 | -0.138 | 0.313 | -13.625 | -340.623 |
| long | 101 | 0.071 | 0.356 | 7.194 | 179.858 |

By entry session:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| ASIA | 64 | 0.175 | 0.375 | 11.174 | 279.347 |
| EUROPE | 31 | -0.347 | 0.226 | -10.744 | -268.602 |
| US | 70 | 0.028 | 0.386 | 1.979 | 49.485 |
| LATE | 35 | -0.253 | 0.257 | -8.840 | -220.994 |

By volatility regime at entry (thresholds from train):

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| low | 32 | -0.261 | 0.250 | -8.336 | -208.412 |
| normal | 58 | -0.160 | 0.293 | -9.271 | -231.779 |
| high | 90 | 0.151 | 0.400 | 13.611 | 340.279 |
| extreme | 20 | -0.122 | 0.300 | -2.434 | -60.852 |

## 3. C on holdout — full metrics

| index | value |
|---|---|
| trades | 255.000 |
| exp_r | -0.158 |
| win_rate | 0.314 |
| avg_win_r | 1.509 |
| avg_loss_r | -0.920 |
| pf | 0.750 |
| total_r | -40.300 |
| gross_r | 0.016 |
| cost_r | 0.174 |
| net_usd | -1007.494 |
| return_pct | -20.150 |
| max_dd_pct | 21.742 |
| sharpe | -1.432 |
| sortino | -2.100 |
| calmar | -0.520 |
| avg_minutes | 262.192 |
| max_consec_wins | 5.000 |
| max_consec_losses | 17.000 |
| n_overlap | 124.000 |
| n_session | 10.000 |
| days | 473.000 |

Long vs short:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| short | 122 | -0.072 | 0.352 | -8.727 | -218.177 |
| long | 133 | -0.237 | 0.278 | -31.573 | -789.318 |

By entry session:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| ASIA | 68 | -0.007 | 0.353 | -0.443 | -11.086 |
| EUROPE | 63 | -0.341 | 0.222 | -21.475 | -536.866 |
| US | 99 | -0.130 | 0.343 | -12.850 | -321.250 |
| LATE | 25 | -0.221 | 0.320 | -5.532 | -138.292 |

By volatility regime at entry (thresholds from train):

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| low | 30 | -0.501 | 0.167 | -15.041 | -376.021 |
| normal | 88 | -0.255 | 0.273 | -22.472 | -561.810 |
| high | 124 | -0.005 | 0.379 | -0.585 | -14.624 |
| extreme | 13 | -0.169 | 0.308 | -2.202 | -55.041 |

## 4. The $5,000 account (whole contracts / 0.1-lot CFD, risk governor ON, compounding)

`rejected_size`: signals whose smallest tradeable size would exceed the risk budget — never taken.

| candidate | period | instrument | risk_pct | signals | filled | rejected_size | blocked_governor | net_usd | return_pct | max_dd_pct | exp_usd | avg_contracts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | test | BTCUSDT | 0.50 | 169 | 123 | 0 | 0 | 229.86 | 4.60 | 4.78 | 1.87 | 0.02 |
| A | test | BTCUSDT | 1.00 | 169 | 123 | 0 | 0 | 463.84 | 9.28 | 9.63 | 3.77 | 0.04 |
| A | holdout | BTCUSDT | 0.50 | 218 | 176 | 0 | 0 | -272.42 | -5.45 | 9.73 | -1.55 | 0.02 |
| A | holdout | BTCUSDT | 1.00 | 218 | 176 | 0 | 0 | -556.00 | -11.12 | 19.09 | -3.16 | 0.04 |
| B | test | BTCUSDT | 0.50 | 2024 | 1358 | 0 | 45 | -3175.10 | -63.50 | 65.23 | -2.34 | 0.03 |
| B | test | BTCUSDT | 1.00 | 2024 | 1260 | 0 | 173 | -4322.49 | -86.45 | 87.79 | -3.43 | 0.04 |
| B | holdout | BTCUSDT | 0.50 | 2612 | 1769 | 0 | 62 | -4091.88 | -81.84 | 82.53 | -2.31 | 0.02 |
| B | holdout | BTCUSDT | 1.00 | 2612 | 1632 | 0 | 243 | -4815.63 | -96.31 | 96.56 | -2.95 | 0.02 |
| C | test | BTCUSDT | 0.50 | 344 | 200 | 0 | 0 | -178.30 | -3.57 | 9.18 | -0.89 | 0.04 |
| C | test | BTCUSDT | 1.00 | 344 | 200 | 0 | 0 | -386.47 | -7.73 | 18.16 | -1.93 | 0.08 |
| C | holdout | BTCUSDT | 0.50 | 389 | 255 | 0 | 0 | -928.85 | -18.58 | 19.87 | -3.64 | 0.04 |
| C | holdout | BTCUSDT | 1.00 | 389 | 254 | 0 | 1 | -1722.27 | -34.45 | 36.26 | -6.78 | 0.07 |

## 5. Monte Carlo (5,000 one-year paths; whole days drawn from test+holdout; 0.5% risk per R)

| index | A | B | C |
|---|---|---|---|
| source_trades | 299.000 | 3214.000 | 455.000 |
| dd_median_pct | 8.975 | 58.239 | 20.858 |
| dd_p95_pct | 17.054 | 67.779 | 33.223 |
| dd_p99_pct | 21.032 | 71.194 | 38.072 |
| p_dd_gt_10pct | 0.399 | 1.000 | 0.956 |
| p_dd_gt_20pct | 0.017 | 1.000 | 0.549 |
| ret_median_pct | -1.040 | -56.976 | -15.975 |
| ret_p05_pct | -13.866 | -67.198 | -30.895 |
| ret_p95_pct | 13.802 | -43.174 | 2.740 |
| p_loss | 0.554 | 1.000 | 0.923 |
| losing_streak_median | 9.000 | 13.000 | 13.000 |
| losing_streak_p95 | 14.000 | 18.000 | 20.000 |
| paths | 5000.000 | 5000.000 | 5000.000 |
| horizon_days | 252.000 | 252.000 | 252.000 |

![equity](figures/equity_BTC.png)
