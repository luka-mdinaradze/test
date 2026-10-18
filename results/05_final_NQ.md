# Experiment 5 — Frozen candidates on unseen data (NQ)

Frozen configuration committed in `69018318d0`; development verdict: **NO strategy passed the development gates -> system outputs NO TRADE**.

R per trade after costs, exact fractional sizing, risk governor off, unless stated.

## 1. Identical-period comparison

| candidate | period | trades | exp_r | ci_lo | ci_hi | p_le0 | win_rate | pf | total_r | sharpe | max_dd_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | train | 232 | 0.020 | -0.173 | 0.216 | 0.418 | 0.332 | 1.032 | 4.699 | 0.165 | 6.604 |
| A | validation | 85 | 0.539 | 0.164 | 0.938 | 0.002 | 0.471 | 2.125 | 45.780 | 3.119 | 2.086 |
| A | test | 63 | -0.100 | -0.443 | 0.288 | 0.716 | 0.286 | 0.856 | -6.311 | -0.616 | 5.972 |
| A | holdout | 24 | -0.076 | -0.662 | 0.543 | 0.620 | 0.250 | 0.880 | -1.822 | -0.510 | 3.553 |
| B | train | 363 | 0.010 | -0.131 | 0.156 | 0.450 | 0.358 | 1.016 | 3.541 | 0.107 | 12.151 |
| B | validation | 182 | 0.153 | -0.049 | 0.362 | 0.070 | 0.407 | 1.270 | 27.913 | 1.669 | 8.005 |
| B | test | 149 | 0.272 | 0.057 | 0.489 | 0.006 | 0.470 | 1.549 | 40.527 | 2.847 | 4.625 |
| B | holdout | 48 | -0.342 | -0.599 | -0.027 | 0.983 | 0.292 | 0.466 | -16.438 | -4.389 | 8.219 |
| C | train | 70 | 0.154 | -0.122 | 0.442 | 0.139 | 0.500 | 1.358 | 10.814 | 0.872 | 5.087 |
| C | validation | 31 | 0.209 | -0.229 | 0.699 | 0.191 | 0.484 | 1.514 | 6.471 | 1.007 | 2.992 |
| C | test | 14 | -0.004 | -0.523 | 0.612 | 0.527 | 0.429 | 0.991 | -0.061 | -0.018 | 1.612 |
| C | holdout | 6 | 0.235 | -0.849 | 1.558 | 0.399 | 0.333 | 1.455 | 1.407 | 0.780 | 1.036 |

## 2. Pre-registered gates

| index | A | B | C |
|---|---|---|---|
| T1 test mean>0, PF>1, p<0.10 | False | True | False |
| H1 holdout mean>0 | False | False | True |
| passed development | False | False | False |
| would be VALIDATED | False | False | False |

**Final verdict: NOT VALIDATED — the system outputs NO TRADE.**

## 3. A on test — full metrics

| index | value |
|---|---|
| trades | 63.000 |
| exp_r | -0.100 |
| win_rate | 0.286 |
| avg_win_r | 2.077 |
| avg_loss_r | -0.971 |
| pf | 0.856 |
| total_r | -6.311 |
| gross_r | -0.078 |
| cost_r | 0.022 |
| net_usd | -157.783 |
| return_pct | -3.156 |
| max_dd_pct | 5.972 |
| sharpe | -0.616 |
| sortino | -1.074 |
| calmar | -0.727 |
| avg_minutes | 269.540 |
| max_consec_wins | 3.000 |
| max_consec_losses | 8.000 |
| n_overlap | 41.000 |
| n_session | 0.000 |
| days | 182.000 |

Long vs short:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| short | 34 | -0.052 | 0.294 | -1.773 | -44.333 |
| long | 29 | -0.156 | 0.276 | -4.538 | -113.450 |

By entry session:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| ASIA | 29 | -0.138 | 0.241 | -3.996 | -99.911 |
| LONDON | 20 | -0.113 | 0.250 | -2.254 | -56.340 |
| RTH_OPEN | 6 | -0.088 | 0.500 | -0.528 | -13.188 |
| RTH_MID | 5 | 0.325 | 0.400 | 1.624 | 40.608 |
| RTH_CLOSE | 3 | -0.386 | 0.333 | -1.158 | -28.952 |

By volatility regime at entry (thresholds from train):

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| low | 17 | -0.101 | 0.353 | -1.724 | -43.102 |
| normal | 9 | 0.013 | 0.222 | 0.119 | 2.985 |
| high | 13 | 0.471 | 0.462 | 6.117 | 152.916 |
| extreme | 24 | -0.451 | 0.167 | -10.823 | -270.582 |

## 3. A on holdout — full metrics

| index | value |
|---|---|
| trades | 24.000 |
| exp_r | -0.076 |
| win_rate | 0.250 |
| avg_win_r | 2.228 |
| avg_loss_r | -0.844 |
| pf | 0.880 |
| total_r | -1.822 |
| gross_r | -0.058 |
| cost_r | 0.017 |
| net_usd | -45.562 |
| return_pct | -0.911 |
| max_dd_pct | 3.553 |
| sharpe | -0.510 |
| sortino | -1.021 |
| calmar | -1.116 |
| avg_minutes | 335.000 |
| max_consec_wins | 2.000 |
| max_consec_losses | 5.000 |
| n_overlap | 17.000 |
| n_session | 2.000 |
| days | 57.000 |

Long vs short:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| short | 14 | -0.013 | 0.286 | -0.184 | -4.600 |
| long | 10 | -0.164 | 0.200 | -1.639 | -40.963 |

By entry session:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| ASIA | 11 | 0.165 | 0.273 | 1.815 | 45.385 |
| LONDON | 5 | -1.000 | 0.000 | -4.998 | -124.948 |
| RTH_OPEN | 3 | 0.324 | 0.333 | 0.971 | 24.266 |
| RTH_MID | 3 | 0.146 | 0.333 | 0.437 | 10.916 |
| POST | 2 | -0.024 | 0.500 | -0.047 | -1.181 |

By volatility regime at entry (thresholds from train):

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| low | 10 | 0.330 | 0.400 | 3.300 | 82.507 |
| normal | 3 | 0.323 | 0.333 | 0.968 | 24.190 |
| high | 7 | -0.434 | 0.143 | -3.041 | -76.028 |
| extreme | 4 | -0.762 | 0.000 | -3.049 | -76.231 |

## 3. B on test — full metrics

| index | value |
|---|---|
| trades | 149.000 |
| exp_r | 0.272 |
| win_rate | 0.470 |
| avg_win_r | 1.634 |
| avg_loss_r | -0.935 |
| pf | 1.549 |
| total_r | 40.527 |
| gross_r | 0.295 |
| cost_r | 0.023 |
| net_usd | 1013.170 |
| return_pct | 20.263 |
| max_dd_pct | 4.625 |
| sharpe | 2.847 |
| sortino | 5.715 |
| calmar | 6.294 |
| avg_minutes | 143.960 |
| max_consec_wins | 5.000 |
| max_consec_losses | 8.000 |
| n_overlap | 1.000 |
| n_session | 1.000 |
| days | 182.000 |

Long vs short:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| short | 60 | 0.189 | 0.417 | 11.354 | 283.843 |
| long | 89 | 0.328 | 0.506 | 29.173 | 729.326 |

By entry session:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| ASIA | 25 | 0.202 | 0.480 | 5.062 | 126.539 |
| LONDON | 51 | 0.631 | 0.569 | 32.192 | 804.797 |
| RTH_OPEN | 45 | -0.192 | 0.267 | -8.644 | -216.088 |
| RTH_MID | 20 | 0.289 | 0.550 | 5.778 | 144.462 |
| RTH_CLOSE | 6 | 0.838 | 0.833 | 5.029 | 125.722 |
| POST | 2 | 0.555 | 0.500 | 1.109 | 27.737 |

By volatility regime at entry (thresholds from train):

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| low | 54 | 0.294 | 0.481 | 15.897 | 397.427 |
| normal | 33 | -0.079 | 0.364 | -2.618 | -65.457 |
| high | 34 | 0.279 | 0.471 | 9.493 | 237.318 |
| extreme | 28 | 0.634 | 0.571 | 17.755 | 443.882 |

## 3. B on holdout — full metrics

| index | value |
|---|---|
| trades | 48.000 |
| exp_r | -0.342 |
| win_rate | 0.292 |
| avg_win_r | 1.025 |
| avg_loss_r | -0.906 |
| pf | 0.466 |
| total_r | -16.438 |
| gross_r | -0.325 |
| cost_r | 0.017 |
| net_usd | -410.957 |
| return_pct | -8.219 |
| max_dd_pct | 8.219 |
| sharpe | -4.389 |
| sortino | -5.139 |
| calmar | -3.840 |
| avg_minutes | 120.146 |
| max_consec_wins | 3.000 |
| max_consec_losses | 7.000 |
| n_overlap | 0.000 |
| n_session | 1.000 |
| days | 57.000 |

Long vs short:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| short | 26 | -0.158 | 0.346 | -4.098 | -102.461 |
| long | 22 | -0.561 | 0.227 | -12.340 | -308.496 |

By entry session:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| ASIA | 6 | -0.213 | 0.333 | -1.278 | -31.944 |
| LONDON | 18 | -0.019 | 0.389 | -0.348 | -8.700 |
| RTH_OPEN | 14 | -0.970 | 0.000 | -13.575 | -339.368 |
| RTH_MID | 6 | -0.241 | 0.500 | -1.443 | -36.081 |
| RTH_CLOSE | 3 | 0.086 | 0.667 | 0.258 | 6.442 |
| POST | 1 | -0.052 | 0.000 | -0.052 | -1.306 |

By volatility regime at entry (thresholds from train):

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| low | 21 | -0.342 | 0.238 | -7.182 | -179.547 |
| normal | 8 | -0.248 | 0.500 | -1.981 | -49.527 |
| high | 13 | -0.504 | 0.231 | -6.550 | -163.758 |
| extreme | 6 | -0.121 | 0.333 | -0.725 | -18.125 |

## 3. C on test — full metrics

| index | value |
|---|---|
| trades | 14.000 |
| exp_r | -0.004 |
| win_rate | 0.429 |
| avg_win_r | 1.115 |
| avg_loss_r | -0.844 |
| pf | 0.991 |
| total_r | -0.061 |
| gross_r | 0.009 |
| cost_r | 0.014 |
| net_usd | -1.532 |
| return_pct | -0.031 |
| max_dd_pct | 1.612 |
| sharpe | -0.018 |
| sortino | -0.031 |
| calmar | -0.026 |
| avg_minutes | 169.857 |
| max_consec_wins | 3.000 |
| max_consec_losses | 4.000 |
| n_overlap | 7.000 |
| n_session | 0.000 |
| days | 182.000 |

Long vs short:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| short | 11 | -0.043 | 0.364 | -0.471 | -11.781 |
| long | 3 | 0.137 | 0.667 | 0.410 | 10.249 |

By entry session:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| RTH_OPEN | 6 | -0.088 | 0.500 | -0.528 | -13.188 |
| RTH_MID | 5 | 0.325 | 0.400 | 1.624 | 40.608 |
| RTH_CLOSE | 3 | -0.386 | 0.333 | -1.158 | -28.952 |

By volatility regime at entry (thresholds from train):

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| low | 5 | 0.094 | 0.600 | 0.472 | 11.812 |
| normal | 4 | -0.688 | 0.000 | -2.751 | -68.780 |
| high | 2 | -0.046 | 0.500 | -0.092 | -2.311 |
| extreme | 3 | 0.770 | 0.667 | 2.310 | 57.747 |

## 3. C on holdout — full metrics

| index | value |
|---|---|
| trades | 6.000 |
| exp_r | 0.235 |
| win_rate | 0.333 |
| avg_win_r | 2.250 |
| avg_loss_r | -0.773 |
| pf | 1.455 |
| total_r | 1.407 |
| gross_r | 0.246 |
| cost_r | 0.012 |
| net_usd | 35.182 |
| return_pct | 0.704 |
| max_dd_pct | 1.036 |
| sharpe | 0.780 |
| sortino | 1.705 |
| calmar | 3.039 |
| avg_minutes | 218.833 |
| max_consec_wins | 1.000 |
| max_consec_losses | 3.000 |
| n_overlap | 9.000 |
| n_session | 0.000 |
| days | 57.000 |

Long vs short:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| short | 4 | 0.220 | 0.250 | 0.878 | 21.951 |
| long | 2 | 0.265 | 0.500 | 0.529 | 13.231 |

By entry session:

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| RTH_OPEN | 3 | 0.324 | 0.333 | 0.971 | 24.266 |
| RTH_MID | 3 | 0.146 | 0.333 | 0.437 | 10.916 |

By volatility regime at entry (thresholds from train):

| index | trades | exp_r | win_rate | total_r | pnl_usd |
|---|---|---|---|---|---|
| low | 4 | 0.852 | 0.500 | 3.409 | 85.218 |
| normal | 1 | -1.001 | 0.000 | -1.001 | -25.036 |
| high | 1 | -1.000 | 0.000 | -1.000 | -25.000 |

## 4. The $5,000 account (whole contracts / 0.1-lot CFD, risk governor ON, compounding)

`rejected_size`: signals whose smallest tradeable size would exceed the risk budget — never taken.

| candidate | period | instrument | risk_pct | signals | filled | rejected_size | blocked_governor | net_usd | return_pct | max_dd_pct | exp_usd | avg_contracts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | test | MNQ | 0.50 | 104 | 0 | 104 | 0 | 0.00 | 0.00 | 0.00 |  | 0.00 |
| A | test | MNQ | 1.00 | 104 | 2 | 97 | 0 | 62.03 | 1.24 | 0.88 | 31.01 | 1.00 |
| A | test | US100 | 0.50 | 104 | 62 | 3 | 0 | -201.98 | -4.04 | 5.47 | -3.26 | 0.35 |
| A | holdout | MNQ | 0.50 | 43 | 0 | 41 | 0 | 0.00 | 0.00 | 0.00 |  | 0.00 |
| A | holdout | MNQ | 1.00 | 43 | 0 | 41 | 0 | 0.00 | 0.00 | 0.00 |  | 0.00 |
| A | holdout | US100 | 0.50 | 43 | 24 | 0 | 0 | -65.51 | -1.31 | 2.95 | -2.73 | 0.25 |
| B | test | MNQ | 0.50 | 151 | 0 | 150 | 0 | 0.00 | 0.00 | 0.00 |  | 0.00 |
| B | test | MNQ | 1.00 | 151 | 4 | 146 | 0 | -23.77 | -0.48 | 1.92 | -5.94 | 1.00 |
| B | test | US100 | 0.50 | 151 | 146 | 3 | 0 | 899.62 | 17.99 | 4.36 | 6.16 | 0.42 |
| B | holdout | MNQ | 0.50 | 49 | 0 | 48 | 0 | 0.00 | 0.00 | 0.00 |  | 0.00 |
| B | holdout | MNQ | 1.00 | 49 | 0 | 48 | 0 | 0.00 | 0.00 | 0.00 |  | 0.00 |
| B | holdout | US100 | 0.50 | 49 | 47 | 1 | 0 | -343.02 | -6.86 | 6.86 | -7.30 | 0.26 |
| C | test | MNQ | 0.50 | 21 | 0 | 21 | 0 | 0.00 | 0.00 | 0.00 |  | 0.00 |
| C | test | MNQ | 1.00 | 21 | 0 | 21 | 0 | 0.00 | 0.00 | 0.00 |  | 0.00 |
| C | test | US100 | 0.50 | 21 | 13 | 2 | 0 | -41.22 | -0.82 | 1.31 | -3.17 | 0.24 |
| C | holdout | MNQ | 0.50 | 15 | 0 | 15 | 0 | 0.00 | 0.00 | 0.00 |  | 0.00 |
| C | holdout | MNQ | 1.00 | 15 | 0 | 15 | 0 | 0.00 | 0.00 | 0.00 |  | 0.00 |
| C | holdout | US100 | 0.50 | 15 | 6 | 0 | 0 | 36.39 | 0.73 | 0.79 | 6.07 | 0.20 |

## 5. Monte Carlo (5,000 one-year paths; whole days drawn from test+holdout; 0.5% risk per R)

| index | A | B | C |
|---|---|---|---|
| source_trades | 87.000 | 197.000 | 20.000 |
| dd_median_pct | 19.517 | 7.726 | 7.735 |
| dd_p95_pct | 31.991 | 13.960 | 14.584 |
| dd_p99_pct | 37.166 | 17.690 | 18.360 |
| p_dd_gt_10pct | 0.942 | 0.235 | 0.252 |
| p_dd_gt_20pct | 0.479 | 0.003 | 0.004 |
| ret_median_pct | -12.918 | 21.153 | 8.806 |
| ret_p05_pct | -29.597 | -0.414 | -7.823 |
| ret_p95_pct | 7.874 | 48.022 | 28.504 |
| p_loss | 0.860 | 0.052 | 0.209 |
| losing_streak_median | 14.000 | 9.000 | 9.000 |
| losing_streak_p95 | 22.000 | 14.000 | 14.000 |
| paths | 5000.000 | 5000.000 | 5000.000 |
| horizon_days | 252.000 | 252.000 | 252.000 |

![equity](figures/equity_NQ.png)
