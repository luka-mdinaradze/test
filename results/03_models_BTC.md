# Experiment 3 — Model investigations (BTC; fit on train, scored on validation)

## M1 Next-bar volatility forecasts (QLIKE, lower is better; validation)

Each model is rescaled by one constant fitted on train. `qlike_gain_vs_atr_pct` > 0 = better than ATR.

| tf | model | qlike_train | qlike_test | mz_r2_test | scale_k | qlike_gain_vs_atr_pct |
|---|---|---|---|---|---|---|
| 1 | atr | 15.0466 | 18.4040 | 0.3439 | 814.3776 | 0.0000 |
| 1 | ewma | 1.6186 | 1.5178 | 0.3297 | 1.0752 | 91.7528 |
| 1 | seasonal | 1.5982 | 1.5163 | 0.3326 | 1.0732 | 91.7611 |
| 1 | garch | 1.5872 | 1.5030 | 0.3488 | 1.1038 | 91.8331 |
| 5 | atr | 1.6957 | 1.5706 | 0.3093 | 0.8357 | 0.0000 |
| 5 | ewma | 1.6836 | 1.6041 | 0.2829 | 1.0746 | -2.1308 |
| 5 | seasonal | 1.6512 | 1.5821 | 0.2930 | 1.0747 | -0.7326 |
| 5 | garch | 1.6787 | 1.5752 | 0.3078 | 1.1382 | -0.2934 |
| 15 | atr | 1.7228 | 1.6571 | 0.2829 | 0.7581 | 0.0000 |
| 15 | ewma | 1.7600 | 1.7233 | 0.2525 | 1.0722 | -3.9993 |
| 15 | seasonal | 1.7143 | 1.6718 | 0.2701 | 1.0748 | -0.8882 |
| 15 | garch | 1.7654 | 1.6517 | 0.2854 | 1.0933 | 0.3236 |
| 60 | atr | 1.8661 | 2.0439 | 0.2281 | 0.7280 | 0.0000 |
| 60 | ewma | 1.8946 | 2.1334 | 0.1986 | 1.0638 | -4.3800 |
| 60 | seasonal | 1.8473 | 2.0535 | 0.2191 | 1.0719 | -0.4690 |
| 60 | garch | 1.8854 | 1.9186 | 0.2383 | 1.0564 | 6.1293 |

GARCH(1,1) on deseasonalised changes (train fit): 1m alpha=0.100 beta=0.880 nu=4.2, 5m alpha=0.200 beta=0.780 nu=4.2, 15m alpha=0.223 beta=0.767 nu=4.0, 60m alpha=0.136 beta=0.855 nu=3.4

## M2 Stop width from ATR vs from the seasonal volatility forecast (validation, per-signal R after costs)

Same median stop width on train; only the time profile of the width differs.

| tf | family | stop_unit | signals | exp_r | ci_lo | ci_hi | r_std | stopped_within_1_bar |
|---|---|---|---|---|---|---|---|---|
| 1 | A | atr | 5801 | -0.6128 | -0.6375 | -0.5878 | 0.6845 | 0.3679 |
| 1 | A | seasonal | 5801 | -0.6087 | -0.6315 | -0.5859 | 0.6688 | 0.3692 |
| 1 | B | atr | 27554 | -0.6483 | -0.6626 | -0.6338 | 0.5877 | 0.3143 |
| 1 | B | seasonal | 27554 | -0.6228 | -0.6362 | -0.6096 | 0.5964 | 0.2545 |
| 5 | A | atr | 1131 | -0.3894 | -0.4515 | -0.3287 | 0.9474 | 0.1804 |
| 5 | A | seasonal | 1131 | -0.3770 | -0.4397 | -0.3150 | 0.9481 | 0.1724 |
| 5 | B | atr | 5750 | -0.3967 | -0.4210 | -0.3728 | 0.8384 | 0.1440 |
| 5 | B | seasonal | 5750 | -0.3623 | -0.3878 | -0.3379 | 0.8456 | 0.1191 |
| 15 | A | atr | 419 | -0.0484 | -0.1663 | 0.0763 | 1.1431 | 0.1384 |
| 15 | A | seasonal | 419 | -0.0606 | -0.1790 | 0.0627 | 1.1342 | 0.1313 |
| 15 | B | atr | 2019 | -0.2271 | -0.2746 | -0.1788 | 0.9791 | 0.1639 |
| 15 | B | seasonal | 2019 | -0.2087 | -0.2546 | -0.1619 | 0.9737 | 0.1332 |
| 60 | A | atr | 112 | -0.0294 | -0.2615 | 0.1986 | 1.0654 | 0.1071 |
| 60 | A | seasonal | 112 | -0.0431 | -0.2753 | 0.1894 | 1.0642 | 0.1339 |
| 60 | B | atr | 537 | -0.1890 | -0.2760 | -0.1035 | 0.9663 | 0.2086 |
| 60 | B | seasonal | 537 | -0.1615 | -0.2402 | -0.0822 | 0.9385 | 0.1750 |

## M3 Regime classifiers vs what happened NEXT

`fwd_vr_minus_1`: realised variance ratio of the next h bars minus 1 (h = 60 min or 5 bars).
> 0 means the next bars trended, < 0 means they reverted. A useful trend/range classifier
shows TREND > 0 > RANGE in BOTH periods. `fwd_vol_vs_fcst`: next-bar realised |change| /
forecast (a vol classifier is useful if this differs across labels).

| ('tf', '') | ('classifier', '') | ('label', '') | ('fwd_vol_vs_fcst', 'train') | ('fwd_vol_vs_fcst', 'validation') | ('fwd_vr_minus_1', 'train') | ('fwd_vr_minus_1', 'validation') | ('t', 'train') | ('t', 'validation') |
|---|---|---|---|---|---|---|---|---|
| 1 | CUSUM transition | stable | 0.779 | 0.771 | -0.294 | -0.243 | -57.906 | -25.768 |
| 1 | CUSUM transition | transition | 0.748 | 0.748 | -0.298 | -0.257 | -44.821 | -23.528 |
| 1 | HMM-3 (filtered) | calm | 0.769 | 0.748 | -0.282 | -0.238 | -23.681 | -16.545 |
| 1 | HMM-3 (filtered) | mid | 0.751 | 0.746 | -0.300 | -0.254 | -59.002 | -26.714 |
| 1 | HMM-3 (filtered) | turbulent | 0.800 | 0.807 | -0.292 | -0.235 | -57.199 | -23.516 |
| 1 | efficiency-ratio rule | NEUTRAL | 0.770 | 0.755 | -0.299 | -0.253 | -58.112 | -27.535 |
| 1 | efficiency-ratio rule | RANGE | 0.765 | 0.742 | -0.302 | -0.243 | -53.874 | -22.201 |
| 1 | efficiency-ratio rule | TREND | 0.786 | 0.799 | -0.282 | -0.238 | -44.955 | -22.398 |
| 1 | session | ETH | 0.770 | 0.762 | -0.305 | -0.250 | -54.376 | -24.548 |
| 1 | session | RTH | 0.781 | 0.778 | -0.269 | -0.234 | -26.820 | -14.314 |
| 1 | variance-ratio rule | NEUTRAL | 0.778 | 0.768 | -0.298 | -0.255 | -37.897 | -22.673 |
| 1 | variance-ratio rule | RANGE | 0.784 | 0.767 | -0.312 | -0.252 | -37.168 | -12.360 |
| 1 | variance-ratio rule | TREND | 0.757 | 0.764 | -0.273 | -0.224 | -28.671 | -12.152 |
| 1 | vol tercile | extreme | 0.600 | 0.592 | -0.272 | -0.254 | -14.595 | -9.790 |
| 1 | vol tercile | high | 0.694 | 0.723 | -0.307 | -0.251 | -37.905 | -18.605 |
| 1 | vol tercile | low | 0.861 | 0.819 | -0.289 | -0.259 | -30.930 | -16.454 |
| 1 | vol tercile | normal | 0.779 | 0.787 | -0.294 | -0.220 | -38.398 | -13.986 |
| 5 | CUSUM transition | stable | 0.763 | 0.773 | -0.217 | -0.191 | -45.834 | -21.665 |
| 5 | CUSUM transition | transition | 0.739 | 0.755 | -0.207 | -0.193 | -22.210 | -12.293 |
| 5 | HMM-3 (filtered) | calm | 0.650 | 0.661 | -0.234 | -0.197 | -27.631 | -15.383 |
| 5 | HMM-3 (filtered) | mid | 0.761 | 0.775 | -0.213 | -0.193 | -45.241 | -22.883 |
| 5 | HMM-3 (filtered) | turbulent | 0.877 | 0.934 | -0.202 | -0.172 | -27.759 | -13.501 |
| 5 | efficiency-ratio rule | NEUTRAL | 0.749 | 0.753 | -0.212 | -0.184 | -38.687 | -17.842 |
| 5 | efficiency-ratio rule | RANGE | 0.737 | 0.735 | -0.223 | -0.189 | -31.342 | -15.582 |
| 5 | efficiency-ratio rule | TREND | 0.793 | 0.830 | -0.211 | -0.203 | -30.031 | -16.830 |
| 5 | session | ETH | 0.752 | 0.753 | -0.224 | -0.198 | -43.939 | -21.504 |
| 5 | session | RTH | 0.775 | 0.813 | -0.191 | -0.174 | -20.737 | -11.826 |
| 5 | variance-ratio rule | NEUTRAL | 0.760 | 0.772 | -0.199 | -0.184 | -27.661 | -13.826 |
| 5 | variance-ratio rule | RANGE | 0.768 | 0.786 | -0.209 | -0.180 | -25.963 | -14.461 |
| 5 | variance-ratio rule | TREND | 0.747 | 0.746 | -0.242 | -0.214 | -30.794 | -14.640 |
| 5 | vol tercile | extreme | 0.606 | 0.601 | -0.222 | -0.172 | -12.035 | -6.154 |
| 5 | vol tercile | high | 0.690 | 0.722 | -0.229 | -0.195 | -26.787 | -16.005 |
| 5 | vol tercile | low | 0.839 | 0.827 | -0.203 | -0.209 | -23.886 | -14.573 |
| 5 | vol tercile | normal | 0.760 | 0.786 | -0.214 | -0.169 | -29.965 | -11.501 |
| 15 | CUSUM transition | stable | 0.744 | 0.766 | -0.131 | -0.129 | -25.469 | -15.560 |
| 15 | CUSUM transition | transition | 0.713 | 0.707 | -0.130 | -0.160 | -11.607 | -9.938 |
| 15 | HMM-3 (filtered) | calm | 0.638 | 0.621 | -0.154 | -0.157 | -19.881 | -13.447 |
| 15 | HMM-3 (filtered) | mid | 0.761 | 0.796 | -0.124 | -0.128 | -22.467 | -13.575 |
| 15 | HMM-3 (filtered) | turbulent | 0.908 | 1.004 | -0.101 | -0.099 | -9.398 | -5.406 |
| 15 | efficiency-ratio rule | NEUTRAL | 0.730 | 0.734 | -0.122 | -0.149 | -18.169 | -15.100 |
| 15 | efficiency-ratio rule | RANGE | 0.703 | 0.700 | -0.147 | -0.141 | -17.624 | -11.127 |
| 15 | efficiency-ratio rule | TREND | 0.788 | 0.842 | -0.126 | -0.109 | -15.485 | -8.146 |
| 15 | session | ETH | 0.728 | 0.722 | -0.142 | -0.144 | -25.745 | -16.206 |
| 15 | session | RTH | 0.766 | 0.832 | -0.103 | -0.113 | -10.842 | -7.851 |
| 15 | variance-ratio rule | NEUTRAL | 0.735 | 0.740 | -0.124 | -0.139 | -15.861 | -11.494 |
| 15 | variance-ratio rule | RANGE | 0.761 | 0.788 | -0.110 | -0.103 | -13.020 | -8.214 |
| 15 | variance-ratio rule | TREND | 0.722 | 0.731 | -0.160 | -0.172 | -19.691 | -12.514 |
| 15 | vol tercile | extreme | 0.639 | 0.589 | -0.114 | -0.151 | -5.038 | -5.281 |
| 15 | vol tercile | high | 0.677 | 0.717 | -0.160 | -0.128 | -19.581 | -9.921 |
| 15 | vol tercile | low | 0.803 | 0.799 | -0.104 | -0.150 | -11.860 | -11.772 |
| 15 | vol tercile | normal | 0.742 | 0.767 | -0.135 | -0.124 | -17.933 | -8.818 |
| 60 | CUSUM transition | stable | 0.742 | 0.739 | -0.152 | -0.151 | -13.320 | -8.115 |
| 60 | CUSUM transition | transition | 0.671 | 0.737 | -0.180 | -0.141 | -7.005 | -4.182 |
| 60 | HMM-3 (filtered) | calm | 0.611 | 0.567 | -0.203 | -0.179 | -11.368 | -7.114 |
| 60 | HMM-3 (filtered) | mid | 0.763 | 0.817 | -0.146 | -0.134 | -11.404 | -6.780 |
| 60 | HMM-3 (filtered) | turbulent | 0.840 | 0.956 | -0.101 | -0.115 | -4.916 | -3.203 |
| 60 | efficiency-ratio rule | NEUTRAL | 0.735 | 0.737 | -0.127 | -0.125 | -8.438 | -5.299 |
| 60 | efficiency-ratio rule | RANGE | 0.700 | 0.688 | -0.167 | -0.153 | -9.763 | -5.978 |
| 60 | efficiency-ratio rule | TREND | 0.755 | 0.805 | -0.183 | -0.179 | -10.244 | -5.903 |
| 60 | session | ETH | 0.731 | 0.737 | -0.161 | -0.143 | -13.802 | -7.534 |
| 60 | session | RTH | 0.730 | 0.745 | -0.142 | -0.166 | -7.089 | -5.379 |
| 60 | variance-ratio rule | NEUTRAL | 0.722 | 0.775 | -0.152 | -0.145 | -9.087 | -5.893 |
| 60 | variance-ratio rule | RANGE | 0.722 | 0.729 | -0.114 | -0.098 | -6.253 | -3.784 |
| 60 | variance-ratio rule | TREND | 0.751 | 0.695 | -0.204 | -0.223 | -10.836 | -6.781 |
| 60 | vol tercile | extreme | 0.684 | 0.587 | -0.141 | -0.236 | -3.041 | -4.762 |
| 60 | vol tercile | high | 0.676 | 0.722 | -0.175 | -0.144 | -8.993 | -4.874 |
| 60 | vol tercile | low | 0.781 | 0.793 | -0.106 | -0.152 | -5.754 | -5.778 |
| 60 | vol tercile | normal | 0.733 | 0.724 | -0.191 | -0.129 | -10.657 | -3.874 |

### HMM (3 states, fitted on train): transition matrices and expected durations (bars)

- 1m: durations [1.4, 2.8, 2.3], diag [0.308, 0.639, 0.563]
- 5m: durations [2.6, 3.8, 1.9], diag [0.616, 0.738, 0.481]
- 15m: durations [3.1, 3.3, 1.7], diag [0.674, 0.693, 0.414]
- 60m: durations [3.2, 3.3, 1.5], diag [0.686, 0.698, 0.354]

### Reference strategy expectancy (R per signal, after costs) by regime label

| ('tf', '') | ('family', '') | ('classifier', '') | ('label', '') | ('exp_r', 'train') | ('exp_r', 'validation') | ('n', 'train') | ('n', 'validation') |
|---|---|---|---|---|---|---|---|
| 1 | A | CUSUM transition | stable | -0.452 | -0.625 | 10056.000 | 4353.000 |
| 1 | A | CUSUM transition | transition | -0.397 | -0.577 | 3396.000 | 1448.000 |
| 1 | A | HMM-3 (filtered) | calm | -0.663 | -0.832 | 354.000 | 273.000 |
| 1 | A | HMM-3 (filtered) | mid | -0.493 | -0.705 | 1599.000 | 982.000 |
| 1 | A | HMM-3 (filtered) | turbulent | -0.424 | -0.580 | 11499.000 | 4546.000 |
| 1 | A | session | ETH | -0.459 | -0.660 | 9571.000 | 4101.000 |
| 1 | A | session | RTH | -0.386 | -0.498 | 3881.000 | 1700.000 |
| 1 | A | variance-ratio rule | NEUTRAL | -0.416 | -0.607 | 5444.000 | 2853.000 |
| 1 | A | variance-ratio rule | RANGE | -0.415 | -0.595 | 3825.000 | 1132.000 |
| 1 | A | variance-ratio rule | TREND | -0.488 | -0.633 | 4183.000 | 1816.000 |
| 1 | A | vol tercile | extreme | -0.263 | -0.340 | 1120.000 | 558.000 |
| 1 | A | vol tercile | high | -0.363 | -0.521 | 5226.000 | 2177.000 |
| 1 | A | vol tercile | low | -0.602 | -0.812 | 2785.000 | 1528.000 |
| 1 | A | vol tercile | normal | -0.469 | -0.644 | 4321.000 | 1538.000 |
| 1 | B | CUSUM transition | stable | -0.508 | -0.647 | 58282.000 | 21883.000 |
| 1 | B | CUSUM transition | transition | -0.535 | -0.652 | 14023.000 | 5671.000 |
| 1 | B | HMM-3 (filtered) | calm | -0.709 | -0.827 | 1862.000 | 1100.000 |
| 1 | B | HMM-3 (filtered) | mid | -0.575 | -0.669 | 9109.000 | 4819.000 |
| 1 | B | HMM-3 (filtered) | turbulent | -0.498 | -0.635 | 61334.000 | 21635.000 |
| 1 | B | session | ETH | -0.526 | -0.671 | 52289.000 | 20014.000 |
| 1 | B | session | RTH | -0.478 | -0.587 | 20016.000 | 7540.000 |
| 1 | B | variance-ratio rule | NEUTRAL | -0.491 | -0.641 | 28875.000 | 13888.000 |
| 1 | B | variance-ratio rule | RANGE | -0.518 | -0.667 | 22649.000 | 5764.000 |
| 1 | B | variance-ratio rule | TREND | -0.537 | -0.648 | 20781.000 | 7902.000 |
| 1 | B | vol tercile | extreme | -0.325 | -0.394 | 3204.000 | 1570.000 |
| 1 | B | vol tercile | high | -0.420 | -0.535 | 19551.000 | 8435.000 |
| 1 | B | vol tercile | low | -0.619 | -0.777 | 24266.000 | 9876.000 |
| 1 | B | vol tercile | normal | -0.507 | -0.659 | 25284.000 | 7673.000 |
| 5 | A | CUSUM transition | stable | -0.209 | -0.406 | 2117.000 | 825.000 |
| 5 | A | CUSUM transition | transition | -0.346 | -0.344 | 697.000 | 306.000 |
| 5 | A | HMM-3 (filtered) | calm | -0.229 | -0.431 | 39.000 | 11.000 |
| 5 | A | HMM-3 (filtered) | mid | -0.330 | -0.384 | 898.000 | 346.000 |
| 5 | A | HMM-3 (filtered) | turbulent | -0.201 | -0.391 | 1877.000 | 774.000 |
| 5 | A | session | ETH | -0.241 | -0.446 | 1893.000 | 702.000 |
| 5 | A | session | RTH | -0.245 | -0.297 | 921.000 | 429.000 |
| 5 | A | variance-ratio rule | NEUTRAL | -0.217 | -0.390 | 1123.000 | 444.000 |
| 5 | A | variance-ratio rule | RANGE | -0.204 | -0.341 | 901.000 | 404.000 |
| 5 | A | variance-ratio rule | TREND | -0.324 | -0.456 | 790.000 | 283.000 |
| 5 | A | vol tercile | extreme | -0.222 | -0.187 | 260.000 | 107.000 |
| 5 | A | vol tercile | high | -0.223 | -0.341 | 1193.000 | 462.000 |
| 5 | A | vol tercile | low | -0.298 | -0.524 | 453.000 | 243.000 |
| 5 | A | vol tercile | normal | -0.246 | -0.425 | 908.000 | 319.000 |
| 5 | B | CUSUM transition | stable | -0.265 | -0.391 | 12095.000 | 4607.000 |
| 5 | B | CUSUM transition | transition | -0.263 | -0.421 | 2649.000 | 1143.000 |
| 5 | B | HMM-3 (filtered) | calm | -0.475 | -0.491 | 150.000 | 133.000 |
| 5 | B | HMM-3 (filtered) | mid | -0.271 | -0.407 | 5654.000 | 2562.000 |
| 5 | B | HMM-3 (filtered) | turbulent | -0.257 | -0.384 | 8940.000 | 3055.000 |
| 5 | B | session | ETH | -0.268 | -0.406 | 10665.000 | 4095.000 |
| 5 | B | session | RTH | -0.256 | -0.374 | 4079.000 | 1655.000 |
| 5 | B | variance-ratio rule | NEUTRAL | -0.245 | -0.383 | 5837.000 | 2257.000 |
| 5 | B | variance-ratio rule | RANGE | -0.299 | -0.425 | 4895.000 | 2021.000 |
| 5 | B | variance-ratio rule | TREND | -0.251 | -0.379 | 4012.000 | 1472.000 |
| 5 | B | vol tercile | extreme | -0.104 | -0.256 | 635.000 | 281.000 |
| 5 | B | vol tercile | high | -0.198 | -0.308 | 4012.000 | 1710.000 |
| 5 | B | vol tercile | low | -0.378 | -0.512 | 5000.000 | 2047.000 |
| 5 | B | vol tercile | normal | -0.226 | -0.370 | 5097.000 | 1712.000 |
| 15 | A | CUSUM transition | stable | -0.037 | -0.049 | 819.000 | 314.000 |
| 15 | A | CUSUM transition | transition | -0.128 | -0.047 | 195.000 | 105.000 |
| 15 | A | HMM-3 (filtered) | calm | -0.185 | 0.165 | 55.000 | 23.000 |
| 15 | A | HMM-3 (filtered) | mid | -0.156 | -0.012 | 349.000 | 141.000 |
| 15 | A | HMM-3 (filtered) | turbulent | 0.016 | -0.088 | 610.000 | 255.000 |
| 15 | A | session | ETH | -0.065 | -0.152 | 702.000 | 245.000 |
| 15 | A | session | RTH | -0.029 | 0.097 | 312.000 | 174.000 |
| 15 | A | variance-ratio rule | NEUTRAL | -0.101 | -0.073 | 427.000 | 177.000 |
| 15 | A | variance-ratio rule | RANGE | -0.117 | 0.065 | 302.000 | 142.000 |
| 15 | A | variance-ratio rule | TREND | 0.081 | -0.166 | 285.000 | 100.000 |
| 15 | A | vol tercile | extreme | 0.095 | 0.128 | 108.000 | 47.000 |
| 15 | A | vol tercile | high | -0.080 | -0.048 | 421.000 | 194.000 |
| 15 | A | vol tercile | low | -0.080 | -0.064 | 188.000 | 80.000 |
| 15 | A | vol tercile | normal | -0.057 | -0.120 | 297.000 | 98.000 |
| 15 | B | CUSUM transition | stable | -0.123 | -0.209 | 4089.000 | 1641.000 |
| 15 | B | CUSUM transition | transition | -0.158 | -0.307 | 822.000 | 378.000 |
| 15 | B | HMM-3 (filtered) | calm | -0.173 | -0.263 | 107.000 | 123.000 |
| 15 | B | HMM-3 (filtered) | mid | -0.162 | -0.258 | 2098.000 | 903.000 |
| 15 | B | HMM-3 (filtered) | turbulent | -0.100 | -0.195 | 2706.000 | 993.000 |
| 15 | B | session | ETH | -0.118 | -0.213 | 3492.000 | 1392.000 |
| 15 | B | session | RTH | -0.155 | -0.258 | 1419.000 | 627.000 |
| 15 | B | variance-ratio rule | NEUTRAL | -0.096 | -0.239 | 1898.000 | 773.000 |
| 15 | B | variance-ratio rule | RANGE | -0.167 | -0.254 | 1776.000 | 787.000 |
| 15 | B | variance-ratio rule | TREND | -0.122 | -0.161 | 1237.000 | 459.000 |
| 15 | B | vol tercile | extreme | -0.035 | -0.229 | 210.000 | 86.000 |
| 15 | B | vol tercile | high | -0.069 | -0.094 | 1366.000 | 556.000 |
| 15 | B | vol tercile | low | -0.225 | -0.311 | 1679.000 | 742.000 |
| 15 | B | vol tercile | normal | -0.091 | -0.245 | 1656.000 | 635.000 |
| 60 | A | CUSUM transition | stable | 0.051 | -0.133 | 225.000 | 78.000 |
| 60 | A | CUSUM transition | transition | -0.003 | 0.207 | 64.000 | 34.000 |
| 60 | A | HMM-3 (filtered) | calm | -0.097 | -0.547 | 30.000 | 7.000 |
| 60 | A | HMM-3 (filtered) | mid | 0.170 | 0.161 | 100.000 | 35.000 |
| 60 | A | HMM-3 (filtered) | turbulent | -0.018 | -0.073 | 159.000 | 70.000 |
| 60 | A | session | ETH | -0.072 | -0.035 | 208.000 | 68.000 |
| 60 | A | session | RTH | 0.324 | -0.022 | 81.000 | 44.000 |
| 60 | A | variance-ratio rule | NEUTRAL | 0.029 | -0.095 | 91.000 | 51.000 |
| 60 | A | variance-ratio rule | RANGE | 0.004 | 0.207 | 91.000 | 37.000 |
| 60 | A | variance-ratio rule | TREND | 0.077 | -0.254 | 107.000 | 24.000 |
| 60 | A | vol tercile | extreme | 0.018 | 0.121 | 32.000 | 16.000 |
| 60 | A | vol tercile | high | 0.014 | 0.065 | 147.000 | 66.000 |
| 60 | A | vol tercile | low | -0.046 | -0.449 | 41.000 | 10.000 |
| 60 | A | vol tercile | normal | 0.152 | -0.253 | 69.000 | 20.000 |
| 60 | B | CUSUM transition | stable | -0.092 | -0.179 | 1077.000 | 434.000 |
| 60 | B | CUSUM transition | transition | 0.084 | -0.230 | 167.000 | 103.000 |
| 60 | B | HMM-3 (filtered) | calm | 0.029 | -0.121 | 20.000 | 36.000 |
| 60 | B | HMM-3 (filtered) | mid | -0.042 | -0.052 | 406.000 | 191.000 |
| 60 | B | HMM-3 (filtered) | turbulent | -0.083 | -0.281 | 818.000 | 310.000 |
| 60 | B | session | ETH | -0.070 | -0.213 | 808.000 | 301.000 |
| 60 | B | session | RTH | -0.064 | -0.158 | 436.000 | 236.000 |
| 60 | B | variance-ratio rule | NEUTRAL | -0.031 | -0.142 | 450.000 | 198.000 |
| 60 | B | variance-ratio rule | RANGE | -0.118 | -0.324 | 536.000 | 249.000 |
| 60 | B | variance-ratio rule | TREND | -0.029 | 0.080 | 258.000 | 90.000 |
| 60 | B | vol tercile | extreme | -0.081 | -0.202 | 49.000 | 24.000 |
| 60 | B | vol tercile | high | -0.083 | -0.312 | 374.000 | 151.000 |
| 60 | B | vol tercile | low | -0.171 | -0.104 | 399.000 | 220.000 |
| 60 | B | vol tercile | normal | 0.044 | -0.188 | 422.000 | 142.000 |

## M4 P(target before stop) — validation scores

Brier skill > 0 = better than the training base rate. ECE = expected calibration error.

| tf | family | model | n | observed_rate | mean_pred | brier_skill | auc | ece | log_loss |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A | base rate | 5801 | 0.2863 | 0.3129 | 0.0000 | 0.5000 | 0.0266 | 0.6005 |
| 1 | A | logistic | 5801 | 0.2863 | 0.3016 | 0.0048 | 0.5209 | 0.0153 | 0.5980 |
| 1 | A | gbm | 5801 | 0.2863 | 0.2981 | 0.0012 | 0.5114 | 0.0127 | 0.6003 |
| 1 | A | bayes beta-binomial | 5801 | 0.2863 | 0.2984 | 0.0044 | 0.5204 | 0.0120 | 0.5984 |
| 1 | B | base rate | 27554 | 0.2515 | 0.2500 | 0.0000 | 0.5000 | 0.0015 | 0.5640 |
| 1 | B | logistic | 27554 | 0.2515 | 0.2272 | -0.0033 | 0.4985 | 0.0244 | 0.5657 |
| 1 | B | gbm | 27554 | 0.2515 | 0.2269 | -0.0036 | 0.4964 | 0.0247 | 0.5660 |
| 1 | B | bayes beta-binomial | 27554 | 0.2515 | 0.2266 | -0.0036 | 0.4988 | 0.0249 | 0.5659 |
| 5 | A | base rate | 1131 | 0.2821 | 0.2935 | 0.0000 | 0.5000 | 0.0115 | 0.5952 |
| 5 | A | logistic | 1131 | 0.2821 | 0.2721 | 0.0039 | 0.5213 | 0.0100 | 0.5923 |
| 5 | A | gbm | 1131 | 0.2821 | 0.2705 | -0.0028 | 0.5115 | 0.0230 | 0.6020 |
| 5 | A | bayes beta-binomial | 1131 | 0.2821 | 0.3028 | -0.0052 | 0.4806 | 0.0293 | 0.5977 |
| 5 | B | base rate | 5750 | 0.2391 | 0.2415 | 0.0000 | 0.5000 | 0.0024 | 0.5501 |
| 5 | B | logistic | 5750 | 0.2391 | 0.2419 | 0.0038 | 0.5455 | 0.0045 | 0.5538 |
| 5 | B | gbm | 5750 | 0.2391 | 0.2407 | 0.0048 | 0.5449 | 0.0028 | 0.5489 |
| 5 | B | bayes beta-binomial | 5750 | 0.2391 | 0.2244 | -0.0010 | 0.5117 | 0.0147 | 0.5507 |
| 15 | A | base rate | 419 | 0.3413 | 0.3008 | 0.0000 | 0.5000 | 0.0405 | 0.6457 |
| 15 | A | logistic | 419 | 0.3413 | 0.3091 | -0.0617 | 0.5040 | 0.0654 | 0.7615 |
| 15 | A | gbm | 419 | 0.3413 | 0.3027 | -0.0411 | 0.4729 | 0.0959 | 0.6695 |
| 15 | A | bayes beta-binomial | 419 | 0.3413 | 0.3137 | 0.0041 | 0.5198 | 0.0276 | 0.6436 |
| 15 | B | base rate | 2019 | 0.2392 | 0.2305 | 0.0000 | 0.5000 | 0.0087 | 0.5504 |
| 15 | B | logistic | 2019 | 0.2392 | 0.2164 | 0.0053 | 0.5704 | 0.0335 | 0.5477 |
| 15 | B | gbm | 2019 | 0.2392 | 0.2109 | -0.0003 | 0.5807 | 0.0390 | 0.5488 |
| 15 | B | bayes beta-binomial | 2019 | 0.2392 | 0.2076 | -0.0062 | 0.4993 | 0.0316 | 0.5539 |
| 60 | B | base rate | 537 | 0.1471 | 0.1487 | 0.0000 | 0.5000 | 0.0016 | 0.4177 |
| 60 | B | logistic | 537 | 0.1471 | 0.1419 | -0.0717 | 0.6301 | 0.0302 | 0.5107 |
| 60 | B | gbm | 537 | 0.1471 | 0.1381 | -0.0267 | 0.6707 | 0.0702 | 0.4672 |
| 60 | B | bayes beta-binomial | 537 | 0.1471 | 0.1358 | 0.0134 | 0.5698 | 0.0177 | 0.4095 |

### Does filtering on the probability raise expectancy? (keep p >= break-even + 2pp)

| tf | family | model | p_breakeven | kept_share | exp_r_all | exp_r_kept | exp_r_dropped |
|---|---|---|---|---|---|---|---|
| 1 | A | logistic | 0.5602 | 0.0000 | -0.6128 |  | -0.6128 |
| 1 | A | gbm | 0.5602 | 0.0000 | -0.6128 |  | -0.6128 |
| 1 | A | bayes beta-binomial | 0.5602 | 0.0000 | -0.6128 |  | -0.6128 |
| 1 | B | logistic | 0.5984 | 0.0000 | -0.6483 |  | -0.6483 |
| 1 | B | gbm | 0.5984 | 0.0000 | -0.6483 |  | -0.6483 |
| 1 | B | bayes beta-binomial | 0.5984 | 0.0000 | -0.6483 |  | -0.6483 |
| 5 | A | logistic | 0.3996 | 0.0000 | -0.3894 |  | -0.3894 |
| 5 | A | gbm | 0.3996 | 0.0000 | -0.3894 |  | -0.3894 |
| 5 | A | bayes beta-binomial | 0.3996 | 0.0000 | -0.3894 |  | -0.3894 |
| 5 | B | logistic | 0.3801 | 0.0000 | -0.3967 |  | -0.3967 |
| 5 | B | gbm | 0.3801 | 0.0050 | -0.3967 | -0.4390 | -0.3965 |
| 5 | B | bayes beta-binomial | 0.3801 | 0.0000 | -0.3967 |  | -0.3967 |
| 15 | A | logistic | 0.3239 | 0.2554 | -0.0484 | -0.0729 | -0.0400 |
| 15 | A | gbm | 0.3239 | 0.2768 | -0.0484 | -0.2486 | 0.0282 |
| 15 | A | bayes beta-binomial | 0.3239 | 0.0883 | -0.0484 | -0.2412 | -0.0297 |
| 15 | B | logistic | 0.2915 | 0.1105 | -0.2271 | -0.3355 | -0.2136 |
| 15 | B | gbm | 0.2915 | 0.0768 | -0.2271 | -0.1741 | -0.2315 |
| 15 | B | bayes beta-binomial | 0.2915 | 0.0000 | -0.2271 |  | -0.2271 |
| 60 | B | logistic | 0.1807 | 0.3222 | -0.1890 | -0.2096 | -0.1791 |
| 60 | B | gbm | 0.1807 | 0.2365 | -0.1890 | -0.2927 | -0.1568 |
| 60 | B | bayes beta-binomial | 0.1807 | 0.0000 | -0.1890 |  | -0.1890 |
