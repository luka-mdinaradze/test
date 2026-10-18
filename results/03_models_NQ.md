# Experiment 3 — Model investigations (NQ; fit on train, scored on validation)

## M1 Next-bar volatility forecasts (QLIKE, lower is better; validation)

Each model is rescaled by one constant fitted on train. `qlike_gain_vs_atr_pct` > 0 = better than ATR.

| tf | model | qlike_train | qlike_test | mz_r2_test | scale_k | qlike_gain_vs_atr_pct |
|---|---|---|---|---|---|---|
| 1 | atr | 1.2643 | 1.3504 | 0.3018 | 0.7100 | 0.0000 |
| 1 | ewma | 1.3489 | 1.4142 | 0.2728 | 1.0681 | -4.7316 |
| 1 | seasonal | 1.1802 | 1.2922 | 0.3171 | 1.0342 | 4.3057 |
| 1 | garch | 1.1723 | 1.2920 | 0.3218 | 1.0043 | 4.3248 |
| 5 | atr | 1.5662 | 1.5806 | 0.2478 | 0.7147 | 0.0000 |
| 5 | ewma | 1.8408 | 1.7854 | 0.1814 | 1.1107 | -12.9590 |
| 5 | seasonal | 1.3728 | 1.4342 | 0.3076 | 1.0467 | 9.2600 |
| 5 | garch | 1.3550 | 1.4210 | 0.3146 | 1.0192 | 10.0946 |
| 15 | atr | 1.8843 | 1.8442 | 0.1682 | 0.7595 | 0.0000 |
| 15 | ewma | 2.1240 | 2.0178 | 0.1033 | 1.1031 | -9.4152 |
| 15 | seasonal | 1.4379 | 1.5395 | 0.2888 | 1.0413 | 16.5201 |
| 15 | garch | 1.4177 | 1.5299 | 0.3001 | 1.0106 | 17.0427 |
| 60 | atr | 2.2005 | 2.1531 | 0.0850 | 0.8178 | 0.0000 |
| 60 | ewma | 2.2675 | 2.2084 | 0.0620 | 1.0930 | -2.5692 |
| 60 | seasonal | 1.5047 | 1.5852 | 0.2886 | 1.0052 | 26.3757 |
| 60 | garch | 1.4884 | 1.5924 | 0.3102 | 0.9736 | 26.0398 |

GARCH(1,1) on deseasonalised changes (train fit): 1m alpha=0.049 beta=0.946 nu=7.0, 5m alpha=0.061 beta=0.927 nu=6.3, 15m alpha=0.059 beta=0.922 nu=5.7, 60m alpha=0.047 beta=0.931 nu=4.9

## M2 Stop width from ATR vs from the seasonal volatility forecast (validation, per-signal R after costs)

Same median stop width on train; only the time profile of the width differs.

| tf | family | stop_unit | signals | exp_r | ci_lo | ci_hi | r_std | stopped_within_1_bar |
|---|---|---|---|---|---|---|---|---|
| 1 | A | atr | 2990 | -0.1144 | -0.1692 | -0.0605 | 1.2526 | 0.1789 |
| 1 | A | seasonal | 2990 | -0.1192 | -0.1697 | -0.0702 | 1.2556 | 0.2037 |
| 1 | B | atr | 14551 | -0.1483 | -0.1682 | -0.1277 | 1.1565 | 0.1904 |
| 1 | B | seasonal | 14551 | -0.1512 | -0.1703 | -0.1312 | 1.1507 | 0.1807 |
| 5 | A | atr | 603 | -0.0001 | -0.1356 | 0.1384 | 1.3149 | 0.0614 |
| 5 | A | seasonal | 603 | -0.0584 | -0.1934 | 0.0756 | 1.3115 | 0.0713 |
| 5 | B | atr | 2949 | -0.0468 | -0.0920 | 0.0011 | 1.2258 | 0.0966 |
| 5 | B | seasonal | 2949 | -0.0688 | -0.1158 | -0.0202 | 1.1865 | 0.0604 |
| 15 | A | atr | 199 | 0.0829 | -0.1309 | 0.3130 | 1.3048 | 0.0905 |
| 15 | A | seasonal | 199 | -0.0901 | -0.3100 | 0.1273 | 1.2279 | 0.0503 |
| 15 | B | atr | 1023 | -0.0476 | -0.1366 | 0.0412 | 1.2591 | 0.1388 |
| 15 | B | seasonal | 1023 | -0.0615 | -0.1420 | 0.0223 | 1.1387 | 0.0538 |
| 60 | A | atr | 56 | 0.2807 | -0.0869 | 0.6179 | 1.2944 | 0.1250 |
| 60 | A | seasonal | 56 | 0.2596 | -0.0366 | 0.5507 | 1.1327 | 0.0536 |
| 60 | B | atr | 302 | -0.0723 | -0.1966 | 0.0534 | 1.1588 | 0.2517 |
| 60 | B | seasonal | 302 | -0.0620 | -0.1586 | 0.0420 | 0.8747 | 0.0894 |

## M3 Regime classifiers vs what happened NEXT

`fwd_vr_minus_1`: realised variance ratio of the next h bars minus 1 (h = 60 min or 5 bars).
> 0 means the next bars trended, < 0 means they reverted. A useful trend/range classifier
shows TREND > 0 > RANGE in BOTH periods. `fwd_vol_vs_fcst`: next-bar realised |change| /
forecast (a vol classifier is useful if this differs across labels).

| ('tf', '') | ('classifier', '') | ('label', '') | ('fwd_vol_vs_fcst', 'train') | ('fwd_vol_vs_fcst', 'validation') | ('fwd_vr_minus_1', 'train') | ('fwd_vr_minus_1', 'validation') | ('t', 'train') | ('t', 'validation') |
|---|---|---|---|---|---|---|---|---|
| 1 | CUSUM transition | stable | 0.843 | 0.824 | -0.103 | -0.091 | -8.740 | -5.684 |
| 1 | CUSUM transition | transition | 0.820 | 0.814 | -0.126 | -0.083 | -8.718 | -3.876 |
| 1 | HMM-3 (filtered) | calm | 0.788 | 0.767 | -0.144 | -0.117 | -9.836 | -5.836 |
| 1 | HMM-3 (filtered) | mid | 0.842 | 0.825 | -0.101 | -0.083 | -8.665 | -5.109 |
| 1 | HMM-3 (filtered) | turbulent | 0.864 | 0.854 | -0.101 | -0.087 | -7.897 | -5.098 |
| 1 | efficiency-ratio rule | NEUTRAL | 0.838 | 0.819 | -0.109 | -0.095 | -9.329 | -5.736 |
| 1 | efficiency-ratio rule | RANGE | 0.839 | 0.811 | -0.111 | -0.092 | -8.941 | -4.926 |
| 1 | efficiency-ratio rule | TREND | 0.839 | 0.837 | -0.099 | -0.078 | -7.131 | -4.538 |
| 1 | session | ETH | 0.870 | 0.849 | -0.145 | -0.113 | -11.741 | -6.285 |
| 1 | session | RTH | 0.761 | 0.754 | -0.012 | -0.029 | -0.521 | -0.965 |
| 1 | variance-ratio rule | NEUTRAL | 0.838 | 0.820 | -0.098 | -0.072 | -5.516 | -3.234 |
| 1 | variance-ratio rule | RANGE | 0.840 | 0.820 | -0.139 | -0.092 | -7.604 | -3.228 |
| 1 | variance-ratio rule | TREND | 0.837 | 0.826 | -0.087 | -0.112 | -4.276 | -3.459 |
| 1 | vol tercile | extreme | 0.639 | 0.707 | -0.162 | -0.103 | -4.065 | -2.272 |
| 1 | vol tercile | high | 0.766 | 0.770 | -0.103 | -0.107 | -5.513 | -4.341 |
| 1 | vol tercile | low | 0.929 | 0.900 | -0.111 | -0.079 | -5.670 | -3.009 |
| 1 | vol tercile | normal | 0.840 | 0.817 | -0.098 | -0.077 | -6.136 | -2.800 |
| 5 | CUSUM transition | stable | 0.827 | 0.799 | -0.063 | -0.054 | -6.039 | -3.569 |
| 5 | CUSUM transition | transition | 0.803 | 0.816 | -0.056 | -0.103 | -2.682 | -4.369 |
| 5 | HMM-3 (filtered) | calm | 0.754 | 0.734 | -0.060 | -0.078 | -4.348 | -4.307 |
| 5 | HMM-3 (filtered) | mid | 0.841 | 0.819 | -0.065 | -0.060 | -6.141 | -3.818 |
| 5 | HMM-3 (filtered) | turbulent | 0.901 | 0.915 | -0.049 | -0.046 | -3.439 | -2.171 |
| 5 | efficiency-ratio rule | NEUTRAL | 0.823 | 0.799 | -0.056 | -0.079 | -4.931 | -4.714 |
| 5 | efficiency-ratio rule | RANGE | 0.816 | 0.788 | -0.075 | -0.049 | -5.267 | -2.369 |
| 5 | efficiency-ratio rule | TREND | 0.829 | 0.823 | -0.055 | -0.061 | -3.461 | -2.969 |
| 5 | session | ETH | 0.855 | 0.829 | -0.092 | -0.074 | -8.319 | -4.544 |
| 5 | session | RTH | 0.742 | 0.737 | 0.014 | -0.039 | 0.710 | -1.535 |
| 5 | variance-ratio rule | NEUTRAL | 0.814 | 0.808 | -0.070 | -0.076 | -5.416 | -3.735 |
| 5 | variance-ratio rule | RANGE | 0.829 | 0.797 | -0.033 | -0.037 | -1.711 | -1.878 |
| 5 | variance-ratio rule | TREND | 0.827 | 0.801 | -0.080 | -0.095 | -4.231 | -2.848 |
| 5 | vol tercile | extreme | 0.622 | 0.711 | -0.074 | -0.062 | -1.598 | -1.403 |
| 5 | vol tercile | high | 0.762 | 0.763 | -0.055 | -0.100 | -3.129 | -4.287 |
| 5 | vol tercile | low | 0.897 | 0.857 | -0.074 | -0.056 | -4.090 | -2.463 |
| 5 | vol tercile | normal | 0.830 | 0.814 | -0.053 | -0.039 | -3.451 | -1.374 |
| 15 | CUSUM transition | stable | 0.823 | 0.791 | -0.015 | -0.052 | -1.455 | -3.229 |
| 15 | CUSUM transition | transition | 0.777 | 0.793 | -0.065 | -0.001 | -3.405 | -0.033 |
| 15 | HMM-3 (filtered) | calm | 0.731 | 0.687 | -0.010 | -0.031 | -0.683 | -1.572 |
| 15 | HMM-3 (filtered) | mid | 0.827 | 0.812 | -0.035 | -0.047 | -3.368 | -2.957 |
| 15 | HMM-3 (filtered) | turbulent | 0.933 | 0.977 | -0.005 | -0.039 | -0.249 | -1.462 |
| 15 | efficiency-ratio rule | NEUTRAL | 0.822 | 0.799 | -0.023 | -0.023 | -1.779 | -1.233 |
| 15 | efficiency-ratio rule | RANGE | 0.817 | 0.785 | -0.038 | -0.076 | -2.527 | -3.504 |
| 15 | efficiency-ratio rule | TREND | 0.800 | 0.789 | -0.013 | -0.028 | -0.775 | -1.111 |
| 15 | session | ETH | 0.849 | 0.836 | -0.044 | -0.045 | -4.170 | -2.840 |
| 15 | session | RTH | 0.725 | 0.679 | 0.027 | -0.029 | 1.410 | -1.178 |
| 15 | variance-ratio rule | NEUTRAL | 0.804 | 0.797 | -0.003 | -0.044 | -0.176 | -2.364 |
| 15 | variance-ratio rule | RANGE | 0.833 | 0.775 | -0.024 | 0.035 | -1.398 | 1.309 |
| 15 | variance-ratio rule | TREND | 0.808 | 0.798 | -0.053 | -0.105 | -2.995 | -3.680 |
| 15 | vol tercile | extreme | 0.625 | 0.731 | -0.011 | -0.025 | -0.247 | -0.554 |
| 15 | vol tercile | high | 0.759 | 0.763 | -0.009 | -0.052 | -0.489 | -2.034 |
| 15 | vol tercile | low | 0.887 | 0.825 | -0.045 | -0.049 | -2.723 | -2.291 |
| 15 | vol tercile | normal | 0.817 | 0.813 | -0.019 | -0.023 | -1.286 | -0.816 |
| 60 | CUSUM transition | stable | 0.921 | 0.908 | -0.014 | -0.047 | -0.646 | -1.631 |
| 60 | CUSUM transition | transition | 0.885 | 0.853 | -0.018 | 0.166 | -0.429 | 2.428 |
| 60 | HMM-3 (filtered) | calm | 0.807 | 0.768 | 0.002 | 0.031 | 0.074 | 0.762 |
| 60 | HMM-3 (filtered) | mid | 0.962 | 0.925 | -0.026 | -0.027 | -1.227 | -0.831 |
| 60 | HMM-3 (filtered) | turbulent | 1.002 | 1.121 | -0.008 | 0.018 | -0.218 | 0.340 |
| 60 | efficiency-ratio rule | NEUTRAL | 0.925 | 0.914 | -0.015 | -0.003 | -0.583 | -0.087 |
| 60 | efficiency-ratio rule | RANGE | 0.904 | 0.863 | -0.008 | 0.009 | -0.253 | 0.210 |
| 60 | efficiency-ratio rule | TREND | 0.909 | 0.907 | -0.021 | -0.005 | -0.607 | -0.096 |
| 60 | session | ETH | 0.946 | 0.942 | -0.024 | 0.010 | -1.231 | 0.315 |
| 60 | session | RTH | 0.724 | 0.622 | 0.042 | -0.057 | 0.936 | -0.937 |
| 60 | variance-ratio rule | NEUTRAL | 0.909 | 0.870 | -0.018 | -0.005 | -0.570 | -0.120 |
| 60 | variance-ratio rule | RANGE | 0.932 | 0.906 | 0.041 | 0.044 | 1.287 | 0.839 |
| 60 | variance-ratio rule | TREND | 0.902 | 0.919 | -0.067 | -0.040 | -2.091 | -0.772 |
| 60 | vol tercile | extreme | 0.753 | 0.897 | -0.004 | -0.016 | -0.065 | -0.215 |
| 60 | vol tercile | high | 0.847 | 0.892 | -0.012 | -0.013 | -0.323 | -0.246 |
| 60 | vol tercile | low | 0.999 | 0.902 | -0.032 | -0.010 | -0.978 | -0.218 |
| 60 | vol tercile | normal | 0.910 | 0.890 | -0.001 | 0.047 | -0.035 | 0.704 |

### HMM (3 states, fitted on train): transition matrices and expected durations (bars)

- 1m: durations [2.0, 2.9, 1.7], diag [0.503, 0.654, 0.399]
- 5m: durations [2.2, 2.6, 1.5], diag [0.541, 0.618, 0.336]
- 15m: durations [2.2, 2.7, 1.5], diag [0.543, 0.623, 0.34]
- 60m: durations [2.3, 2.4, 1.4], diag [0.563, 0.587, 0.305]

### Reference strategy expectancy (R per signal, after costs) by regime label

| ('tf', '') | ('family', '') | ('classifier', '') | ('label', '') | ('exp_r', 'train') | ('exp_r', 'validation') | ('n', 'train') | ('n', 'validation') |
|---|---|---|---|---|---|---|---|
| 1 | A | CUSUM transition | stable | -0.200 | -0.101 | 4743.000 | 2297.000 |
| 1 | A | CUSUM transition | transition | -0.236 | -0.158 | 1321.000 | 693.000 |
| 1 | A | HMM-3 (filtered) | calm | -0.168 | 0.000 | 205.000 | 92.000 |
| 1 | A | HMM-3 (filtered) | mid | -0.187 | -0.133 | 2355.000 | 1147.000 |
| 1 | A | HMM-3 (filtered) | turbulent | -0.224 | -0.108 | 3504.000 | 1751.000 |
| 1 | A | session | ETH | -0.273 | -0.137 | 4236.000 | 2081.000 |
| 1 | A | session | RTH | -0.057 | -0.062 | 1828.000 | 909.000 |
| 1 | A | variance-ratio rule | NEUTRAL | -0.168 | -0.116 | 2448.000 | 1273.000 |
| 1 | A | variance-ratio rule | RANGE | -0.274 | -0.096 | 1679.000 | 927.000 |
| 1 | A | variance-ratio rule | TREND | -0.201 | -0.134 | 1937.000 | 790.000 |
| 1 | A | vol tercile | extreme | -0.248 | -0.032 | 468.000 | 451.000 |
| 1 | A | vol tercile | high | -0.174 | -0.099 | 2200.000 | 858.000 |
| 1 | A | vol tercile | low | -0.257 | -0.113 | 1349.000 | 917.000 |
| 1 | A | vol tercile | normal | -0.203 | -0.183 | 2047.000 | 764.000 |
| 1 | B | CUSUM transition | stable | -0.230 | -0.149 | 23769.000 | 11649.000 |
| 1 | B | CUSUM transition | transition | -0.217 | -0.147 | 5224.000 | 2902.000 |
| 1 | B | HMM-3 (filtered) | calm | -0.209 | -0.238 | 318.000 | 182.000 |
| 1 | B | HMM-3 (filtered) | mid | -0.248 | -0.155 | 9633.000 | 4779.000 |
| 1 | B | HMM-3 (filtered) | turbulent | -0.218 | -0.143 | 19042.000 | 9590.000 |
| 1 | B | session | ETH | -0.289 | -0.193 | 20487.000 | 10274.000 |
| 1 | B | session | RTH | -0.082 | -0.040 | 8506.000 | 4277.000 |
| 1 | B | variance-ratio rule | NEUTRAL | -0.221 | -0.143 | 11537.000 | 6019.000 |
| 1 | B | variance-ratio rule | RANGE | -0.223 | -0.182 | 8898.000 | 4741.000 |
| 1 | B | variance-ratio rule | TREND | -0.243 | -0.114 | 8558.000 | 3791.000 |
| 1 | B | vol tercile | extreme | -0.151 | -0.088 | 1288.000 | 1710.000 |
| 1 | B | vol tercile | high | -0.193 | -0.113 | 8230.000 | 3952.000 |
| 1 | B | vol tercile | low | -0.270 | -0.198 | 9288.000 | 5369.000 |
| 1 | B | vol tercile | normal | -0.228 | -0.142 | 10187.000 | 3520.000 |
| 5 | A | CUSUM transition | stable | -0.026 | 0.060 | 974.000 | 428.000 |
| 5 | A | CUSUM transition | transition | -0.073 | -0.146 | 297.000 | 175.000 |
| 5 | A | HMM-3 (filtered) | calm | 0.066 | 0.119 | 109.000 | 40.000 |
| 5 | A | HMM-3 (filtered) | mid | -0.032 | -0.014 | 546.000 | 252.000 |
| 5 | A | HMM-3 (filtered) | turbulent | -0.060 | -0.004 | 616.000 | 311.000 |
| 5 | A | session | ETH | -0.061 | -0.049 | 890.000 | 419.000 |
| 5 | A | session | RTH | 0.020 | 0.111 | 381.000 | 184.000 |
| 5 | A | variance-ratio rule | NEUTRAL | -0.088 | -0.035 | 476.000 | 249.000 |
| 5 | A | variance-ratio rule | RANGE | 0.035 | -0.010 | 402.000 | 238.000 |
| 5 | A | variance-ratio rule | TREND | -0.049 | 0.094 | 393.000 | 116.000 |
| 5 | A | vol tercile | extreme | -0.125 | 0.107 | 119.000 | 112.000 |
| 5 | A | vol tercile | high | -0.037 | -0.131 | 454.000 | 190.000 |
| 5 | A | vol tercile | low | -0.098 | -0.028 | 276.000 | 165.000 |
| 5 | A | vol tercile | normal | 0.028 | 0.129 | 422.000 | 136.000 |
| 5 | B | CUSUM transition | stable | -0.119 | -0.057 | 4891.000 | 2279.000 |
| 5 | B | CUSUM transition | transition | -0.080 | -0.011 | 1025.000 | 670.000 |
| 5 | B | HMM-3 (filtered) | calm | -0.094 | 0.114 | 235.000 | 171.000 |
| 5 | B | HMM-3 (filtered) | mid | -0.125 | -0.058 | 2841.000 | 1443.000 |
| 5 | B | HMM-3 (filtered) | turbulent | -0.101 | -0.055 | 2840.000 | 1335.000 |
| 5 | B | session | ETH | -0.158 | -0.070 | 4248.000 | 2096.000 |
| 5 | B | session | RTH | 0.006 | 0.011 | 1668.000 | 853.000 |
| 5 | B | variance-ratio rule | NEUTRAL | -0.103 | -0.021 | 2315.000 | 1138.000 |
| 5 | B | variance-ratio rule | RANGE | -0.153 | -0.063 | 1945.000 | 1230.000 |
| 5 | B | variance-ratio rule | TREND | -0.076 | -0.063 | 1656.000 | 581.000 |
| 5 | B | vol tercile | extreme | -0.071 | -0.020 | 245.000 | 346.000 |
| 5 | B | vol tercile | high | -0.050 | -0.042 | 1680.000 | 791.000 |
| 5 | B | vol tercile | low | -0.130 | -0.091 | 1893.000 | 1066.000 |
| 5 | B | vol tercile | normal | -0.150 | -0.001 | 2098.000 | 746.000 |
| 15 | A | CUSUM transition | stable | 0.025 | 0.182 | 355.000 | 144.000 |
| 15 | A | CUSUM transition | transition | -0.063 | -0.176 | 98.000 | 55.000 |
| 15 | A | HMM-3 (filtered) | calm | -0.029 | 0.021 | 63.000 | 23.000 |
| 15 | A | HMM-3 (filtered) | mid | 0.138 | 0.236 | 193.000 | 78.000 |
| 15 | A | HMM-3 (filtered) | turbulent | -0.112 | -0.025 | 197.000 | 98.000 |
| 15 | A | session | ETH | -0.029 | 0.151 | 327.000 | 140.000 |
| 15 | A | session | RTH | 0.097 | -0.080 | 126.000 | 59.000 |
| 15 | A | variance-ratio rule | NEUTRAL | -0.054 | -0.145 | 184.000 | 83.000 |
| 15 | A | variance-ratio rule | RANGE | -0.020 | 0.498 | 129.000 | 59.000 |
| 15 | A | variance-ratio rule | TREND | 0.109 | -0.014 | 140.000 | 57.000 |
| 15 | A | vol tercile | extreme | 0.098 | 0.076 | 41.000 | 30.000 |
| 15 | A | vol tercile | high | -0.049 | 0.079 | 153.000 | 59.000 |
| 15 | A | vol tercile | low | 0.099 | -0.141 | 99.000 | 50.000 |
| 15 | A | vol tercile | normal | -0.023 | 0.276 | 160.000 | 60.000 |
| 15 | B | CUSUM transition | stable | -0.055 | -0.049 | 1710.000 | 796.000 |
| 15 | B | CUSUM transition | transition | -0.010 | -0.042 | 359.000 | 227.000 |
| 15 | B | HMM-3 (filtered) | calm | -0.009 | 0.044 | 99.000 | 98.000 |
| 15 | B | HMM-3 (filtered) | mid | -0.076 | -0.032 | 1018.000 | 511.000 |
| 15 | B | HMM-3 (filtered) | turbulent | -0.020 | -0.088 | 952.000 | 414.000 |
| 15 | B | session | ETH | -0.065 | -0.042 | 1418.000 | 678.000 |
| 15 | B | session | RTH | -0.009 | -0.059 | 651.000 | 345.000 |
| 15 | B | variance-ratio rule | NEUTRAL | -0.037 | -0.094 | 788.000 | 473.000 |
| 15 | B | variance-ratio rule | RANGE | -0.057 | -0.003 | 729.000 | 296.000 |
| 15 | B | variance-ratio rule | TREND | -0.049 | -0.012 | 552.000 | 254.000 |
| 15 | B | vol tercile | extreme | 0.027 | -0.017 | 92.000 | 140.000 |
| 15 | B | vol tercile | high | -0.047 | 0.053 | 598.000 | 271.000 |
| 15 | B | vol tercile | low | -0.084 | -0.123 | 687.000 | 385.000 |
| 15 | B | vol tercile | normal | -0.021 | -0.057 | 692.000 | 227.000 |
| 60 | A | CUSUM transition | stable | -0.065 | 0.117 | 106.000 | 42.000 |
| 60 | A | CUSUM transition | transition | 0.057 | 0.772 | 19.000 | 14.000 |
| 60 | A | HMM-3 (filtered) | calm | -0.077 | 0.165 | 26.000 | 8.000 |
| 60 | A | HMM-3 (filtered) | mid | -0.087 | 0.196 | 53.000 | 22.000 |
| 60 | A | HMM-3 (filtered) | turbulent | 0.017 | 0.388 | 46.000 | 26.000 |
| 60 | A | session | ETH | -0.107 | 0.316 | 68.000 | 27.000 |
| 60 | A | session | RTH | 0.026 | 0.248 | 57.000 | 29.000 |
| 60 | A | variance-ratio rule | NEUTRAL | 0.128 | 0.543 | 47.000 | 24.000 |
| 60 | A | variance-ratio rule | RANGE | -0.215 | 0.353 | 27.000 | 11.000 |
| 60 | A | variance-ratio rule | TREND | -0.118 | -0.057 | 51.000 | 21.000 |
| 60 | A | vol tercile | extreme | 0.182 | 0.734 | 11.000 | 12.000 |
| 60 | A | vol tercile | high | -0.381 | 0.213 | 36.000 | 18.000 |
| 60 | A | vol tercile | low | 0.549 | 0.305 | 25.000 | 16.000 |
| 60 | A | vol tercile | normal | -0.148 | -0.180 | 53.000 | 10.000 |
| 60 | B | CUSUM transition | stable | -0.055 | -0.049 | 503.000 | 231.000 |
| 60 | B | CUSUM transition | transition | 0.049 | -0.148 | 108.000 | 71.000 |
| 60 | B | HMM-3 (filtered) | calm | -0.149 | -0.321 | 55.000 | 36.000 |
| 60 | B | HMM-3 (filtered) | mid | -0.035 | 0.037 | 360.000 | 157.000 |
| 60 | B | HMM-3 (filtered) | turbulent | -0.009 | -0.147 | 196.000 | 109.000 |
| 60 | B | session | ETH | 0.001 | -0.089 | 243.000 | 128.000 |
| 60 | B | session | RTH | -0.062 | -0.060 | 368.000 | 174.000 |
| 60 | B | variance-ratio rule | NEUTRAL | -0.025 | 0.065 | 245.000 | 111.000 |
| 60 | B | variance-ratio rule | RANGE | -0.105 | -0.108 | 227.000 | 115.000 |
| 60 | B | variance-ratio rule | TREND | 0.054 | -0.219 | 139.000 | 76.000 |
| 60 | B | vol tercile | extreme | -0.384 | -0.432 | 28.000 | 40.000 |
| 60 | B | vol tercile | high | 0.064 | 0.114 | 166.000 | 86.000 |
| 60 | B | vol tercile | low | -0.054 | -0.045 | 203.000 | 118.000 |
| 60 | B | vol tercile | normal | -0.053 | -0.157 | 214.000 | 58.000 |

## M4 P(target before stop) — validation scores

Brier skill > 0 = better than the training base rate. ECE = expected calibration error.

| tf | family | model | n | observed_rate | mean_pred | brier_skill | auc | ece | log_loss |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A | base rate | 2990 | 0.2970 | 0.2761 | 0.0000 | 0.5000 | 0.0209 | 0.6094 |
| 1 | A | logistic | 2990 | 0.2970 | 0.3007 | 0.0020 | 0.5313 | 0.0175 | 0.6199 |
| 1 | A | gbm | 2990 | 0.2970 | 0.2986 | -0.0022 | 0.5134 | 0.0355 | 0.6126 |
| 1 | A | bayes beta-binomial | 2990 | 0.2970 | 0.2778 | -0.0035 | 0.4893 | 0.0331 | 0.6112 |
| 1 | B | base rate | 14551 | 0.2311 | 0.2192 | 0.0000 | 0.5000 | 0.0119 | 0.5410 |
| 1 | B | logistic | 14551 | 0.2311 | 0.2110 | -0.0031 | 0.5190 | 0.0208 | 0.5460 |
| 1 | B | gbm | 14551 | 0.2311 | 0.2112 | -0.0015 | 0.5304 | 0.0219 | 0.5433 |
| 1 | B | bayes beta-binomial | 14551 | 0.2311 | 0.2200 | -0.0006 | 0.5076 | 0.0111 | 0.5414 |
| 5 | A | base rate | 603 | 0.3035 | 0.2793 | 0.0000 | 0.5000 | 0.0242 | 0.6152 |
| 5 | A | logistic | 603 | 0.3035 | 0.2497 | -0.0631 | 0.5144 | 0.0887 | 0.6623 |
| 5 | A | gbm | 603 | 0.3035 | 0.2354 | -0.0306 | 0.5285 | 0.0800 | 0.6749 |
| 5 | A | bayes beta-binomial | 603 | 0.3035 | 0.2807 | -0.0139 | 0.4729 | 0.0456 | 0.6229 |
| 5 | B | base rate | 2949 | 0.2418 | 0.2299 | 0.0000 | 0.5000 | 0.0119 | 0.5535 |
| 5 | B | logistic | 2949 | 0.2418 | 0.2317 | 0.0045 | 0.5557 | 0.0142 | 0.5778 |
| 5 | B | gbm | 2949 | 0.2418 | 0.2375 | 0.0019 | 0.5674 | 0.0213 | 0.5575 |
| 5 | B | bayes beta-binomial | 2949 | 0.2418 | 0.2313 | -0.0033 | 0.4694 | 0.0105 | 0.5553 |
| 15 | A | base rate | 199 | 0.2915 | 0.2759 | 0.0000 | 0.5000 | 0.0155 | 0.6040 |
| 15 | A | logistic | 199 | 0.2915 | 0.2527 | 0.0095 | 0.6399 | 0.0437 | 0.5984 |
| 15 | A | gbm | 199 | 0.2915 | 0.2456 | -0.0090 | 0.5000 | 0.0458 | 0.6089 |
| 15 | A | bayes beta-binomial | 199 | 0.2915 | 0.2728 | 0.0096 | 0.5570 | 0.0504 | 0.5986 |
| 15 | B | base rate | 1023 | 0.2532 | 0.2678 | 0.0000 | 0.5000 | 0.0146 | 0.5663 |
| 15 | B | logistic | 1023 | 0.2532 | 0.2730 | 0.0365 | 0.6167 | 0.0314 | 0.5586 |
| 15 | B | gbm | 1023 | 0.2532 | 0.2800 | 0.0210 | 0.5934 | 0.0381 | 0.5681 |
| 15 | B | bayes beta-binomial | 1023 | 0.2532 | 0.2657 | -0.0024 | 0.5001 | 0.0125 | 0.5675 |
| 60 | B | base rate | 302 | 0.1854 | 0.2029 | 0.0000 | 0.5000 | 0.0175 | 0.4805 |
| 60 | B | logistic | 302 | 0.1854 | 0.2476 | -0.0026 | 0.6673 | 0.0820 | 0.5208 |
| 60 | B | gbm | 302 | 0.1854 | 0.2609 | -0.1823 | 0.6470 | 0.0834 | 0.7542 |
| 60 | B | bayes beta-binomial | 302 | 0.1854 | 0.1978 | 0.0442 | 0.6318 | 0.0464 | 0.4601 |

### Does filtering on the probability raise expectancy? (keep p >= break-even + 2pp)

| tf | family | model | p_breakeven | kept_share | exp_r_all | exp_r_kept | exp_r_dropped |
|---|---|---|---|---|---|---|---|
| 1 | A | logistic | 0.3578 | 0.0221 | -0.1144 | 0.1553 | -0.1205 |
| 1 | A | gbm | 0.3578 | 0.0194 | -0.1144 | 0.0832 | -0.1183 |
| 1 | A | bayes beta-binomial | 0.3578 | 0.0000 | -0.1144 |  | -0.1144 |
| 1 | B | logistic | 0.3178 | 0.0003 | -0.1483 | 0.2810 | -0.1485 |
| 1 | B | gbm | 0.3178 | 0.0100 | -0.1483 | -0.0388 | -0.1494 |
| 1 | B | bayes beta-binomial | 0.3178 | 0.0000 | -0.1483 |  | -0.1483 |
| 5 | A | logistic | 0.2937 | 0.1111 | -0.0001 | 0.1023 | -0.0129 |
| 5 | A | gbm | 0.2937 | 0.1393 | -0.0001 | -0.0351 | 0.0056 |
| 5 | A | bayes beta-binomial | 0.2937 | 0.2985 | -0.0001 | -0.0565 | 0.0239 |
| 5 | B | logistic | 0.2752 | 0.0624 | -0.0468 | 0.0316 | -0.0521 |
| 5 | B | gbm | 0.2752 | 0.2506 | -0.0468 | -0.0630 | -0.0414 |
| 5 | B | bayes beta-binomial | 0.2752 | 0.0000 | -0.0468 |  | -0.0468 |
| 15 | A | logistic | 0.2737 | 0.0402 | 0.0829 | 0.4489 | 0.0676 |
| 15 | A | gbm | 0.2737 | 0.0000 | 0.0829 |  | 0.0829 |
| 15 | A | bayes beta-binomial | 0.2737 | 0.3317 | 0.0829 | 0.0209 | 0.1137 |
| 15 | B | logistic | 0.2857 | 0.4809 | -0.0476 | -0.0123 | -0.0803 |
| 15 | B | gbm | 0.2857 | 0.5640 | -0.0476 | -0.0570 | -0.0354 |
| 15 | B | bayes beta-binomial | 0.2857 | 0.0156 | -0.0476 | -0.2200 | -0.0448 |
| 60 | B | logistic | 0.2177 | 0.5232 | -0.0723 | -0.1151 | -0.0255 |
| 60 | B | gbm | 0.2177 | 0.4669 | -0.0723 | -0.0362 | -0.1040 |
| 60 | B | bayes beta-binomial | 0.2177 | 0.2583 | -0.0723 | 0.0583 | -0.1178 |

## M5 Estimated order-flow imbalance (tick rule on 1-second bars, NQ-B, Mar-Apr 2026)

ESTIMATE, not measured order flow: no bid/ask or aggressor flags exist in any reachable source.

| index | all hours | RTH |
|---|---|---|
| minutes | 13229 | 6598 |
| contemp_corr | 0.2655 | 0.6157 |
| slope_in | 0.2363 | 1.7212 |
| t_in | 0.3738 | 1.7269 |
| corr_out | 0.0153 | 0.0322 |
| r2_oos | -0.0026 | -0.0040 |
| sign_hit_out | 0.4908 | 0.5095 |
| out_minutes | 5292 | 2640 |
| start | 2026-03-09 12:10:00 | 2026-03-09 12:10:00 |
| end | 2026-04-15 19:28:00 | 2026-04-15 15:59:00 |
