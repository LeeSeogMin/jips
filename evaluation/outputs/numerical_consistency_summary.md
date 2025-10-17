# Numerical Consistency Analysis Report
Generated: 2025-10-18 00:04:48

## Overall Correlation Analysis

- **Pearson correlation**: r = 0.8464 (p = 0.0005)
- **Spearman correlation**: ρ = 0.7692 (p = 0.0034)
- **Kendall correlation**: τ = 0.5758 (p = 0.0088)

## Metric-wise Correlations

- **Coherence**: r = 0.9996 (p = 0.0177)
- **Distinctiveness**: r = 0.2472 (p = 0.8410)
- **Diversity**: r = 0.9144 (p = 0.2653)
- **Overall_Score**: r = 0.9707 (p = 0.1545)

## Dataset Trend Analysis

- **Coherence**: decreasing vs decreasing ✓
- **Distinctiveness**: increasing vs decreasing ✗
- **Diversity**: decreasing vs decreasing ✓
- **Overall_Score**: decreasing vs decreasing ✓

## Statistical Significance

- **Paired t-test**: t = -0.6385, p = 0.5362
- **Effect size (Cohen's d)**: -0.1925

## Consistency Metrics

- **Mean Absolute Error**: 0.0845
- **Root Mean Square Error**: 0.1187
- **Mean Absolute Percentage Error**: 16.65%
- **Agreement Score**: 0.5833

## Conclusions

1. **Correlation**: strong positive correlation between ST and DL metrics (r = 0.846)
2. **Agreement**: low rank agreement between methods (58.3%)
3. **Significance**: No statistically significant differences (p = 0.5362)