#!/usr/bin/env python3
"""
Numerical Consistency Analysis

수치 일관성 검증 및 ST/DL 메트릭 간 비교 분석
- 통계적 유의성 검증
- 메트릭 간 상관관계 분석
- 데이터셋별 트렌드 일관성 확인
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

class NumericalConsistencyAnalyzer:
    """수치 일관성 및 메트릭 비교 분석기"""
    
    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def load_evaluation_results(self):
        """ST와 DL 평가 결과 로드"""
        # ST 결과 로드
        with open(self.output_dir / 'st_results.json', 'r', encoding='utf-8') as f:
            self.st_results = json.load(f)
            
        # DL 결과 로드
        with open(self.output_dir / 'dl_results.json', 'r', encoding='utf-8') as f:
            self.dl_results = json.load(f)
            
        print("✓ Evaluation results loaded successfully")
        
    def extract_metric_values(self):
        """메트릭 값들을 추출하여 비교 가능한 형태로 정리"""
        datasets = ['distinct', 'similar', 'more_similar']
        
        # ST 메트릭 추출
        st_metrics = {}
        for dataset in datasets:
            dataset_key = dataset if dataset in self.st_results['datasets'] else dataset.replace('_', ' ').title()
            if dataset_key in self.st_results['datasets']:
                st_metrics[dataset] = self.st_results['datasets'][dataset_key]
            elif dataset in self.st_results['datasets']:
                st_metrics[dataset] = self.st_results['datasets'][dataset]
                
        # DL 메트릭 추출 (Mean 값 사용)
        dl_metrics = {}
        for dataset in datasets:
            dataset_key = dataset if dataset in self.dl_results['datasets'] else dataset.replace('_', ' ').title()
            if dataset_key in self.dl_results['datasets']:
                mean_values = self.dl_results['datasets'][dataset_key]['metrics']['Mean']
                dl_metrics[dataset] = {
                    'coherence': mean_values['Coherence'],
                    'distinctiveness': mean_values['Distinctiveness'],
                    'diversity': mean_values['Diversity'],
                    'overall_score': mean_values['Overall Score']
                }
            elif dataset in self.dl_results['datasets']:
                mean_values = self.dl_results['datasets'][dataset]['metrics']['Mean']
                dl_metrics[dataset] = {
                    'coherence': mean_values['Coherence'],
                    'distinctiveness': mean_values['Distinctiveness'],
                    'diversity': mean_values['Diversity'],
                    'overall_score': mean_values['Overall Score']
                }
                
        self.st_metrics = st_metrics
        self.dl_metrics = dl_metrics
        
        print("✓ Metric values extracted and organized")
        
    def create_comparison_dataframe(self):
        """비교 분석을 위한 DataFrame 생성"""
        datasets = ['distinct', 'similar', 'more_similar']
        metrics = ['coherence', 'distinctiveness', 'diversity', 'overall_score']
        
        # ST와 DL 데이터를 하나의 DataFrame으로 결합
        comparison_data = []
        
        for dataset in datasets:
            for metric in metrics:
                st_value = self.st_metrics[dataset][metric]
                dl_value = self.dl_metrics[dataset][metric]
                
                comparison_data.append({
                    'Dataset': dataset,
                    'Metric': metric,
                    'ST_Value': st_value,
                    'DL_Value': dl_value,
                    'Difference': dl_value - st_value,
                    'Relative_Difference': (dl_value - st_value) / st_value * 100 if st_value != 0 else 0
                })
                
        self.comparison_df = pd.DataFrame(comparison_data)
        print("✓ Comparison DataFrame created")
        
    def compute_correlation_analysis(self):
        """ST와 DL 메트릭 간 상관관계 분석"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS: ST vs DL Metrics")
        print("="*60)
        
        # 전체 메트릭에 대한 상관관계
        st_values = self.comparison_df['ST_Value'].values
        dl_values = self.comparison_df['DL_Value'].values
        
        # 다양한 상관계수 계산
        pearson_r, pearson_p = pearsonr(st_values, dl_values)
        spearman_r, spearman_p = spearmanr(st_values, dl_values)
        kendall_tau, kendall_p = kendalltau(st_values, dl_values)
        
        self.results['overall_correlation'] = {
            'pearson': {'correlation': pearson_r, 'p_value': pearson_p},
            'spearman': {'correlation': spearman_r, 'p_value': spearman_p},
            'kendall': {'correlation': kendall_tau, 'p_value': kendall_p}
        }
        
        print(f"Overall Correlation between ST and DL metrics:")
        print(f"  Pearson r = {pearson_r:.4f} (p = {pearson_p:.4f})")
        print(f"  Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.4f})")
        print(f"  Kendall τ = {kendall_tau:.4f} (p = {kendall_p:.4f})")
        
        # 메트릭별 상관관계
        print(f"\nMetric-wise Correlations:")
        print("-" * 40)
        
        metric_correlations = {}
        for metric in ['coherence', 'distinctiveness', 'diversity', 'overall_score']:
            metric_data = self.comparison_df[self.comparison_df['Metric'] == metric]
            if len(metric_data) >= 3:  # 최소 3개 데이터포인트 필요
                st_vals = metric_data['ST_Value'].values
                dl_vals = metric_data['DL_Value'].values
                
                r, p = pearsonr(st_vals, dl_vals)
                metric_correlations[metric] = {'correlation': r, 'p_value': p}
                
                print(f"  {metric.title()}: r = {r:.4f} (p = {p:.4f})")
            else:
                print(f"  {metric.title()}: insufficient data")
                
        self.results['metric_correlations'] = metric_correlations
        
    def analyze_dataset_trends(self):
        """데이터셋별 트렌드 일관성 분석"""
        print(f"\n" + "="*60)
        print("DATASET TREND ANALYSIS")
        print("="*60)
        
        datasets = ['distinct', 'similar', 'more_similar']
        metrics = ['coherence', 'distinctiveness', 'diversity', 'overall_score']
        
        trend_analysis = {}
        
        for metric in metrics:
            st_trend = [self.st_metrics[dataset][metric] for dataset in datasets]
            dl_trend = [self.dl_metrics[dataset][metric] for dataset in datasets]
            
            # 트렌드 방향 비교
            st_direction = self._get_trend_direction(st_trend)
            dl_direction = self._get_trend_direction(dl_trend)
            
            # 순위 상관관계 (rank correlation)
            rank_corr, rank_p = spearmanr(st_trend, dl_trend)
            
            trend_analysis[metric] = {
                'st_trend': st_trend,
                'dl_trend': dl_trend,
                'st_direction': st_direction,
                'dl_direction': dl_direction,
                'trend_agreement': st_direction == dl_direction,
                'rank_correlation': rank_corr,
                'rank_p_value': rank_p
            }
            
            print(f"{metric.title()}:")
            print(f"  ST trend: {st_direction} {st_trend}")
            print(f"  DL trend: {dl_direction} {dl_trend}")
            print(f"  Agreement: {'✓' if st_direction == dl_direction else '✗'}")
            print(f"  Rank correlation: ρ = {rank_corr:.4f} (p = {rank_p:.4f})")
            print()
            
        self.results['trend_analysis'] = trend_analysis
        
    def _get_trend_direction(self, values):
        """값들의 전체적인 트렌드 방향 결정"""
        if len(values) < 2:
            return "insufficient_data"
            
        # 첫 번째와 마지막 값 비교
        if values[-1] > values[0]:
            return "increasing"
        elif values[-1] < values[0]:
            return "decreasing"
        else:
            return "stable"
            
    def statistical_significance_testing(self):
        """통계적 유의성 검증"""
        print(f"\n" + "="*60)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("="*60)
        
        # 쌍대 t-검정 (paired t-test)
        st_values = self.comparison_df['ST_Value'].values
        dl_values = self.comparison_df['DL_Value'].values
        
        t_stat, t_p = stats.ttest_rel(dl_values, st_values)
        
        # Wilcoxon signed-rank test (비모수 검정)
        w_stat, w_p = stats.wilcoxon(dl_values, st_values)
        
        # 평균 차이 및 효과 크기
        mean_diff = np.mean(dl_values - st_values)
        effect_size = mean_diff / np.std(dl_values - st_values)  # Cohen's d
        
        self.results['significance_testing'] = {
            'paired_ttest': {'t_statistic': t_stat, 'p_value': t_p},
            'wilcoxon_test': {'w_statistic': w_stat, 'p_value': w_p},
            'mean_difference': mean_diff,
            'effect_size': effect_size
        }
        
        print(f"Paired t-test: t = {t_stat:.4f}, p = {t_p:.4f}")
        print(f"Wilcoxon test: W = {w_stat:.4f}, p = {w_p:.4f}")
        print(f"Mean difference (DL - ST): {mean_diff:.4f}")
        print(f"Effect size (Cohen's d): {effect_size:.4f}")
        
        # 해석
        if t_p < 0.001:
            significance = "highly significant (p < 0.001)"
        elif t_p < 0.01:
            significance = "very significant (p < 0.01)"
        elif t_p < 0.05:
            significance = "significant (p < 0.05)"
        else:
            significance = "not significant (p ≥ 0.05)"
            
        print(f"Result: {significance}")
        
    def consistency_metrics(self):
        """일관성 메트릭 계산"""
        print(f"\n" + "="*60)
        print("CONSISTENCY METRICS")
        print("="*60)
        
        # 절대 평균 오차 (MAE)
        mae = np.mean(np.abs(self.comparison_df['Difference']))
        
        # 평균 제곱근 오차 (RMSE)  
        rmse = np.sqrt(np.mean(self.comparison_df['Difference']**2))
        
        # 평균 절대 백분율 오차 (MAPE)
        mape = np.mean(np.abs(self.comparison_df['Relative_Difference']))
        
        # 일치도 (Agreement) - 동일한 순위를 갖는 비율
        agreement_score = self._calculate_agreement_score()
        
        consistency_metrics = {
            'mae': mae,
            'rmse': rmse, 
            'mape': mape,
            'agreement_score': agreement_score
        }
        
        self.results['consistency_metrics'] = consistency_metrics
        
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Agreement Score: {agreement_score:.4f}")
        
    def _calculate_agreement_score(self):
        """ST와 DL 메트릭 간 순위 일치도 계산"""
        datasets = ['distinct', 'similar', 'more_similar']
        metrics = ['coherence', 'distinctiveness', 'diversity', 'overall_score']
        
        agreements = []
        
        for metric in metrics:
            st_values = [self.st_metrics[dataset][metric] for dataset in datasets]
            dl_values = [self.dl_metrics[dataset][metric] for dataset in datasets]
            
            # 순위 계산 (높은 값이 1순위)
            st_ranks = stats.rankdata(-np.array(st_values))
            dl_ranks = stats.rankdata(-np.array(dl_values))
            
            # 순위 일치 여부 확인
            rank_agreement = np.mean(st_ranks == dl_ranks)
            agreements.append(rank_agreement)
            
        return np.mean(agreements)
        
    def generate_visualizations(self):
        """결과 시각화 생성"""
        print(f"\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # 1. 산점도 - ST vs DL 전체 비교
        plt.figure(figsize=(12, 10))
        
        # 서브플롯 1: 전체 비교
        plt.subplot(2, 2, 1)
        plt.scatter(self.comparison_df['ST_Value'], self.comparison_df['DL_Value'], 
                   alpha=0.7, s=100, c='blue')
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect Agreement')
        plt.xlabel('Statistical Metrics (ST)')
        plt.ylabel('Deep Learning Metrics (DL)')
        plt.title('ST vs DL Metrics Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 2: 메트릭별 비교
        plt.subplot(2, 2, 2)
        metric_colors = {'coherence': 'red', 'distinctiveness': 'green', 
                        'diversity': 'blue', 'overall_score': 'orange'}
        
        for metric in metric_colors.keys():
            metric_data = self.comparison_df[self.comparison_df['Metric'] == metric]
            plt.scatter(metric_data['ST_Value'], metric_data['DL_Value'], 
                       c=metric_colors[metric], label=metric.title(), s=100, alpha=0.7)
                       
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('Statistical Metrics (ST)')
        plt.ylabel('Deep Learning Metrics (DL)')
        plt.title('Metric-wise Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 3: 차이 분포
        plt.subplot(2, 2, 3)
        plt.hist(self.comparison_df['Difference'], bins=10, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='No Difference')
        plt.xlabel('Difference (DL - ST)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Differences')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 4: 데이터셋별 트렌드
        plt.subplot(2, 2, 4)
        datasets = ['distinct', 'similar', 'more_similar']
        x_pos = np.arange(len(datasets))
        
        # Overall Score만 표시
        st_overall = [self.st_metrics[dataset]['overall_score'] for dataset in datasets]
        dl_overall = [self.dl_metrics[dataset]['overall_score'] for dataset in datasets]
        
        plt.plot(x_pos, st_overall, 'o-', label='Statistical (ST)', linewidth=2, markersize=8)
        plt.plot(x_pos, dl_overall, 's-', label='Deep Learning (DL)', linewidth=2, markersize=8)
        plt.xticks(x_pos, [d.replace('_', ' ').title() for d in datasets])
        plt.ylabel('Overall Score')
        plt.title('Dataset Trends Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'numerical_consistency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Visualizations saved to numerical_consistency_analysis.png")
        
    def save_results(self):
        """분석 결과 저장"""
        # JSON 형태로 상세 결과 저장
        output_file = self.output_dir / 'numerical_consistency_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
            
        # 요약 보고서 생성
        self.generate_summary_report()
        
        print(f"✓ Detailed results saved to {output_file}")
        
    def generate_summary_report(self):
        """요약 보고서 생성"""
        report_lines = []
        report_lines.append("# Numerical Consistency Analysis Report")
        report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 전체 상관관계
        report_lines.append("## Overall Correlation Analysis")
        report_lines.append("")
        corr = self.results['overall_correlation']
        report_lines.append(f"- **Pearson correlation**: r = {corr['pearson']['correlation']:.4f} (p = {corr['pearson']['p_value']:.4f})")
        report_lines.append(f"- **Spearman correlation**: ρ = {corr['spearman']['correlation']:.4f} (p = {corr['spearman']['p_value']:.4f})")
        report_lines.append(f"- **Kendall correlation**: τ = {corr['kendall']['correlation']:.4f} (p = {corr['kendall']['p_value']:.4f})")
        report_lines.append("")
        
        # 메트릭별 상관관계
        report_lines.append("## Metric-wise Correlations")
        report_lines.append("")
        if 'metric_correlations' in self.results:
            for metric, data in self.results['metric_correlations'].items():
                report_lines.append(f"- **{metric.title()}**: r = {data['correlation']:.4f} (p = {data['p_value']:.4f})")
        report_lines.append("")
        
        # 트렌드 분석
        report_lines.append("## Dataset Trend Analysis")
        report_lines.append("")
        trend = self.results['trend_analysis']
        for metric, data in trend.items():
            agreement = "✓" if data['trend_agreement'] else "✗"
            report_lines.append(f"- **{metric.title()}**: {data['st_direction']} vs {data['dl_direction']} {agreement}")
        report_lines.append("")
        
        # 통계적 유의성
        report_lines.append("## Statistical Significance")
        report_lines.append("")
        sig = self.results['significance_testing']
        report_lines.append(f"- **Paired t-test**: t = {sig['paired_ttest']['t_statistic']:.4f}, p = {sig['paired_ttest']['p_value']:.4f}")
        report_lines.append(f"- **Effect size (Cohen's d)**: {sig['effect_size']:.4f}")
        report_lines.append("")
        
        # 일관성 메트릭
        report_lines.append("## Consistency Metrics")
        report_lines.append("")
        cons = self.results['consistency_metrics']
        report_lines.append(f"- **Mean Absolute Error**: {cons['mae']:.4f}")
        report_lines.append(f"- **Root Mean Square Error**: {cons['rmse']:.4f}")
        report_lines.append(f"- **Mean Absolute Percentage Error**: {cons['mape']:.2f}%")
        report_lines.append(f"- **Agreement Score**: {cons['agreement_score']:.4f}")
        report_lines.append("")
        
        # 결론
        report_lines.append("## Conclusions")
        report_lines.append("")
        
        # 상관관계 기반 결론
        pearson_r = corr['pearson']['correlation']
        if pearson_r > 0.8:
            corr_strength = "strong positive"
        elif pearson_r > 0.5:
            corr_strength = "moderate positive"  
        elif pearson_r > 0.2:
            corr_strength = "weak positive"
        elif pearson_r > -0.2:
            corr_strength = "negligible"
        else:
            corr_strength = "negative"
            
        report_lines.append(f"1. **Correlation**: {corr_strength} correlation between ST and DL metrics (r = {pearson_r:.3f})")
        
        # 일치도 기반 결론
        agreement = cons['agreement_score']
        if agreement > 0.8:
            agreement_level = "high"
        elif agreement > 0.6:
            agreement_level = "moderate"
        else:
            agreement_level = "low"
            
        report_lines.append(f"2. **Agreement**: {agreement_level} rank agreement between methods ({agreement:.1%})")
        
        # 유의성 기반 결론
        p_value = sig['paired_ttest']['p_value']
        if p_value < 0.05:
            report_lines.append(f"3. **Significance**: Statistically significant differences detected (p = {p_value:.4f})")
        else:
            report_lines.append(f"3. **Significance**: No statistically significant differences (p = {p_value:.4f})")
            
        report_content = "\n".join(report_lines)
        
        # 보고서 저장
        with open(self.output_dir / 'numerical_consistency_summary.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        print("✓ Summary report saved to numerical_consistency_summary.md")
        
    def run_full_analysis(self):
        """전체 분석 실행"""
        print("Starting Numerical Consistency Analysis...")
        print("="*60)
        
        self.load_evaluation_results()
        self.extract_metric_values()
        self.create_comparison_dataframe()
        self.compute_correlation_analysis()
        self.analyze_dataset_trends()
        self.statistical_significance_testing()
        self.consistency_metrics()
        self.generate_visualizations()
        self.save_results()
        
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved in: {self.output_dir}")
        

def main():
    """메인 함수"""
    analyzer = NumericalConsistencyAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()