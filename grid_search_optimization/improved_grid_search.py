import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from itertools import product
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

class ImprovedSemanticMetricsCalculator:
    """개선된 semantic metrics 계산 클래스"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def calculate_semantic_coherence(self, topic_keywords, alpha=0.4, damping=0.85, threshold=0.3):
        """개선된 Semantic Coherence 계산"""
        if len(topic_keywords) < 2:
            return 0.0
            
        try:
            # 키워드 임베딩 생성
            embeddings = self.model.encode(topic_keywords, normalize_embeddings=True)
            n = len(embeddings)
            
            # 계층적 유사도 계산
            direct_sim = cosine_similarity(embeddings)
            indirect_sim = np.dot(direct_sim, direct_sim) / n
            hierarchical_sim = 0.7 * direct_sim + 0.3 * indirect_sim
            
            # PageRank 가중치 계산
            G = nx.Graph()
            for i in range(n):
                for j in range(i + 1, n):
                    if direct_sim[i, j] > threshold:
                        G.add_edge(i, j, weight=direct_sim[i, j])
            
            if len(G.nodes()) == 0:
                # 그래프가 비어있으면 균등 가중치 사용
                weights = np.ones(n) / n
            else:
                pagerank_weights = nx.pagerank(G, alpha=damping)
                weights = np.array([pagerank_weights.get(i, 1.0/n) for i in range(n)])
            
            # 중요도 행렬 계산
            importance_matrix = np.outer(weights, weights)
            
            # Semantic Coherence 계산 (α 파라미터 적용)
            coherence = np.sum(hierarchical_sim * importance_matrix) / np.sum(importance_matrix)
            
            # α 파라미터로 가중치 적용
            weighted_coherence = alpha * coherence + (1 - alpha) * np.mean(hierarchical_sim)
            
            return float(np.clip(weighted_coherence, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error in coherence calculation: {e}")
            return 0.0
    
    def calculate_semantic_distinctiveness(self, topic_centroids):
        """개선된 Semantic Distinctiveness 계산"""
        if len(topic_centroids) < 2:
            return 0.0
            
        try:
            n_topics = len(topic_centroids)
            distinctiveness_scores = []
            
            for i in range(n_topics):
                for j in range(i + 1, n_topics):
                    cos_sim = np.dot(topic_centroids[i], topic_centroids[j]) / (
                        np.linalg.norm(topic_centroids[i]) * np.linalg.norm(topic_centroids[j])
                    )
                    sd = (1 - cos_sim) / 2
                    distinctiveness_scores.append(sd)
            
            return float(np.mean(distinctiveness_scores))
            
        except Exception as e:
            print(f"Error in distinctiveness calculation: {e}")
            return 0.0
    
    def calculate_semantic_diversity(self, topic_centroids, topic_assignments, gamma=0.2):
        """개선된 Semantic Diversity 계산"""
        try:
            # Semantic component
            semantic_div = self.calculate_semantic_distinctiveness(topic_centroids)
            
            # Distribution component
            if len(topic_assignments) == 0:
                return semantic_div
                
            topic_counts = np.bincount(topic_assignments, minlength=len(topic_centroids))
            P_T = topic_counts / np.sum(topic_counts)
            H_T = -np.sum(P_T * np.log(P_T + 1e-12))
            H_max = np.log(len(topic_centroids))
            distribution_div = H_T / H_max if H_max > 0 else 0.0
            
            # γ 파라미터로 가중치 적용
            diversity = gamma * semantic_div + (1 - gamma) * distribution_div
            
            return float(np.clip(diversity, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error in diversity calculation: {e}")
            return 0.0

class ImprovedGridSearchOptimizer:
    """개선된 Grid Search 최적화 클래스"""
    
    def __init__(self):
        self.metrics_calc = ImprovedSemanticMetricsCalculator()
        self.sample_topics = self._create_improved_sample_topics()
        self.sample_llm_scores = self._create_improved_llm_scores()
        
    def _create_improved_sample_topics(self):
        """개선된 샘플 토픽 데이터 생성 (더 다양한 주제)"""
        return [
            # Machine Learning (높은 일관성)
            ["machine", "learning", "algorithm", "neural", "network", "deep", "model", "training"],
            # Biology (중간 일관성)
            ["biology", "cell", "dna", "protein", "genetic", "organism", "evolution", "species"],
            # History (낮은 일관성)
            ["history", "ancient", "civilization", "culture", "tradition", "war", "empire", "kingdom"],
            # Physics (높은 일관성)
            ["physics", "quantum", "energy", "particle", "theory", "force", "matter", "space"],
            # Chemistry (중간 일관성)
            ["chemistry", "molecule", "reaction", "compound", "element", "chemical", "bond", "atomic"],
            # Literature (낮은 일관성)
            ["literature", "novel", "poetry", "writing", "author", "story", "book", "character"],
            # Technology (높은 일관성)
            ["technology", "computer", "software", "hardware", "digital", "internet", "data", "system"],
            # Art (중간 일관성)
            ["art", "painting", "sculpture", "design", "creative", "artist", "gallery", "museum"]
        ]
    
    def _create_improved_llm_scores(self):
        """개선된 LLM 점수 생성 (토픽 품질에 따른 차별화)"""
        # 토픽별 품질 점수 (일관성 높은 토픽이 높은 점수)
        quality_scores = [0.9, 0.7, 0.5, 0.9, 0.7, 0.5, 0.9, 0.7]  # 8개 토픽
        
        # 노이즈 추가
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, len(quality_scores))
        final_scores = np.clip(np.array(quality_scores) + noise, 0.0, 1.0)
        
        return final_scores
    
    def evaluate_parameters(self, alpha, gamma, damping, threshold):
        """개선된 파라미터 평가"""
        try:
            # Semantic metrics 계산
            coherence_scores = []
            distinctiveness_scores = []
            diversity_scores = []
            
            # 각 토픽에 대해 계산
            for i, topic_keywords in enumerate(self.sample_topics):
                # Coherence 계산
                sc = self.metrics_calc.calculate_semantic_coherence(
                    topic_keywords, alpha, damping, threshold
                )
                coherence_scores.append(sc)
            
            # Topic centroids 계산
            topic_centroids = []
            for topic_keywords in self.sample_topics:
                embeddings = self.metrics_calc.model.encode(topic_keywords, normalize_embeddings=True)
                centroid = np.mean(embeddings, axis=0)
                topic_centroids.append(centroid)
            
            # Distinctiveness 계산
            sd = self.metrics_calc.calculate_semantic_distinctiveness(topic_centroids)
            
            # Diversity 계산
            topic_assignments = np.random.randint(0, len(self.sample_topics), 100)
            semdiv = self.metrics_calc.calculate_semantic_diversity(topic_centroids, topic_assignments, gamma)
            
            # LLM 상관계수 계산 (개선된 방식)
            correlation = self._calculate_improved_correlation(coherence_scores, self.sample_llm_scores)
            
            # 차별화 범위 계산
            discrimination = self._calculate_improved_discrimination(coherence_scores)
            
            # 안정성 계산
            stability = self._calculate_stability(coherence_scores)
            
            # 전체 점수 (개선된 가중치)
            overall_score = self._calculate_overall_score(correlation, discrimination, stability)
            
            return {
                'correlation': correlation,
                'discrimination': discrimination,
                'stability': stability,
                'overall_score': overall_score,
                'coherence_scores': coherence_scores,
                'distinctiveness': sd,
                'diversity': semdiv
            }
            
        except Exception as e:
            print(f"Error evaluating parameters: {e}")
            return {
                'correlation': 0.0,
                'discrimination': 0.0,
                'stability': 0.0,
                'overall_score': 0.0,
                'coherence_scores': [],
                'distinctiveness': 0.0,
                'diversity': 0.0
            }
    
    def _calculate_improved_correlation(self, semantic_scores, llm_scores):
        """개선된 상관계수 계산"""
        if len(semantic_scores) != len(llm_scores) or len(semantic_scores) < 2:
            return 0.0
        
        # Spearman 상관계수 계산
        correlation, p_value = spearmanr(semantic_scores, llm_scores)
        
        # p-value가 유의하지 않으면 상관계수를 0으로 설정
        if p_value > 0.05:
            correlation = 0.0
            
        return float(correlation)
    
    def _calculate_improved_discrimination(self, scores):
        """개선된 차별화 범위 계산"""
        if len(scores) < 2:
            return 0.0
        
        # 표준편차 기반 차별화 계산
        std_dev = np.std(scores)
        mean_score = np.mean(scores)
        
        # 정규화된 차별화 점수
        discrimination = std_dev / (mean_score + 1e-6)
        
        return float(np.clip(discrimination, 0.0, 1.0))
    
    def _calculate_stability(self, scores):
        """안정성 계산"""
        if len(scores) < 2:
            return 1.0
        
        # 변동계수의 역수로 안정성 계산
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if mean_score == 0:
            return 1.0
        
        cv = std_score / mean_score
        stability = 1.0 / (1.0 + cv)  # 0-1 범위로 정규화
        
        return float(stability)
    
    def _calculate_overall_score(self, correlation, discrimination, stability):
        """전체 점수 계산 (개선된 가중치)"""
        # 상관계수는 절댓값 사용
        abs_correlation = abs(correlation)
        
        # 가중치: 상관계수 50%, 차별화 30%, 안정성 20%
        overall_score = 0.5 * abs_correlation + 0.3 * discrimination + 0.2 * stability
        
        return float(overall_score)
    
    def run_improved_grid_search(self):
        """개선된 Grid Search 실행"""
        print("Starting Improved Grid Search...")
        
        # 파라미터 범위 정의 (더 세밀하게)
        alpha_values = [0.3, 0.4, 0.5, 0.6, 0.7]
        gamma_values = [0.1, 0.2, 0.3]
        damping_values = [0.75, 0.80, 0.85, 0.90]
        threshold_values = [0.2, 0.3, 0.4]
        
        results = []
        total_combinations = len(alpha_values) * len(gamma_values) * len(damping_values) * len(threshold_values)
        
        print(f"Total combinations to test: {total_combinations}")
        
        for i, (alpha, gamma, damping, threshold) in enumerate(product(alpha_values, gamma_values, damping_values, threshold_values)):
            if (i + 1) % 20 == 0:
                print(f"Testing combination {i+1}/{total_combinations}: α={alpha}, γ={gamma}, damping={damping}, threshold={threshold}")
            
            # 파라미터 평가
            evaluation = self.evaluate_parameters(alpha, gamma, damping, threshold)
            
            results.append({
                'alpha': alpha,
                'gamma': gamma,
                'damping': damping,
                'threshold': threshold,
                'correlation': evaluation['correlation'],
                'discrimination': evaluation['discrimination'],
                'stability': evaluation['stability'],
                'overall_score': evaluation['overall_score'],
                'distinctiveness': evaluation['distinctiveness'],
                'diversity': evaluation['diversity']
            })
        
        return pd.DataFrame(results)
    
    def find_optimal_parameters(self, results_df):
        """최적 파라미터 찾기 (개선된 조건)"""
        # 더 엄격한 조건 적용
        valid_results = results_df[
            (results_df['correlation'] >= 0.3) &  # 상관계수 조건
            (results_df['discrimination'] >= 0.1) &  # 차별화 조건
            (results_df['overall_score'] >= 0.2)  # 전체 점수 조건
        ]
        
        if len(valid_results) > 0:
            best_idx = valid_results['overall_score'].idxmax()
            return valid_results.loc[best_idx]
        else:
            # 조건을 만족하는 것이 없으면 전체에서 최고점
            best_idx = results_df['overall_score'].idxmax()
            return results_df.loc[best_idx]
    
    def analyze_improved_results(self, results_df):
        """개선된 결과 분석"""
        print("\n" + "="*60)
        print("IMPROVED GRID SEARCH RESULTS")
        print("="*60)
        
        # 최적 파라미터 찾기
        optimal = self.find_optimal_parameters(results_df)
        
        print(f"\nOptimal Parameters:")
        print(f"  α (coherence weight): {optimal['alpha']:.1f}")
        print(f"  γ (diversity weight): {optimal['gamma']:.1f}")
        print(f"  PageRank damping: {optimal['damping']:.2f}")
        print(f"  Similarity threshold: {optimal['threshold']:.1f}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Correlation with LLM: {optimal['correlation']:.3f}")
        print(f"  Discrimination range: {optimal['discrimination']:.3f}")
        print(f"  Stability score: {optimal['stability']:.3f}")
        print(f"  Overall score: {optimal['overall_score']:.3f}")
        print(f"  Distinctiveness: {optimal['distinctiveness']:.3f}")
        print(f"  Diversity: {optimal['diversity']:.3f}")
        
        # 통계 요약
        print(f"\nSummary Statistics:")
        print(f"  Mean correlation: {results_df['correlation'].mean():.3f} ± {results_df['correlation'].std():.3f}")
        print(f"  Mean discrimination: {results_df['discrimination'].mean():.3f} ± {results_df['discrimination'].std():.3f}")
        print(f"  Mean overall score: {results_df['overall_score'].mean():.3f} ± {results_df['overall_score'].std():.3f}")
        
        # 상위 10개 결과
        print(f"\nTop 10 Results:")
        top_10 = results_df.nlargest(10, 'overall_score')
        print(top_10[['alpha', 'gamma', 'damping', 'threshold', 'correlation', 'discrimination', 'overall_score']].to_string(index=False))
        
        return optimal

if __name__ == "__main__":
    # 개선된 Grid Search 실행
    optimizer = ImprovedGridSearchOptimizer()
    results_df = optimizer.run_improved_grid_search()
    
    # 결과 분석
    optimal_params = optimizer.analyze_improved_results(results_df)
    
    # 결과 저장
    results_df.to_csv('grid_search_optimization/improved_grid_search_results.csv', index=False)
    print(f"\nImproved results saved to 'grid_search_optimization/improved_grid_search_results.csv'")
    
    print(f"\nImproved Grid Search completed successfully!")
