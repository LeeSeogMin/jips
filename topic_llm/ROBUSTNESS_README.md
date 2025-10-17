# LLM Robustness Analysis Framework

3개의 독립적인 분석 스크립트로 구성된 LLM 평가 강건성 검증 프레임워크

---

## 📁 파일 구조

```
topic_llm/
├── AI_eval.py                      # 메인 평가 스크립트 (이미 존재)
├── llm_metadata_logger.py          # 메타데이터 수집 및 로깅
├── cohens_kappa_analysis.py        # Cohen's κ 계산 (inter-rater reliability)
├── llm_robustness_analysis.py      # 강건성 테스트 (temperature, prompt, multi-LLM)
└── ROBUSTNESS_README.md            # 이 파일
```

---

## 1. LLM Metadata Logger

### 목적
모든 LLM API 호출의 메타데이터를 수집하고 문서화합니다.

### 주요 기능
- ✅ 모델명/버전 기록
- ✅ API 파라미터 (temperature, max_tokens, top_p) 기록
- ✅ 호출 타임스탬프
- ✅ 평가 설계 파라미터 (datasets, metrics, aggregation method)
- ✅ Deterministic sampling 여부 (temperature=0)
- ✅ JSON 형식 저장 (재분석 가능)

### 사용 방법

#### 기본 사용 예제
```python
from topic_llm.llm_metadata_logger import LLMMetadataLogger

# 로거 생성
logger = LLMMetadataLogger(output_dir="llm_evaluation_logs")

# 평가 설계 파라미터 설정
logger.set_evaluation_parameters(
    datasets=["distinct", "similar", "more_similar"],
    num_topics_per_dataset={"distinct": 15, "similar": 15, "more_similar": 16},
    metrics_evaluated=["coherence", "distinctiveness", "diversity", "semantic_integration"],
    aggregation_method="mean",
    num_iterations=1
)

# API 호출 로깅
logger.log_api_call(
    evaluator_name="AnthropicEval",
    model_name="claude-sonnet-4-5-20250929",
    metric="coherence_batch",
    prompt="Rate the coherence...",
    response="<topic1>0.92</topic1>...",
    api_parameters={
        "temperature": 0,
        "max_tokens": 1024,
        "top_p": None,
    },
    dataset_name="distinct",
    topic_indices=list(range(15))
)

# 세션 종료 및 저장
metadata_path = logger.finalize_session()

# 리포트 생성
from topic_llm.llm_metadata_logger import create_metadata_report
report = create_metadata_report(metadata_path)
print(report)
```

#### AI_eval.py에 통합하기

```python
# AI_eval.py 상단에 추가
from topic_llm.llm_metadata_logger import LLMMetadataLogger

def run_llm_evaluation() -> pd.DataFrame:
    # 로거 초기화
    logger = LLMMetadataLogger()
    logger.set_evaluation_parameters(
        datasets=["distinct", "similar", "more_similar"],
        num_topics_per_dataset={"distinct": 15, "similar": 15, "more_similar": 16},
        metrics_evaluated=["coherence", "distinctiveness", "diversity", "semantic_integration"],
        aggregation_method="mean",
        num_iterations=1
    )

    # 기존 평가 코드...

    # 세션 종료
    logger.finalize_session()
```

### 출력 예시

**Metadata JSON 파일** (`metadata_20250117_120000.json`):
```json
{
  "session_id": "20250117_120000",
  "start_time": "2025-01-17T12:00:00",
  "total_api_calls": 12,
  "deterministic": true,
  "aggregation_method": "mean",
  "api_calls": [
    {
      "timestamp": "2025-01-17T12:00:05",
      "evaluator": "AnthropicEval",
      "model": "claude-sonnet-4-5-20250929",
      "metric": "coherence_batch",
      "dataset": "distinct",
      "api_parameters": {
        "temperature": 0,
        "max_tokens": 1024
      }
    }
  ]
}
```

**Report Markdown**:
```markdown
# LLM Evaluation Metadata Report

**Session ID**: 20250117_120000
**Date**: 2025-01-17T12:00:00
**Total API Calls**: 12

## API Configuration

- **Model**: claude-sonnet-4-5-20250929
- **Temperature**: 0
- **Max Tokens**: 1024
- **Deterministic**: Yes (temperature=0)

## Evaluation Design

- **Datasets**: distinct, similar, more_similar
- **Metrics**: coherence, distinctiveness, diversity, semantic_integration
- **Aggregation Method**: mean
- **Iterations per Item**: 1
```

---

## 2. Cohen's Kappa Analysis

### 목적
연속형 LLM 점수를 범주형 레이블로 변환하고 Cohen's κ를 계산하여 inter-rater reliability를 측정합니다.

### 주요 기능
- ✅ 연속형 → 범주형 변환 (customizable thresholds)
- ✅ Cohen's κ 계산 (unweighted, linear, quadratic)
- ✅ Confusion matrix 생성
- ✅ Multi-rater analysis (3+ LLMs)
- ✅ 의사코드 + 실제 구현 포함

### 사용 방법

#### 단일 분석
```python
from topic_llm.cohens_kappa_analysis import analyze_llm_agreement
import numpy as np

# LLM 점수 준비 (예: Anthropic vs OpenAI)
scores_anthropic = np.array([0.92, 0.85, 0.88, ...])  # 45개 topics
scores_openai = np.array([0.89, 0.82, 0.91, ...])

# 분석 실행
results = analyze_llm_agreement(
    scores_anthropic,
    scores_openai,
    thresholds=[0.60, 0.80],  # 3 categories: poor(<0.6), acceptable(0.6-0.8), excellent(>0.8)
    labels=['poor', 'acceptable', 'excellent'],
    weights=None  # Unweighted (nominal categories)
)

# 리포트 생성
from topic_llm.cohens_kappa_analysis import generate_kappa_report
report = generate_kappa_report(results)
print(report)
```

#### Multi-rater 분석
```python
from topic_llm.cohens_kappa_analysis import multi_rater_analysis

# 3개 LLM 점수
scores_dict = {
    "Anthropic": scores_anthropic,
    "OpenAI": scores_openai,
    "Gemini": scores_gemini,
}

# Pairwise kappa 계산
kappa_matrix = multi_rater_analysis(scores_dict, thresholds=[0.60, 0.80])
print(kappa_matrix)
```

### 출력 예시

```
# Cohen's Kappa Analysis Report

## Inter-Rater Reliability

**Cohen's κ**: 0.742
**Interpretation**: Substantial agreement

### Agreement Statistics

- Observed Agreement (p_o): 0.822
- Expected Agreement (p_e): 0.289
- Number of Samples: 45

### Confusion Matrix

```
         poor  acceptable  excellent
poor        5           2          0
acceptable  1          18          3
excellent   0           2         14
```

## Continuous Score Analysis

- Pearson Correlation: 0.865
- Mean Absolute Difference: 0.042

### Score Distributions

- LLM 1: μ=0.857, σ=0.068
- LLM 2: μ=0.840, σ=0.095
```

### Cohen's κ 해석 (Landis & Koch, 1977)
- **< 0.00**: Poor (random보다 나쁨)
- **0.00 - 0.20**: Slight agreement
- **0.21 - 0.40**: Fair agreement
- **0.41 - 0.60**: Moderate agreement
- **0.61 - 0.80**: Substantial agreement
- **0.81 - 1.00**: Almost perfect agreement

---

## 3. LLM Robustness Analysis

### 목적
Temperature, prompt variation, multi-LLM 등 다양한 조건에서 LLM 평가의 강건성을 테스트합니다.

### 주요 기능
- ✅ Temperature sensitivity analysis (0.0, 0.3, 0.7, 1.0)
- ✅ Prompt variation testing (5 semantic variations)
- ✅ Multi-LLM comparison (Anthropic, OpenAI, Gemini)
- ✅ Multi-iteration stability (5-10 runs)
- ✅ Statistical analysis (std, CV, ANOVA)
- ✅ Mitigation strategy recommendations

### ⚠️ 주의사항

**API 호출 비용이 높습니다!**
- Temperature sweep: 4 × 12 = 48 calls
- Prompt variations: 5 × 12 = 60 calls
- Multi-LLM: 3 × 12 = 36 calls
- Multi-iteration: 5 × 12 = 60 calls
- **Total: ~200-300 API calls**
- **Estimated cost: $0.60 - $4.50**

### 사용 방법

#### 전체 분석 (주의: 비용 발생)
```python
from topic_llm.llm_robustness_analysis import RobustnessAnalyzer
from topic_llm.anthropic_topic_evaluator import TopicEvaluatorLLM as AnthropicEval

# Analyzer 초기화
analyzer = RobustnessAnalyzer(output_dir="robustness_results")

# 1. Temperature sensitivity
analyzer.temperature_sensitivity_analysis(
    AnthropicEval,
    temperatures=[0.0, 0.3, 0.7, 1.0],
    dataset="distinct"
)

# 2. Prompt variations
analyzer.prompt_variation_analysis(
    AnthropicEval,
    dataset="distinct",
    num_variations=5
)

# 3. Multi-LLM comparison
from topic_llm.openai_topic_evaluator import TopicEvaluatorLLM as OpenAIEval
from topic_llm.gemini_topic_evaluator import TopicEvaluatorLLM as GeminiEval

analyzer.multi_llm_comparison(
    [AnthropicEval, OpenAIEval, GeminiEval],
    ["Anthropic", "OpenAI", "Gemini"],
    dataset="distinct"
)

# 4. Multi-iteration stability
analyzer.multi_iteration_stability(
    AnthropicEval,
    dataset="distinct",
    num_iterations=5
)

# 5. Mitigation strategies
analyzer.propose_mitigation_strategies()

# Generate report
report = analyzer.generate_report(output_path="robustness_report.md")
print(report)

# Save results
analyzer.save_results("robustness_results.pkl")
```

#### 단계별 테스트 (권장)
```python
# 먼저 단일 dataset으로 테스트
analyzer = RobustnessAnalyzer()

# 1. Temperature만 테스트 (12 API calls)
temp_results = analyzer.temperature_sensitivity_analysis(
    AnthropicEval,
    temperatures=[0.0, 0.3, 0.7, 1.0],
    dataset="distinct"
)

# 결과 확인
print(f"Mean range: {temp_results['statistics']['mean_range']:.3f}")
print(f"Mean CV: {temp_results['statistics']['mean_cv']:.3f}")

# 2. 결과가 만족스러우면 전체 분석 실행
# ...
```

### 출력 예시

```markdown
# LLM Robustness Analysis Report

**Generated**: 2025-01-17 14:30:00

## 1. Temperature Sensitivity Analysis

**Temperatures tested**: [0.0, 0.3, 0.7, 1.0]

### Results

| Temperature | Mean Score | Std Dev | CV |
|-------------|------------|---------|------|
| 0.0 | 0.887 | 0.068 | 0.077 |
| 0.3 | 0.891 | 0.072 | 0.081 |
| 0.7 | 0.875 | 0.095 | 0.109 |
| 1.0 | 0.862 | 0.118 | 0.137 |

**Mean Range**: 0.029 (low sensitivity ✓)
**Mean CV**: 0.033

## 2. Prompt Variation Analysis

**Variations tested**: 5

| Variation | Mean Score | Std Dev | CV |
|-----------|------------|---------|------|
| variation_1 | 0.887 | 0.068 | 0.077 |
| variation_2 | 0.892 | 0.065 | 0.073 |
| variation_3 | 0.883 | 0.071 | 0.080 |
| variation_4 | 0.889 | 0.069 | 0.078 |
| variation_5 | 0.885 | 0.070 | 0.079 |

**Mean Range**: 0.009 (very stable ✓)
**Mean CV**: 0.010

## 3. Multi-LLM Comparison

**Models tested**: Anthropic, OpenAI, Gemini

### Inter-Model Agreement

- **Mean Pairwise Correlation**: 0.823 (high ✓)
- **Mean Topic Variance**: 0.005 (low ✓)

## 4. Mitigation Strategies

### Temperature

- **Recommendation**: Temperature-insensitive (robust)
- **Suggested Value**: 0.0
- **Rationale**: Minimal variation across temperatures, use deterministic (T=0)

### Multi-Model Consensus

- **Recommendation**: High inter-model agreement
- **Method**: Use any single model (cost-effective)
- **Rationale**: Strong correlation (r=0.823) between models

### Ensemble Methods (Recommended)

- Temperature ensemble: Average scores from T∈{0, 0.3, 0.7}
- Model ensemble: Average scores from Anthropic + OpenAI + Gemini
- Prompt ensemble: Average scores from 3-5 prompt variations
- Iteration ensemble: Average scores from 5+ independent runs

**Expected Improvement**: 15-30% reduction in score variance

## Conclusions

1. **Bias Mitigation**: Use ensemble methods to reduce single-model bias
2. **Hallucination Risk**: Temperature=0 minimizes hallucination
3. **Score Stability**: Multi-iteration averaging improves reliability
4. **Best Practice**: Combine 2+ LLMs with prompt ensemble for critical evaluations
```

---

## 📊 실행 워크플로우

### 논문을 위한 권장 워크플로우

```
1. AI_eval.py 실행 (baseline 결과)
   └─> 12 API calls per LLM
   └─> Output: baseline scores

2. llm_metadata_logger.py 통합
   └─> AI_eval.py에 로깅 추가
   └─> Output: metadata JSON + report

3. cohens_kappa_analysis.py 실행
   └─> 2+ LLM 결과로 Cohen's κ 계산
   └─> Output: inter-rater reliability report

4. llm_robustness_analysis.py 실행 (단계별)
   4a. Temperature sensitivity (12 calls)
       └─> Temperature=0 사용 정당화

   4b. Prompt variations (60 calls, 선택사항)
       └─> Prompt stability 검증

   4c. Multi-LLM comparison (36 calls)
       └─> Model agreement 검증

   4d. Multi-iteration stability (60 calls, 선택사항)
       └─> Score stability 검증

5. Mitigation strategies 문서화
   └─> 최종 논문에 포함
```

### 최소 요구사항 (비용 절감)

```
✅ 필수:
- AI_eval.py + metadata logging (12 calls)
- Cohen's kappa (2 LLMs, 24 calls total)
- Temperature sensitivity (12 calls)

Total: ~48 API calls, ~$0.15-0.75

📝 논문에 포함:
- Baseline scores with full metadata
- Cohen's κ showing inter-model agreement
- Temperature sensitivity showing robustness
```

### 전체 분석 (논문 완성도 최대화)

```
✅ 전체:
- AI_eval.py + metadata (12 calls)
- Cohen's kappa (3 LLMs, 36 calls total)
- Temperature sensitivity (48 calls)
- Prompt variations (60 calls)
- Multi-iteration (60 calls)

Total: ~216 API calls, ~$0.65-3.25

📝 논문에 포함:
- 완전한 robustness analysis
- Multiple mitigation strategies
- Comprehensive validation
```

---

## 📖 논문 작성 가이드

### Reviewer Requirement (2) 대응

**요구사항**:
> "Report the exact LLM model/version used, date of calls, API parameters
> (temperature, top_p, max_tokens), how many times each item was evaluated,
> and how results were aggregated (mean/median), and whether sampling was deterministic."

**대응 방법**:

1. **llm_metadata_logger.py** 사용
   - 모든 메타데이터를 자동 수집
   - JSON 형식으로 저장 (재현 가능)
   - 논문 Methods 섹션에 포함

**논문 예시**:
```
We used Claude Sonnet 4.5 (claude-sonnet-4-5-20250929) with deterministic
sampling (temperature=0, max_tokens=1024). Each topic was evaluated once
with batch API calls. All API parameters and timestamps are documented
in our metadata logs (see Supplementary Materials).
```

2. **cohens_kappa_analysis.py** 사용
   - 의사코드와 실제 구현 모두 제공
   - Confusion matrix 포함
   - 논문 Methods 섹션에 포함

**논문 예시**:
```
To assess inter-rater reliability, we converted continuous LLM scores [0, 1]
to categorical labels using thresholds [0.60, 0.80], yielding three categories:
poor (<0.60), acceptable (0.60-0.80), and excellent (>0.80). We computed
Cohen's κ = 0.742 (substantial agreement) between Anthropic and OpenAI
evaluators (see Algorithm 1 for pseudocode).
```

### Reviewer Requirement (4) 대응

**요구사항**:
> "Acknowledge LLM bias and hallucination risks, and test robustness: run
> sensitivity analyses across different temperature settings, prompt variations,
> and ideally across ≥2 LLMs. Show how much scores fluctuate and discuss
> mitigation strategies."

**대응 방법**:

1. **llm_robustness_analysis.py** 사용
   - Temperature sensitivity (4 values)
   - Prompt variations (5 versions)
   - Multi-LLM comparison (3 models)
   - Statistical analysis + mitigation

**논문 예시**:
```
We conducted comprehensive robustness testing across temperature settings
(0.0, 0.3, 0.7, 1.0), finding minimal score variation (mean range=0.029,
CV=0.033). Prompt variation analysis (5 semantic versions) showed high
stability (mean range=0.009). Multi-model comparison (Anthropic, OpenAI,
Gemini) revealed strong inter-model correlation (r=0.823), suggesting
low model-specific bias. We recommend temperature=0 for deterministic
evaluation and multi-model ensemble for critical assessments (see
Section 4.5 for mitigation strategies).
```

---

## 🛠️ 개발자 가이드

### 새로운 evaluator 추가하기

1. **base_topic_evaluator.py** 수정하여 temperature/prompt 설정 가능하게:

```python
class BaseLLMEvaluator(ABC):
    def __init__(self):
        self.system_prompt = """..."""
        self._temperature = 0  # Default

    def set_temperature(self, temperature: float):
        """Set temperature for API calls"""
        self._temperature = temperature

    def set_system_prompt(self, prompt: str):
        """Set custom system prompt"""
        self.system_prompt = prompt
```

2. **각 evaluator** (anthropic, openai, etc.)에서 temperature 사용:

```python
def _call_api(self, metric: str, prompt: str) -> str:
    message = self.client.messages.create(
        model=self.model,
        temperature=self._temperature,  # Use instance variable
        # ...
    )
```

### 커스텀 분석 추가하기

```python
# custom_analysis.py
from topic_llm.llm_robustness_analysis import RobustnessAnalyzer

class CustomAnalyzer(RobustnessAnalyzer):
    def custom_analysis(self, ...):
        # Your custom analysis logic
        pass
```

---

## 📚 참고문헌

- **Cohen's Kappa**: Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159-174.
- **Inter-rater Reliability**: Fleiss, J. L., Levin, B., & Paik, M. C. (2003). *Statistical methods for rates and proportions*. John Wiley & Sons.
- **LLM Evaluation**: Zheng, L., et al. (2023). Judging LLM-as-a-judge with MT-bench and chatbot arena. *NeurIPS*.

---

## ✅ 체크리스트

논문 제출 전 확인사항:

- [ ] AI_eval.py 실행 및 baseline 결과 확보
- [ ] llm_metadata_logger 통합 및 메타데이터 저장
- [ ] cohens_kappa_analysis로 inter-rater reliability 계산
- [ ] Temperature sensitivity 분석 완료
- [ ] Multi-LLM comparison 완료 (2개 이상)
- [ ] Mitigation strategies 문서화
- [ ] 모든 결과를 논문 Methods/Results 섹션에 통합
- [ ] Supplementary materials에 전체 로그 포함
- [ ] 재현 가능성 확보 (코드 + 데이터 + 메타데이터)

---

## 🆘 문제 해결

### Q1: Temperature 설정이 작동하지 않음
**A**: `base_topic_evaluator.py`와 각 evaluator에 `set_temperature()` 메서드를 추가해야 합니다. (위의 개발자 가이드 참조)

### Q2: API 비용이 너무 높음
**A**:
- 단일 dataset (distinct)만 테스트
- Temperature sweep만 실행 (가장 중요)
- Prompt variation 생략 (선택사항)
- Multi-iteration을 3회로 줄임

### Q3: Gemini evaluator가 없음
**A**:
- Anthropic + OpenAI만 사용 (충분)
- 또는 `gemini_topic_evaluator.py` 생성 필요

### Q4: 결과 재현이 안됨
**A**:
- Temperature=0 사용 확인
- 메타데이터 JSON 파일 확인
- 동일한 모델 버전 사용 확인

---

## 📧 연락처

질문이나 문제가 있으면 이슈를 제출하거나 개발자에게 연락하세요.

**Generated**: 2025-01-17
**Version**: 1.0.0
