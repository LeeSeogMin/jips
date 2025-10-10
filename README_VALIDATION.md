# Multi-LLM Manuscript Validation Framework

논문의 타당성을 검증하기 위한 4개 LLM 기반 독립 분석 프레임워크

## 개요

이 프레임워크는 4개의 최신 LLM을 사용하여 topic modeling 평가 연구 논문을 독립적으로 분석하고, 최종적으로 Claude를 통해 종합 검증을 수행합니다.

**사용 모델 (2025-10 최신 버전)**:
- **OpenAI GPT-4.1** (gpt-4.1) - 1M 토큰 컨텍스트, 향상된 코딩/추론 성능
- **Anthropic Claude Sonnet 4.5** (claude-sonnet-4-5-20250929) - 최첨단 코딩/에이전트 작업
- **xAI Grok 4** (grok-4-0709) - 256K 토큰 컨텍스트, 실시간 검색 통합
- **Google Gemini 2.5 Flash** (gemini-2.5-flash-preview-09-2025) - 향상된 품질/효율성

## 구조

```
llm_analyzers/
├── __init__.py              # 패키지 초기화
├── base_analyzer.py         # 기본 분석 클래스
├── openai_analyzer.py       # OpenAI GPT-4 분석기
├── anthropic_analyzer.py    # Anthropic Claude 분석기
├── grok_analyzer.py         # xAI Grok 분석기
└── gemini_analyzer.py       # Google Gemini 분석기

validation_prompts.py        # 구조화된 프롬프트 템플릿
manuscript_validator.py      # 병렬 실행 오케스트레이터
synthesis_engine.py          # 최종 종합 분석 엔진
```

## 필요한 패키지 설치

```bash
pip install openai anthropic google-generativeai python-dotenv
```

## 환경 변수 설정

`.env` 파일에 API 키를 설정하세요:

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GROK_API_KEY=your_grok_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

## 사용 방법

### 1단계: 4개 LLM 독립 분석 실행

```bash
python manuscript_validator.py
```

이 스크립트는:
- 기존 평가 결과(pkl 파일)를 로드
- 데이터셋 정보를 수집
- 4개 LLM을 병렬로 실행하여 각각 독립적으로 분석
- 개별 결과를 `validation_results/` 디렉토리에 저장

**분석 차원** (각 LLM이 평가):
1. **방법론 타당성** (methodology_validity)
   - 선택된 메트릭의 적절성
   - 통계적 + LLM 기반 평가의 정당성

2. **통계적 엄밀성** (statistical_rigor)
   - 정규화 방법의 수학적 건전성
   - 가중치 스킴의 타당성

3. **실험 설계** (experimental_design)
   - 3가지 데이터셋의 적합성
   - 재현 가능성

4. **결과 해석** (results_interpretation)
   - 결론의 논리적 연결성
   - 대안 설명 고려

5. **기여도 평가** (contribution_assessment)
   - 연구의 novelty
   - 학술적 가치

6. **한계점 식별** (limitations_identification)
   - 주요 한계점
   - 개선 방향

### 2단계: 최종 종합 분석

```bash
python synthesis_engine.py
```

이 스크립트는:
- 4개 LLM의 분석 결과를 통합
- 합의점(consensus)과 불일치점(disagreement) 식별
- Claude를 통해 최종 종합 검증 수행
- 실행 가능한 권장사항 생성
- 최종 검증 리포트 생성

## 출력 결과

### 개별 LLM 분석 결과
`validation_results/openai_YYYYMMDD_HHMMSS.json`
`validation_results/anthropic_YYYYMMDD_HHMMSS.json`
`validation_results/grok_YYYYMMDD_HHMMSS.json`
`validation_results/gemini_YYYYMMDD_HHMMSS.json`

각 파일 구조:
```json
{
  "model": "OpenAI-gpt-4-turbo-preview",
  "analyses": {
    "methodology_validity": {
      "dimension": "methodology_validity",
      "score": 8.5,
      "confidence": 0.9,
      "reasoning": "...",
      "recommendations": ["...", "..."]
    },
    ...
  },
  "overall_assessment": {
    "overall_score": 7.8,
    "recommendation": "MINOR_REVISIONS",
    "key_strengths": [...],
    "key_weaknesses": [...],
    "avg_confidence": 0.85
  }
}
```

### 통합 결과
`validation_results/combined_analysis_YYYYMMDD_HHMMSS.json`

### 최종 검증 리포트
`validation_results/final_validation_report_YYYYMMDD_HHMMSS.json`

구조:
```json
{
  "metadata": {...},
  "consensus_metrics": {
    "overall_scores": {...},
    "mean_score": 7.5,
    "std_score": 0.8,
    "agreements": [...],
    "disagreements": [...]
  },
  "categorized_recommendations": {
    "methodology": [...],
    "statistics": [...],
    "experiments": [...],
    "writing": [...]
  },
  "claude_synthesis": "종합 분석 결과...",
  "individual_analyses": {...}
}
```

## 커스터마이징

### 논문 텍스트 제공

`manuscript_validator.py`의 `main()` 함수에서:

```python
# 방법 1: 텍스트 파일에서 로드
with open('manuscript_text.txt', 'r', encoding='utf-8') as f:
    manuscript_text = f.read()
context = validator.prepare_context(manuscript_text)

# 방법 2: 직접 입력
manuscript_text = """
Your manuscript text here...
"""
context = validator.prepare_context(manuscript_text)
```

### 특정 LLM만 실행

```python
# OpenAI와 Claude만 실행
selected = ['openai', 'anthropic']
results = validator.run_parallel_analysis(context, selected_analyzers=selected)
```

### 분석 차원 수정

`llm_analyzers/base_analyzer.py`의 `analysis_dimensions` 리스트를 수정하고, `validation_prompts.py`에 해당 프롬프트를 추가하세요.

## 예상 실행 시간

- 각 LLM 분석: 3-5분
- 병렬 실행 총 시간: 약 5-7분
- 최종 종합: 1-2분
- **전체 프로세스**: 약 10분

## 비용 예상

차원당 약 500-1000 토큰, 6개 차원 × 4개 LLM:
- OpenAI GPT-4: ~$0.50
- Anthropic Claude: ~$0.40
- Grok: ~$0.30
- Gemini: ~$0.20
- **총 예상 비용**: ~$1.50

## 문제 해결

### API 키 오류
`.env` 파일의 API 키가 올바른지 확인하세요.

### 특정 LLM 실패
개별 LLM 실패 시 나머지는 계속 실행됩니다. 실패한 모델은 `errors` 섹션에 기록됩니다.

### 메모리 부족
한 번에 하나씩 실행하려면:

```python
for name in ['openai', 'anthropic', 'grok', 'gemini']:
    results = validator.run_parallel_analysis(context, selected_analyzers=[name])
```

## 다음 단계

1. 최종 검증 리포트 검토
2. 합의점과 불일치점 분석
3. 우선순위 권장사항 구현
4. 필요시 추가 분석 수행

## 라이센스

이 프레임워크는 연구 목적으로 자유롭게 사용 가능합니다.
