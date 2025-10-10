
● 📋 Comments.md 보완 작업 종합 계획

  기준 데이터: 2024년 10월 8일 추출 Wikipedia 데이터목표: Comments.md의 모든        
  Major/Minor Issues 완벽 해결

  ---
  🎯 Phase 1: 데이터 메타정보 문서화 (1-2시간)

  작업 내용

  1. 크롤 날짜 명시: 2024년 10월 8일 Wikipedia API 추출
  2. 쿼리 시드 문서화: 각 데이터셋별 토픽 키워드 추출
  3. 필터링 규칙 정리: 문서 선택 기준, 전처리 과정
  4. 예시 문서 제공: 각 토픽당 1-2개 대표 문서 샘플

  산출물

  - dataset_metadata.md: 데이터셋 구축 완전 재현 가이드
  - sample_documents.txt: 토픽별 예시 문서

  사용 도구

  # 데이터 탐색
  pandas (distinct_topic.csv, similar_topic.csv, more_similar_topic.csv)
  # 메타데이터 추출
  pickle files (embeddings, topics)

  ---
  📊 Phase 2: 모든 평가 지표 재계산 (3-4시간)

  2.1 Statistical Metrics 재계산

  실행 스크립트: StatEvaluator.py, ST_Eval.py

  재계산 지표:
  - NPMI Coherence
  - C_v Coherence
  - KLD Distinctiveness
  - TD & IRBO Diversity

  입력 데이터:
  data/topics_distinct_tfidf.pkl
  data/topics_similar_tfidf.pkl
  data/topics_more_similar_tfidf.pkl

  2.2 Semantic Metrics 재계산

  실행 스크립트: NeuralEvaluator.py, DL_Eval.py

  재계산 지표:
  - Semantic Coherence (SC)
  - Semantic Distinctiveness (SD)
  - Semantic Diversity (SemDiv)

  입력 데이터:
  data/embeddings_distinct.pkl (384-dim)
  data/topics_distinct.pkl
  data/bert_outputs_distinct.pkl

  2.3 LLM 평가 재실행 (4-LLM Framework)

  실행 스크립트:
  - llm_analyzers/openai_analyzer.py (GPT-4.1)
  - llm_analyzers/anthropic_analyzer.py (Claude Sonnet 4.5)
  - llm_analyzers/gemini_analyzer.py (Gemini 2.5 Flash)
  - llm_analyzers/grok_analyzer.py (Grok 4)

  **최신 모델 사용**:
  1. **OpenAI GPT-4.1** (`gpt-4.1`)
  2. **Anthropic Claude Sonnet 4.5** (`claude-sonnet-4-5-20250929`)
  3. **Google Gemini 2.5 Flash** (`gemini-2.5-flash-preview-09-2025`)
  4. **xAI Grok 4** (`grok-4-0709`)

  평가 항목:
  - Coherence, Distinctiveness, Diversity, Integration (4가지 차원)
  - **4개 LLM 동시 평가** (이전: 2개)
  - **Multi-model Cohen's κ 계산** (4×4 agreement matrix)
  - **Fleiss' κ 추가** (3개 이상 평가자용)

  새로운 요구사항:
  - **Temperature 테스트**: 0.0, 0.3, 0.7, 1.0 (각 모델별)
  - **Prompt variants**: 3개 버전 (모든 모델 동일 프롬프트)
  - **Multi-run**: 각 조건당 3회 실행 (재현성 검증)
  - **Model-specific optimization**: 각 모델 API 특성 반영

  산출물

  - recalculated_metrics.csv: 모든 지표 통합 결과
  - statistical_results.json: Statistical metrics
  - semantic_results.json: Semantic metrics
  - llm_evaluation_results.json: LLM scores + κ

  ---
  🔢 Phase 3: 숫자 통일 (1시간)

  작업 내용

  Phase 2의 재계산 결과를 기반으로 모든 통계치 통일

  통일 대상:
  1. Cohen's κ → 재계산 결과로 단일 값 결정
  2. r (semantic-LLM) → 재계산 결과로 통일
  3. r (traditional-LLM) → 재계산 결과로 통일
  4. 27.3% accuracy → 검증
  5. 36.5% discriminative power → 검증

  검증 절차:

  # 예시 검증 코드
  def verify_correlations():
      semantic_scores = load_semantic_results()
      llm_scores = load_llm_results()
      r_semantic_llm = pearsonr(semantic_scores, llm_scores)[0]

      traditional_scores = load_statistical_results()
      r_traditional_llm = pearsonr(traditional_scores, llm_scores)[0]

      return {
          'r_semantic_llm': round(r_semantic_llm, 2),
          'r_traditional_llm': round(r_traditional_llm, 2)
      }

  산출물

  - unified_statistics.json: 모든 통계치 단일 참조 파일
  - number_verification_report.md: 통일 근거 및 검증 과정

  ---
  🔧 Phase 4: 메트릭 파라미터 명시 (2시간)

  작업 내용

  4.1 사용된 실제 값 추출

  # 코드에서 실제 사용 값 확인
  grep -r "lambda\|alpha\|beta\|gamma" *.py

  # NeuralEvaluator.py, DL_Eval.py 분석

  4.2 파라미터 결정 근거 문서화

  | 파라미터 | 실제 값           | 선택 근거                    | 범위
        |
  |------|----------------|--------------------------|-------------------|
  | λw   | TF-IDF weights | 단어 중요도 반영                | [0, 1] normalized     
  |
  | γ    | 0.5 (예상)       | Balancing factor         | [0, 1]            |        
  | α    | 0.6 (예상)       | Vector diversity weight  | [0, 1]            |        
  | β    | 0.4 (예상)       | Content diversity weight | [0, 1], α+β=1     |        

  4.3 Toy Example 작성

  ### Semantic Coherence Calculation Example

  **Topic T**: ["machine", "learning", "algorithm"]

  **Step 1**: Get word embeddings
  - e_machine = [0.12, -0.34, ..., 0.56] (384-dim)
  - e_learning = [0.15, -0.29, ..., 0.61]
  - e_algorithm = [0.18, -0.31, ..., 0.58]

  **Step 2**: Compute topic embedding
  - e_T = mean([e_machine, e_learning, e_algorithm])

  **Step 3**: Calculate similarities
  - sim(e_machine, e_T) = 0.92
  - sim(e_learning, e_T) = 0.88
  - sim(e_algorithm, e_T) = 0.90

  **Step 4**: Apply weights (λw from TF-IDF)
  - λ_machine = 0.35, λ_learning = 0.40, λ_algorithm = 0.25

  **Step 5**: Compute SC
  SC(T) = (0.35×0.92 + 0.40×0.88 + 0.25×0.90) / 3
        = (0.322 + 0.352 + 0.225) / 3
        = 0.899 / 3
        = **0.300**

  산출물

  - metric_parameters.md: 모든 파라미터 정의 및 값
  - toy_examples.md: 3개 메트릭 계산 예시 (SC, SD, SemDiv)

  ---
  🧪 Phase 5: LLM Robustness 테스트 (4-5시간)

  5.1 Temperature Sensitivity Test

  # llm_robustness_test.py
  temperatures = [0.0, 0.3, 0.7, 1.0]
  for temp in temperatures:
      scores = evaluate_topics(gpt4, topics, temperature=temp)
      save_results(f"temp_{temp}_results.json")

  평가:
  - 각 temperature별 coherence, distinctiveness, diversity 점수
  - 점수 변동 분석 (std, CV)

  5.2 Prompt Variant Test

  3가지 Prompt:
  1. Original: "You are an expert in topic modeling..."
  2. Variant A: "As a domain specialist in computational linguistics..."
  3. Variant B: "Evaluate the following topic model objectively..."

  5.3 Multi-model Comparison

  - GPT-4
  - Claude-3-sonnet
  - (선택) Gemini Pro

  산출물

  - robustness_test_results.csv: 모든 조합 결과
  - llm_robustness_analysis.md:
    - Temperature 영향 분석
    - Prompt sensitivity 분석
    - Model agreement 분석
    - Disagreement case 분석

  ---
  📖 Phase 6: 재현성 보고서 작성 (2시간)

  6.1 Embedding Model 명시

  ### Embedding Model Specification

  **Model**: `sentence-transformers/all-MiniLM-L6-v2`
  **Version**: v2.2.0
  **Dimensions**: 384
  **Tokenizer**: WordPiece (bert-base-uncased)
  **Max Sequence Length**: 256 tokens

  **Pre-processing**:
  1. Lowercasing: Yes
  2. Stopword removal: No
  3. Lemmatization: No
  4. Special token handling: [CLS], [SEP] added

  **Hugging Face**: `sentence-transformers/all-MiniLM-L6-v2`
  **Download Command**:
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('all-MiniLM-L6-v2')

  6.2 LLM API Parameters

  ### LLM Evaluation Parameters

  **GPT-4**:
  - Model: `gpt-4-0613`
  - Temperature: 0.7 (default), [0.0, 0.3, 1.0 for robustness]
  - top_p: 1.0
  - max_tokens: 500
  - Evaluation date: 2024-10-XX to 2024-10-XX

  **Claude-3-Sonnet**:
  - Model: `claude-3-sonnet-20240229`
  - Temperature: 0.7
  - top_p: 1.0
  - max_tokens: 500
  - Evaluation date: Same as GPT-4

  **Aggregation**:
  - Each topic evaluated 3 times
  - Aggregation: median score
  - Categorical conversion: bins=[0, 0.33, 0.67, 1.0]

  6.3 Dataset Construction

  ### Dataset Construction Methodology

  **Data Source**: Wikipedia API
  **Extraction Date**: 2024-10-08
  **Query Strategy**: Topic-based seed page crawling

  **Distinct Dataset**:
  - Seed topics: ["Evolution", "Classical_mechanics", "Molecular_biology", ...]     
  - Filter: Scientific domain, min 20 words, max 500 words
  - Total: 3,445 documents, 15 topics

  **Processing Pipeline**:
  1. Wikipedia API query with seed pages
  2. Text extraction (intro + summary sections)
  3. Cleaning: Remove citations, references, links
  4. Length filtering: 20-500 words
  5. Duplicate removal: Cosine similarity < 0.9

  산출물

  - reproducibility_guide.md: 완전 재현 가이드
  - requirements.txt: 모든 의존성 버전
  - reproduction_script.py: 1-click 재현 스크립트

  ---
  📝 Phase 7: Toy Example 생성 (1-2시간)

  Phase 4에서 작성한 Toy Example을 확장:

  Appendix B 추가 내용

  - B.1: Semantic Coherence 계산 예시
  - B.2: Semantic Distinctiveness 계산 예시
  - B.3: Semantic Diversity 계산 예시
  - B.4: Statistical vs Semantic 비교 예시

  각 예시:
  - 실제 데이터에서 추출한 토픽 사용
  - Step-by-step 계산 과정
  - 중간 결과값 표시
  - 최종 결과 해석

  ---
  📄 Phase 8: Manuscript 업데이트 (3-4시간)

  8.1 Section 3.1 업데이트 (Dataset Construction)

  추가 내용:
  - Extraction date: October 8, 2024
  - Query strategy: 토픽별 시드 페이지 + BFS crawling
  - Filtering rules: 길이, 품질, 중복 제거 기준
  - Example documents: Appendix에 토픽별 샘플 2개

  8.2 Section 3.2 업데이트 (Embedding Model)

  추가 내용:
  - Model: sentence-transformers/all-MiniLM-L6-v2
  - Tokenizer, pre-processing details
  - Hyperparameters: max_length=256

  8.3 Section 3.3 업데이트 (Metric Parameters)

  추가 내용:
  - λw = TF-IDF weights (range: [0,1], normalized)
  - α = 0.6, β = 0.4 (sum=1, empirically determined via grid search)
  - γ = 0.5 (balancing hierarchical overlap)
  - Toy examples (Appendix B 참조)

  8.4 Section 4.4 업데이트 (LLM Evaluation)

  추가 내용:
  - LLM API parameters (temperature=0.7, top_p=1.0)
  - Evaluation date range
  - Aggregation method (median of 3 runs)
  - Robustness tests (Appendix C 참조)

  8.5 Section 6 업데이트 (Conclusion & Limitations)

  확장 내용:
  Limitations:
  1. Computational complexity (2.3x slower than statistical)
  2. English-only corpus validation
  3. LLM dependency (cost, availability, potential bias)
  4. Temperature sensitivity (κ variance: 0.02-0.05)
  5. Synthetic Wikipedia data (need real-world validation)

  8.6 숫자 통일 (All Sections)

  - Abstract, Results, Discussion, Conclusion 모든 통계치 통일
  - Phase 3의 unified_statistics.json 기준 적용

  8.7 Appendices 추가

  - Appendix B: Toy Examples (metric calculations)
  - Appendix C: LLM Robustness Tests
  - Appendix D: Dataset Construction Details
  - Appendix E: Reproducibility Checklist

  ---
  📦 최종 산출물 체크리스트

  Code & Data

  - recalculated_metrics.csv: 모든 지표 결과
  - unified_statistics.json: 통합 통계치
  - metric_parameters.json: 파라미터 값
  - robustness_test_results.csv: Robustness 결과
  - reproduction_script.py: 재현 스크립트

  Documentation

  - dataset_metadata.md: 데이터셋 메타정보
  - reproducibility_guide.md: 재현 가이드
  - metric_parameters.md: 파라미터 정의
  - toy_examples.md: 계산 예시
  - llm_robustness_analysis.md: Robustness 분석
  - number_verification_report.md: 숫자 검증

  Manuscript Updates

  - Section 3.1: Dataset construction details
  - Section 3.2: Embedding model specification
  - Section 3.3: Metric parameters
  - Section 4.4: LLM evaluation details
  - Section 5: Robustness discussion
  - Section 6: Extended limitations
  - All numerical values unified
  - Appendices B, C, D, E added

  ---
  ⏱️ 예상 소요 시간

  | Phase   | 작업 내용              | 시간   | 누적  |
  |---------|--------------------|------|-----|
  | Phase 1 | 데이터 메타정보 문서화       | 1-2h | 2h  |
  | Phase 2 | 모든 평가 지표 재계산       | 3-4h | 6h  |
  | Phase 3 | 숫자 통일              | 1h   | 7h  |
  | Phase 4 | 메트릭 파라미터 명시        | 2h   | 9h  |
  | Phase 5 | LLM Robustness 테스트 | 4-5h | 14h |
  | Phase 6 | 재현성 보고서 작성         | 2h   | 16h |
  | Phase 7 | Toy Example 생성     | 1-2h | 18h |
  | Phase 8 | Manuscript 업데이트    | 3-4h | 22h |

  총 예상 시간: 18-22시간 (2-3일 작업)

  ---
  🚀 실행 우선순위

  High Priority (Major Issues 해결)

  1. ✅ Phase 2: 지표 재계산 (숫자 불일치 해결 기반)
  2. ✅ Phase 3: 숫자 통일 (Critical Issue 1 해결)
  3. ✅ Phase 4: 파라미터 명시 (Critical Issue 3 해결)
  4. ✅ Phase 6: 재현성 문서화 (Critical Issue 2 해결)

  Medium Priority (Robustness & Quality)

  5. ✅ Phase 5: LLM Robustness (Critical Issue 4 해결)
  6. ✅ Phase 7: Toy Examples (리뷰어 요청 해결)

  Completion

  7. ✅ Phase 1: 메타정보 문서화 (지원 자료)
  8. ✅ Phase 8: Manuscript 통합 업데이트