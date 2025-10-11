# Reviewer Comments

## Major Issues

Major issues that need to be resolved in the revision:

### 1. Inconsistent reported numbers — unify and verify all values
Several key statistics appear with different values in different places (for example, reported Cohen's κ and correlation coefficients appear as different numbers in the response letter, abstract, body, and conclusion). The authors must verify and unify all reported numbers across the manuscript and response letter. If different computation methods were used, explicitly state and compare them. Provide the raw summary tables or scripts used to compute each aggregated number.

### 2. Reproducibility and methodological detail are insufficient
(1) Embedding model and hyperparameters: Specify exactly which embedding model(s) were used (model name and checkpoint), tokenizer settings, any fine-tuning, and preprocessing steps (lowercasing, stopword removal, lemmatization, frequency thresholds).

(2) LLM call details: Report the exact LLM model/version used, date of calls, API parameters (temperature, top_p, max_tokens), number of evaluations per item, aggregation method (mean/median), and whether sampling was deterministic. Include pseudo-code or actual scripts showing how continuous LLM scores were converted to categorical labels and then to Cohen's κ.

(3) Dataset construction & availability: For any datasets built (Wikipedia-derived or otherwise), provide the crawl date, query seeds or page lists, filtering rules, and several example documents per topic. Ideally release the dataset or provide code to reproduce it.

### 3. Metric definitions and normalization are unclear
Some custom metrics (e.g., Semantic Coherence, Semantic Distinctiveness, SemDiv with α/β/γ/λ parameters) lack full mathematical specification, parameter values, and value ranges. Provide precise formulas, show chosen parameter values with justification, and include a small worked example (toy data) to demonstrate calculation end-to-end.

### 4. Discussion of LLM evaluation limitations & robustness tests
Acknowledge bias and hallucination risks of LLMs and test robustness: run sensitivity analyses across different temperature settings, prompt variants, and ideally across more than one LLM. Present how much scores vary and discuss mitigation strategies (e.g., multi-model consensus, prompt ensembling).

## Minor Issues

Minor issues that recommended fixes:

### (1) Table and figure clarity
Improve table layout (clear headers, consistent column alignment) and add a concise caption and a one-sentence reader takeaway for each table.
For t-SNE plots, add hyperparameters (perplexity, learning rate, seed) and consider adding a UMAP comparison and/or multiple seeds to demonstrate stability.

### (2) Terminology and abbreviation consistency
Ensure every abbreviation (e.g., NPMI, IRBO) is defined upon first use. Run a global search to catch missed items.

### (3) Appendix code and pseudo-code
The appendix pseudocode is useful — complement it with minimal runnable examples for key routines: computing semantic metrics, LLM scoring, and Cohen's κ aggregation.

### (4) Language polish
The manuscript reads well overall but contains occasional redundancy and formatting issues. A final round of native-English proofreading (or professional editing) is recommended.

### (5) Conclusion alignment
Make sure the conclusion's numeric claims and limitation statements match the body and abstract. Explicitly list main limitations and concrete future work items (multi-lingual extension, low-resource behavior, reducing LLM cost).

## Additional Comments

1. Please add at least one simple public real-world dataset, because relying solely on three Wikipedia-based synthetic datasets limits external validity.

2. Clarify Related Work regarding Ref. 15. You cite LLM-based evaluation in Related Work, but §2.2 frames the "existing metrics" as statistical. Please simply state how your metric differs and why it is more important.

3. Specify metric details in §3.3. State exactly which neural embedding model you use (only the 384-dimensional setting is given), how λw is chosen or learned, and what values or selection process you used for α, β, γ.

4. Fix numeric inconsistencies. For example, the Conclusion reports κ = 0.89, which differs from other sections.

---

# 리뷰어 코멘트 (한글 번역)

## 주요 이슈

개정에서 반드시 해결해야 할 주요 문제들:

### 1. 보고된 수치의 불일치 — 모든 값을 통일하고 검증하라
여러 핵심 통계치가 다른 장소에서 다른 값으로 나타납니다 (예: Cohen's κ와 상관계수가 답변서, 초록, 본문, 결론에서 서로 다른 숫자로 보고됨). 저자들은 원고와 답변서 전체에 걸쳐 보고된 모든 숫자를 검증하고 통일해야 합니다. 만약 다른 계산 방법이 사용되었다면, 명시적으로 설명하고 비교하십시오. 각 집계된 숫자를 계산하는 데 사용된 원시 요약 테이블이나 스크립트를 제공하십시오.

### 2. 재현성과 방법론적 세부사항이 불충분함
(1) 임베딩 모델 및 하이퍼파라미터: 어떤 임베딩 모델이 사용되었는지 정확히 명시하십시오 (모델 이름과 체크포인트), 토크나이저 설정, 파인튜닝 여부, 전처리 단계 (소문자 변환, 불용어 제거, 표제어 추출, 빈도 임계값).

(2) LLM 호출 세부사항: 사용된 정확한 LLM 모델/버전, 호출 날짜, API 파라미터 (temperature, top_p, max_tokens), 항목당 평가 횟수, 집계 방법 (평균/중앙값), 샘플링의 결정론적 여부를 보고하십시오. 연속형 LLM 점수가 범주형 레이블로 변환되고 Cohen's κ로 계산되는 과정을 보여주는 의사코드 또는 실제 스크립트를 포함하십시오.

(3) 데이터셋 구축 및 가용성: 구축된 모든 데이터셋 (Wikipedia 기반 또는 기타)에 대해, 크롤링 날짜, 쿼리 시드 또는 페이지 목록, 필터링 규칙, 토픽당 여러 예시 문서를 제공하십시오. 이상적으로는 데이터셋을 공개하거나 재현할 수 있는 코드를 제공하십시오.

### 3. 메트릭 정의와 정규화가 불명확함
일부 커스텀 메트릭 (예: Semantic Coherence, Semantic Distinctiveness, α/β/γ/λ 파라미터를 가진 SemDiv)이 완전한 수학적 명세, 파라미터 값, 값 범위가 부족합니다. 정확한 수식을 제공하고, 선택된 파라미터 값을 정당화와 함께 보여주며, 계산을 처음부터 끝까지 보여주는 작은 예시 (toy data)를 포함하십시오.

### 4. LLM 평가의 한계 및 강건성 테스트에 대한 논의
LLM의 편향과 환각 위험을 인정하고 강건성을 테스트하십시오: 다양한 temperature 설정, 프롬프트 변형에 걸쳐 민감도 분석을 실행하고, 이상적으로는 둘 이상의 LLM에 걸쳐 테스트하십시오. 점수가 얼마나 변동하는지 제시하고 완화 전략 (예: 다중 모델 합의, 프롬프트 앙상블)을 논의하십시오.

## 부차적 이슈

수정이 권장되는 부차적 문제들:

### (1) 표와 그림의 명확성
표 레이아웃을 개선하고 (명확한 헤더, 일관된 열 정렬), 각 표에 간결한 캡션과 독자를 위한 한 문장 핵심 요약을 추가하십시오.
t-SNE 플롯의 경우, 하이퍼파라미터 (perplexity, learning rate, seed)를 추가하고 안정성을 보여주기 위해 UMAP 비교 및/또는 여러 시드 추가를 고려하십시오.

### (2) 용어와 약어의 일관성
모든 약어 (예: NPMI, IRBO)가 처음 사용될 때 정의되도록 하십시오. 누락된 항목을 찾기 위해 전역 검색을 실행하십시오.

### (3) 부록 코드와 의사코드
부록의 의사코드는 유용합니다 — 핵심 루틴에 대한 최소한의 실행 가능한 예제로 보완하십시오: 시맨틱 메트릭 계산, LLM 채점, Cohen's κ 집계.

### (4) 언어 다듬기
원고는 전반적으로 잘 읽히지만 가끔 중복과 형식 문제가 있습니다. 영어 원어민 교정 (또는 전문 편집)의 최종 라운드가 권장됩니다.

### (5) 결론 정렬
결론의 수치 주장과 한계 진술이 본문 및 초록과 일치하는지 확인하십시오. 주요 한계점과 구체적인 향후 연구 항목 (다국어 확장, 저자원 상황에서의 동작, LLM 비용 절감)을 명시적으로 나열하십시오.

## 추가 코멘트

1. 세 개의 Wikipedia 기반 합성 데이터셋에만 의존하는 것은 외부 타당성을 제한하므로, 적어도 하나의 간단한 공개 실제 데이터셋을 추가하십시오.

2. Ref. 15와 관련하여 관련 연구를 명확히 하십시오. 관련 연구에서 LLM 기반 평가를 인용하지만, §2.2는 "기존 메트릭"을 통계적인 것으로 프레임합니다. 귀하의 메트릭이 어떻게 다르고 왜 더 중요한지 간단히 설명하십시오.

3. §3.3에서 메트릭 세부사항을 명시하십시오. 사용하는 정확한 신경 임베딩 모델 (384차원 설정만 제공됨), λw가 어떻게 선택되거나 학습되는지, α, β, γ에 사용된 값 또는 선택 프로세스를 명시하십시오.

4. 수치 불일치를 수정하십시오. 예를 들어, 결론에서 κ = 0.89로 보고되는데, 이는 다른 섹션과 다릅니다.

