# 데이터셋 타당성 분석: "실세계 데이터셋" 요구의 적절성 검토

## 📋 리뷰어 코멘트

> "Please add at least one simple public real-world dataset, because relying solely on three Wikipedia-based synthetic datasets limits external validity."

## 🔍 논문의 데이터셋 구성 방법 상세 분석

### 실제 구성 방법 (논문 §3.1 기반)

**데이터 출처**: Wikipedia API (실제 텍스트)

**3개 데이터셋 구성**:

| 데이터셋 | 문서 수 | 토픽 수 | 실제 토픽 예시 | Inter-topic Similarity |
|---------|---------|---------|---------------|----------------------|
| **Distinct** | 3,445 | 15 | • Evolution theory (636 docs)<br>• Classical mechanics (405 docs)<br>• Molecular biology (375 docs) | 0.21 |
| **Similar** | 2,719 | 15 | • Artificial intelligence (366 docs)<br>• Robotics (309 docs)<br>• Neural networks (254 docs) | 0.48 |
| **More Similar** | 3,444 | 15 | • Big data analytics (506 docs)<br>• Speech recognition (480 docs)<br>• AI (365 docs) | 0.67 |

### 핵심 발견: "Synthetic" vs "Real-world"의 정의 혼란

**논문의 표현**: "synthetic datasets"
**실제 내용**: **실제 Wikipedia 문서를 사용**, 다만 **의도적으로 토픽 중복도를 조절한 구성**

---

## 💡 "Synthetic"의 정확한 의미 분석

### 이 연구에서 "Synthetic"의 의미:

**NOT**:
- ❌ 인공적으로 생성된 가짜 텍스트
- ❌ GPT로 만든 합성 문서
- ❌ 템플릿 기반 자동 생성 문서

**YES**:
- ✅ **실제 Wikipedia 문서** 사용
- ✅ **의도적으로 선택/구성**한 토픽 조합
- ✅ **통제된 실험 설계**를 위한 데이터셋 구성

### 유사 사례: 학술 연구의 "Controlled Dataset"

이는 다음과 같은 경우와 유사합니다:

```
Example 1: 이미지 분류 연구
- ImageNet의 실제 이미지를 사용
- 하지만 "쉬운 클래스 5개", "중간 클래스 5개", "어려운 클래스 5개"로 구성
- 이것을 "synthetic dataset"이라고 부르지 않음
- → "Controlled subset" 또는 "Curated dataset"

Example 2: 감정 분석 연구
- Twitter 실제 트윗 사용
- 하지만 positive/neutral/negative를 균등 비율로 샘플링
- → "Balanced dataset" 또는 "Stratified dataset"
```

---

## 🎯 본 연구의 데이터셋: 정확한 분류

### 올바른 표현:

| 잘못된 표현 | 올바른 표현 | 이유 |
|------------|-----------|------|
| ❌ "Synthetic datasets" | ✅ "**Controlled real-world datasets**" | 실제 Wikipedia 문서 사용 |
| ❌ "Artificially generated" | ✅ "**Curated from Wikipedia**" | 선택/구성했을 뿐 생성 안 함 |
| ❌ "Simulated data" | ✅ "**Stratified by topic similarity**" | 실제 데이터를 계층화 |

### 데이터 특성:

**Real-world 특성** (이미 가지고 있음):
- ✅ 실제 Wikipedia 문서 (진짜 텍스트)
- ✅ 자연스러운 언어 (인간이 작성)
- ✅ 다양한 도메인 (과학, 기술, AI 등)
- ✅ 실제 사용되는 백과사전 콘텐츠

**Controlled 특성** (실험 설계를 위해):
- ✅ 의도적 토픽 선택 (중복도 조절)
- ✅ 단계적 유사도 설계 (0.21 → 0.48 → 0.67)
- ✅ 균형잡힌 문서 수 (~3,000 docs per dataset)

---

## 🔬 리뷰어 요구의 타당성 분석

### 리뷰어가 "실세계 데이터셋"을 요구한 이유 (추정):

#### 가능성 1: 용어 혼동 ❓
- "Synthetic"이라는 표현 때문에 **가짜 데이터**로 오해
- 실제로는 **실제 Wikipedia 문서**인데 표현이 모호함

#### 가능성 2: External Validity 우려 ✅ (타당한 우려)
- Wikipedia만 사용 → 다른 도메인에서도 작동하는지?
- 학술 논문, 뉴스, 소셜미디어 등에서도 유효한지?

#### 가능성 3: Task-specific Dataset 부재 ⚠️
- 일반적인 topic modeling 연구는 **특정 응용 분야**에서 검증
  - 예: 의료 문서 (PubMed)
  - 예: 뉴스 기사 (NYT, Reuters)
  - 예: 학술 논문 (ArXiv)
- 본 연구는 **범용 평가 지표** 제안 → 다양한 도메인 검증 필요

---

## 📊 "External Validity" 개념 정리

### External Validity란?

**정의**: 연구 결과가 다른 상황, 다른 집단, 다른 시간에도 일반화될 수 있는가?

### 본 연구의 External Validity 평가:

**현재 상황**:
- ✅ 실제 텍스트 사용 (Wikipedia)
- ✅ 다양한 도메인 커버 (과학, 기술, AI)
- ❌ **단일 소스** (Wikipedia only)
- ❌ **단일 스타일** (백과사전 스타일)
- ❌ **단일 언어** (영어)

**잠재적 한계**:
1. **도메인 한계**: Wikipedia 스타일만
   - 뉴스, 학술 논문, 블로그 등 다른 텍스트 유형에서는?
2. **언어 한계**: 영어만
   - 다른 언어에서도 작동하는가?
3. **응용 한계**: 범용 토픽만
   - 전문 도메인 (의료, 법률)에서도 유효한가?

---

## 🎯 논문 저자의 논리적 반박 가능성

### 반박 포인트 1: "이미 실세계 데이터입니다"

**주장**:
> "Our datasets are NOT synthetic in the traditional sense. We use **real Wikipedia documents** - actual text written by humans for real-world encyclopedia articles. We merely **curated** these documents to create controlled conditions for systematic evaluation."

**근거**:
- Wikipedia는 세계에서 가장 큰 실제 백과사전
- 실제 사람들이 실제 목적으로 작성한 텍스트
- 단지 **실험 통제를 위해 선택적으로 구성**했을 뿐

### 반박 포인트 2: "통제된 실험이 필수입니다"

**주장**:
> "Our research objective requires **controlled conditions** to systematically evaluate metric effectiveness. Using existing 'real-world' datasets (e.g., 20 Newsgroups, Reuters) would introduce **uncontrolled variables** that confound our evaluation."

**근거**:
- **연구 목적**: 의미론 지표 vs 통계 지표의 **체계적 비교**
- **필요 조건**: 토픽 중복도를 **단계적으로 조절**한 데이터셋
- **기존 데이터셋 문제**:
  - 20 Newsgroups: 토픽 중복도가 **고정**되어 있음
  - Reuters: 토픽 분포가 **불균형** (뉴스 특성)
  - PubMed: **전문 도메인**으로 일반화 어려움

### 반박 포인트 3: "방법론적 우선순위"

**주장**:
> "The **validity of our semantic metrics** is not dependent on dataset source, but on **controlled experimental design**. Our three-tier similarity structure (0.21 → 0.48 → 0.67) is the critical factor, not whether documents come from Wikipedia vs. other sources."

**근거**:
- **평가 대상**: Semantic metrics의 **discriminative power**
- **핵심 변수**: Inter-topic similarity (통제 필요)
- **부차적 변수**: 텍스트 출처 (통제 불필요)

---

## 🔍 리뷰어 요구의 진정한 의미 재해석

### 리뷰어가 *실제로* 원하는 것:

#### Option A: 용어 정정 ✍️
- "Synthetic" → "Controlled real-world" or "Curated Wikipedia"
- 데이터가 **실제 문서**임을 명확히

#### Option B: 일반화 가능성 증명 🌍
- **다른 텍스트 유형**에서도 작동 확인
- 예: 뉴스, 학술 논문, 소셜미디어
- **단, 통제 조건 유지 어려움** (토픽 중복도 조절 불가)

#### Option C: 보편성 검증 📈
- **기존 벤치마크 데이터셋**에서도 테스트
- 예: 20 Newsgroups, Reuters-21578
- **한계**: 토픽 중복도 통제 불가 → 본 연구의 핵심 설계 손상

---

## 💡 최선의 대응 전략

### 전략 1: 용어 명확화 + 정당화 (추천 ⭐⭐⭐⭐⭐)

**수정 내용**:
1. "Synthetic datasets" → **"Controlled real-world datasets curated from Wikipedia"**
2. Methodology 섹션에 명확히 설명:
   ```
   "While we use the term 'controlled datasets' to emphasize our
   systematic design, we stress that all documents are **real Wikipedia
   articles written by humans**. We curated these documents specifically
   to create three tiers of topic overlap (0.21, 0.48, 0.67), which is
   essential for evaluating the discriminative power of semantic metrics."
   ```

3. **Justification 추가**:
   ```
   "We chose Wikipedia as our data source because:
   (1) It provides diverse, high-quality real-world text across domains
   (2) It enables controlled selection of topic combinations
   (3) Using existing benchmark datasets (e.g., 20 Newsgroups) would
       not allow the graduated similarity structure essential to our
       experimental design"
   ```

### 전략 2: 보조 검증 실험 추가 (선택적 ⭐⭐⭐)

**타협안**: 기존 벤치마크 1개 추가하되, **한계 명시**

예시:
```
"To demonstrate broader applicability, we additionally tested our
metrics on 20 Newsgroups dataset. However, we note that this dataset
has **fixed topic overlap** (avg similarity = 0.34) and thus cannot
validate our key finding regarding discriminative power across varying
similarity levels. Results show [X], consistent with our controlled
experiments."
```

**장점**:
- ✅ 리뷰어 요구 충족
- ✅ 일반화 가능성 일부 입증

**단점**:
- ❌ 핵심 실험 설계 희석
- ❌ 추가 실험 필요 (시간/노력)

### 전략 3: Discussion에 한계점 명시 (필수 ⭐⭐⭐⭐⭐)

**추가 내용**:
```
"Limitations:
While our datasets use real Wikipedia documents, the **exclusive use
of encyclopedia-style text** may limit generalizability to other genres
(e.g., conversational text, technical jargon-heavy domains). Future
work should validate our metrics on diverse text types while maintaining
controlled similarity structures."
```

---

## 📋 구체적 수정 권장사항

### 📝 수정 1: Abstract
**Before**:
> "experiments with three synthetic datasets"

**After**:
> "experiments with three **controlled real-world datasets curated from Wikipedia**, representing varying degrees of topic overlap"

### 📝 수정 2: §3.1 제목
**Before**:
> "3.1 Experimental Data Construction"

**After**:
> "3.1 Controlled Dataset Construction from Wikipedia"

### 📝 수정 3: §3.1 첫 문단
**Before**:
> "This study employs three carefully constructed synthetic datasets..."

**After**:
> "This study employs three **controlled real-world datasets** constructed from Wikipedia articles. While we systematically curated these datasets to represent varying degrees of topic overlap, **all documents are authentic Wikipedia articles** written by human contributors. This approach enables controlled experimental conditions while maintaining ecological validity through the use of real-world text."

### 📝 수정 4: Justification 단락 추가 (§3.1 끝)

**Add**:
```
"We selected Wikipedia as our data source for three reasons:
(1) **Authenticity**: Wikipedia provides high-quality, peer-reviewed
    real-world text across diverse domains
(2) **Control**: The broad topic coverage enables systematic selection
    of topic combinations with graduated similarity levels
(3) **Reproducibility**: Wikipedia's public accessibility ensures
    experimental reproducibility

While existing benchmark datasets (e.g., 20 Newsgroups, Reuters-21578)
are widely used in topic modeling research, they possess **fixed topic
overlap patterns** that would not allow us to evaluate metric performance
across our three-tier similarity structure (0.21 → 0.48 → 0.67). Our
controlled curation approach addresses this methodological requirement."
```

### 📝 수정 5: Limitations (§6)

**Add**:
```
"Although our datasets comprise authentic Wikipedia articles, the
**exclusive use of encyclopedia-style text** represents a limitation.
Future research should validate our semantic metrics on diverse text
genres (news articles, academic papers, social media) while maintaining
controlled similarity structures to ensure broader generalizability."
```

---

## 🎓 학술적 관점: 리뷰어 vs 저자

### 리뷰어 관점 (External Validity 강조)
- ✅ 다양한 도메인에서 검증 필요
- ✅ 실용적 적용 가능성 증명 필요
- ⚠️ "Synthetic" 표현에 오해 가능성

### 저자 관점 (Internal Validity 강조)
- ✅ 통제된 실험이 우선
- ✅ 변수 통제를 위한 설계 정당
- ⚠️ 용어 선택 불명확

### 균형잡힌 해결책:
1. **용어 명확화** (synthetic → controlled real-world)
2. **정당화 강화** (왜 Wikipedia인가? 왜 curated인가?)
3. **한계 인정** (encyclopedia-style 한정)
4. **(선택) 보조 검증** (1개 벤치마크 추가)

---

## ✅ 결론: 리뷰어 요구의 타당성 평가

### 요구의 타당성: ⭐⭐⭐ (3/5) - 부분적으로 타당

**타당한 부분**:
- ✅ External validity 우려는 정당
- ✅ "Synthetic" 용어는 오해 소지
- ✅ 다양한 도메인 검증은 이상적

**과도한 부분**:
- ❌ 데이터가 이미 "real-world" (Wikipedia 실제 문서)
- ❌ 통제된 실험 설계의 필요성 간과
- ❌ 기존 벤치마크는 연구 설계에 부적합

### 최종 권장 조치:

**필수 (Must)**:
1. ✅ 용어 변경: "synthetic" → "controlled real-world curated from Wikipedia"
2. ✅ Justification 추가: 왜 Wikipedia? 왜 curated?
3. ✅ Limitation 명시: Encyclopedia-style text only

**권장 (Should)**:
4. ✅ Discussion에 generalizability 논의 추가

**선택 (Optional)**:
5. ⚠️ 기존 벤치마크 1개 추가 (시간 있으면)
   - 20 Newsgroups 또는 Reuters-21578
   - 단, **한계 명확히** 명시

---

## 📌 핵심 메시지

> **"우리는 이미 실세계 데이터를 사용하고 있습니다. 단지 실험 통제를 위해 의도적으로 구성했을 뿐입니다. 이는 'synthetic'가 아니라 'controlled curation'입니다."**

리뷰어의 요구는 **용어 명확화와 정당화 강화**로 충분히 해결 가능하며, 반드시 새로운 데이터셋을 추가할 필요는 없습니다.
