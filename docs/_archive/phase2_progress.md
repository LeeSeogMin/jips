# Phase 2 진행 상황 보고서

**날짜**: 2025-10-11
**작성 시점**: Phase 2.3 실행 중 오류 발생

---

## ✅ 완료된 작업

### Phase 2.1: Statistical Metrics 재계산 (완료)

**실행 파일**: `ST_Eval.py`
**실행 환경**: Conda environment `jips` (Python 3.11.13)
**실행 시간**: 5회 반복 평가 완료

**결과 요약**:

| Dataset | NPMI | Coherence | Diversity | KLD | JSD | IRBO | Overall |
|---------|------|-----------|-----------|-----|-----|------|---------|
| Distinct Topics | 0.635 | 0.597 | 0.914 | 0.950 | 0.950 | 0.986 | 0.816 |
| Similar Topics | 0.586 | 0.631 | 0.894 | 0.900 | 0.900 | 0.970 | 0.793 |
| More Similar Topics | 0.585 | 0.622 | 0.900 | 0.901 | 0.901 | 0.963 | 0.791 |

**생성된 파일**:
- `stat_evaluation_comparison.csv`
- `stat_evaluation_details.json`

**주요 발견**:
- Distinct Topics이 가장 높은 Overall Score (0.816) 기록
- 모든 데이터셋에서 높은 Diversity (0.89-0.91) 유지
- KLD, JSD, IRBO 지표에서 우수한 성능 (0.90-0.98)

---

### Phase 2.2: Semantic Metrics 재계산 (완료)

**실행 파일**: `DL_Eval.py`
**실행 환경**: Conda environment `jips` (Python 3.11.13)
**실행 시간**: 5회 반복 평가 완료

**결과 요약**:

| Dataset | Coherence | Distinctiveness | Diversity | Semantic Integration | Overall |
|---------|-----------|-----------------|-----------|---------------------|---------|
| Distinct Topics | 0.940 | 0.205 | 0.571 | 0.131 | 0.484 |
| Similar Topics | 0.575 | 0.142 | 0.550 | 0.083 | 0.342 |
| More Similar Topics | 0.559 | 0.136 | 0.536 | 0.078 | 0.331 |

**생성된 파일**:
- Semantic evaluation results (embedded in output)

**주요 발견**:
- Distinct Topics이 매우 높은 Coherence (0.940) 기록
- Distinctiveness는 전반적으로 낮은 편 (0.13-0.20)
- Similar/More Similar Topics 간 성능 차이 미미

**CV (Coefficient of Variation) 분석**:
- 모든 지표에서 CV = 0.000% → 완벽한 일관성
- 5회 반복 평가에서 동일한 결과 산출 (재현성 확보)

---


