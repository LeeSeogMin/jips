# Evaluation Results (Root data folder: distinct/similar/more_similar)

The following results were produced by running `DL_Eval.py` (semantic) and `ST_Eval.py` (statistical) on the three synthetic datasets in the root `data` folder.

## Overall (per dataset)

| Dataset | DL Overall | ST Overall |
|--------|-----------:|-----------:|
| Distinct | 0.484 | 0.533 |
| Similar | 0.342 | 0.469 |
| More Similar | 0.331 | 0.481 |

## Coherence

| Dataset | DL | ST |
|--------|----:|----:|
| Distinct | 0.940 | 0.635 |
| Similar | 0.575 | 0.586 |
| More Similar | 0.559 | 0.585 |

## Distinctiveness

| Dataset | DL | ST |
|--------|----:|----:|
| Distinct | 0.205 | 0.203 |
| Similar | 0.142 | 0.168 |
| More Similar | 0.136 | 0.212 |

## Diversity

| Dataset | DL | ST |
|--------|----:|----:|
| Distinct | 0.571 | 0.773 |
| Similar | 0.550 | 0.627 |
| More Similar | 0.536 | 0.625 |

## Notes
- Semantic Integration is reported by DL only:
  - Distinct: 0.131
  - Similar: 0.083
  - More Similar: 0.078
