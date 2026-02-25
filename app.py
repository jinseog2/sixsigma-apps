import streamlit as st

st.set_page_config(page_title="6시그마 MBB 분석 도구", layout="wide", page_icon="📊")

st.title("6시그마 MBB 분석 도구")
st.markdown("---")

st.markdown("""
### DMAIC 단계별 분석 도구

왼쪽 사이드바에서 분석 도구를 선택하세요.

| 단계 | 도구 | 설명 |
|------|------|------|
| **기초** | 확률분포 계산기 | 정규분포, 이항분포, 포아송분포, 초기하분포 |
| **Measure** | Gage R&R | Type 1, Crossed, Nested, Attribute Agreement |
| **Measure** | 공정능력 분석 | Cp, Cpk, Pp, Ppk, 정규성 검정 |
| **Analyze** | 가설검정 | t-test, 카이제곱, 정규성, 등분산 검정 |
| **Analyze** | ANOVA | One-Way, Two-Way ANOVA |
| **Analyze** | 회귀분석 | 상관분석, 단순/다중 선형 회귀 |
| **Improve** | DOE | 완전요인, 부분요인, 반응표면법 |
| **Control** | 관리도 | Xbar-R, I-MR, P, NP, C, U 관리도 |

---
*6시그마 MBB 교육 과정 실습용*
""")
