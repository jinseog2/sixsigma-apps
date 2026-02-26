import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="파레토 차트", layout="wide")
st.title("파레토 차트 (Pareto Chart)")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_stb(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

st.markdown("""
> **파레토 원칙 (Pareto Rule)**: 결과의 80%는 20%의 소수 원인에 의해 결정된다.
>
> 파레토 차트는 문제가 되는 요인들을 중요도 순으로 나타내어, 해결에 집중해야 할 **핵심 소수(Vital Few)**를 식별하는 도구입니다.
""")

st.divider()

# --- 데이터 소스 선택 ---
data_source = st.sidebar.selectbox(
    "데이터 소스",
    ["STB 1.1 (Issue Vote)", "직접 입력", "CSV 업로드"]
)

if data_source == "STB 1.1 (Issue Vote)":
    df = load_stb('STB_1.1.csv')
    if df is None:
        st.error("STB_1.1.csv를 찾을 수 없습니다.")
        st.stop()
    cause_col = 'Cause'
    effect_col = 'Effect'
    st.sidebar.success(f"STB 1.1 로드 ({len(df)}행)")

elif data_source == "직접 입력":
    st.sidebar.markdown("**항목과 빈도를 입력하세요**")
    raw_causes = st.sidebar.text_area(
        "항목 (쉼표 구분)",
        "배송지연, 제품파손, 오배송, 포장불량, CS응대, 기타"
    )
    raw_effects = st.sidebar.text_area(
        "빈도 (쉼표 구분)",
        "82, 65, 48, 13, 11, 9"
    )
    try:
        causes = [c.strip() for c in raw_causes.split(",") if c.strip()]
        effects = [float(v.strip()) for v in raw_effects.split(",") if v.strip()]
        if len(causes) != len(effects):
            st.error("항목 수와 빈도 수가 일치해야 합니다.")
            st.stop()
        df = pd.DataFrame({'Cause': causes, 'Effect': effects})
        cause_col = 'Cause'
        effect_col = 'Effect'
    except ValueError:
        st.error("빈도는 숫자여야 합니다.")
        st.stop()

else:
    uploaded = st.sidebar.file_uploader("CSV 파일 업로드", type="csv")
    if not uploaded:
        st.info("CSV 파일을 업로드하세요. 항목 열과 빈도 열이 포함되어야 합니다.")
        st.stop()
    df = pd.read_csv(uploaded)
    cause_col = st.sidebar.selectbox("항목 열", df.columns)
    effect_col = st.sidebar.selectbox("빈도 열", df.columns)

cutoff_pct = st.sidebar.slider("누적 기준선 (%)", min_value=50, max_value=95, value=80, step=5)

if st.button("▶ 파레토 차트 생성"):
    # Sort by effect descending
    df_sorted = df[[cause_col, effect_col]].copy()
    df_sorted[effect_col] = pd.to_numeric(df_sorted[effect_col], errors='coerce')
    df_sorted = df_sorted.dropna().sort_values(effect_col, ascending=False).reset_index(drop=True)
    total = df_sorted[effect_col].sum()

    if total == 0:
        st.error("빈도의 합이 0입니다.")
        st.stop()

    df_sorted['비율(%)'] = (df_sorted[effect_col] / total * 100).round(1)
    df_sorted['누적비율(%)'] = df_sorted['비율(%)'].cumsum().round(1)

    # --- Pareto Chart ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df_sorted[cause_col],
            y=df_sorted[effect_col],
            name="빈도",
            marker_color='#2e75b6',
            text=df_sorted[effect_col],
            textposition='outside'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=df_sorted[cause_col],
            y=df_sorted['누적비율(%)'],
            name="누적비율 (%)",
            mode='lines+markers+text',
            line=dict(color='#e74c3c', width=2.5),
            marker=dict(size=8),
            text=[f"{v}%" for v in df_sorted['누적비율(%)']],
            textposition='top center',
            textfont=dict(size=10)
        ),
        secondary_y=True
    )

    # Cutoff line
    fig.add_hline(
        y=cutoff_pct, line_dash="dash", line_color="green",
        annotation_text=f"{cutoff_pct}% 기준선",
        annotation_position="top right",
        secondary_y=True
    )

    fig.update_layout(
        title=f"Pareto Chart",
        xaxis_title="항목",
        height=520,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    fig.update_yaxes(title_text="빈도", secondary_y=False)
    fig.update_yaxes(title_text="누적비율 (%)", range=[0, 110], secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # --- Vital Few ---
    vital_few = df_sorted[df_sorted['누적비율(%)'] <= cutoff_pct]
    if len(vital_few) == 0:
        vital_few = df_sorted.head(1)

    st.subheader(f"핵심 소수 (Vital Few) — 누적 {cutoff_pct}% 이내")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(
            vital_few[[cause_col, effect_col, '비율(%)', '누적비율(%)']],
            hide_index=True, use_container_width=True
        )
    with col2:
        st.metric("Vital Few 항목 수", f"{len(vital_few)} / {len(df_sorted)}")
        st.metric("Vital Few 누적 비율", f"{vital_few['누적비율(%)'].iloc[-1]:.1f}%")

    st.divider()

    st.subheader("전체 데이터 요약")
    st.dataframe(
        df_sorted[[cause_col, effect_col, '비율(%)', '누적비율(%)']],
        hide_index=True, use_container_width=True
    )

    # --- 해석 가이드 ---
    with st.expander("📖 해석 방법"):
        st.markdown(f"""
**작성 방법:**
1. Issue/불량/결함 등의 데이터 또는 Vote를 수집
2. 각 항목 또는 유사 항목으로 그룹화하여 발생 빈도를 집계
3. 빈도가 큰 순서대로 나열하여 막대그림으로 표현
4. 누적 빈도와 백분율을 계산하여 꺾은선으로 도시

**해석 기준:**
- 어떤 항목이 가장 빈도가 높은가?
- 가장 빈도가 높은 항목은 전체 중 어느 정도의 비중을 갖는가?
- 만일 특정 비중의 Issue를 해결하기 위해서는 어떤 항목까지 고려해야 하는가?

**현재 결과:**
- **"{df_sorted[cause_col].iloc[0]}"** 항목이 가장 높은 빈도({df_sorted[effect_col].iloc[0]})를 기록
- 전체의 약 {df_sorted['비율(%)'].iloc[0]}%를 차지
- {cutoff_pct}% 이상 개선을 위해서는 **{', '.join(vital_few[cause_col].tolist())}** 항목을 고려해야 함
        """)
