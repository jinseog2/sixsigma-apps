import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
import os

st.set_page_config(page_title="ANOVA 분석", layout="wide")
st.title("분산분석 (ANOVA)")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_stb(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

analysis_type = st.sidebar.selectbox("분석 유형", ["One-Way ANOVA", "Two-Way ANOVA"])

# ==============================================================
# One-Way ANOVA
# ==============================================================
if analysis_type == "One-Way ANOVA":
    st.header("One-Way ANOVA")

    data_source = st.sidebar.radio("데이터", ["STB 3.6 (Platform 비교)", "직접 입력", "CSV 업로드"])

    if data_source == "STB 3.6 (Platform 비교)":
        df_raw = load_stb('STB_3.6.csv')
        if df_raw is None:
            st.stop()
        # Melt: Platform1/2/3를 Group, Value로 변환
        plat_cols = [c for c in df_raw.columns if c.startswith('Platform')]
        df = pd.melt(df_raw, value_vars=plat_cols, var_name='Group', value_name='Value').dropna()
        grp_col, val_col = 'Group', 'Value'
        st.sidebar.success(f"STB 3.6 로드 ({len(df)}행, {len(plat_cols)}그룹)")
    elif data_source == "직접 입력":
        n_groups = st.sidebar.number_input("그룹 수", value=3, min_value=2, max_value=10)
        all_data = []
        for i in range(n_groups):
            raw = st.sidebar.text_area(f"그룹 {i+1}",
                f"{', '.join([str(round(50+i+np.random.randn(),1)) for _ in range(8)])}", key=f"grp_{i}")
            for v in [float(x.strip()) for x in raw.split(",") if x.strip()]:
                all_data.append({'Group': f'G{i+1}', 'Value': v})
        df = pd.DataFrame(all_data)
        grp_col, val_col = 'Group', 'Value'
    else:
        uploaded = st.sidebar.file_uploader("CSV", type="csv")
        if not uploaded:
            st.stop()
        df = pd.read_csv(uploaded)
        val_col = st.sidebar.selectbox("측정값 컬럼", df.columns)
        grp_col = st.sidebar.selectbox("그룹 컬럼", df.columns)

    alpha = st.sidebar.number_input("유의수준 (α)", value=0.05, format="%.2f")

    if st.button("▶ 분석 실행", type="primary"):
        groups = df[grp_col].unique()
        group_data = [df[df[grp_col] == g][val_col].dropna().values.astype(float) for g in groups]

        f_stat, p_val = stats.f_oneway(*group_data)
        grand_mean = np.mean(np.concatenate(group_data))
        n_total = sum(len(g) for g in group_data)
        k = len(groups)

        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_data)
        ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in group_data)
        ss_total = ss_between + ss_within
        df_between, df_within = k - 1, n_total - k
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within

        st.subheader("ANOVA 테이블")
        st.dataframe(pd.DataFrame({
            'Source': ['Between', 'Within', 'Total'],
            'DF': [df_between, df_within, n_total - 1],
            'SS': [f"{ss_between:.4f}", f"{ss_within:.4f}", f"{ss_total:.4f}"],
            'MS': [f"{ms_between:.4f}", f"{ms_within:.4f}", ""],
            'F': [f"{f_stat:.4f}", "", ""],
            'p-value': [f"{p_val:.4f}", "", ""]
        }), hide_index=True, use_container_width=True)

        if p_val < alpha:
            st.error(f"p = {p_val:.4f} < {alpha} → 적어도 하나의 그룹 평균이 다름")
        else:
            st.success(f"p = {p_val:.4f} ≥ {alpha} → 유의한 차이 없음")

        st.subheader("그룹별 기술 통계")
        desc = [{'그룹': str(g), 'n': len(gd), '평균': f"{np.mean(gd):.4f}", '표준편차': f"{np.std(gd,ddof=1):.4f}"}
                for g, gd in zip(groups, group_data)]
        st.dataframe(pd.DataFrame(desc), hide_index=True, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            for g, gd in zip(groups, group_data):
                fig.add_trace(go.Box(y=gd, name=str(g)))
            fig.update_layout(title="그룹별 박스플롯", height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            means = [np.mean(gd) for gd in group_data]
            cis = [1.96 * np.std(gd, ddof=1) / np.sqrt(len(gd)) for gd in group_data]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[str(g) for g in groups], y=means,
                error_y=dict(type='data', array=cis, visible=True),
                mode='markers', marker=dict(size=10, color='#2e75b6')))
            fig.add_hline(y=grand_mean, line_dash="dash", line_color="red",
                          annotation_text=f"Grand Mean={grand_mean:.2f}")
            fig.update_layout(title="평균 비교 (95% CI)", height=400)
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# Two-Way ANOVA
# ==============================================================
else:
    st.header("Two-Way ANOVA")

    data_source = st.sidebar.radio("데이터", ["STB 3.7-1 (Fan/Chamber)", "CSV 업로드"])

    if data_source == "STB 3.7-1 (Fan/Chamber)":
        df = load_stb('STB_3.7-1.csv')
        if df is None:
            st.stop()
        st.sidebar.success(f"STB 3.7-1 로드 ({len(df)}행)")
        factor1_col = st.sidebar.selectbox("인자 1", df.columns, index=df.columns.get_loc('PictureMode') if 'PictureMode' in df.columns else 0)
        factor2_col = st.sidebar.selectbox("인자 2", df.columns, index=df.columns.get_loc('Chamber No') if 'Chamber No' in df.columns else 1)
        response_col = st.sidebar.selectbox("반응변수", df.columns, index=df.columns.get_loc('Response') if 'Response' in df.columns else 2)
    else:
        uploaded = st.sidebar.file_uploader("CSV", type="csv")
        if not uploaded:
            st.stop()
        df = pd.read_csv(uploaded)
        factor1_col = st.sidebar.selectbox("인자 1", df.columns, index=0)
        factor2_col = st.sidebar.selectbox("인자 2", df.columns, index=1)
        response_col = st.sidebar.selectbox("반응변수", df.columns, index=2)

    if st.button("▶ 분석 실행"):
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        df_clean = df[[factor1_col, factor2_col, response_col]].dropna()
        df_clean.columns = ['F1', 'F2', 'Y']
        df_clean['F1'] = df_clean['F1'].astype(str)
        df_clean['F2'] = df_clean['F2'].astype(str)

        model = ols('Y ~ C(F1) + C(F2) + C(F1):C(F2)', data=df_clean).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        st.subheader("ANOVA 테이블")
        st.dataframe(anova_table.round(4), use_container_width=True)

        st.subheader("교호작용 플롯")
        interaction = df_clean.groupby(['F1', 'F2'])['Y'].mean().reset_index()
        fig = go.Figure()
        for f2_val in df_clean['F2'].unique():
            sub = interaction[interaction['F2'] == f2_val]
            fig.add_trace(go.Scatter(x=sub['F1'], y=sub['Y'], mode='lines+markers',
                                     name=f"{factor2_col}={f2_val}"))
        fig.update_layout(xaxis_title=factor1_col, yaxis_title="평균", height=450)
        st.plotly_chart(fig, use_container_width=True)
