import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="가설검정 도구", layout="wide")
st.title("가설검정 도구")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_stb(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

test_type = st.sidebar.selectbox(
    "검정 유형",
    ["정규성 검정", "1-Sample t Test", "2-Sample t Test", "Paired t Test",
     "등분산 검정", "카이제곱 검정"]
)

def parse_data(raw_text):
    return [float(v.strip()) for v in raw_text.split(",") if v.strip()]

# ==============================================================
# 정규성 검정
# ==============================================================
if test_type == "정규성 검정":
    st.header("정규성 검정 (Anderson-Darling / Shapiro-Wilk)")

    data_source = st.sidebar.radio("데이터", ["STB 2.8 (RAM used)", "직접 입력", "CSV 업로드"])
    if data_source == "STB 2.8 (RAM used)":
        df_raw = load_stb('STB_2.8.csv')
        if df_raw is None:
            st.error("STB_2.8.csv를 찾을 수 없습니다.")
            st.stop()
        data = df_raw['RAM used'].dropna().values.astype(float)
        st.sidebar.success(f"STB 2.8 로드 ({len(data)}행)")
    elif data_source == "직접 입력":
        raw = st.sidebar.text_area("데이터 (쉼표 구분)",
            "49.2, 50.1, 51.3, 48.7, 50.5, 49.8, 50.2, 51.0, 49.5, 50.8")
        data = np.array(parse_data(raw))
    else:
        uploaded = st.sidebar.file_uploader("CSV", type="csv")
        if not uploaded:
            st.stop()
        df = pd.read_csv(uploaded)
        col = st.sidebar.selectbox("컬럼", df.columns)
        data = df[col].dropna().values.astype(float)

    alpha = st.sidebar.number_input("유의수준 (α)", value=0.05, min_value=0.01, max_value=0.20, format="%.2f")

    if st.button("▶ 검정 실행"):
        sw_stat, sw_p = stats.shapiro(data)
        ad_result = stats.anderson(data, dist='norm')

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Shapiro-Wilk 검정")
            st.metric("W 통계량", f"{sw_stat:.4f}")
            st.metric("p-value", f"{sw_p:.4f}")
            if sw_p > alpha:
                st.success(f"p = {sw_p:.4f} > {alpha} → 정규성 가정 채택")
            else:
                st.warning(f"p = {sw_p:.4f} ≤ {alpha} → 정규분포 아님")
        with col2:
            st.subheader("Anderson-Darling 검정")
            st.metric("AD 통계량", f"{ad_result.statistic:.4f}")
            ad_5pct = ad_result.critical_values[2]
            if ad_result.statistic < ad_5pct:
                st.success(f"AD < 임계값({ad_5pct:.3f}) → 정규성 채택")
            else:
                st.warning(f"AD ≥ 임계값({ad_5pct:.3f}) → 정규분포 아닐 수 있음")

        st.subheader("정규확률도")
        sorted_data = np.sort(data)
        n = len(sorted_data)
        theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=theoretical, y=sorted_data, mode='markers', marker=dict(color='#2e75b6')))
        slope, intercept = np.std(data, ddof=1), np.mean(data)
        x_line = np.array([theoretical[0], theoretical[-1]])
        fig.add_trace(go.Scatter(x=x_line, y=intercept + slope * x_line, mode='lines', line=dict(color='red', dash='dash')))
        fig.update_layout(xaxis_title="이론적 분위수", yaxis_title="실측값", height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# 1-Sample t Test
# ==============================================================
elif test_type == "1-Sample t Test":
    st.header("1-Sample t Test")

    data_source = st.sidebar.radio("데이터", ["STB 2.8 (RAM used)", "직접 입력", "CSV 업로드"])
    if data_source == "STB 2.8 (RAM used)":
        df_raw = load_stb('STB_2.8.csv')
        if df_raw is None:
            st.stop()
        data = df_raw['RAM used'].dropna().values.astype(float)
        st.sidebar.success(f"STB 2.8 로드 ({len(data)}행)")
    elif data_source == "직접 입력":
        raw = st.sidebar.text_area("데이터", "49.2, 50.1, 51.3, 48.7, 50.5, 49.8, 50.2, 51.0, 49.5, 50.8")
        data = np.array(parse_data(raw))
    else:
        uploaded = st.sidebar.file_uploader("CSV", type="csv")
        if not uploaded:
            st.stop()
        df = pd.read_csv(uploaded)
        col = st.sidebar.selectbox("컬럼", df.columns)
        data = df[col].dropna().values.astype(float)

    mu0 = st.sidebar.number_input("목표값 (μ₀)", value=float(np.mean(data)) if len(data) > 0 else 50.0, format="%.3f")
    alt = st.sidebar.selectbox("대립가설", ["μ ≠ μ₀ (양측)", "μ > μ₀ (우측)", "μ < μ₀ (좌측)"])
    alpha = st.sidebar.number_input("유의수준", value=0.05, format="%.2f")

    if st.button("▶ 검정 실행"):
        n = len(data)
        mean_val, std_val = np.mean(data), np.std(data, ddof=1)
        se = std_val / np.sqrt(n)
        t_stat = (mean_val - mu0) / se
        if "양측" in alt:
            p_val = 2 * stats.t.sf(abs(t_stat), n - 1)
        elif "우측" in alt:
            p_val = stats.t.sf(t_stat, n - 1)
        else:
            p_val = stats.t.cdf(t_stat, n - 1)
        ci_low, ci_high = stats.t.interval(1 - alpha, n - 1, loc=mean_val, scale=se)

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pd.DataFrame({
                '항목': ['n', '평균', '표준편차', 'SE', 't', 'p-value', 'CI 하한', 'CI 상한'],
                '값': [n, f"{mean_val:.4f}", f"{std_val:.4f}", f"{se:.4f}",
                       f"{t_stat:.4f}", f"{p_val:.4f}", f"{ci_low:.4f}", f"{ci_high:.4f}"]
            }), hide_index=True, use_container_width=True)
        with col2:
            if p_val < alpha:
                st.error(f"p = {p_val:.4f} < {alpha} → H₀ 기각")
            else:
                st.success(f"p = {p_val:.4f} ≥ {alpha} → H₀ 채택")
            d = abs(mean_val - mu0) / std_val
            st.metric("Cohen's d", f"{d:.3f}")

# ==============================================================
# 2-Sample t Test
# ==============================================================
elif test_type == "2-Sample t Test":
    st.header("2-Sample t Test")

    data_source = st.sidebar.radio("데이터", ["STB 3.2 (Strength by Supplier)", "직접 입력", "CSV 업로드"])
    if data_source == "STB 3.2 (Strength by Supplier)":
        df_raw = load_stb('STB_3.2.csv')
        if df_raw is None:
            st.stop()
        df_raw = df_raw.dropna(subset=['Strength', 'Suplier'])
        groups = df_raw['Suplier'].unique()
        data1 = df_raw[df_raw['Suplier'] == groups[0]]['Strength'].values.astype(float)
        data2 = df_raw[df_raw['Suplier'] == groups[1]]['Strength'].values.astype(float)
        st.sidebar.success(f"STB 3.2: {groups[0]}({len(data1)}), {groups[1]}({len(data2)})")
    elif data_source == "직접 입력":
        raw1 = st.sidebar.text_area("그룹 1", "50.1, 49.8, 50.5, 51.0, 49.9, 50.3")
        raw2 = st.sidebar.text_area("그룹 2", "48.5, 49.2, 48.8, 49.0, 48.7, 49.1")
        data1, data2 = np.array(parse_data(raw1)), np.array(parse_data(raw2))
    else:
        uploaded = st.sidebar.file_uploader("CSV", type="csv")
        if not uploaded:
            st.stop()
        df = pd.read_csv(uploaded)
        val_col = st.sidebar.selectbox("측정값", df.columns)
        grp_col = st.sidebar.selectbox("그룹", df.columns)
        groups = df[grp_col].unique()
        data1 = df[df[grp_col] == groups[0]][val_col].dropna().values.astype(float)
        data2 = df[df[grp_col] == groups[1]][val_col].dropna().values.astype(float)

    alt = st.sidebar.selectbox("대립가설", ["μ₁ ≠ μ₂ (양측)", "μ₁ > μ₂", "μ₁ < μ₂"])
    alpha = st.sidebar.number_input("유의수준", value=0.05, format="%.2f")
    equal_var = st.sidebar.checkbox("등분산 가정 (Pooled)", value=False)

    if st.button("▶ 검정 실행"):
        alt_map = {"μ₁ ≠ μ₂ (양측)": "two-sided", "μ₁ > μ₂": "greater", "μ₁ < μ₂": "less"}
        t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=equal_var, alternative=alt_map[alt])
        lev_stat, lev_p = stats.levene(data1, data2)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("기술 통계")
            st.dataframe(pd.DataFrame({
                '': ['그룹 1', '그룹 2'], 'n': [len(data1), len(data2)],
                '평균': [f"{np.mean(data1):.4f}", f"{np.mean(data2):.4f}"],
                '표준편차': [f"{np.std(data1,ddof=1):.4f}", f"{np.std(data2,ddof=1):.4f}"]
            }), hide_index=True, use_container_width=True)
        with col2:
            st.metric("t 통계량", f"{t_stat:.4f}")
            st.metric("p-value", f"{p_val:.4f}")
            if p_val < alpha:
                st.error(f"p < {alpha} → 유의한 차이 있음")
            else:
                st.success(f"p ≥ {alpha} → 유의한 차이 없음")

        st.subheader("등분산 검정 (Levene)")
        st.markdown(f"F = {lev_stat:.4f}, p = {lev_p:.4f}")
        if lev_p > 0.05:
            st.success("등분산 가정 채택 → Pooled t Test 적합")
        else:
            st.warning("등분산 가정 기각 → Welch's t Test 권장")

        fig = go.Figure()
        fig.add_trace(go.Box(y=data1, name="그룹 1", marker_color='#2e75b6'))
        fig.add_trace(go.Box(y=data2, name="그룹 2", marker_color='#e74c3c'))
        fig.update_layout(height=400, yaxis_title="측정값")
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# Paired t Test
# ==============================================================
elif test_type == "Paired t Test":
    st.header("Paired t Test")

    data_source = st.sidebar.radio("데이터", ["직접 입력", "CSV 업로드"])
    if data_source == "직접 입력":
        raw1 = st.sidebar.text_area("Before", "50.1, 49.8, 50.5, 51.0, 49.9, 50.3, 50.7, 49.6")
        raw2 = st.sidebar.text_area("After", "48.5, 48.2, 49.0, 49.5, 48.4, 48.8, 49.2, 48.1")
        data1, data2 = np.array(parse_data(raw1)), np.array(parse_data(raw2))
    else:
        uploaded = st.sidebar.file_uploader("CSV", type="csv")
        if not uploaded:
            st.stop()
        df = pd.read_csv(uploaded)
        c1 = st.sidebar.selectbox("Before", df.columns, index=0)
        c2 = st.sidebar.selectbox("After", df.columns, index=1)
        data1, data2 = df[c1].dropna().values.astype(float), df[c2].dropna().values.astype(float)

    alpha = st.sidebar.number_input("유의수준", value=0.05, format="%.2f")

    if st.button("▶ 검정 실행"):
        if len(data1) != len(data2):
            st.error("두 데이터의 길이가 같아야 합니다.")
            st.stop()
        diff = data1 - data2
        t_stat, p_val = stats.ttest_rel(data1, data2)
        mean_diff = np.mean(diff)
        se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))
        ci_low, ci_high = stats.t.interval(1 - alpha, len(diff) - 1, loc=mean_diff, scale=se_diff)

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pd.DataFrame({
                '항목': ['n (쌍)', '차이 평균', 't', 'p-value', 'CI 하한', 'CI 상한'],
                '값': [len(diff), f"{mean_diff:.4f}", f"{t_stat:.4f}", f"{p_val:.4f}", f"{ci_low:.4f}", f"{ci_high:.4f}"]
            }), hide_index=True, use_container_width=True)
        with col2:
            if p_val < alpha:
                st.error(f"p = {p_val:.4f} → 유의한 차이 있음")
            else:
                st.success(f"p = {p_val:.4f} → 유의한 차이 없음")

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Before vs After", "차이 분포"))
        fig.add_trace(go.Scatter(y=data1, mode='lines+markers', name='Before'), row=1, col=1)
        fig.add_trace(go.Scatter(y=data2, mode='lines+markers', name='After'), row=1, col=1)
        fig.add_trace(go.Histogram(x=diff, nbinsx=15, marker_color='#2e75b6', showlegend=False), row=1, col=2)
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# 등분산 검정
# ==============================================================
elif test_type == "등분산 검정":
    st.header("등분산 검정 (Levene / Bartlett)")

    data_source = st.sidebar.radio("데이터", ["STB 3.2 (Strength by Supplier)", "직접 입력", "CSV 업로드"])
    if data_source == "STB 3.2 (Strength by Supplier)":
        df_raw = load_stb('STB_3.2.csv')
        if df_raw is None:
            st.stop()
        df_raw = df_raw.dropna(subset=['Strength', 'Suplier'])
        grp_names = df_raw['Suplier'].unique()
        groups_data = [df_raw[df_raw['Suplier'] == g]['Strength'].values.astype(float) for g in grp_names]
        st.sidebar.success(f"STB 3.2: {len(grp_names)}그룹 로드")
    elif data_source == "직접 입력":
        n_groups = st.sidebar.number_input("그룹 수", value=3, min_value=2, max_value=10)
        groups_data = []
        for i in range(n_groups):
            raw = st.sidebar.text_area(f"그룹 {i+1}", f"{', '.join([str(round(50+i+np.random.randn(),1)) for _ in range(8)])}", key=f"grp_{i}")
            groups_data.append(np.array(parse_data(raw)))
    else:
        uploaded = st.sidebar.file_uploader("CSV", type="csv")
        if not uploaded:
            st.stop()
        df = pd.read_csv(uploaded)
        val_col = st.sidebar.selectbox("측정값", df.columns)
        grp_col = st.sidebar.selectbox("그룹", df.columns)
        grp_names = df[grp_col].unique()
        groups_data = [df[df[grp_col] == g][val_col].dropna().values.astype(float) for g in grp_names]

    if st.button("▶ 검정 실행"):
        lev_stat, lev_p = stats.levene(*groups_data)
        bart_stat, bart_p = stats.bartlett(*groups_data)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Levene 검정")
            st.metric("F", f"{lev_stat:.4f}")
            st.metric("p-value", f"{lev_p:.4f}")
            st.success("등분산 채택") if lev_p > 0.05 else st.warning("등분산 기각")
        with col2:
            st.subheader("Bartlett 검정")
            st.metric("χ²", f"{bart_stat:.4f}")
            st.metric("p-value", f"{bart_p:.4f}")
            st.success("등분산 채택") if bart_p > 0.05 else st.warning("등분산 기각")

        fig = go.Figure()
        for i, gd in enumerate(groups_data):
            fig.add_trace(go.Box(y=gd, name=f"그룹 {i+1}"))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# 카이제곱 검정
# ==============================================================
elif test_type == "카이제곱 검정":
    st.header("카이제곱 검정 (Chi-Square Test)")

    data_source = st.sidebar.radio("데이터", ["STB 3.5 (PASS vs 전송방식)", "직접 입력"])

    if data_source == "STB 3.5 (PASS vs 전송방식)":
        df_raw = load_stb('STB_3.5.csv')
        if df_raw is None:
            st.stop()
        # 한글 컬럼명 처리
        cols = df_raw.columns.tolist()
        pass_col = 'PASS'
        grp_col = cols[2] if len(cols) > 2 else cols[1]  # 전송방식
        st.sidebar.success(f"STB 3.5 로드 ({len(df_raw)}행)")

        if st.button("▶ 검정 실행"):
            ct = pd.crosstab(df_raw[pass_col], df_raw[grp_col])
            st.subheader("교차표 (관측 빈도)")
            st.dataframe(ct, use_container_width=True)

            chi2, p_val, dof, expected = stats.chi2_contingency(ct.values)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("χ² 통계량", f"{chi2:.4f}")
                st.metric("자유도", dof)
                st.metric("p-value", f"{p_val:.4f}")
                if p_val < 0.05:
                    st.error("p < 0.05 → 유의한 연관성 있음")
                else:
                    st.success("p ≥ 0.05 → 유의한 연관성 없음")
            with col2:
                st.subheader("기대 빈도")
                st.dataframe(pd.DataFrame(np.round(expected, 2), index=ct.index, columns=ct.columns), use_container_width=True)
    else:
        st.markdown("교차표 데이터를 입력하세요:")
        rows = st.sidebar.number_input("행 수", value=2, min_value=2, max_value=10)
        cols_n = st.sidebar.number_input("열 수", value=2, min_value=2, max_value=10)
        table_data = []
        for i in range(rows):
            row_data = []
            col_inputs = st.columns(cols_n)
            for j in range(cols_n):
                with col_inputs[j]:
                    val = st.number_input(f"({i+1},{j+1})", value=10, min_value=0, key=f"chi_{i}_{j}")
                    row_data.append(val)
            table_data.append(row_data)

        if st.button("▶ 검정 실행"):
            observed = np.array(table_data)
            chi2, p_val, dof, expected = stats.chi2_contingency(observed)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("χ²", f"{chi2:.4f}")
                st.metric("자유도", dof)
                st.metric("p-value", f"{p_val:.4f}")
                if p_val < 0.05:
                    st.error("유의한 연관성 있음")
                else:
                    st.success("유의한 연관성 없음")
            with col2:
                st.subheader("기대 빈도")
                st.dataframe(pd.DataFrame(np.round(expected, 2)), use_container_width=True)
