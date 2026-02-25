import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import product
import os

st.set_page_config(page_title="실험계획법 (DOE)", layout="wide")
st.title("실험계획법 (DOE) 분석 도구")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_stb(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def code_factor(series):
    """텍스트/숫자 인자를 -1/+1로 코딩"""
    unique = sorted(series.unique())
    if len(unique) == 2:
        return series.map({unique[0]: -1, unique[1]: 1}).astype(float)
    elif len(unique) == 3:
        return series.map({unique[0]: -1, unique[1]: 0, unique[2]: 1}).astype(float)
    mid = (series.max() + series.min()) / 2
    half = (series.max() - series.min()) / 2
    return ((series - mid) / half).astype(float) if half > 0 else series.astype(float)

analysis_type = st.sidebar.selectbox("분석 유형",
    ["2수준 완전 요인 배치", "2수준 부분 요인 배치", "반응표면법 (RSM)"])

# ==============================================================
# Full Factorial
# ==============================================================
if analysis_type == "2수준 완전 요인 배치":
    st.header("2수준 완전 요인 배치 (2ᵏ Full Factorial)")
    mode = st.sidebar.radio("모드", ["데이터 분석", "설계 생성"])

    if mode == "데이터 분석":
        data_source = st.sidebar.radio("데이터",
            ["STB 4.1 (2⁴ 배치)", "STB 4.2 (Frame Rate×조도)", "STB 4.3 (Supplier×Temp)", "CSV 업로드"])

        if data_source == "STB 4.1 (2⁴ 배치)":
            df = load_stb('STB_4.1.csv')
            if df is None:
                st.stop()
            st.sidebar.success(f"STB 4.1 로드 ({len(df)}행)")
            factor_cols_raw = ['A', 'B', 'C', 'D']
            response_col = 'Response'
        elif data_source == "STB 4.2 (Frame Rate×조도)":
            df = load_stb('STB_4.2.csv')
            if df is None:
                st.stop()
            st.sidebar.success(f"STB 4.2 로드 ({len(df)}행)")
            factor_cols_raw = ['Frame Rate']
            # 조도 컬럼명 처리
            for c in df.columns:
                if '조도' in str(c) or c == '조도':
                    factor_cols_raw.append(c)
                    break
            response_col = 'Response'
        elif data_source == "STB 4.3 (Supplier×Temp)":
            df = load_stb('STB_4.3.csv')
            if df is None:
                st.stop()
            st.sidebar.success(f"STB 4.3 로드 ({len(df)}행)")
            factor_cols_raw = ['Supplier', 'Temp']
            response_col = 'Response'
        else:
            uploaded = st.sidebar.file_uploader("CSV", type="csv")
            if not uploaded:
                st.stop()
            df = pd.read_csv(uploaded)
            all_cols = df.columns.tolist()
            response_col = st.sidebar.selectbox("반응변수", all_cols)
            factor_cols_raw = st.sidebar.multiselect("인자 컬럼", [c for c in all_cols if c != response_col])

        if len(factor_cols_raw) >= 1:
            st.dataframe(df[factor_cols_raw + [response_col]].head(20), use_container_width=True, hide_index=True)

        if len(factor_cols_raw) >= 1 and st.button("▶ 분석 실행", type="primary"):
            import statsmodels.api as sm

            # 코딩
            df_coded = pd.DataFrame()
            factor_cols = []
            for col in factor_cols_raw:
                coded = code_factor(df[col])
                coded_name = f"{col}_coded"
                df_coded[coded_name] = coded
                factor_cols.append(coded_name)

            Y = df[response_col].values.astype(float)
            X = df_coded[factor_cols].values
            k = len(factor_cols)

            # 주효과 + 2인자 교호작용
            col_names = list(factor_cols_raw)
            X_full = X.copy()
            for i in range(k):
                for j in range(i+1, k):
                    X_full = np.column_stack([X_full, X[:, i] * X[:, j]])
                    col_names.append(f"{factor_cols_raw[i]}×{factor_cols_raw[j]}")

            X_const = sm.add_constant(X_full)
            model = sm.OLS(Y, X_const).fit()
            effects = model.params[1:] * 2

            st.subheader("효과 추정")
            st.dataframe(pd.DataFrame({
                '인자': col_names,
                '계수': [f"{c:.4f}" for c in model.params[1:]],
                '효과': [f"{e:.4f}" for e in effects],
                'p-value': [f"{p:.4f}" for p in model.pvalues[1:]]
            }), hide_index=True, use_container_width=True)
            st.markdown(f"**R² = {model.rsquared:.4f}**, **Adj R² = {model.rsquared_adj:.4f}**")

            # Pareto
            st.subheader("효과 Pareto 차트")
            abs_eff = np.abs(effects)
            sorted_idx = np.argsort(abs_eff)[::-1]
            t_crit = stats.t.ppf(0.975, model.df_resid) if model.df_resid > 0 else 2
            colors = ['#e74c3c' if abs(model.tvalues[1:][i]) > t_crit else '#bdc3c7' for i in sorted_idx]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[col_names[i] for i in sorted_idx], y=[abs_eff[i] for i in sorted_idx], marker_color=colors))
            fig.update_layout(height=400, xaxis_title="인자", yaxis_title="|효과|")
            st.plotly_chart(fig, use_container_width=True)

            # 주효과 플롯
            st.subheader("주효과 플롯")
            n_f = min(k, 6)
            fig = make_subplots(rows=1, cols=n_f, subplot_titles=[factor_cols_raw[i] for i in range(n_f)])
            for idx in range(n_f):
                low_mean = np.mean(Y[X[:, idx] <= 0]) if np.any(X[:, idx] <= 0) else np.mean(Y)
                high_mean = np.mean(Y[X[:, idx] > 0]) if np.any(X[:, idx] > 0) else np.mean(Y)
                fig.add_trace(go.Scatter(x=['-1', '+1'], y=[low_mean, high_mean],
                    mode='lines+markers', marker=dict(size=10), showlegend=False), row=1, col=idx+1)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    else:  # 설계 생성
        k = st.sidebar.number_input("인자 수 (k)", value=3, min_value=2, max_value=7)
        n_center = st.sidebar.number_input("중심점 수", value=0, min_value=0)
        if st.button("▶ 설계 생성"):
            levels = list(product([-1, 1], repeat=k))
            design_df = pd.DataFrame(levels, columns=[chr(65+i) for i in range(k)])
            if n_center > 0:
                center = pd.DataFrame([[0]*k]*n_center, columns=design_df.columns)
                design_df = pd.concat([design_df, center], ignore_index=True)
            design_df = design_df.sample(frac=1, random_state=42).reset_index(drop=True)
            design_df.index += 1
            design_df['Response'] = ''
            st.dataframe(design_df, use_container_width=True)
            st.download_button("CSV 다운로드", design_df.to_csv(index=True), "doe_design.csv")

# ==============================================================
# Fractional Factorial
# ==============================================================
elif analysis_type == "2수준 부분 요인 배치":
    st.header("2수준 부분 요인 배치 (Fractional Factorial)")
    k = st.sidebar.number_input("인자 수 (k)", value=5, min_value=3, max_value=10)
    p = st.sidebar.number_input("생략 수 (p)", value=2, min_value=1, max_value=k-2)
    n_runs = 2 ** (k - p)
    st.sidebar.info(f"실험 수: {n_runs} runs")

    if st.button("▶ 설계 생성"):
        base_k = k - p
        base = list(product([-1, 1], repeat=base_k))
        df = pd.DataFrame(base, columns=[chr(65+i) for i in range(base_k)])
        for i in range(p):
            gen_cols = list(range(min(i+2, base_k)))
            vals = np.ones(n_runs)
            for gc in gen_cols:
                vals *= df.iloc[:, gc].values
            df[chr(65 + base_k + i)] = vals.astype(int)
        df.index += 1
        df['Response'] = ''
        st.dataframe(df, use_container_width=True)
        st.download_button("CSV 다운로드", df.to_csv(index=True), "fractional_design.csv")

# ==============================================================
# RSM
# ==============================================================
else:
    st.header("반응표면법 (RSM)")

    data_source = st.sidebar.radio("데이터",
        ["STB 4.4 (Temp×Cycle)", "STB 4.7 (Temp×Cycle 확장)", "STB 4.8 (LL×ML)", "CSV 업로드"])

    if data_source == "STB 4.4 (Temp×Cycle)":
        df = load_stb('STB_4.4.csv')
        if df is None:
            st.stop()
        x_cols = ['Temp', 'Cycle']
        y_col = 'Response'
        st.sidebar.success(f"STB 4.4 로드 ({len(df)}행)")
    elif data_source == "STB 4.7 (Temp×Cycle 확장)":
        df = load_stb('STB_4.7.csv')
        if df is None:
            st.stop()
        x_cols = ['Temp', 'Cycle']
        y_col = 'Response'
        st.sidebar.success(f"STB 4.7 로드 ({len(df)}행)")
    elif data_source == "STB 4.8 (LL×ML)":
        df = load_stb('STB_4.8.csv')
        if df is None:
            st.stop()
        x_cols = ['LL', 'ML']
        y_col = 'Project Y'
        st.sidebar.success(f"STB 4.8 로드 ({len(df)}행)")
    else:
        uploaded = st.sidebar.file_uploader("CSV", type="csv")
        if not uploaded:
            st.stop()
        df = pd.read_csv(uploaded)
        y_col = st.sidebar.selectbox("반응변수", df.columns)
        x_cols = st.sidebar.multiselect("인자", [c for c in df.columns if c != y_col])

    if len(x_cols) >= 2 and st.button("▶ 분석 실행", type="primary"):
        import statsmodels.api as sm

        sub = df[x_cols + [y_col]].dropna()
        X_raw = sub[x_cols].values.astype(float)
        Y = sub[y_col].values.astype(float)
        k = len(x_cols)

        # 코딩 (중심/범위 기준)
        X_coded = np.zeros_like(X_raw)
        for i in range(k):
            mid = (X_raw[:, i].max() + X_raw[:, i].min()) / 2
            half = (X_raw[:, i].max() - X_raw[:, i].min()) / 2
            X_coded[:, i] = (X_raw[:, i] - mid) / half if half > 0 else 0

        # 2차 모형: X, X², X_i*X_j
        col_names = list(x_cols)
        X_model = X_coded.copy()
        for i in range(k):
            X_model = np.column_stack([X_model, X_coded[:, i] ** 2])
            col_names.append(f"{x_cols[i]}²")
        for i in range(k):
            for j in range(i+1, k):
                X_model = np.column_stack([X_model, X_coded[:, i] * X_coded[:, j]])
                col_names.append(f"{x_cols[i]}×{x_cols[j]}")

        X_const = sm.add_constant(X_model)
        model = sm.OLS(Y, X_const).fit()

        st.subheader("2차 반응표면 모형")
        st.dataframe(pd.DataFrame({
            'Term': ['Constant'] + col_names,
            'Coefficient': [f"{c:.4f}" for c in model.params],
            'p-value': [f"{p:.4f}" for p in model.pvalues]
        }), hide_index=True, use_container_width=True)
        st.markdown(f"**R² = {model.rsquared:.4f}**, **Adj R² = {model.rsquared_adj:.4f}**")

        # 3D + 등고선
        if k >= 2:
            st.subheader("반응표면 & 등고선")
            x1r = np.linspace(X_coded[:, 0].min() - 0.3, X_coded[:, 0].max() + 0.3, 50)
            x2r = np.linspace(X_coded[:, 1].min() - 0.3, X_coded[:, 1].max() + 0.3, 50)
            X1g, X2g = np.meshgrid(x1r, x2r)

            grid = np.column_stack([X1g.ravel(), X2g.ravel()])
            if k > 2:
                grid = np.column_stack([grid, np.zeros((grid.shape[0], k - 2))])
            grid_model = grid.copy()
            for i in range(k):
                grid_model = np.column_stack([grid_model, grid[:, i] ** 2])
            for i in range(k):
                for j in range(i+1, k):
                    grid_model = np.column_stack([grid_model, grid[:, i] * grid[:, j]])
            Z = model.predict(sm.add_constant(grid_model)).reshape(X1g.shape)

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(data=[go.Surface(x=x1r, y=x2r, z=Z, colorscale='Viridis')])
                fig.update_layout(title="3D 반응표면", height=500,
                    scene=dict(xaxis_title=x_cols[0], yaxis_title=x_cols[1], zaxis_title=y_col))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = go.Figure(data=[go.Contour(x=x1r, y=x2r, z=Z, colorscale='Viridis')])
                fig.add_trace(go.Scatter(x=X_coded[:, 0], y=X_coded[:, 1], mode='markers',
                    marker=dict(color='red', size=8), name='실험점'))
                fig.update_layout(title="등고선 플롯", height=500, xaxis_title=x_cols[0], yaxis_title=x_cols[1])
                st.plotly_chart(fig, use_container_width=True)

            opt_type = st.sidebar.radio("최적화", ["최대화", "최소화"])
            opt_idx = np.argmax(Z) if opt_type == "최대화" else np.argmin(Z)
            r, c = np.unravel_index(opt_idx, Z.shape)
            st.markdown(f"**최적 {x_cols[0]}**: {x1r[c]:.3f} (코딩값), **최적 {x_cols[1]}**: {x2r[r]:.3f} (코딩값)")
            st.markdown(f"**예측 {y_col}**: {Z[r, c]:.3f}")
