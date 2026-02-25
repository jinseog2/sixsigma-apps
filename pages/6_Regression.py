import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="회귀분석", layout="wide")
st.title("회귀분석 도구")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_stb(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

analysis_type = st.sidebar.selectbox("분석 유형", ["상관분석", "단순 선형 회귀", "다중 선형 회귀"])

# ==============================================================
# 상관분석
# ==============================================================
if analysis_type == "상관분석":
    st.header("상관분석 (Correlation Analysis)")

    data_source = st.sidebar.radio("데이터", ["STB 3.1 (카메라 모듈)", "CSV 업로드"])
    if data_source == "STB 3.1 (카메라 모듈)":
        df = load_stb('STB_3.1.csv')
        if df is None:
            st.stop()
        st.sidebar.success(f"STB 3.1 로드 ({len(df)}행)")
    else:
        uploaded = st.sidebar.file_uploader("CSV", type="csv")
        if not uploaded:
            st.stop()
        df = pd.read_csv(uploaded)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected = st.sidebar.multiselect("변수 선택", num_cols, default=num_cols[:min(4, len(num_cols))])

    if len(selected) >= 2 and st.button("▶ 분석 실행"):
        sub = df[selected].dropna()
        corr_matrix = sub.corr()
        st.subheader("상관계수 행렬 (Pearson r)")
        st.dataframe(corr_matrix.round(4), use_container_width=True)

        p_matrix = pd.DataFrame(np.zeros((len(selected), len(selected))), columns=selected, index=selected)
        for i in range(len(selected)):
            for j in range(len(selected)):
                if i != j:
                    _, p = stats.pearsonr(sub[selected[i]], sub[selected[j]])
                    p_matrix.iloc[i, j] = p
        st.subheader("p-value 행렬")
        st.dataframe(p_matrix.round(4), use_container_width=True)

        fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=selected, y=selected,
            colorscale='RdBu_r', zmin=-1, zmax=1,
            text=corr_matrix.round(3).values, texttemplate="%{text}"))
        fig.update_layout(title="상관계수 히트맵", height=450)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# 단순 선형 회귀
# ==============================================================
elif analysis_type == "단순 선형 회귀":
    st.header("단순 선형 회귀")

    data_source = st.sidebar.radio("데이터", ["STB 3.7 (디스크 성능)", "직접 입력", "CSV 업로드"])
    if data_source == "STB 3.7 (디스크 성능)":
        df = load_stb('STB_3.7.csv')
        if df is None:
            st.stop()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        x_col = st.sidebar.selectbox("X (독립변수)", num_cols, index=num_cols.index('WriteCount') if 'WriteCount' in num_cols else 0)
        y_col = st.sidebar.selectbox("Y (종속변수)", num_cols, index=num_cols.index('Response') if 'Response' in num_cols else 1)
        st.sidebar.success(f"STB 3.7 로드 ({len(df)}행)")
    elif data_source == "직접 입력":
        raw_x = st.sidebar.text_area("X 값", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10")
        raw_y = st.sidebar.text_area("Y 값", "2.1, 4.0, 5.8, 8.1, 10.2, 11.8, 14.1, 16.0, 17.9, 20.1")
        df = pd.DataFrame({'X': [float(v) for v in raw_x.split(",") if v.strip()],
                          'Y': [float(v) for v in raw_y.split(",") if v.strip()]})
        x_col, y_col = 'X', 'Y'
    else:
        uploaded = st.sidebar.file_uploader("CSV", type="csv")
        if not uploaded:
            st.stop()
        df = pd.read_csv(uploaded)
        x_col = st.sidebar.selectbox("X", df.columns, index=0)
        y_col = st.sidebar.selectbox("Y", df.columns, index=1)

    if st.button("▶ 분석 실행", type="primary"):
        sub = df[[x_col, y_col]].dropna()
        x, y = sub[x_col].values.astype(float), sub[y_col].values.astype(float)
        n = len(x)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        y_pred = intercept + slope * x
        residuals = y - y_pred
        r_sq = r_value ** 2
        rmse = np.sqrt(np.mean(residuals ** 2))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("회귀 계수")
            st.dataframe(pd.DataFrame({
                '항목': ['절편 (b₀)', '기울기 (b₁)', 'SE', 'p-value'],
                '값': [f"{intercept:.4f}", f"{slope:.4f}", f"{std_err:.4f}", f"{p_value:.6f}"]
            }), hide_index=True, use_container_width=True)
            st.markdown(f"**Ŷ = {intercept:.3f} + {slope:.3f} × X**")
        with col2:
            st.subheader("적합도")
            st.dataframe(pd.DataFrame({
                '항목': ['R', 'R²', 'RMSE', 'n'],
                '값': [f"{r_value:.4f}", f"{r_sq:.4f}", f"{rmse:.4f}", n]
            }), hide_index=True, use_container_width=True)

        x_line = np.linspace(min(x), max(x), 100)
        y_line = intercept + slope * x_line
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='데이터', marker=dict(color='#2e75b6', size=8)))
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='회귀선', line=dict(color='red', width=2)))
        fig.update_layout(height=450, xaxis_title=x_col, yaxis_title=y_col)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("잔차 분석")
        fig = make_subplots(rows=2, cols=2, subplot_titles=("정규확률도", "잔차 vs 적합값", "잔차 히스토그램", "잔차 순서도"))
        sorted_res = np.sort(residuals)
        theoretical = stats.norm.ppf((np.arange(1, n+1) - 0.375) / (n + 0.25))
        fig.add_trace(go.Scatter(x=theoretical, y=sorted_res, mode='markers', marker=dict(color='#2e75b6'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', marker=dict(color='#2e75b6'), showlegend=False), row=1, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_trace(go.Histogram(x=residuals, nbinsx=15, marker_color='#2e75b6', showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(y=residuals, mode='lines+markers', marker=dict(color='#2e75b6'), showlegend=False), row=2, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# 다중 선형 회귀
# ==============================================================
else:
    st.header("다중 선형 회귀")

    data_source = st.sidebar.radio("데이터", ["STB 3.8 (배합 실험)", "STB 3.3 (강도 분석)", "CSV 업로드"])
    if data_source == "STB 3.8 (배합 실험)":
        df = load_stb('STB_3.8.csv')
        if df is None:
            st.stop()
        st.sidebar.success(f"STB 3.8 로드 ({len(df)}행)")
    elif data_source == "STB 3.3 (강도 분석)":
        df = load_stb('STB_3.3.csv')
        if df is None:
            st.stop()
        st.sidebar.success(f"STB 3.3 로드 ({len(df)}행)")
    else:
        uploaded = st.sidebar.file_uploader("CSV", type="csv")
        if not uploaded:
            st.stop()
        df = pd.read_csv(uploaded)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    y_col = st.sidebar.selectbox("Y (종속변수)", num_cols,
        index=num_cols.index('Response') if 'Response' in num_cols else (num_cols.index('Strength') if 'Strength' in num_cols else 0))
    x_cols = st.sidebar.multiselect("X (독립변수)", [c for c in num_cols if c != y_col],
        default=[c for c in num_cols if c != y_col and c not in ['Id', 'ID', 'id', 'Spot.25']][:5])

    if len(x_cols) >= 1 and st.button("▶ 분석 실행", type="primary"):
        import statsmodels.api as sm

        sub = df[[y_col] + x_cols].dropna()
        Y = sub[y_col].values.astype(float)
        X = sub[x_cols].values.astype(float)
        X_const = sm.add_constant(X)
        model = sm.OLS(Y, X_const).fit()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("회귀 계수")
            coef_names = ['Constant'] + x_cols
            st.dataframe(pd.DataFrame({
                '변수': coef_names,
                '계수': [f"{c:.4f}" for c in model.params],
                'SE': [f"{s:.4f}" for s in model.bse],
                't': [f"{t:.4f}" for t in model.tvalues],
                'p-value': [f"{p:.4f}" for p in model.pvalues]
            }), hide_index=True, use_container_width=True)
        with col2:
            st.subheader("적합도")
            st.dataframe(pd.DataFrame({
                '항목': ['R²', 'Adj R²', 'F', 'p(F)', 'n'],
                '값': [f"{model.rsquared:.4f}", f"{model.rsquared_adj:.4f}",
                       f"{model.fvalue:.4f}", f"{model.f_pvalue:.6f}", len(Y)]
            }), hide_index=True, use_container_width=True)

        # VIF
        if len(x_cols) >= 2:
            st.subheader("VIF (다중공선성)")
            vif_data = []
            for i, col in enumerate(x_cols):
                others = [j for j in range(len(x_cols)) if j != i]
                X_o = sm.add_constant(X[:, others])
                r2j = sm.OLS(X[:, i], X_o).fit().rsquared
                vif = 1 / (1 - r2j) if r2j < 1 else np.inf
                vif_data.append({'변수': col, 'VIF': f"{vif:.2f}", '판정': "✅" if vif < 5 else ("⚠️" if vif < 10 else "❌")})
            st.dataframe(pd.DataFrame(vif_data), hide_index=True, use_container_width=True)

        # 잔차 4종
        st.subheader("잔차 분석")
        residuals = model.resid
        fitted = model.fittedvalues
        n = len(residuals)
        fig = make_subplots(rows=2, cols=2, subplot_titles=("정규확률도", "잔차 vs 적합값", "히스토그램", "순서도"))
        sorted_res = np.sort(residuals)
        theoretical = stats.norm.ppf((np.arange(1, n+1) - 0.375) / (n + 0.25))
        fig.add_trace(go.Scatter(x=theoretical, y=sorted_res, mode='markers', marker=dict(color='#2e75b6'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=fitted, y=residuals, mode='markers', marker=dict(color='#2e75b6'), showlegend=False), row=1, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_trace(go.Histogram(x=residuals, nbinsx=15, marker_color='#2e75b6', showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(y=residuals, mode='lines+markers', marker=dict(color='#2e75b6'), showlegend=False), row=2, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
