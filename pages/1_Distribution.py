import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="확률분포 계산기", layout="wide")
st.title("확률분포 계산기")

dist_type = st.sidebar.selectbox(
    "분포 유형 선택",
    ["정규분포", "이항분포", "포아송분포", "초기하분포"]
)

# --- 정규분포 ---
if dist_type == "정규분포":
    st.header("정규분포 (Normal Distribution)")

    col1, col2 = st.columns([1, 2])
    with col1:
        mu = st.number_input("평균 (μ)", value=0.0, format="%.2f")
        sigma = st.number_input("표준편차 (σ)", value=1.0, min_value=0.01, format="%.2f")

        calc_type = st.radio("계산 유형", ["P(X ≤ x)", "P(X ≥ x)", "P(a ≤ X ≤ b)"])

        if calc_type == "P(a ≤ X ≤ b)":
            a_val = st.number_input("a (하한)", value=mu - sigma, format="%.2f")
            b_val = st.number_input("b (상한)", value=mu + sigma, format="%.2f")
        else:
            x_val = st.number_input("x 값", value=mu, format="%.2f")

        if st.button("▶ 계산", key="norm_calc"):
            if calc_type == "P(X ≤ x)":
                prob = stats.norm.cdf(x_val, mu, sigma)
                st.success(f"P(X ≤ {x_val}) = **{prob:.6f}**")
            elif calc_type == "P(X ≥ x)":
                prob = 1 - stats.norm.cdf(x_val, mu, sigma)
                st.success(f"P(X ≥ {x_val}) = **{prob:.6f}**")
            else:
                prob = stats.norm.cdf(b_val, mu, sigma) - stats.norm.cdf(a_val, mu, sigma)
                st.success(f"P({a_val} ≤ X ≤ {b_val}) = **{prob:.6f}**")

    with col2:
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
        y = stats.norm.pdf(x, mu, sigma)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='PDF',
                                 line=dict(color='#2e75b6', width=2)))

        # 영역 음영
        if calc_type == "P(X ≤ x)":
            x_fill = x[x <= x_val]
            y_fill = stats.norm.pdf(x_fill, mu, sigma)
            fig.add_trace(go.Scatter(x=np.concatenate([x_fill, [x_fill[-1], x_fill[0]]]),
                                     y=np.concatenate([y_fill, [0, 0]]),
                                     fill='toself', fillcolor='rgba(46,117,182,0.3)',
                                     line=dict(width=0), name='확률 영역'))
        elif calc_type == "P(X ≥ x)":
            x_fill = x[x >= x_val]
            y_fill = stats.norm.pdf(x_fill, mu, sigma)
            fig.add_trace(go.Scatter(x=np.concatenate([x_fill, [x_fill[-1], x_fill[0]]]),
                                     y=np.concatenate([y_fill, [0, 0]]),
                                     fill='toself', fillcolor='rgba(46,117,182,0.3)',
                                     line=dict(width=0), name='확률 영역'))
        else:
            x_fill = x[(x >= a_val) & (x <= b_val)]
            y_fill = stats.norm.pdf(x_fill, mu, sigma)
            fig.add_trace(go.Scatter(x=np.concatenate([x_fill, [x_fill[-1], x_fill[0]]]),
                                     y=np.concatenate([y_fill, [0, 0]]),
                                     fill='toself', fillcolor='rgba(46,117,182,0.3)',
                                     line=dict(width=0), name='확률 영역'))

        # 시그마 수준 표시
        for i in range(1, 4):
            fig.add_vline(x=mu + i * sigma, line_dash="dot", line_color="gray",
                          annotation_text=f"+{i}σ", annotation_position="top")
            fig.add_vline(x=mu - i * sigma, line_dash="dot", line_color="gray",
                          annotation_text=f"-{i}σ", annotation_position="top")

        fig.update_layout(title=f"정규분포 N({mu}, {sigma}²)",
                          xaxis_title="X", yaxis_title="확률밀도",
                          height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # 시그마 수준 표
    st.subheader("시그마 수준별 확률")
    sigma_data = []
    for s in range(1, 7):
        p_within = stats.norm.cdf(s) - stats.norm.cdf(-s)
        ppm = (1 - p_within) * 1_000_000
        p_shifted = stats.norm.cdf(s - 1.5) - stats.norm.cdf(-s - 1.5)
        ppm_shifted = (1 - p_shifted) * 1_000_000
        sigma_data.append({
            "시그마 수준": f"{s}σ",
            "수율(%) - 중심": f"{p_within * 100:.4f}%",
            "PPM - 중심": f"{ppm:.1f}",
            "수율(%) - 1.5σ Shift": f"{p_shifted * 100:.4f}%",
            "PPM - 1.5σ Shift": f"{ppm_shifted:.1f}"
        })
    st.dataframe(pd.DataFrame(sigma_data), use_container_width=True, hide_index=True)

# --- 이항분포 ---
elif dist_type == "이항분포":
    st.header("이항분포 (Binomial Distribution)")

    col1, col2 = st.columns([1, 2])
    with col1:
        n = st.number_input("시행 횟수 (n)", value=10, min_value=1, max_value=1000)
        p = st.number_input("성공 확률 (p)", value=0.5, min_value=0.0, max_value=1.0, format="%.3f")

        calc_type = st.radio("계산 유형", ["P(X = k)", "P(X ≤ k)", "P(X ≥ k)"])
        k = st.number_input("k 값", value=5, min_value=0, max_value=n)

        if st.button("▶ 계산", key="binom_calc"):
            if calc_type == "P(X = k)":
                prob = stats.binom.pmf(k, n, p)
                st.success(f"P(X = {k}) = **{prob:.6f}**")
            elif calc_type == "P(X ≤ k)":
                prob = stats.binom.cdf(k, n, p)
                st.success(f"P(X ≤ {k}) = **{prob:.6f}**")
            else:
                prob = 1 - stats.binom.cdf(k - 1, n, p)
                st.success(f"P(X ≥ {k}) = **{prob:.6f}**")

        st.markdown(f"**평균**: μ = np = {n * p:.2f}")
        st.markdown(f"**분산**: σ² = np(1-p) = {n * p * (1 - p):.2f}")

    with col2:
        x_vals = np.arange(0, n + 1)
        pmf_vals = stats.binom.pmf(x_vals, n, p)

        colors = ['rgba(46,117,182,0.3)'] * len(x_vals)
        if calc_type == "P(X = k)":
            colors[k] = 'rgba(46,117,182,0.9)'
        elif calc_type == "P(X ≤ k)":
            for i in range(k + 1):
                colors[i] = 'rgba(46,117,182,0.9)'
        else:
            for i in range(k, n + 1):
                colors[i] = 'rgba(46,117,182,0.9)'

        fig = go.Figure()
        fig.add_trace(go.Bar(x=x_vals, y=pmf_vals, marker_color=colors, name='PMF'))
        fig.update_layout(title=f"이항분포 B({n}, {p})",
                          xaxis_title="X (성공 횟수)", yaxis_title="확률",
                          height=450)
        st.plotly_chart(fig, use_container_width=True)

# --- 포아송분포 ---
elif dist_type == "포아송분포":
    st.header("포아송분포 (Poisson Distribution)")

    col1, col2 = st.columns([1, 2])
    with col1:
        lam = st.number_input("λ (평균 발생 횟수)", value=3.0, min_value=0.1, format="%.2f")

        calc_type = st.radio("계산 유형", ["P(X = k)", "P(X ≤ k)", "P(X ≥ k)"])
        k = st.number_input("k 값", value=3, min_value=0, max_value=100)

        if st.button("▶ 계산", key="pois_calc"):
            if calc_type == "P(X = k)":
                prob = stats.poisson.pmf(k, lam)
                st.success(f"P(X = {k}) = **{prob:.6f}**")
            elif calc_type == "P(X ≤ k)":
                prob = stats.poisson.cdf(k, lam)
                st.success(f"P(X ≤ {k}) = **{prob:.6f}**")
            else:
                prob = 1 - stats.poisson.cdf(k - 1, lam)
                st.success(f"P(X ≥ {k}) = **{prob:.6f}**")

        st.markdown(f"**평균**: μ = λ = {lam:.2f}")
        st.markdown(f"**분산**: σ² = λ = {lam:.2f}")

    with col2:
        x_max = int(lam + 5 * np.sqrt(lam)) + 1
        x_vals = np.arange(0, x_max)
        pmf_vals = stats.poisson.pmf(x_vals, lam)

        colors = ['rgba(46,117,182,0.3)'] * len(x_vals)
        if calc_type == "P(X = k)" and k < len(colors):
            colors[k] = 'rgba(46,117,182,0.9)'
        elif calc_type == "P(X ≤ k)":
            for i in range(min(k + 1, len(colors))):
                colors[i] = 'rgba(46,117,182,0.9)'
        else:
            for i in range(k, len(colors)):
                colors[i] = 'rgba(46,117,182,0.9)'

        fig = go.Figure()
        fig.add_trace(go.Bar(x=x_vals, y=pmf_vals, marker_color=colors, name='PMF'))
        fig.update_layout(title=f"포아송분포 Pois({lam})",
                          xaxis_title="X (발생 횟수)", yaxis_title="확률",
                          height=450)
        st.plotly_chart(fig, use_container_width=True)

# --- 초기하분포 ---
elif dist_type == "초기하분포":
    st.header("초기하분포 (Hypergeometric Distribution)")

    col1, col2 = st.columns([1, 2])
    with col1:
        N_pop = st.number_input("모집단 크기 (N)", value=100, min_value=1)
        K_success = st.number_input("모집단 내 성공 수 (K)", value=5, min_value=0, max_value=N_pop)
        n_draw = st.number_input("추출 수 (n)", value=10, min_value=1, max_value=N_pop)

        calc_type = st.radio("계산 유형", ["P(X = k)", "P(X ≤ k)", "P(X ≥ k)"])
        k_max = min(K_success, n_draw)
        k = st.number_input("k 값", value=min(1, k_max), min_value=0, max_value=k_max)

        if st.button("▶ 계산", key="hyper_calc"):
            if calc_type == "P(X = k)":
                prob = stats.hypergeom.pmf(k, N_pop, K_success, n_draw)
                st.success(f"P(X = {k}) = **{prob:.6f}**")
            elif calc_type == "P(X ≤ k)":
                prob = stats.hypergeom.cdf(k, N_pop, K_success, n_draw)
                st.success(f"P(X ≤ {k}) = **{prob:.6f}**")
            else:
                prob = 1 - stats.hypergeom.cdf(k - 1, N_pop, K_success, n_draw)
                st.success(f"P(X ≥ {k}) = **{prob:.6f}**")

        mean_val = n_draw * K_success / N_pop
        st.markdown(f"**평균**: μ = nK/N = {mean_val:.2f}")

    with col2:
        x_lo = max(0, n_draw - (N_pop - K_success))
        x_hi = min(K_success, n_draw)
        x_vals = np.arange(x_lo, x_hi + 1)
        pmf_vals = stats.hypergeom.pmf(x_vals, N_pop, K_success, n_draw)

        colors = ['rgba(46,117,182,0.3)'] * len(x_vals)
        for idx, xv in enumerate(x_vals):
            if calc_type == "P(X = k)" and xv == k:
                colors[idx] = 'rgba(46,117,182,0.9)'
            elif calc_type == "P(X ≤ k)" and xv <= k:
                colors[idx] = 'rgba(46,117,182,0.9)'
            elif calc_type == "P(X ≥ k)" and xv >= k:
                colors[idx] = 'rgba(46,117,182,0.9)'

        fig = go.Figure()
        fig.add_trace(go.Bar(x=x_vals, y=pmf_vals, marker_color=colors, name='PMF'))
        fig.update_layout(title=f"초기하분포 HG(N={N_pop}, K={K_success}, n={n_draw})",
                          xaxis_title="X (성공 수)", yaxis_title="확률",
                          height=450)
        st.plotly_chart(fig, use_container_width=True)
