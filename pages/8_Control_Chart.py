import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="관리도 분석", layout="wide")
st.title("관리도 (Control Charts) 분석 도구")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_stb(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

chart_type = st.sidebar.selectbox("관리도 유형",
    ["Xbar-R", "I-MR", "P 관리도", "NP 관리도", "C 관리도", "U 관리도"])

CONTROL_CONSTANTS = {
    2: {'A2': 1.880, 'D3': 0, 'D4': 3.267, 'd2': 1.128},
    3: {'A2': 1.023, 'D3': 0, 'D4': 2.575, 'd2': 1.693},
    4: {'A2': 0.729, 'D3': 0, 'D4': 2.282, 'd2': 2.059},
    5: {'A2': 0.577, 'D3': 0, 'D4': 2.115, 'd2': 2.326},
    6: {'A2': 0.483, 'D3': 0, 'D4': 2.004, 'd2': 2.534},
    7: {'A2': 0.419, 'D3': 0.076, 'D4': 1.924, 'd2': 2.704},
    8: {'A2': 0.373, 'D3': 0.136, 'D4': 1.864, 'd2': 2.847},
    9: {'A2': 0.337, 'D3': 0.184, 'D4': 1.816, 'd2': 2.970},
}

def detect_we_rules(data, cl, ucl, lcl):
    n = len(data)
    violations = {i: [] for i in range(n)}
    sigma = (ucl - cl) / 3 if ucl != cl else 1
    zone_a_upper = cl + 2 * sigma
    zone_a_lower = cl - 2 * sigma

    for i in range(n):
        if data[i] > ucl or data[i] < lcl:
            violations[i].append("R1: 관리한계 이탈")
        if i >= 2:
            window = data[max(0, i-2):i+1]
            if sum(1 for v in window if v > zone_a_upper) >= 2 or sum(1 for v in window if v < zone_a_lower) >= 2:
                violations[i].append("R2: Zone A 2/3")
        if i >= 7:
            window = data[i-7:i+1]
            if all(v > cl for v in window) or all(v < cl for v in window):
                violations[i].append("R4: 같은 쪽 8점")
        if i >= 5:
            window = data[i-5:i+1]
            if all(window[j] < window[j+1] for j in range(5)) or all(window[j] > window[j+1] for j in range(5)):
                violations[i].append("R5: 연속 6점 추세")
    return violations

def plot_control_chart(data, cl, ucl, lcl, title, violations=None):
    fig = go.Figure()
    n = len(data)
    x = list(range(1, n + 1))
    colors = ['red' if violations and violations.get(i, []) else '#2e75b6' for i in range(n)]
    fig.add_trace(go.Scatter(x=x, y=data, mode='lines+markers',
        marker=dict(color=colors, size=6), line=dict(color='#2e75b6', width=1)))
    fig.add_hline(y=cl, line_color="green", line_width=2, annotation_text=f"CL={cl:.3f}")
    fig.add_hline(y=ucl, line_color="red", line_dash="dash", annotation_text=f"UCL={ucl:.3f}")
    fig.add_hline(y=lcl, line_color="red", line_dash="dash", annotation_text=f"LCL={lcl:.3f}")
    if violations:
        for i, rules in violations.items():
            if rules:
                fig.add_annotation(x=x[i], y=data[i], text="⚠", showarrow=True, arrowhead=2,
                    arrowcolor="red", font=dict(size=14, color="red"))
    fig.update_layout(title=title, height=400, xaxis_title="Sample", yaxis_title="값", showlegend=False)
    return fig

# ==============================================================
# Xbar-R
# ==============================================================
if chart_type == "Xbar-R":
    st.header("Xbar-R 관리도")

    data_source = st.sidebar.radio("데이터", ["STB 5.2 (Noise)", "CSV 업로드"])
    subgroup_size = st.sidebar.number_input("부분군 크기", value=5, min_value=2, max_value=9)

    if data_source == "STB 5.2 (Noise)":
        df_raw = load_stb('STB_5.2.csv')
        if df_raw is None:
            st.stop()
        data_flat = df_raw['Noise'].dropna().values.astype(float)
        st.sidebar.success(f"STB 5.2 로드 ({len(data_flat)}행)")
    else:
        uploaded = st.sidebar.file_uploader("CSV", type="csv")
        if not uploaded:
            st.stop()
        df_up = pd.read_csv(uploaded)
        # 여러 컬럼이면 행이 부분군
        if len(df_up.columns) >= subgroup_size:
            data_flat = df_up.select_dtypes(include=[np.number]).values.flatten()
        else:
            col = st.sidebar.selectbox("측정값", df_up.columns)
            data_flat = df_up[col].dropna().values.astype(float)

    if st.button("▶ 분석 실행", type="primary"):
        n_sg = len(data_flat) // subgroup_size
        data_matrix = data_flat[:n_sg * subgroup_size].reshape(n_sg, subgroup_size)

        if subgroup_size not in CONTROL_CONSTANTS:
            st.error(f"부분군 크기 {subgroup_size} 미지원 (2~9)")
            st.stop()

        consts = CONTROL_CONSTANTS[subgroup_size]
        xbars = np.mean(data_matrix, axis=1)
        ranges = np.ptp(data_matrix, axis=1)
        xbar_bar, r_bar = np.mean(xbars), np.mean(ranges)

        xbar_ucl = xbar_bar + consts['A2'] * r_bar
        xbar_lcl = xbar_bar - consts['A2'] * r_bar
        r_ucl = consts['D4'] * r_bar
        r_lcl = consts['D3'] * r_bar

        xbar_v = detect_we_rules(xbars, xbar_bar, xbar_ucl, xbar_lcl)
        r_v = detect_we_rules(ranges, r_bar, r_ucl, r_lcl)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("X̄̄", f"{xbar_bar:.4f}")
            st.metric("UCL", f"{xbar_ucl:.4f}")
            st.metric("LCL", f"{xbar_lcl:.4f}")
        with col2:
            st.metric("R̄", f"{r_bar:.4f}")
            st.metric("R UCL", f"{r_ucl:.4f}")
            st.metric("R LCL", f"{r_lcl:.4f}")

        st.plotly_chart(plot_control_chart(xbars, xbar_bar, xbar_ucl, xbar_lcl, "Xbar 관리도", xbar_v), use_container_width=True)
        st.plotly_chart(plot_control_chart(ranges, r_bar, r_ucl, r_lcl, "R 관리도", r_v), use_container_width=True)

        all_v = []
        for i, rules in xbar_v.items():
            for r in rules:
                all_v.append({'관리도': 'Xbar', '부분군': i+1, '값': f"{xbars[i]:.4f}", '규칙': r})
        for i, rules in r_v.items():
            for r in rules:
                all_v.append({'관리도': 'R', '부분군': i+1, '값': f"{ranges[i]:.4f}", '규칙': r})
        if all_v:
            st.subheader("이상 패턴 검출")
            st.dataframe(pd.DataFrame(all_v), hide_index=True, use_container_width=True)
        else:
            st.success("공정이 관리 상태입니다.")

# ==============================================================
# I-MR
# ==============================================================
elif chart_type == "I-MR":
    st.header("I-MR 관리도")

    data_source = st.sidebar.radio("데이터", ["STB 5.1 (Noise)", "STB 5.3 (측정값)", "직접 입력", "CSV 업로드"])

    if data_source == "STB 5.1 (Noise)":
        df_raw = load_stb('STB_5.1.csv')
        if df_raw is None:
            st.stop()
        data = df_raw['Noise'].dropna().values.astype(float)
        st.sidebar.success(f"STB 5.1 로드 ({len(data)}행)")
    elif data_source == "STB 5.3 (측정값)":
        df_raw = load_stb('STB_5.3.csv')
        if df_raw is None:
            st.stop()
        # 한글 컬럼 처리
        meas_col = [c for c in df_raw.columns if '측정' in str(c)]
        col = meas_col[0] if meas_col else df_raw.columns[1]
        data = df_raw[col].dropna().values.astype(float)
        st.sidebar.success(f"STB 5.3 로드 ({len(data)}행)")
    elif data_source == "직접 입력":
        raw = st.sidebar.text_area("데이터 (쉼표 구분)",
            "50.1, 49.3, 51.2, 48.8, 50.5, 49.7, 51.0, 50.3, 49.5, 50.8")
        data = np.array([float(v.strip()) for v in raw.split(",") if v.strip()])
    else:
        uploaded = st.sidebar.file_uploader("CSV", type="csv")
        if not uploaded:
            st.stop()
        df = pd.read_csv(uploaded)
        col = st.sidebar.selectbox("측정값", df.columns)
        data = df[col].dropna().values.astype(float)

    if st.button("▶ 분석 실행", type="primary"):
        mr = np.abs(np.diff(data))
        x_bar, mr_bar = np.mean(data), np.mean(mr)
        d2 = 1.128
        i_ucl = x_bar + 3 * mr_bar / d2
        i_lcl = x_bar - 3 * mr_bar / d2
        mr_ucl = 3.267 * mr_bar

        i_v = detect_we_rules(data, x_bar, i_ucl, i_lcl)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("CL", f"{x_bar:.4f}")
            st.metric("UCL", f"{i_ucl:.4f}")
            st.metric("LCL", f"{i_lcl:.4f}")
        with col2:
            st.metric("MR̄", f"{mr_bar:.4f}")
            st.metric("MR UCL", f"{mr_ucl:.4f}")
            st.metric("σ (MR̄/d2)", f"{mr_bar/d2:.4f}")

        st.plotly_chart(plot_control_chart(data, x_bar, i_ucl, i_lcl, "I 관리도", i_v), use_container_width=True)
        st.plotly_chart(plot_control_chart(mr, mr_bar, mr_ucl, 0, "MR 관리도"), use_container_width=True)

# ==============================================================
# P 관리도
# ==============================================================
elif chart_type == "P 관리도":
    st.header("P 관리도 (불량률)")

    data_source = st.sidebar.radio("데이터", ["STB 5.5 (불량률)", "직접 입력", "CSV 업로드"])

    if data_source == "STB 5.5 (불량률)":
        df_raw = load_stb('STB_5.5.csv')
        if df_raw is None:
            st.stop()
        n_vals = df_raw['Sample'].dropna().values.astype(int)
        d_vals = df_raw['NC'].dropna().values.astype(int)
        st.sidebar.success(f"STB 5.5 로드 ({len(n_vals)}행)")
    elif data_source == "직접 입력":
        raw_n = st.sidebar.text_area("검사 수 (n)", "100, 100, 100, 100, 100, 100, 100, 100, 100, 100")
        raw_d = st.sidebar.text_area("불량 수", "3, 5, 2, 4, 6, 3, 4, 5, 2, 8")
        n_vals = np.array([int(v.strip()) for v in raw_n.split(",") if v.strip()])
        d_vals = np.array([int(v.strip()) for v in raw_d.split(",") if v.strip()])
    else:
        uploaded = st.sidebar.file_uploader("CSV", type="csv")
        if not uploaded:
            st.stop()
        df = pd.read_csv(uploaded)
        n_col = st.sidebar.selectbox("검사 수", df.columns)
        d_col = st.sidebar.selectbox("불량 수", df.columns)
        n_vals = df[n_col].dropna().values.astype(int)
        d_vals = df[d_col].dropna().values.astype(int)

    if st.button("▶ 분석 실행", type="primary"):
        p_vals = d_vals / n_vals
        p_bar = np.sum(d_vals) / np.sum(n_vals)
        ucl_vals = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / n_vals)
        lcl_vals = np.maximum(0, p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / n_vals))

        st.metric("P̄", f"{p_bar:.4f} ({p_bar*100:.2f}%)")

        x = list(range(1, len(p_vals) + 1))
        colors = ['red' if p_vals[i] > ucl_vals[i] or p_vals[i] < lcl_vals[i] else '#2e75b6' for i in range(len(p_vals))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=p_vals, mode='lines+markers', marker=dict(color=colors, size=6), line=dict(color='#2e75b6')))
        fig.add_trace(go.Scatter(x=x, y=ucl_vals, mode='lines', line=dict(color='red', dash='dash'), name='UCL'))
        fig.add_trace(go.Scatter(x=x, y=lcl_vals, mode='lines', line=dict(color='red', dash='dash'), name='LCL'))
        fig.add_hline(y=p_bar, line_color="green", annotation_text=f"P̄={p_bar:.4f}")
        fig.update_layout(title="P 관리도", height=450)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# NP 관리도
# ==============================================================
elif chart_type == "NP 관리도":
    st.header("NP 관리도")

    data_source = st.sidebar.radio("데이터", ["STB 5.4 (불량 개수)", "직접 입력"])

    if data_source == "STB 5.4 (불량 개수)":
        df_raw = load_stb('STB_5.4.csv')
        if df_raw is None:
            st.stop()
        d_vals = df_raw['NC'].dropna().values.astype(int)
        n_fixed = st.sidebar.number_input("부분군 크기 (n)", value=100, min_value=1)
        st.sidebar.success(f"STB 5.4 로드 ({len(d_vals)}행)")
    else:
        raw_d = st.sidebar.text_area("불량 수", "3, 5, 2, 4, 6, 3, 4, 5, 2, 8, 3, 4, 2, 5, 3")
        d_vals = np.array([int(v.strip()) for v in raw_d.split(",") if v.strip()])
        n_fixed = st.sidebar.number_input("부분군 크기 (n)", value=100, min_value=1)

    if st.button("▶ 분석 실행", type="primary"):
        p_bar = np.sum(d_vals) / (len(d_vals) * n_fixed)
        np_bar = n_fixed * p_bar
        ucl = np_bar + 3 * np.sqrt(np_bar * (1 - p_bar))
        lcl = max(0, np_bar - 3 * np.sqrt(np_bar * (1 - p_bar)))

        st.metric("NP̄", f"{np_bar:.2f}")
        st.metric("UCL", f"{ucl:.2f}")
        st.metric("LCL", f"{lcl:.2f}")

        v = detect_we_rules(d_vals.astype(float), np_bar, ucl, lcl)
        st.plotly_chart(plot_control_chart(d_vals.astype(float), np_bar, ucl, lcl, "NP 관리도", v), use_container_width=True)

# ==============================================================
# C 관리도
# ==============================================================
elif chart_type == "C 관리도":
    st.header("C 관리도 (결점 수)")

    data_source = st.sidebar.radio("데이터", ["STB 5.6 (결점 수)", "직접 입력"])

    if data_source == "STB 5.6 (결점 수)":
        df_raw = load_stb('STB_5.6.csv')
        if df_raw is None:
            st.stop()
        c_vals = df_raw['NC'].dropna().values.astype(float)
        st.sidebar.success(f"STB 5.6 로드 ({len(c_vals)}행)")
    else:
        raw_c = st.sidebar.text_area("결점 수", "3, 5, 2, 7, 4, 6, 3, 5, 2, 4, 8, 3, 5, 4, 2")
        c_vals = np.array([float(v.strip()) for v in raw_c.split(",") if v.strip()])

    if st.button("▶ 분석 실행", type="primary"):
        c_bar = np.mean(c_vals)
        ucl = c_bar + 3 * np.sqrt(c_bar)
        lcl = max(0, c_bar - 3 * np.sqrt(c_bar))

        st.metric("C̄", f"{c_bar:.2f}")
        st.metric("UCL", f"{ucl:.2f}")
        st.metric("LCL", f"{lcl:.2f}")

        v = detect_we_rules(c_vals, c_bar, ucl, lcl)
        st.plotly_chart(plot_control_chart(c_vals, c_bar, ucl, lcl, "C 관리도", v), use_container_width=True)

# ==============================================================
# U 관리도
# ==============================================================
elif chart_type == "U 관리도":
    st.header("U 관리도 (단위당 결점 수)")

    data_source = st.sidebar.radio("데이터", ["STB 5.7 (단위당 결점)", "직접 입력"])

    if data_source == "STB 5.7 (단위당 결점)":
        df_raw = load_stb('STB_5.7.csv')
        if df_raw is None:
            st.stop()
        c_vals = df_raw['NC'].dropna().values.astype(float)
        n_vals = df_raw['No.Unit'].dropna().values.astype(float)
        st.sidebar.success(f"STB 5.7 로드 ({len(c_vals)}행)")
    else:
        raw_c = st.sidebar.text_area("결점 수", "3, 5, 2, 7, 4, 6, 3, 5, 2, 4")
        raw_n = st.sidebar.text_area("검사 단위 수", "10, 12, 8, 15, 10, 11, 9, 13, 10, 12")
        c_vals = np.array([float(v.strip()) for v in raw_c.split(",") if v.strip()])
        n_vals = np.array([float(v.strip()) for v in raw_n.split(",") if v.strip()])

    if st.button("▶ 분석 실행", type="primary"):
        u_vals = c_vals / n_vals
        u_bar = np.sum(c_vals) / np.sum(n_vals)
        ucl_vals = u_bar + 3 * np.sqrt(u_bar / n_vals)
        lcl_vals = np.maximum(0, u_bar - 3 * np.sqrt(u_bar / n_vals))

        st.metric("Ū", f"{u_bar:.4f}")

        x = list(range(1, len(u_vals) + 1))
        colors = ['red' if u_vals[i] > ucl_vals[i] or u_vals[i] < lcl_vals[i] else '#2e75b6' for i in range(len(u_vals))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=u_vals, mode='lines+markers', marker=dict(color=colors, size=6), line=dict(color='#2e75b6')))
        fig.add_trace(go.Scatter(x=x, y=ucl_vals, mode='lines', line=dict(color='red', dash='dash'), name='UCL'))
        fig.add_trace(go.Scatter(x=x, y=lcl_vals, mode='lines', line=dict(color='red', dash='dash'), name='LCL'))
        fig.add_hline(y=u_bar, line_color="green", annotation_text=f"Ū={u_bar:.4f}")
        fig.update_layout(title="U 관리도", height=450)
        st.plotly_chart(fig, use_container_width=True)
