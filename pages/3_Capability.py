import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import os

st.set_page_config(page_title="공정능력분석", layout="wide")
st.title("공정능력분석 (Process Capability)")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_stb(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# --- 데이터 입력 ---
data_source = st.sidebar.radio("데이터 입력", ["STB 2.5 (샘플)", "STB 2.8 (RAM used)", "CSV 업로드", "직접 입력"])

if data_source == "STB 2.5 (샘플)":
    df_raw = load_stb('STB_2.5.csv')
    if df_raw is None:
        st.error("STB_2.5.csv를 찾을 수 없습니다.")
        st.stop()
    num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    val_col = st.sidebar.selectbox("측정값 컬럼", num_cols, index=num_cols.index('KO') if 'KO' in num_cols else 0)
    df = df_raw[[val_col]].rename(columns={val_col: 'Value'}).dropna()
    data = df['Value'].values.astype(float)
    usl_default, lsl_default, target_default = float(np.mean(data) + 3*np.std(data)), float(np.mean(data) - 3*np.std(data)), float(np.mean(data))
    st.sidebar.success(f"STB 2.5 [{val_col}] 로드 완료 ({len(df)}행)")
elif data_source == "STB 2.8 (RAM used)":
    df_raw = load_stb('STB_2.8.csv')
    if df_raw is None:
        st.error("STB_2.8.csv를 찾을 수 없습니다.")
        st.stop()
    df = df_raw[['RAM used']].rename(columns={'RAM used': 'Value'}).dropna()
    data = df['Value'].values.astype(float)
    usl_default, lsl_default, target_default = 500.0, 200.0, 350.0
    st.sidebar.success(f"STB 2.8 로드 완료 ({len(df)}행)")
elif data_source == "CSV 업로드":
    uploaded = st.sidebar.file_uploader("CSV 파일", type="csv")
    if uploaded:
        df_raw = pd.read_csv(uploaded)
    else:
        st.info("CSV 파일을 업로드하세요.")
        st.stop()
    val_col = st.sidebar.selectbox("측정값 컬럼", df_raw.columns)
    df = df_raw[[val_col]].rename(columns={val_col: 'Value'}).dropna()
    data = df['Value'].values.astype(float)
    usl_default = float(np.max(data) + np.std(data))
    lsl_default = float(np.min(data) - np.std(data))
    target_default = float(np.mean(data))
else:
    raw = st.sidebar.text_area("측정값 (쉼표 구분)",
        "49.2, 50.1, 51.3, 48.7, 50.5, 49.8, 50.2, 51.0, 49.5, 50.8")
    vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
    df = pd.DataFrame({'Value': vals})
    data = df['Value'].values
    usl_default, lsl_default, target_default = max(vals) + 2, min(vals) - 2, np.mean(vals)

st.sidebar.markdown("---")
st.sidebar.subheader("규격 한계")
usl = st.sidebar.number_input("USL (상한 규격)", value=usl_default, format="%.3f")
lsl = st.sidebar.number_input("LSL (하한 규격)", value=lsl_default, format="%.3f")
target = st.sidebar.number_input("Target (목표값)", value=target_default, format="%.3f")
subgroup_size = st.sidebar.number_input("부분군 크기 (Within σ 추정)", value=5, min_value=2, max_value=50)

if st.button("▶ 분석 실행", type="primary"):
    data = df['Value'].values.astype(float)
    n = len(data)
    mean_val = np.mean(data)
    overall_std = np.std(data, ddof=1)

    d2_table = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704,
                8: 2.847, 9: 2.970, 10: 3.078, 15: 3.472, 20: 3.735, 25: 3.931, 50: 4.498}
    d2_keys = sorted(d2_table.keys())
    if subgroup_size in d2_table:
        d2 = d2_table[subgroup_size]
    else:
        lower = max(k for k in d2_keys if k <= subgroup_size)
        upper = min(k for k in d2_keys if k >= subgroup_size)
        d2 = d2_table[lower] + (d2_table[upper] - d2_table[lower]) * (subgroup_size - lower) / (upper - lower) if lower != upper else d2_table[lower]

    n_subgroups = n // subgroup_size
    if n_subgroups >= 2:
        ranges = [np.ptp(data[i * subgroup_size:(i + 1) * subgroup_size]) for i in range(n_subgroups)]
        rbar = np.mean(ranges)
        within_std = rbar / d2
    else:
        within_std = overall_std

    cp = (usl - lsl) / (6 * within_std) if within_std > 0 else np.inf
    cpu = (usl - mean_val) / (3 * within_std) if within_std > 0 else np.inf
    cpl = (mean_val - lsl) / (3 * within_std) if within_std > 0 else np.inf
    cpk = min(cpu, cpl)

    pp = (usl - lsl) / (6 * overall_std) if overall_std > 0 else np.inf
    ppu = (usl - mean_val) / (3 * overall_std) if overall_std > 0 else np.inf
    ppl = (mean_val - lsl) / (3 * overall_std) if overall_std > 0 else np.inf
    ppk = min(ppu, ppl)

    z_bench_within = min((usl - mean_val) / within_std, (mean_val - lsl) / within_std) if within_std > 0 else np.inf
    z_bench_overall = min((usl - mean_val) / overall_std, (mean_val - lsl) / overall_std) if overall_std > 0 else np.inf

    ppm_usl = stats.norm.sf((usl - mean_val) / overall_std) * 1_000_000 if overall_std > 0 else 0
    ppm_lsl = stats.norm.sf((mean_val - lsl) / overall_std) * 1_000_000 if overall_std > 0 else 0
    ppm_total = ppm_usl + ppm_lsl

    ad_result = stats.anderson(data, dist='norm')
    ad_stat = ad_result.statistic
    ad_critical = ad_result.critical_values[2]
    ad_pass = ad_stat < ad_critical

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("기본 통계")
        st.dataframe(pd.DataFrame({
            '항목': ['n', '평균', 'σ_within', 'σ_overall', 'USL', 'LSL'],
            '값': [n, f"{mean_val:.4f}", f"{within_std:.4f}", f"{overall_std:.4f}", f"{usl:.3f}", f"{lsl:.3f}"]
        }), hide_index=True, use_container_width=True)

    with col2:
        st.subheader("공정능력 지수")
        st.dataframe(pd.DataFrame({
            '지수': ['Cp', 'CPU', 'CPL', '**Cpk**', 'Pp', 'PPU', 'PPL', '**Ppk**'],
            '값': [f"{cp:.3f}", f"{cpu:.3f}", f"{cpl:.3f}", f"**{cpk:.3f}**",
                   f"{pp:.3f}", f"{ppu:.3f}", f"{ppl:.3f}", f"**{ppk:.3f}**"]
        }), hide_index=True, use_container_width=True)

    with col3:
        st.subheader("시그마 수준 & PPM")
        st.dataframe(pd.DataFrame({
            '항목': ['Z.bench (Within)', 'Z.bench (Overall)', 'PPM (합계)'],
            '값': [f"{z_bench_within:.2f}", f"{z_bench_overall:.2f}", f"{ppm_total:.1f}"]
        }), hide_index=True, use_container_width=True)
        if cpk >= 1.67:
            st.success("Cpk ≥ 1.67: 우수")
        elif cpk >= 1.33:
            st.info("Cpk ≥ 1.33: 양호")
        elif cpk >= 1.0:
            st.warning("Cpk ≥ 1.00: 최소 허용")
        else:
            st.error("Cpk < 1.00: 개선 필요")

    st.subheader("정규성 검정 (Anderson-Darling)")
    if ad_pass:
        st.success(f"AD = {ad_stat:.4f} (임계값 {ad_critical:.3f}) → 정규성 채택")
    else:
        st.warning(f"AD = {ad_stat:.4f} (임계값 {ad_critical:.3f}) → 정규분포 아닐 수 있음")

    st.subheader("공정능력 히스토그램")
    x_range = np.linspace(min(lsl - 2*overall_std, min(data) - overall_std),
                          max(usl + 2*overall_std, max(data) + overall_std), 500)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=25, name='데이터',
                               marker_color='rgba(46,117,182,0.5)', histnorm='probability density'))
    fig.add_trace(go.Scatter(x=x_range, y=stats.norm.pdf(x_range, mean_val, within_std),
                             mode='lines', name='Within', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=x_range, y=stats.norm.pdf(x_range, mean_val, overall_std),
                             mode='lines', name='Overall', line=dict(color='blue', width=2, dash='dash')))
    fig.add_vline(x=usl, line_color="red", line_dash="dash", annotation_text=f"USL={usl}")
    fig.add_vline(x=lsl, line_color="red", line_dash="dash", annotation_text=f"LSL={lsl}")
    fig.add_vline(x=mean_val, line_color="black", annotation_text=f"Mean={mean_val:.3f}")
    fig.update_layout(height=500, xaxis_title="측정값", yaxis_title="밀도", barmode='overlay')
    st.plotly_chart(fig, use_container_width=True)
