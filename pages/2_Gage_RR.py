import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="Gage R&R 분석", layout="wide")
st.title("Gage R&R 분석 도구")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_stb(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

analysis_type = st.sidebar.selectbox(
    "분석 유형",
    ["Type 1 Gage Study", "Type 2 - Crossed", "Type 2 - Nested", "Attribute Agreement"]
)

# ==============================================================
# Type 1 Gage Study
# ==============================================================
if analysis_type == "Type 1 Gage Study":
    st.header("Type 1 Gage Study")
    st.markdown("단일 측정자가 기준값이 있는 하나의 부품을 반복 측정하여 측정기 능력을 평가합니다.")

    data_source = st.sidebar.radio("데이터", ["STB 2.1 (샘플)", "CSV 업로드", "직접 입력"])

    if data_source == "STB 2.1 (샘플)":
        df = load_stb('STB_2.1.csv')
        if df is not None:
            df = df[['Data']].rename(columns={'Data': 'Measurement'}).dropna()
            st.sidebar.success(f"STB 2.1 로드 완료 ({len(df)}행)")
        else:
            st.error("STB_2.1.csv 파일을 찾을 수 없습니다.")
            st.stop()
        ref_value = st.sidebar.number_input("기준값 (Reference)", value=50.0, format="%.3f")
        tolerance = st.sidebar.number_input("공차 (Tolerance = USL - LSL)", value=6.0, format="%.3f")
    elif data_source == "CSV 업로드":
        uploaded = st.sidebar.file_uploader("CSV 파일", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            st.info("CSV 파일을 업로드하세요.")
            st.stop()
        meas_col = st.sidebar.selectbox("측정값 컬럼", df.columns)
        df = df[[meas_col]].rename(columns={meas_col: 'Measurement'})
        ref_value = st.sidebar.number_input("기준값", value=0.0, format="%.3f")
        tolerance = st.sidebar.number_input("공차 (USL - LSL)", value=6.0, format="%.3f")
    else:
        raw = st.sidebar.text_area("측정값 (쉼표 구분)", "50.1, 49.8, 50.3, 50.0, 49.9, 50.2")
        vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
        df = pd.DataFrame({'Measurement': vals})
        ref_value = st.sidebar.number_input("기준값", value=50.0, format="%.3f")
        tolerance = st.sidebar.number_input("공차 (USL - LSL)", value=6.0, format="%.3f")

    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    if st.button("▶ 분석 실행"):
        data = df['Measurement'].values.astype(float)
        n = len(data)
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        bias = mean_val - ref_value

        cg = (0.2 * tolerance) / (6 * std_val) if std_val > 0 else np.inf
        cgk = (0.1 * tolerance - abs(bias)) / (3 * std_val) if std_val > 0 else np.inf

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("기본 통계")
            stats_df = pd.DataFrame({
                '항목': ['측정 수', '평균', '표준편차', '기준값', 'Bias', '공차'],
                '값': [n, f"{mean_val:.4f}", f"{std_val:.4f}", f"{ref_value:.4f}",
                       f"{bias:.4f}", f"{tolerance:.4f}"]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

        with col2:
            st.subheader("Gage 능력 지수")
            judge_cg = "✅ 적합" if cg >= 1.33 else "❌ 부적합"
            judge_cgk = "✅ 적합" if cgk >= 1.33 else "❌ 부적합"
            cap_df = pd.DataFrame({
                '지수': ['Cg', 'Cgk'],
                '값': [f"{cg:.3f}", f"{cgk:.3f}"],
                '기준': ['≥ 1.33', '≥ 1.33'],
                '판정': [judge_cg, judge_cgk]
            })
            st.dataframe(cap_df, hide_index=True, use_container_width=True)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("측정값 Run Chart", "히스토그램"))
        fig.add_trace(go.Scatter(y=data, mode='lines+markers', name='측정값',
                                 marker=dict(color='#2e75b6')), row=1, col=1)
        fig.add_hline(y=ref_value, line_dash="dash", line_color="red",
                       annotation_text="기준값", row=1, col=1)
        fig.add_hline(y=mean_val, line_dash="dot", line_color="green",
                       annotation_text="평균", row=1, col=1)
        fig.add_trace(go.Histogram(x=data, nbinsx=15, marker_color='#2e75b6',
                                    name='분포'), row=1, col=2)
        fig.add_vline(x=ref_value, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_vline(x=mean_val, line_dash="dot", line_color="green", row=1, col=2)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# Type 2 - Crossed Gage R&R
# ==============================================================
elif analysis_type == "Type 2 - Crossed":
    st.header("Type 2 Gage R&R - Crossed")
    st.markdown("여러 측정자가 동일한 부품들을 반복 측정하여 측정시스템 변동을 분석합니다.")

    data_source = st.sidebar.radio("데이터", ["STB 2.2 (샘플)", "CSV 업로드"])

    if data_source == "STB 2.2 (샘플)":
        df = load_stb('STB_2.2.csv')
        if df is not None:
            st.sidebar.success(f"STB 2.2 로드 완료 ({len(df)}행)")
        else:
            st.error("STB_2.2.csv 파일을 찾을 수 없습니다.")
            st.stop()
        part_col = st.sidebar.selectbox("부품 컬럼", df.columns, index=df.columns.get_loc('Part') if 'Part' in df.columns else 0)
        op_col = st.sidebar.selectbox("측정자 컬럼", df.columns, index=df.columns.get_loc('Appraiser') if 'Appraiser' in df.columns else 1)
        meas_col = st.sidebar.selectbox("측정값 컬럼", df.columns, index=df.columns.get_loc('Response') if 'Response' in df.columns else 2)
    else:
        uploaded = st.sidebar.file_uploader("CSV 파일", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            st.info("CSV 파일을 업로드하세요.")
            st.stop()
        part_col = st.sidebar.selectbox("부품 컬럼", df.columns, index=0)
        op_col = st.sidebar.selectbox("측정자 컬럼", df.columns, index=1)
        meas_col = st.sidebar.selectbox("측정값 컬럼", df.columns, index=2)

    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    if st.button("▶ 분석 실행"):
        parts = df[part_col].values
        operators = df[op_col].values
        measurements = df[meas_col].values.astype(float)

        unique_parts = np.unique(parts)
        unique_ops = np.unique(operators)
        p = len(unique_parts)
        o = len(unique_ops)
        n_total = len(measurements)
        r = n_total // (p * o)

        grand_mean = np.mean(measurements)
        part_means = np.array([np.mean(measurements[parts == pt]) for pt in unique_parts])
        op_means = np.array([np.mean(measurements[operators == op]) for op in unique_ops])

        ss_part = o * r * np.sum((part_means - grand_mean) ** 2)
        ss_op = p * r * np.sum((op_means - grand_mean) ** 2)

        ss_interaction = 0
        for pt in unique_parts:
            for op in unique_ops:
                mask = (parts == pt) & (operators == op)
                cell_mean = np.mean(measurements[mask])
                pt_mean = np.mean(measurements[parts == pt])
                op_mean = np.mean(measurements[operators == op])
                ss_interaction += r * (cell_mean - pt_mean - op_mean + grand_mean) ** 2

        ss_total = np.sum((measurements - grand_mean) ** 2)
        ss_repeat = ss_total - ss_part - ss_op - ss_interaction

        df_part = p - 1
        df_op = o - 1
        df_inter = (p - 1) * (o - 1)
        df_repeat = p * o * (r - 1)
        df_total = n_total - 1

        ms_part = ss_part / df_part if df_part > 0 else 0
        ms_op = ss_op / df_op if df_op > 0 else 0
        ms_inter = ss_interaction / df_inter if df_inter > 0 else 0
        ms_repeat = ss_repeat / df_repeat if df_repeat > 0 else 0

        f_part = ms_part / ms_inter if ms_inter > 0 else np.inf
        f_op = ms_op / ms_inter if ms_inter > 0 else np.inf
        f_inter = ms_inter / ms_repeat if ms_repeat > 0 else np.inf

        p_part = 1 - stats.f.cdf(f_part, df_part, df_inter)
        p_op = 1 - stats.f.cdf(f_op, df_op, df_inter)
        p_inter = 1 - stats.f.cdf(f_inter, df_inter, df_repeat)

        var_repeat = ms_repeat
        var_inter = max(0, (ms_inter - ms_repeat) / r)
        var_op = max(0, (ms_op - ms_inter) / (p * r))
        var_part = max(0, (ms_part - ms_inter) / (o * r))

        var_gage_rr = var_repeat + var_op + var_inter
        var_total = var_part + var_gage_rr

        pct_contrib_repeat = (var_repeat / var_total * 100) if var_total > 0 else 0
        pct_contrib_reprod = ((var_op + var_inter) / var_total * 100) if var_total > 0 else 0
        pct_contrib_gage = (var_gage_rr / var_total * 100) if var_total > 0 else 0
        pct_contrib_part = (var_part / var_total * 100) if var_total > 0 else 0

        sd_total = np.sqrt(var_total) if var_total > 0 else 1
        pct_sv_repeat = (np.sqrt(var_repeat) / sd_total * 100)
        pct_sv_reprod = (np.sqrt(var_op + var_inter) / sd_total * 100)
        pct_sv_gage = (np.sqrt(var_gage_rr) / sd_total * 100)
        pct_sv_part = (np.sqrt(var_part) / sd_total * 100)

        ndc = max(1, int(np.sqrt(2) * np.sqrt(var_part) / np.sqrt(var_gage_rr))) if var_gage_rr > 0 else 999

        st.subheader("ANOVA 테이블")
        anova_df = pd.DataFrame({
            'Source': ['Part', 'Operator', 'Part × Operator', 'Repeatability', 'Total'],
            'DF': [df_part, df_op, df_inter, df_repeat, df_total],
            'SS': [f"{ss_part:.4f}", f"{ss_op:.4f}", f"{ss_interaction:.4f}",
                   f"{ss_repeat:.4f}", f"{ss_total:.4f}"],
            'MS': [f"{ms_part:.4f}", f"{ms_op:.4f}", f"{ms_inter:.4f}",
                   f"{ms_repeat:.4f}", ""],
            'F': [f"{f_part:.2f}", f"{f_op:.2f}", f"{f_inter:.2f}", "", ""],
            'p-value': [f"{p_part:.4f}", f"{p_op:.4f}", f"{p_inter:.4f}", "", ""]
        })
        st.dataframe(anova_df, hide_index=True, use_container_width=True)

        st.subheader("분산 성분 분석")
        col1, col2 = st.columns(2)
        with col1:
            var_df = pd.DataFrame({
                'Source': ['Gage R&R', '  Repeatability', '  Reproducibility', 'Part-to-Part', 'Total'],
                'VarComp': [f"{var_gage_rr:.6f}", f"{var_repeat:.6f}",
                           f"{var_op + var_inter:.6f}", f"{var_part:.6f}", f"{var_total:.6f}"],
                '%Contribution': [f"{pct_contrib_gage:.2f}%", f"{pct_contrib_repeat:.2f}%",
                                 f"{pct_contrib_reprod:.2f}%", f"{pct_contrib_part:.2f}%", "100%"],
                '%Study Var': [f"{pct_sv_gage:.2f}%", f"{pct_sv_repeat:.2f}%",
                              f"{pct_sv_reprod:.2f}%", f"{pct_sv_part:.2f}%", "100%"]
            })
            st.dataframe(var_df, hide_index=True, use_container_width=True)

        with col2:
            rr_pct = pct_sv_gage
            if rr_pct < 10:
                judge = "✅ 우수 (< 10%)"
            elif rr_pct < 30:
                judge = "⚠️ 조건부 수용 (10~30%)"
            else:
                judge = "❌ 부적합 (> 30%)"
            st.metric("%R&R (Study Var 기준)", f"{rr_pct:.1f}%")
            st.markdown(f"**판정**: {judge}")
            st.metric("ndc (Number of Distinct Categories)", ndc)

        st.subheader("분석 차트")
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("부품별 측정값", "측정자별 측정값",
                                          "%Contribution", "부품 × 측정자 교호작용"))
        for op in unique_ops:
            mask = operators == op
            fig.add_trace(go.Scatter(x=parts[mask].astype(str), y=measurements[mask],
                                     mode='markers', name=str(op)), row=1, col=1)
        for op in unique_ops:
            mask = operators == op
            fig.add_trace(go.Box(y=measurements[mask], name=str(op), showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=['Gage R&R', 'Repeat', 'Reprod', 'Part'],
                             y=[pct_contrib_gage, pct_contrib_repeat, pct_contrib_reprod, pct_contrib_part],
                             marker_color=['#e74c3c', '#f39c12', '#f39c12', '#2ecc71'],
                             showlegend=False), row=2, col=1)
        for op in unique_ops:
            op_part_means = [np.mean(measurements[(parts == pt) & (operators == op)]) for pt in unique_parts]
            fig.add_trace(go.Scatter(x=unique_parts.astype(str), y=op_part_means,
                                     mode='lines+markers', name=f"Op {op}", showlegend=False), row=2, col=2)
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# Type 2 - Nested
# ==============================================================
elif analysis_type == "Type 2 - Nested":
    st.header("Type 2 Gage R&R - Nested")
    st.markdown("측정자가 각자 다른 부품을 측정하는 경우 (파괴 검사 등)")

    data_source = st.sidebar.radio("데이터", ["STB 2.3 (샘플)", "CSV 업로드"])

    if data_source == "STB 2.3 (샘플)":
        df = load_stb('STB_2.3.csv')
        if df is not None:
            st.sidebar.success(f"STB 2.3 로드 완료 ({len(df)}행)")
        else:
            st.error("STB_2.3.csv 파일을 찾을 수 없습니다.")
            st.stop()
    else:
        uploaded = st.sidebar.file_uploader("CSV 파일", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            st.info("CSV 파일을 업로드하세요.")
            st.stop()

    part_col = st.sidebar.selectbox("부품 컬럼", df.columns, index=df.columns.get_loc('Part') if 'Part' in df.columns else 0)
    op_col = st.sidebar.selectbox("측정자 컬럼", df.columns, index=df.columns.get_loc('Operator') if 'Operator' in df.columns else 1)
    meas_col = st.sidebar.selectbox("측정값 컬럼", df.columns, index=df.columns.get_loc('Response') if 'Response' in df.columns else 2)

    st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    st.info("Nested Gage R&R: 부품이 측정자에 내포(nested)되어 교호작용 추정이 불가합니다.")

# ==============================================================
# Attribute Agreement Analysis
# ==============================================================
elif analysis_type == "Attribute Agreement":
    st.header("이산형 Gage R&R (Attribute Agreement Analysis)")
    st.markdown("합격/불합격 같은 범주형 판정의 일관성을 평가합니다.")

    data_source = st.sidebar.radio("데이터", ["STB 2.4 (샘플)", "CSV 업로드"])

    if data_source == "STB 2.4 (샘플)":
        df = load_stb('STB_2.4.csv')
        if df is not None:
            st.sidebar.success(f"STB 2.4 로드 완료 ({len(df)}행)")
        else:
            st.error("STB_2.4.csv 파일을 찾을 수 없습니다.")
            st.stop()
    else:
        uploaded = st.sidebar.file_uploader("CSV 파일", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            st.info("CSV 파일을 업로드하세요.")
            st.stop()

    part_col = st.sidebar.selectbox("부품 컬럼", df.columns, index=df.columns.get_loc('Part_No') if 'Part_No' in df.columns else 0)
    op_col = st.sidebar.selectbox("측정자 컬럼", df.columns, index=df.columns.get_loc('Appraiser') if 'Appraiser' in df.columns else 1)
    rating_col = st.sidebar.selectbox("판정 컬럼", df.columns, index=df.columns.get_loc('Data') if 'Data' in df.columns else 2)
    std_col = st.sidebar.selectbox("기준 컬럼", df.columns, index=df.columns.get_loc('Standard') if 'Standard' in df.columns else 3) if len(df.columns) > 3 else None

    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    if st.button("▶ 분석 실행"):
        unique_ops = df[op_col].unique()

        st.subheader("측정자 내 일치율 (Within Appraiser)")
        for op in unique_ops:
            op_data = df[df[op_col] == op]
            parts = op_data[part_col].unique()
            agree_count = 0
            for pt in parts:
                ratings = op_data[op_data[part_col] == pt][rating_col].values
                if len(set(ratings)) == 1:
                    agree_count += 1
            pct = agree_count / len(parts) * 100 if len(parts) > 0 else 0
            st.markdown(f"**{op}**: {agree_count}/{len(parts)} ({pct:.1f}%)")

        if std_col:
            st.subheader("기준 대비 일치율 (vs Standard)")
            for op in unique_ops:
                op_data = df[df[op_col] == op]
                matches = (op_data[rating_col] == op_data[std_col]).sum()
                total = len(op_data)
                pct = matches / total * 100 if total > 0 else 0
                st.markdown(f"**{op}**: {matches}/{total} ({pct:.1f}%)")
