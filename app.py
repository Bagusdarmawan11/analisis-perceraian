import os
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import tensorflow as tf

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# OPSIONAL: cek apakah statsmodels tersedia (untuk trendline di scatter)
# ----------------------------------------------------------------------
try:
    import statsmodels.api as sm  # noqa: F401
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ======================================================================
# KONFIGURASI HALAMAN
# ======================================================================
st.set_page_config(
    page_title="Sistem Prediksi Perceraian Jawa Barat",
    page_icon="📊",
    layout="wide",
)

# Sedikit CSS supaya tampilan lebih modern & responsif
st.markdown(
    """
    <style>
    .big-metric {
        font-size: 42px;
        font-weight: 700;
        margin-top: -10px;
    }
    .small-label {
        font-size: 14px;
        color: #777777;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    .stMetric {
        text-align: center;
    }
    [data-testid="stSidebar"] {
        min-width: 260px;
        max-width: 260px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================================
# KONFIGURASI PATH
# ======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DATA_FILE = None
if os.path.isdir(DATA_DIR):
    csv_list = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    if csv_list:
        # ambil CSV pertama di folder data
        DATA_FILE = os.path.join(DATA_DIR, csv_list[0])

PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "model_perceraian.h5")


# ======================================================================
# FUNGSI LOAD DATA & MODEL (CACHE)
# ======================================================================
@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_resource(show_spinner=True)
def load_artifacts(preproc_path: str, model_path: str):
    if not os.path.exists(preproc_path) or not os.path.exists(model_path):
        raise FileNotFoundError("preprocessor.joblib atau model_perceraian.h5 tidak ditemukan.")

    preprocessor = joblib.load(preproc_path)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "mse": tf.keras.losses.MeanSquaredError,
            "mae": tf.keras.metrics.MeanAbsoluteError,
        },
    )
    return preprocessor, model


def format_number(n: float) -> str:
    try:
        return f"{int(round(n)):,}".replace(",", ".")
    except Exception:
        return str(n)


# ======================================================================
# CEK KELENGKAPAN FILE
# ======================================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Aplikasi")
    st.caption("Pastikan struktur folder sudah sesuai:")

    st.code(
        "project-perceraian/\n"
        "├── app.py\n"
        "├── data/\n"
        "│   └── data.csv\n"
        "└── models/\n"
        "    ├── preprocessor.joblib\n"
        "    └── model_perceraian.h5",
        language="text",
    )

    if DATA_FILE:
        st.success(f"Dataset: `{os.path.basename(DATA_FILE)}` ditemukan.")
    else:
        st.error("❌ Tidak ada file CSV di folder `data/`.")

    if os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODEL_PATH):
        st.success("Model & preprocessor ditemukan di folder `models/`.")
    else:
        st.error("❌ Model atau preprocessor belum lengkap di folder `models/`.")

# Kalau dataset belum ada, hentikan
if DATA_FILE is None:
    st.stop()

# ======================================================================
# LOAD DATA & ARTIFACT
# ======================================================================
try:
    df = load_data(DATA_FILE)
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.stop()

try:
    preprocessor, model = load_artifacts(PREPROCESSOR_PATH, MODEL_PATH)
    model_loaded = True
except Exception as e:
    st.warning(
        f"Model tidak bisa dimuat: {e}\n\n"
        "Tab prediksi tetap muncul, tapi tombol prediksi akan dinonaktifkan."
    )
    model = None
    preprocessor = None
    model_loaded = False

# ======================================================================
# SIAPKAN INFORMASI KOLOM
# ======================================================================
target_col = "Jumlah"
year_col = "Tahun"
region_col = "Kabupaten/Kota"

if target_col not in df.columns or year_col not in df.columns or region_col not in df.columns:
    st.error(
        "Pastikan dataset memiliki kolom minimal: "
        f"`{target_col}`, `{year_col}`, `{region_col}`."
    )
    st.stop()

feature_cols = [c for c in df.columns if c != target_col]
categorical_cols = [region_col]
numeric_cols = [c for c in feature_cols if c not in categorical_cols]
factor_cols = [c for c in numeric_cols if c != year_col]

min_year = int(df[year_col].min())
max_year = int(df[year_col].max())
max_future_year = max_year + 6  # boleh prediksi beberapa tahun ke depan

region_list = sorted(df[region_col].dropna().unique().tolist())


# ======================================================================
# FUNGSI PREDIKSI
# ======================================================================
def predict_perceraian(input_dict: dict) -> float:
    if not model_loaded:
        raise RuntimeError("Model belum berhasil dimuat.")

    df_input = pd.DataFrame([input_dict])
    X = preprocessor.transform(df_input)
    if hasattr(X, "toarray"):
        X = X.toarray()
    y_pred = model.predict(X, verbose=0)
    return float(y_pred[0][0])


# ======================================================================
# HEADER HALAMAN
# ======================================================================
st.title("📊 Sistem Prediksi & Visualisasi Perceraian Jawa Barat")
st.caption("Model Deep Learning (MLP) dengan data 2019–2024")

# Ringkasan cepat
total_cases = df[target_col].sum()
num_regions = df[region_col].nunique()
year_span = f"{min_year}–{max_year}"

col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("Total Kasus di Dataset", format_number(total_cases))
col_s2.metric("Jumlah Kabupaten/Kota", int(num_regions))
col_s3.metric("Rentang Tahun Data", year_span)

st.markdown("---")

# ======================================================================
# TAB UTAMA
# ======================================================================
tab_pred, tab_wilayah, tab_faktor, tab_data = st.tabs(
    ["🔮 Prediksi", "🏙️ Grafik Wilayah", "📌 Analisis Faktor", "📁 Data"]
)

# ----------------------------------------------------------------------
# TAB 1: PREDIKSI
# ----------------------------------------------------------------------
with tab_pred:
    st.subheader("🔮 Prediksi Jumlah Perceraian")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### 🎯 Input Prediksi")

        # Pilih kabupaten/kota
        kabupaten_input = st.selectbox(
            "Pilih Kabupaten/Kota",
            options=region_list,
            index=0 if region_list else None,
        )

        # Pilih tahun prediksi
        tahun_pred = st.number_input(
            "Pilih Tahun Prediksi",
            min_value=min_year,
            max_value=max_future_year,
            value=max_year,
            step=1,
        )

        # Ambil nilai default faktor berdasarkan median wilayah & seluruh data
        df_region = df[df[region_col] == kabupaten_input] if kabupaten_input else df
        default_values_region = df_region[factor_cols].median(numeric_only=True)
        default_values_global = df[factor_cols].median(numeric_only=True)

        st.markdown("**🔧 Atur Faktor-faktor (opsional)**")
        faktor_values = {}
        for col in factor_cols:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            if pd.isna(col_min) or pd.isna(col_max) or col_min == col_max:
                col_min, col_max = 0.0, float(default_values_global.get(col, 1.0) * 2)

            default_val = float(
                default_values_region.get(col, default_values_global.get(col, (col_min + col_max) / 2))
            )
            step_val = (col_max - col_min) / 100 if col_max != col_min else 1.0

            faktor_values[col] = st.slider(
                label=col,
                min_value=float(col_min),
                max_value=float(col_max),
                value=float(default_val),
                step=float(step_val),
            )

        st.markdown("")
        pred_btn = st.button("🚀 Hitung Prediksi", use_container_width=True)

    with col_right:
        st.markdown("#### 📈 Hasil & Tren")

        if pred_btn:
            if not model_loaded:
                st.error("Model belum siap digunakan. Cek kembali folder `models/`.")
            else:
                # siapkan dictionary input
                input_dict = {
                    year_col: tahun_pred,
                    region_col: kabupaten_input,
                }
                for k, v in faktor_values.items():
                    input_dict[k] = v

                try:
                    y_pred = predict_perceraian(input_dict)
                    st.markdown('<p class="small-label">Perkiraan Jumlah Perceraian</p>', unsafe_allow_html=True)
                    st.markdown(
                        f'<p class="big-metric">{format_number(y_pred)}</p>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Gagal melakukan prediksi: {e}")
                    y_pred = None

                # Tampilkan tren historis + titik prediksi
                if y_pred is not None:
                    df_hist = df[df[region_col] == kabupaten_input].sort_values(year_col)

                    df_plot = df_hist[[year_col, target_col]].copy()
                    df_plot["Tipe"] = "Aktual"

                    # tambahkan titik prediksi (kalau tahun ke depan)
                    if tahun_pred not in df_plot[year_col].values:
                        df_pred_row = pd.DataFrame(
                            {
                                year_col: [tahun_pred],
                                target_col: [y_pred],
                                "Tipe": ["Prediksi"],
                            }
                        )
                        df_plot = pd.concat([df_plot, df_pred_row], ignore_index=True)
                    else:
                        # jika tahun sudah ada, tandai sebagai "Prediksi" tapi tetap tampil
                        df_plot.loc[df_plot[year_col] == tahun_pred, "Tipe"] = "Prediksi"

                    fig_line = px.line(
                        df_plot.sort_values(year_col),
                        x=year_col,
                        y=target_col,
                        color="Tipe",
                        markers=True,
                        title=f"Tren Jumlah Perceraian di {kabupaten_input}",
                    )
                    fig_line.update_layout(legend_title_text="", height=380)
                    st.plotly_chart(fig_line, use_container_width=True)

        else:
            st.info("Masukkan parameter di kiri lalu klik **🚀 Hitung Prediksi** untuk melihat hasil.")

# ----------------------------------------------------------------------
# TAB 2: GRAFIK WILAYAH
# ----------------------------------------------------------------------
with tab_wilayah:
    st.subheader("🏙️ Distribusi Perceraian per Kabupaten/Kota")

    c1, c2 = st.columns([2, 1])

    with c1:
        year_range = st.slider(
            "Pilih Rentang Tahun",
            min_value=min_year,
            max_value=max_year,
            value=(max(min_year, max_year - 2), max_year),
            step=1,
        )

    with c2:
        chart_type = st.radio(
            "Jenis Grafik",
            options=["Bar", "Line"],
            horizontal=True,
        )

    df_range = df[(df[year_col] >= year_range[0]) & (df[year_col] <= year_range[1])]

    # agregasi total per wilayah
    df_agg = (
        df_range.groupby(region_col)[target_col]
        .sum()
        .reset_index()
        .sort_values(target_col, ascending=False)
    )

    col_g1, col_g2 = st.columns([3, 2])

    with col_g1:
        if chart_type == "Bar":
            fig_bar = px.bar(
                df_agg,
                x=region_col,
                y=target_col,
                title=f"Total Perceraian per Wilayah ({year_range[0]}–{year_range[1]})",
            )
            fig_bar.update_layout(xaxis_tickangle=-45, height=450)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            fig_line = px.line(
                df_range,
                x=year_col,
                y=target_col,
                color=region_col,
                markers=True,
                title=f"Tren Perceraian per Wilayah ({year_range[0]}–{year_range[1]})",
            )
            fig_line.update_layout(height=450)
            st.plotly_chart(fig_line, use_container_width=True)

    with col_g2:
        st.markdown("#### 🔝 10 Wilayah dengan Kasus Tertinggi")
        st.dataframe(
            df_agg.head(10).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

# ----------------------------------------------------------------------
# TAB 3: ANALISIS FAKTOR
# ----------------------------------------------------------------------
with tab_faktor:
    st.subheader("📌 Analisis Faktor-faktor yang Berkaitan")

    if not factor_cols:
        st.info("Tidak ada kolom faktor numerik selain tahun. Cek lagi dataset kamu.")
    else:
        if not HAS_STATSMODELS:
            st.info(
                "Garis trendline membutuhkan paket `statsmodels`. "
                "Pastikan sudah ditambahkan di `requirements.txt` jika ingin menampilkannya."
            )

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.markdown("#### 🔗 Korelasi dengan Jumlah Perceraian")

            corr_cols = factor_cols + [target_col]
            df_corr = df[corr_cols].corr(numeric_only=True)

            fig_corr = px.imshow(
                df_corr[[target_col]].T,
                text_auto=".2f",
                aspect="auto",
                labels=dict(color="Korelasi"),
                title="Korelasi Faktor terhadap Jumlah Perceraian",
            )
            fig_corr.update_layout(height=420)
            st.plotly_chart(fig_corr, use_container_width=True)

        with col_f2:
            st.markdown("#### 📉 Scatter Plot Faktor vs Jumlah Perceraian")

            selected_factor = st.selectbox(
                "Pilih Faktor",
                options=factor_cols,
            )

            df_sample = df[[selected_factor, target_col]].dropna()
            if len(df_sample) > 1000:
                df_sample = df_sample.sample(1000, random_state=42)

            # gunakan trendline hanya jika statsmodels tersedia
            trendline_opt = "ols" if HAS_STATSMODELS else None

            fig_scatter = px.scatter(
                df_sample,
                x=selected_factor,
                y=target_col,
                trendline=trendline_opt,
                title=f"{selected_factor} vs {target_col}",
            )
            fig_scatter.update_layout(height=420)
            st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 4: DATA
# ----------------------------------------------------------------------
with tab_data:
    st.subheader("📁 Data Mentah")

    st.write(
        "Tabel di bawah adalah data yang digunakan oleh model. "
        "Kamu bisa filter berdasarkan kabupaten/kota dan tahun, lalu download."
    )

    col1, col2 = st.columns(2)

    with col1:
        region_filter = st.selectbox(
            "Filter Kabupaten/Kota",
            options=["(Semua)"] + region_list,
            index=0,
        )

    with col2:
        years_available = sorted(df[year_col].unique())
        year_filter = st.multiselect(
            "Filter Tahun",
            options=years_available,
            default=years_available,
        )

    df_view = df[df[year_col].isin(year_filter)]
    if region_filter != "(Semua)":
        df_view = df_view[df_view[region_col] == region_filter]

    st.dataframe(df_view, use_container_width=True, hide_index=True)

    csv = df_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Data yang Difilter",
        data=csv,
        file_name="data_perceraian_jabar_filtered.csv",
        mime="text/csv",
    )
