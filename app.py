import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import urllib.parse
import warnings

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Dashboard Ratu Jaya",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 15px;
        font-weight: 700;
    }

    div[data-testid="stMetricLabel"] * {
        font-size: 18px !important;
        font-weight: 800 !important;
        color: #1f3864 !important;
        opacity: 1 !important;
    }
    div[data-testid="stMetricValue"] * {
        font-size: 30px !important;
        font-weight: 900 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 30px !important;
        font-weight: 900 !important;
    }

    .section-heading {
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 12px;
        color: #1f3864;
    }

    .big-total {
        font-size: 32px;
        font-weight: 900;
        color: #1f3864;
        margin: 4px 0 16px 0;
    }

    .hero-row {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 6px;
    }
    .hero-card {
        flex: 1;
        min-width: 230px;
        border-radius: 16px;
        padding: 24px 28px;
        color: white;
    }
    .hero-label {
        font-size: 18px;
        font-weight: 700;
        opacity: 0.95;
        margin-bottom: 8px;
    }
    .hero-value {
        font-size: 38px;
        font-weight: 900;
        line-height: 1.15;
    }
    .hero-blue   { background: linear-gradient(135deg, #1f77b4, #14507a); }
    .hero-green  { background: linear-gradient(135deg, #2ca02c, #1d7a1d); }
    .hero-orange { background: linear-gradient(135deg, #ff7f0e, #b35900); }
    .hero-red    { background: linear-gradient(135deg, #d62728, #8b1a1a); }

    .income-card-title {
        font-size: 19px;
        font-weight: 800;
        color: #1f3864;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 2px solid #e0e6f0;
    }
</style>
""", unsafe_allow_html=True)

# KONFIGURASI SUMBER DATA
if "spreadsheet_id" not in st.secrets:
    st.error("Secret 'spreadsheet_id' belum diatur di settings/secrets.toml.")
    st.stop()

SPREADSHEET_ID = st.secrets.get("spreadsheet_id")

# Sheet PENJUALAN (Pendapatan Lapak & Analisa Lapak) sekarang diambil dari
# spreadsheet terpisah ini, bukan dari SPREADSHEET_ID utama.
PENJUALAN_LAPAK_SPREADSHEET_ID = "1p0swTGBCLA0XjNOU-bXYu1a4ECc4R2DKW1VvX7y4fWA"

SHEET_ARUS_KAS        = "ARUS KAS"
SHEET_PENGELUARAN     = "PENGELUARAN LAPAK"
SHEET_PENJUALAN       = "PENJUALAN"
SHEET_PENJUALAN_LUAR  = "PENJUALAN LAPAK LUAR"
SHEET_PIUTANG         = "PIUTANG LAPAK"
SHEET_PIUTANG_LUAR    = "PIUTANG LAPAK LUAR"
SHEET_HUTANG_PETANI   = "HUTANG PETANI"
SHEET_EKSPEDISI       = "EKSPEDISI"
SHEET_TANAMAN_BELUM   = "TANAMAN BELUM PANEN"
SHEET_TANAMAN_SUDAH   = "TANAMAN PANEN"
SHEET_KERUGIAN_GUDANG = "KERUGIAN GUDANG"
SHEET_HUTANG_PAKE_TANI = "HUTANG PAK'E TANI"
SHEET_STOK_LAPAK      = "STOK LAPAK"
SHEET_STOK_GUDANG     = "STOK GUDANG"
SHEET_BARANG_MASUK    = "BARANG_MASUK"

GH_SPREADSHEET_ID  = "17MhBomkR5qaLs0tOu6CO4H1pDTn1BV6_7hxOrvcVWSE"
SHEET_GH_BAHAN     = "BAHAN"
SHEET_GH_PEMUPUKAN = "PEMUPUKAN"
SHEET_GH_TENAGA    = "TENAGA"
SHEET_GH_LOKASI    = "LOKASI"
SHEET_GH_KAS       = "GREEN HOUSE KAS"
SHEET_GH_TANAMAN   = "GREEN HOUSE TANAMAN"

HARGA_COK_AB  = 4428
HARGA_COK_RUT = 2214

# LOGIN
def check_password() -> bool:
    def password_entered():
        if st.session_state.get("password") == st.secrets.get("app_password"):
            st.session_state["password_correct"] = True
            st.session_state.pop("password", None)
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct"):
        return True

    st.title("🔒 Login Dashboard Ratu Jaya")
    st.text_input("Masukkan password", type="password", on_change=password_entered, key="password")
    if st.session_state.get("password_correct") is False:
        st.error("Password salah, coba lagi.")
    return False

if "app_password" not in st.secrets:
    st.error("Secret 'app_password' belum diatur di settings/secrets.toml.")
    st.stop()

if not check_password():
    st.stop()

# HELPER FUNGSI
def to_number(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    def parse_val(x):
        if pd.isna(x): return np.nan
        x = str(x).strip()
        if x == "" or x.lower() in ("nan", "none", "-", "rp -", "rp-"): return np.nan
        x = x.replace("Rp", "").replace("rp", "").strip()
        x = "".join(ch for ch in x if ch.isdigit() or ch in ",.-")
        if x in ("", "-", ".", ","): return np.nan
        has_comma, has_dot = "," in x, "." in x
        if has_comma and has_dot:
            x = x.replace(".", "").replace(",", ".")
        elif has_comma:
            x = x.replace(",", ".")
        elif has_dot:
            parts = x.split(".")
            if len(parts) > 2: x = x.replace(".", "")
            elif len(parts[-1]) == 3: x = x.replace(".", "")
        return x
    return pd.to_numeric(series.map(parse_val), errors="coerce")

def is_filled(x) -> bool:
    return not pd.isna(x) and str(x).strip() != "" and str(x).strip().lower() not in ("nan", "none", "-")

def rp(x) -> str:
    if pd.isna(x) or x is None: return "Rp 0"
    return "Rp " + f"{x:,.0f}".replace(",", ".")

def rp_short(x) -> str:
    if pd.isna(x) or x is None: return "Rp 0"
    if abs(x) >= 1_000_000_000:
        return f"Rp {x/1_000_000_000:.1f}M"
    elif abs(x) >= 1_000_000:
        return f"Rp {x/1_000_000:.1f}Jt"
    elif abs(x) >= 1_000:
        return f"Rp {x/1_000:.1f}Rb"
    return f"Rp {x:,.0f}"

def drop_placeholder_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    drop_cols = [c for c in out.columns if str(c).startswith("_col_")]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    empty_cols = [c for c in out.columns if not out[c].apply(is_filled).any()]
    if empty_cols:
        out = out.drop(columns=empty_cols)
    return out

def format_money_table(df: pd.DataFrame, extra_keywords=None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    money_keywords = [
        "HUTANG", "PIUTANG", "TERBAYAR", "PAYMENT", "BAYAR", "PEMBAYARAN",
        "ANGSURAN", "SISA", "NOMINAL", "OUTSTANDING", "TOTAL", "PINJAMAN",
        "KERUGIAN", "OMZET", "LABA", "MODAL", "BIAYA", "HARGA",
    ]
    if extra_keywords:
        money_keywords = money_keywords + [k.upper() for k in extra_keywords]
    out = df.copy()
    for c in out.columns:
        cu = str(c).strip().upper()
        if any(k in cu for k in money_keywords):
            num = to_number(out[c])
            if num.notna().any():
                out[c] = num.apply(rp)
    return out

def hero_card(label: str, value: str, css_class: str = "hero-blue") -> str:
    return f"""<div class="hero-card {css_class}">
        <div class="hero-label">{label}</div>
        <div class="hero-value">{value}</div>
    </div>"""

def section_heading(text: str):
    st.markdown(f'<div class="section-heading">{text}</div>', unsafe_allow_html=True)

def pad_yaxis(fig, max_value, pad: float = 0.22):
    if max_value and max_value > 0:
        fig.update_yaxes(range=[0, max_value * (1 + pad)])
    fig.update_traces(cliponaxis=False)
    return fig

LGB_PARAMS = dict(n_estimators=500, learning_rate=0.03, max_depth=5)
PROPHET_PARAMS = dict(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.15)

# ENGINE PREDIKSI HARGA (tab Prediksi Harga)
def create_time_features(df):
    df = df.copy()
    dow = df["TANGGAL"].dt.dayofweek
    doy = df["TANGGAL"].dt.dayofyear
    df["DOW_SIN"] = np.sin(2 * np.pi * dow / 7)
    df["DOW_COS"] = np.cos(2 * np.pi * dow / 7)
    df["DOY_SIN"] = np.sin(2 * np.pi * doy / 365.25)
    df["DOY_COS"] = np.cos(2 * np.pi * doy / 365.25)
    return df

@st.cache_data(show_spinner=False)
def run_lightgbm(df_known, df_future, use_mbg=False):
    df_feat = create_time_features(df_known)

    for i in range(1, 8):
        df_feat[f"LAG_{i}"] = df_feat["HARGA"].shift(i)
    df_feat["TARGET_DELTA"] = df_feat["HARGA"] - df_feat["LAG_1"]

    feature_cols = [f"LAG_{i}" for i in range(1, 8)] + ["DOW_SIN", "DOW_COS", "DOY_SIN", "DOY_COS"]
    if use_mbg:
        feature_cols.append("MBG")

    df_train = df_feat.dropna(subset=feature_cols + ["TARGET_DELTA"]).reset_index(drop=True)

    model = LGBMRegressor(**LGB_PARAMS, random_state=42, verbose=-1)
    model.fit(df_train[feature_cols], df_train["TARGET_DELTA"])

    history_prices = list(df_feat["HARGA"].values[-7:])

    df_future_feat = create_time_features(df_future).reset_index(drop=True)
    future_dates = df_future_feat["TANGGAL"].values

    predictions = []
    for idx in range(len(future_dates)):
        row = {f"LAG_{i}": history_prices[-i] for i in range(1, 8)}
        row["DOW_SIN"] = df_future_feat.loc[idx, "DOW_SIN"]
        row["DOW_COS"] = df_future_feat.loc[idx, "DOW_COS"]
        row["DOY_SIN"] = df_future_feat.loc[idx, "DOY_SIN"]
        row["DOY_COS"] = df_future_feat.loc[idx, "DOY_COS"]
        if use_mbg:
            row["MBG"] = df_future_feat.loc[idx, "MBG"]

        x_pred = pd.DataFrame([row])[feature_cols]
        pred_delta = model.predict(x_pred)[0]
        pred_val = history_prices[-1] + float(pred_delta)

        predictions.append(pred_val)
        history_prices.append(pred_val)

    return pd.DataFrame({"TANGGAL": future_dates, "PREDIKSI": np.round(predictions, 2)})

@st.cache_data(show_spinner=False)
def run_prophet(df_known, df_future, use_mbg=False):
    df_p = df_known.rename(columns={"TANGGAL": "ds", "HARGA": "y"})
    model = Prophet(daily_seasonality=False, **PROPHET_PARAMS)

    cols = ["ds", "y"]
    if use_mbg:
        model.add_regressor("MBG")
        cols.append("MBG")
    model.fit(df_p[cols])

    df_p_hist = df_known[["TANGGAL"]].rename(columns={"TANGGAL": "ds"})
    if use_mbg:
        df_p_hist["MBG"] = df_known["MBG"]

    df_p_fut = df_future[["TANGGAL"]].rename(columns={"TANGGAL": "ds"})
    if use_mbg:
        df_p_fut["MBG"] = df_future["MBG"]

    future = pd.concat([df_p_hist, df_p_fut], ignore_index=True)

    forecast = model.predict(future)
    future_only = forecast.tail(len(df_future))

    return pd.DataFrame({
        "TANGGAL": pd.to_datetime(future_only["ds"]),
        "PREDIKSI": np.round(future_only["yhat"].values, 2)
    })

@st.cache_data(ttl=300, show_spinner=False)
# BACA G-SHEETS
def fetch_raw_csv(sheet_name: str, spreadsheet_id: str = None) -> pd.DataFrame:
    sid = spreadsheet_id or SPREADSHEET_ID
    encoded = urllib.parse.quote(sheet_name)
    url = f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={encoded}"
    try:
        df = pd.read_csv(url, dtype=str, header=0)
    except Exception as e:
        st.warning(f"Gagal membaca sheet '{sheet_name}': {e}")
        return pd.DataFrame()

    def col_letter(n):
        result = ""
        while n >= 0:
            result = chr(n % 26 + ord("A")) + result
            n = n // 26 - 1
        return result

    new_cols = []
    for i, c in enumerate(df.columns):
        if str(c).startswith("Unnamed") or str(c).strip() == "" or str(c).lower() in ["nan", "none"]:
            new_cols.append(f"_col_{col_letter(i)}")
        else:
            new_cols.append(str(c).strip())
    df.columns = new_cols
    return df.dropna(how="all").reset_index(drop=True)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_clean_csv(sheet_name: str) -> pd.DataFrame:
    encoded = urllib.parse.quote(sheet_name)
    url = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/gviz/tq?tqx=out:csv&sheet={encoded}"
    try:
        df = pd.read_csv(url, dtype=str)
    except Exception as e:
        st.warning(f"Gagal membaca sheet '{sheet_name}': {e}")
        return pd.DataFrame()
    valid_cols = [c for c in df.columns if not str(c).startswith("Unnamed") and str(c).lower() not in ["nan", "none", ""]]
    df = df[valid_cols]
    df = df.loc[:, ~df.columns.duplicated()]
    return df.dropna(how="all").reset_index(drop=True)

# LOADERS
def _parse_tanggal(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    if all(c in cols for c in ["Tanggal", "Bulan", "Tahun"]):
        def _safe_date(row):
            try:
                return pd.Timestamp(year=int(float(row["Tahun"])),
                                    month=int(float(row["Bulan"])),
                                    day=int(float(row["Tanggal"])))
            except:
                return pd.NaT
        df["Tanggal_Lengkap"] = df.apply(_safe_date, axis=1)
    else:
        tgl_col = next(
            (c for c in cols if c.strip().upper() in ["TANGGAL", "TGL", "DATE", "DD/MM/YYYY"]),
            None
        )
        df["Tanggal_Lengkap"] = pd.to_datetime(df[tgl_col], dayfirst=True, errors="coerce") if tgl_col else pd.NaT
    return df

@st.cache_data(ttl=300, show_spinner="Memuat Penjualan Lapak...")
def load_penjualan_lapak() -> pd.DataFrame:
    df = fetch_raw_csv(SHEET_PENJUALAN, spreadsheet_id=PENJUALAN_LAPAK_SPREADSHEET_ID)
    if df.empty:
        return df

    all_cols = list(df.columns)

    def _col_at(idx):
        return all_cols[idx] if idx < len(all_cols) else None

    def _get_col_by_pos_or_name(idx, name_candidates):
        for name in name_candidates:
            matches = [c for c in all_cols if c.strip().lower() == name.lower()]
            if matches:
                return matches[0]
        if idx < len(all_cols):
            return all_cols[idx]
        return None

    # Omzet/Laba/Kredit/Tunai sudah dikonfirmasi posisinya persis di sheet
    # PENJUALAN: S=Omzet, T=Laba, U=Kredit, V=Tunai. Diambil langsung by posisi
    # (bukan dicari by nama dulu) supaya tidak salah ambil kolom lain yang
    # kebetulan juga bernama umum seperti "Total".
    col_omzet   = _col_at(18)
    col_laba    = _col_at(19)
    col_kredit  = _col_at(20)
    col_tunai   = _col_at(21)
    # TANGGAL sebelumnya tidak pernah di-rename eksplisit di loader ini (beda dari
    # load_penjualan_lapak_luar/load_stok_lapak/load_stok_gudang yang semuanya sudah
    # menangani ini) -- kalau header asli kolom tanggal di sheet bukan persis
    # "TANGGAL", _parse_tanggal() gagal menemukannya sama sekali, Tanggal_Lengkap
    # jadi kosong untuk SEMUA baris, dan filter tanggal di sidebar otomatis
    # membuang semua data (itu sebabnya Omzet/Laba/Tunai/Kredit Lapak kebaca 0).
    col_tanggal = _get_col_by_pos_or_name(0, ["tanggal", "tgl", "date", "timestamp"])
    # Kolom B = "Kode Lapak": nilainya bisa berupa kode Gudang (mis. GDC) kalau
    # terjual langsung di gudang, atau kode Lapak (mis. CKP/JTK2) kalau sudah
    # di-moving -- satu kolom yang sama dipakai untuk pencocokan Terjual/Pendapatan
    # Gudang maupun Terjual/Pendapatan Lapak di Rincian per Invoice.
    col_kode_lapak = _get_col_by_pos_or_name(1, ["kode lapak", "kode_lapak", "lapak"])
    col_jenis   = _get_col_by_pos_or_name(13, ["jenis", "jenis tanaman", "nama barang", "produk", "komoditas"])
    col_grade   = _get_col_by_pos_or_name(14, ["grade", "kelas", "mutu"])
    col_kg      = _get_col_by_pos_or_name(15, ["jumlah (kg)", "jumlah kg", "kg", "jumlah", "berat"])
    col_invoice = _get_col_by_pos_or_name(5,  ["invoice", "no invoice", "nomor invoice"])

    def _find(names):
        for n in names:
            m = [c for c in all_cols if c.strip().lower() == n.lower()]
            if m: return m[0]
        return None

    col_keterangan = _find(["keterangan", "ket", "note", "catatan"])

    rename_map = {}
    if col_tanggal and col_tanggal != "TANGGAL":
        rename_map[col_tanggal] = "TANGGAL"
    if col_omzet and col_omzet != "Total harga":
        rename_map[col_omzet] = "Total harga"
    if col_laba and col_laba != "Keuntungan":
        rename_map[col_laba] = "Keuntungan"
    if col_jenis and col_jenis != "JENIS":
        rename_map[col_jenis] = "JENIS"
    if col_grade and col_grade != "GRADE":
        rename_map[col_grade] = "GRADE"
    if col_kg and col_kg != "Jumlah (KG)":
        rename_map[col_kg] = "Jumlah (KG)"
    if col_invoice and col_invoice != "INVOICE":
        rename_map[col_invoice] = "INVOICE"
    if col_tunai and col_tunai != "Tunai":
        rename_map[col_tunai] = "Tunai"
    if col_kredit and col_kredit != "Kredit":
        rename_map[col_kredit] = "Kredit"
    if col_kode_lapak and col_kode_lapak != "KODE LAPAK":
        rename_map[col_kode_lapak] = "KODE LAPAK"
    if col_keterangan and col_keterangan != "Keterangan":
        rename_map[col_keterangan] = "Keterangan"
    if rename_map:
        df = df.rename(columns=rename_map)
        # Jaga-jaga: kalau salah satu target rename di atas kebetulan sudah ada
        # sebagai nama kolom asli di posisi lain, df[col] bisa mengembalikan
        # DataFrame (bukan Series) dan bikin to_number() error. Kolom pertama
        # (posisi paling kiri) yang dipakai.
        df = df.loc[:, ~df.columns.duplicated()]

    for col in ["Total harga", "Keuntungan", "Tunai", "Kredit", "Jumlah (KG)"]:
        if col in df.columns:
            df[col] = to_number(df[col])
    # Distandarkan (strip spasi) supaya pencocokan Jenis/Grade/Gudang/Kode
    # Lapak/Invoice di Rincian per Invoice tidak meleset gara-gara spasi
    # tersembunyi.
    for col in ["JENIS", "GRADE", "KODE LAPAK", "INVOICE"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df = _parse_tanggal(df)

    df["Is_Dibuang"] = df["Keterangan"].apply(is_filled) if "Keterangan" in df.columns else False
    if "KODE LAPAK" in df.columns:
        df = df[df["KODE LAPAK"].apply(is_filled)].reset_index(drop=True)

    return df

@st.cache_data(ttl=300, show_spinner="Memuat Penjualan Lapak Luar...")
def load_penjualan_lapak_luar() -> pd.DataFrame:
    df = fetch_raw_csv(SHEET_PENJUALAN_LUAR)
    if df.empty:
        return df

    all_cols = list(df.columns)

    def _col_at(idx):
        return all_cols[idx] if idx < len(all_cols) else None

    # Posisi kolom sheet PENJUALAN LAPAK LUAR:
    # A=Tanggal, B=Tanggal Nota Balik, C=Invoice, D=Nama Pelanggan,
    # H=Grade, J=Tonnase Nota Balik, R=Omzet, S=Laba.
    # Omzet/Laba sudah 2x pindah posisi (S/T -> R/S) sementara header teksnya tetap
    # "OMZET"/"LABA" -- jadi keduanya dicocokkan by nama dulu, posisi cuma cadangan.
    col_tanggal       = _col_at(0)
    col_tgl_notabalik = _col_at(1)
    col_invoice       = _col_at(2)
    col_nama          = _col_at(3)
    col_grade         = _col_at(7)
    col_tonase_lahan  = _col_at(8)
    col_kg            = _col_at(9)
    col_harga_beli    = _col_at(11)
    col_modal         = _col_at(15)
    col_nota_balik    = _col_at(17)

    def _find_or_pos(names, idx):
        for n in names:
            m = [c for c in all_cols if c.strip().lower() == n.lower()]
            if m:
                return m[0]
        return _col_at(idx)

    col_omzet = _find_or_pos(["omzet", "total harga", "total_harga"], 17)
    col_laba  = _find_or_pos(["laba", "keuntungan", "profit"], 18)

    def _find(names):
        for n in names:
            m = [c for c in all_cols if c.strip().lower() == n.lower()]
            if m: return m[0]
        return None

    col_tunai      = _find(["tunai", "cash"])
    col_kredit     = _find(["kredit", "credit", "piutang"])
    col_keterangan = _find(["keterangan", "ket", "note", "catatan"])

    rename_map = {}
    if col_tanggal and col_tanggal != "TANGGAL":
        rename_map[col_tanggal] = "TANGGAL"
    if col_tgl_notabalik and col_tgl_notabalik != "TANGGAL NOTA BALIK":
        rename_map[col_tgl_notabalik] = "TANGGAL NOTA BALIK"
    if col_invoice and col_invoice != "INVOICE":
        rename_map[col_invoice] = "INVOICE"
    if col_nama and col_nama != "NAMA PELANGGAN":
        rename_map[col_nama] = "NAMA PELANGGAN"
    if col_grade and col_grade != "GRADE":
        rename_map[col_grade] = "GRADE"
    if col_tonase_lahan and col_tonase_lahan != "_RAW_LUAR_TONASE_LAHAN_KG":
        rename_map[col_tonase_lahan] = "_RAW_LUAR_TONASE_LAHAN_KG"
    if col_kg and col_kg != "Jumlah (KG)":
        rename_map[col_kg] = "Jumlah (KG)"
    if col_harga_beli and col_harga_beli != "_RAW_LUAR_HARGA_BELI":
        rename_map[col_harga_beli] = "_RAW_LUAR_HARGA_BELI"
    if col_modal and col_modal != "_RAW_LUAR_MODAL":
        rename_map[col_modal] = "_RAW_LUAR_MODAL"
    if col_nota_balik and col_nota_balik != "_RAW_LUAR_NOTA_BALIK":
        rename_map[col_nota_balik] = "_RAW_LUAR_NOTA_BALIK"
    if col_omzet and col_omzet != "Total harga":
        rename_map[col_omzet] = "Total harga"
    if col_laba and col_laba != "Keuntungan":
        rename_map[col_laba] = "Keuntungan"
    if col_tunai and col_tunai != "Tunai":
        rename_map[col_tunai] = "Tunai"
    if col_kredit and col_kredit != "Kredit":
        rename_map[col_kredit] = "Kredit"
    if col_keterangan and col_keterangan != "Keterangan":
        rename_map[col_keterangan] = "Keterangan"
    if rename_map:
        df = df.rename(columns=rename_map)
        # Jaga-jaga umum: kalau salah satu target rename di atas kebetulan sudah ada
        # sebagai nama kolom asli di sheet pada posisi lain, df[col] bisa mengembalikan
        # DataFrame (bukan Series) dan bikin to_number() error "arg must be a list,
        # tuple, 1-d array, or Series". Kolom pertama (posisi paling kiri) yang dipakai.
        # (4 kolom baru posisi I/L/P/R sudah pakai nama internal unik _RAW_LUAR_... di
        # atas justru supaya TIDAK pernah bentrok dengan kolom "Harga Beli"/"Modal"/dst
        # yang sudah ada duluan di sheet pada posisi lain.)
        df = df.loc[:, ~df.columns.duplicated()]

    # Tanggal/Invoice/Nama Pelanggan hanya terisi di baris pertama tiap invoice di
    # sheet (baris grade berikutnya kosong, gaya sel gabungan). Isi ke bawah supaya
    # tiap baris grade tetap terhubung ke invoice yang benar -- ini akar penyebab
    # Total Omzet/Laba (dan rekap per Nama Pelanggan) sempat salah, karena baris
    # lanjutan yang Nama Pelanggan-nya kosong ikut terbuang oleh filter lama.
    for col in ["TANGGAL", "INVOICE", "NAMA PELANGGAN"]:
        if col in df.columns:
            df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
            df[col] = df[col].ffill()

    # Tanggal Nota Balik ikut gaya sel gabungan yang sama, tapi beda dari Tanggal/
    # Invoice/Nama Pelanggan: 1 invoice bisa saja memang belum punya nota balik sama
    # sekali. Makanya ffill-nya dikelompokkan per INVOICE (bukan ke seluruh tabel),
    # supaya invoice yang belum ada tanggal nota baliknya tetap kosong apa adanya,
    # tidak ikut kebawa tanggal dari invoice sebelumnya.
    if "TANGGAL NOTA BALIK" in df.columns:
        df["TANGGAL NOTA BALIK"] = df["TANGGAL NOTA BALIK"].replace(r"^\s*$", np.nan, regex=True)
        if "INVOICE" in df.columns:
            df["TANGGAL NOTA BALIK"] = df.groupby("INVOICE")["TANGGAL NOTA BALIK"].ffill()
        else:
            df["TANGGAL NOTA BALIK"] = df["TANGGAL NOTA BALIK"].ffill()

    for col in ["Total harga", "Keuntungan", "Tunai", "Kredit", "Jumlah (KG)", "_RAW_LUAR_TONASE_LAHAN_KG", "_RAW_LUAR_HARGA_BELI", "_RAW_LUAR_MODAL", "_RAW_LUAR_NOTA_BALIK"]:
        if col in df.columns:
            df[col] = to_number(df[col])

    df = _parse_tanggal(df)

    if "TANGGAL NOTA BALIK" in df.columns:
        df["Tanggal_Nota_Balik"] = pd.to_datetime(df["TANGGAL NOTA BALIK"], dayfirst=True, errors="coerce")

    df["Is_Dibuang"] = df["Keterangan"].apply(is_filled) if "Keterangan" in df.columns else False
    if "INVOICE" in df.columns:
        df = df[df["INVOICE"].apply(is_filled)].reset_index(drop=True)

    return df


@st.cache_data(ttl=300, show_spinner="Memuat Arus Kas...")
def load_arus_kas() -> pd.DataFrame:
    df = fetch_clean_csv(SHEET_ARUS_KAS)
    if df.empty:
        return df
    for col in ["KAS MASUK", "KAS KELUAR", "SALDO"]:
        if col in df.columns:
            df[col] = to_number(df[col])
    df["Tanggal_Kas"] = pd.to_datetime(df["TANGGAL"], errors="coerce") if "TANGGAL" in df.columns else pd.NaT
    if "JENIS" in df.columns:
        df["JENIS"] = df["JENIS"].astype(str).str.strip().str.upper()
    return df

@st.cache_data(ttl=300, show_spinner="Memuat Pengeluaran Lapak...")
def load_pengeluaran_lapak() -> pd.DataFrame:
    df = fetch_clean_csv(SHEET_PENGELUARAN)
    if df.empty:
        return df
    col_map = {c: c.strip() for c in df.columns}
    df.rename(columns=col_map, inplace=True)
    nominal_col = next((c for c in df.columns if c.strip().upper() == "NOMINAL"), None)
    if nominal_col:
        df[nominal_col] = to_number(df[nominal_col])
    if nominal_col and nominal_col != "NOMINAL":
        df.rename(columns={nominal_col: "NOMINAL"}, inplace=True)
    tgl_col = next((c for c in df.columns if c.strip().upper() in ["TANGGAL", "TGL", "DATE"]), None)
    df["Tanggal_Lengkap"] = pd.to_datetime(df[tgl_col], errors="coerce") if tgl_col else pd.NaT
    lokasi_col = next((c for c in df.columns if "lokasi" in c.strip().lower() and "lapak" in c.strip().lower()), None)
    if lokasi_col:
        if lokasi_col != "LOKASI LAPAK":
            df.rename(columns={lokasi_col: "LOKASI LAPAK"}, inplace=True)
        df = df[df["LOKASI LAPAK"].apply(is_filled)].reset_index(drop=True)

    kategori_col = None
    for _cand in ["JENIS PENGELUARAN", "KATEGORI PENGELUARAN", "TIPE PENGELUARAN",
                  "JENIS BIAYA", "KATEGORI BIAYA", "JENIS", "KATEGORI", "TIPE"]:
        _matches = [c for c in df.columns if c.strip().upper() == _cand]
        if _matches:
            kategori_col = _matches[0]
            break

    if kategori_col is None:
        _kolom_dikecualikan = {"NOMINAL", "LOKASI LAPAK", "Tanggal_Lengkap"}
        if tgl_col:
            _kolom_dikecualikan.add(tgl_col)
        for c in df.columns:
            if c in _kolom_dikecualikan:
                continue
            _vals = df[c].dropna().astype(str).str.strip()
            _vals = _vals[_vals.str.len() > 0]
            if _vals.empty:
                continue
            _vals_upper = _vals.str.upper()
            _cocok = _vals_upper.str.contains("OVERHEAD") | _vals_upper.str.contains("HPP")
            if _vals_upper.str.contains("OVERHEAD").any() and _vals_upper.str.contains("HPP").any() and _cocok.mean() >= 0.8:
                kategori_col = c
                break

    if kategori_col and kategori_col != "JENIS PENGELUARAN":
        df = df.rename(columns={kategori_col: "JENIS PENGELUARAN"})

    if "JENIS PENGELUARAN" not in df.columns and "NOMINAL" not in df.columns:
        overhead_col = next((c for c in df.columns if c.strip().upper() == "OVERHEAD"), None)
        hpp_col      = next((c for c in df.columns if c.strip().upper() == "HPP"), None)
        if overhead_col and hpp_col:
            df[overhead_col] = to_number(df[overhead_col])
            df[hpp_col] = to_number(df[hpp_col])
            id_cols = [c for c in df.columns if c not in [overhead_col, hpp_col]]
            df = df.melt(
                id_vars=id_cols, value_vars=[overhead_col, hpp_col],
                var_name="JENIS PENGELUARAN", value_name="NOMINAL"
            )
            df["JENIS PENGELUARAN"] = df["JENIS PENGELUARAN"].str.strip().str.upper().map(
                {"OVERHEAD": "Overhead", "HPP": "HPP"}
            )

    return df

@st.cache_data(ttl=300, show_spinner="Memuat Ekspedisi...")
def load_ekspedisi() -> pd.DataFrame:
    df = fetch_clean_csv(SHEET_EKSPEDISI)
    if df.empty:
        return df
    for col in ["PENDAPATAN", "PENGELUARAN"]:
        if col in df.columns:
            df[col] = to_number(df[col])
    tgl_col = next((c for c in df.columns if c.strip().upper() in ["TANGGAL", "TGL", "DATE"]), None)
    df["Tanggal_Lengkap"] = pd.to_datetime(df[tgl_col], dayfirst=True, errors="coerce") if tgl_col else pd.NaT
    nama_col = next((c for c in df.columns if c.strip().upper() == "NAMA"), None)
    if nama_col:
        df = df[df[nama_col].apply(is_filled)].reset_index(drop=True)
    return df

@st.cache_data(ttl=300, show_spinner="Memuat Piutang Lapak...")
def load_piutang_lapak() -> pd.DataFrame:
    df = fetch_raw_csv(SHEET_PIUTANG)
    if df.empty:
        return df

    all_cols = list(df.columns)

    kode_col = next((c for c in all_cols if c.strip().upper() == "KODE"), None)
    if kode_col:
        if kode_col != "KODE":
            df = df.rename(columns={kode_col: "KODE"})
        df = df[df["KODE"].apply(is_filled)].reset_index(drop=True)
        all_cols = list(df.columns)

    nama_col = next((c for c in all_cols if c.strip().upper() in ["NAMA", "NAMA PELANGGAN", "CUSTOMER", "PELANGGAN"]), None)
    if nama_col is None and len(all_cols) > 1:
        nama_col = all_cols[1]
    if nama_col and nama_col != "NAMA":
        df = df.rename(columns={nama_col: "NAMA"})
        all_cols = list(df.columns)

    hutang_col = next((c for c in all_cols if c.strip().upper() in ["HUTANG", "TOTAL HUTANG", "PIUTANG", "TOTAL PIUTANG"]), None)
    payment_col = next((c for c in all_cols if c.strip().upper() in ["PAYMENT", "BAYAR", "TERBAYAR", "PEMBAYARAN"]), None)
    sisa_col = next((c for c in all_cols if c.strip().upper() in ["SISA HUTANG", "SISA PIUTANG", "SISA", "OUTSTANDING"]), None)

    rename_map = {}
    if hutang_col and hutang_col != "Hutang":
        rename_map[hutang_col] = "Hutang"
    if payment_col and payment_col != "Payment":
        rename_map[payment_col] = "Payment"
    if sisa_col and sisa_col != "Sisa Hutang":
        rename_map[sisa_col] = "Sisa Hutang"
    if rename_map:
        df = df.rename(columns=rename_map)

    for col in ["Hutang", "Payment", "Sisa Hutang"]:
        if col in df.columns:
            df[col] = to_number(df[col])

    if "Sisa Hutang" not in df.columns and "Hutang" in df.columns and "Payment" in df.columns:
        df["Sisa Hutang"] = df["Hutang"].fillna(0) - df["Payment"].fillna(0)

    tgl_col = next((c for c in df.columns if c.strip().upper() in ["TANGGAL", "TGL", "DATE"]), None)
    df["Tanggal_Lengkap"] = pd.to_datetime(df[tgl_col], dayfirst=True, errors="coerce") if tgl_col else pd.NaT
    return df

@st.cache_data(ttl=300, show_spinner="Memuat Piutang Lapak Luar...")
def load_piutang_lapak_luar() -> pd.DataFrame:
    df = fetch_raw_csv(SHEET_PIUTANG_LUAR)
    if df.empty:
        return df

    all_cols = list(df.columns)

    kode_col = next((c for c in all_cols if c.strip().upper() == "KODE"), None)
    if kode_col:
        df = df[df[kode_col].apply(is_filled)].reset_index(drop=True)
        if kode_col != "KODE":
            df = df.rename(columns={kode_col: "KODE"})
        all_cols = list(df.columns)

    nama_col = next((c for c in all_cols if c.strip().upper() in ["NAMA", "NAMA PELANGGAN", "CUSTOMER", "PELANGGAN"]), None)
    if nama_col and nama_col != "NAMA":
        df = df.rename(columns={nama_col: "NAMA"})
        all_cols = list(df.columns)

    hutang_col = next((c for c in all_cols if c.strip().upper() in ["HUTANG", "TOTAL HUTANG", "PIUTANG", "TOTAL PIUTANG"]), None)
    payment_col = next((c for c in all_cols if c.strip().upper() in ["PAYMENT", "BAYAR", "TERBAYAR", "PEMBAYARAN"]), None)
    if hutang_col is None and len(all_cols) > 2:
        hutang_col = all_cols[2]
    if payment_col is None and len(all_cols) > 3:
        payment_col = all_cols[3]

    rename_map = {}
    if hutang_col and hutang_col != "Hutang":
        rename_map[hutang_col] = "Hutang"
    if payment_col and payment_col != "Payment":
        rename_map[payment_col] = "Payment"
    if rename_map:
        df = df.rename(columns=rename_map)

    for col in ["Hutang", "Payment"]:
        if col in df.columns:
            df[col] = to_number(df[col])

    if "Hutang" in df.columns and "Payment" in df.columns:
        df["Sisa Hutang"] = df["Hutang"].fillna(0) - df["Payment"].fillna(0)
    elif "Sisa Hutang" in df.columns:
        df["Sisa Hutang"] = to_number(df["Sisa Hutang"])

    tgl_col = next((c for c in df.columns if c.strip().upper() in ["TANGGAL", "TGL", "DATE"]), None)
    df["Tanggal_Lengkap"] = pd.to_datetime(df[tgl_col], dayfirst=True, errors="coerce") if tgl_col else pd.NaT
    return df

@st.cache_data(ttl=300, show_spinner="Memuat Hutang Petani...")
def load_hutang_petani() -> pd.DataFrame:
    df = fetch_clean_csv(SHEET_HUTANG_PETANI)
    if df.empty:
        return df
    for col in df.columns:
        if any(k in col.upper() for k in ["HUTANG", "PAYMENT", "BAYAR", "SISA", "NOMINAL", "OUTSTANDING", "JUMLAH"]):
            df[col] = to_number(df[col])
    return df

@st.cache_data(ttl=300, show_spinner="Memuat Kerugian Gudang...")
def load_kerugian_gudang() -> pd.DataFrame:
    df = fetch_raw_csv(SHEET_KERUGIAN_GUDANG)
    if df.empty:
        return df
    all_cols = list(df.columns)

    hutang_col   = next((c for c in all_cols if c.strip().upper() in ["HUTANG", "KERUGIAN", "TOTAL"]), None)
    terbayar_col = next((c for c in all_cols if c.strip().upper() in ["TERBAYAR", "PAYMENT", "BAYAR", "PEMBAYARAN"]), None)
    if hutang_col is None and len(all_cols) > 1:
        hutang_col = all_cols[1]
    if terbayar_col is None and len(all_cols) > 2:
        terbayar_col = all_cols[2]

    rename_map = {}
    if hutang_col and hutang_col != "HUTANG":
        rename_map[hutang_col] = "HUTANG"
    if terbayar_col and terbayar_col != "TERBAYAR":
        rename_map[terbayar_col] = "TERBAYAR"
    if rename_map:
        df = df.rename(columns=rename_map)

    for col in ["HUTANG", "TERBAYAR"]:
        if col in df.columns:
            df[col] = to_number(df[col])

    df = drop_placeholder_cols(df)
    return df

@st.cache_data(ttl=300, show_spinner="Memuat Hutang Pak'e Tani...")
def load_hutang_pake_tani() -> pd.DataFrame:
    df = fetch_raw_csv(SHEET_HUTANG_PAKE_TANI)
    if df.empty:
        return df
    all_cols = list(df.columns)

    hutang_col   = next((c for c in all_cols if c.strip().upper() in ["HUTANG", "JUMLAH HUTANG", "TOTAL HUTANG", "PINJAMAN"]), None)
    terbayar_col = next((c for c in all_cols if c.strip().upper() in ["TERBAYAR", "PAYMENT", "BAYAR", "PEMBAYARAN", "ANGSURAN"]), None)
    if hutang_col is None and len(all_cols) > 1:
        hutang_col = all_cols[1]
    if terbayar_col is None and len(all_cols) > 2:
        terbayar_col = all_cols[2]

    rename_map = {}
    if hutang_col and hutang_col != "HUTANG":
        rename_map[hutang_col] = "HUTANG"
    if terbayar_col and terbayar_col != "TERBAYAR":
        rename_map[terbayar_col] = "TERBAYAR"
    if rename_map:
        df = df.rename(columns=rename_map)

    for col in ["HUTANG", "TERBAYAR"]:
        if col in df.columns:
            df[col] = to_number(df[col])

    df = drop_placeholder_cols(df)
    return df

@st.cache_data(ttl=300, show_spinner="Memuat Data Tanaman...")
def load_tanaman_belum_panen() -> pd.DataFrame:
    return fetch_clean_csv(SHEET_TANAMAN_BELUM)

@st.cache_data(ttl=300, show_spinner="Memuat Data Tanaman...")
def load_tanaman_sudah_panen() -> pd.DataFrame:
    return fetch_clean_csv(SHEET_TANAMAN_SUDAH)

_BULAN_ID_MAP = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11, "desember": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "agt": 8, "agu": 8, "aug": 8, "sep": 9, "sept": 9, "okt": 10, "oct": 10,
    "nov": 11, "des": 12, "dec": 12,
}

def _parse_bulan_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    low = s.lower()
    if low in _BULAN_ID_MAP:
        return _BULAN_ID_MAP[low]
    for token in low.replace("-", " ").replace("/", " ").replace(",", " ").split():
        if token in _BULAN_ID_MAP:
            return _BULAN_ID_MAP[token]
    if s.isdigit():
        n = int(s)
        if 1 <= n <= 12:
            return n
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if pd.notna(dt):
        return dt.month
    return np.nan

@st.cache_data(ttl=300, show_spinner="Memuat Biaya Berjalan Bulanan...")
def load_biaya_berjalan_bulanan() -> pd.DataFrame:
    df = fetch_raw_csv(SHEET_TANAMAN_BELUM)
    if df.empty or len(df.columns) <= 15:
        return pd.DataFrame()
    all_cols = list(df.columns)
    out = df[[all_cols[14], all_cols[15]]].copy()
    out.columns = ["BULAN", "BIAYA BERJALAN"]
    out["BIAYA BERJALAN"] = to_number(out["BIAYA BERJALAN"])
    out = out[out["BULAN"].apply(is_filled) & out["BIAYA BERJALAN"].notna()].reset_index(drop=True)
    out["Bulan_Num"] = out["BULAN"].apply(_parse_bulan_number)
    return out

@st.cache_data(ttl=300, show_spinner="Memuat Stok Lapak...")
def load_stok_lapak() -> pd.DataFrame:
    df = fetch_raw_csv(SHEET_STOK_LAPAK)
    if df.empty:
        return df

    all_cols = list(df.columns)

    def _get_col(idx, name_candidates):
        for name in name_candidates:
            matches = [c for c in all_cols if c.strip().lower() == name.lower()]
            if matches:
                return matches[0]
        if idx < len(all_cols):
            return all_cols[idx]
        return None

    col_tanggal = _get_col(0,  ["tanggal", "tgl", "date", "timestamp"])
    col_tujuan  = _get_col(7,  ["tujuan"])
    col_jenis   = _get_col(10, ["jenis", "jenis tanaman", "nama barang", "produk", "komoditas"])
    col_grade   = _get_col(11, ["grade", "kelas", "mutu"])
    col_stok    = _get_col(19, ["stok lapak", "stok"])

    rename_map = {}
    if col_tanggal and col_tanggal != "TANGGAL":
        rename_map[col_tanggal] = "TANGGAL"
    if col_tujuan and col_tujuan != "TUJUAN":
        rename_map[col_tujuan] = "TUJUAN"
    if col_jenis and col_jenis != "JENIS":
        rename_map[col_jenis] = "JENIS"
    if col_grade and col_grade != "GRADE":
        rename_map[col_grade] = "GRADE"
    if col_stok and col_stok != "STOK LAPAK":
        rename_map[col_stok] = "STOK LAPAK"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "STOK LAPAK" in df.columns:
        df["STOK LAPAK"] = to_number(df["STOK LAPAK"])
    if "JENIS" in df.columns:
        df["JENIS"] = df["JENIS"].astype(str).str.strip()
    if "GRADE" in df.columns:
        df["GRADE"] = df["GRADE"].astype(str).str.strip()
    if "TUJUAN" in df.columns:
        df = df[df["TUJUAN"].apply(is_filled)].reset_index(drop=True)

    if "TANGGAL" in df.columns:
        df["Tanggal_Lengkap"] = pd.to_datetime(df["TANGGAL"], dayfirst=True, errors="coerce")
        batas_awal_stok = pd.Timestamp.now().normalize() - pd.Timedelta(days=14)
        df = df[df["Tanggal_Lengkap"] >= batas_awal_stok].reset_index(drop=True)

    return df

@st.cache_data(ttl=300, show_spinner="Memuat Stok Gudang...")
def load_stok_gudang() -> pd.DataFrame:
    df = fetch_raw_csv(SHEET_STOK_GUDANG)
    if df.empty:
        return df

    all_cols = list(df.columns)

    def _get_col(idx, name_candidates):
        for name in name_candidates:
            matches = [c for c in all_cols if c.strip().lower() == name.lower()]
            if matches:
                return matches[0]
        if idx < len(all_cols):
            return all_cols[idx]
        return None

    def _find_name_only(names):
        for n in names:
            m = [c for c in all_cols if c.strip().lower() == n.lower()]
            if m:
                return m[0]
        return None

    col_tanggal        = _get_col(0,  ["tanggal", "tgl", "date", "timestamp"])
    col_invoice        = _get_col(1,  ["invoice", "no invoice", "nomor invoice"])
    col_asal           = _get_col(5,  ["asal barang", "asal"])
    col_penimbang      = _get_col(6,  ["penimbang"])
    col_tujuan         = _get_col(7,  ["tujuan", "kode gudang", "gudang", "lokasi"])
    col_nopol          = _get_col(8,  ["nopol", "no polisi", "plat nomor"])
    col_jenis          = _get_col(10, ["jenis", "jenis tanaman", "nama barang", "produk", "komoditas"])
    col_grade          = _get_col(11, ["grade", "kelas", "mutu"])
    col_tonnase_lahan  = _get_col(12, ["tonnase lahan", "tonase lahan"])
    col_jumlah_grading = _get_col(13, ["jumlah grading", "jumlah_grading"])
    col_stok           = _get_col(19, ["stok gudang", "stok"])
    col_harga_beli     = _find_name_only(["harga beli"])
    col_harga_modal    = _find_name_only(["harga modal"])

    rename_map = {}
    if col_tanggal and col_tanggal != "TANGGAL":
        rename_map[col_tanggal] = "TANGGAL"
    if col_invoice and col_invoice != "INVOICE":
        rename_map[col_invoice] = "INVOICE"
    if col_asal and col_asal != "ASAL BARANG":
        rename_map[col_asal] = "ASAL BARANG"
    if col_penimbang and col_penimbang != "PENIMBANG":
        rename_map[col_penimbang] = "PENIMBANG"
    if col_tujuan and col_tujuan != "TUJUAN":
        rename_map[col_tujuan] = "TUJUAN"
    if col_nopol and col_nopol != "NOPOL":
        rename_map[col_nopol] = "NOPOL"
    if col_jenis and col_jenis != "JENIS":
        rename_map[col_jenis] = "JENIS"
    if col_grade and col_grade != "GRADE":
        rename_map[col_grade] = "GRADE"
    if col_tonnase_lahan and col_tonnase_lahan != "TONNASE LAHAN":
        rename_map[col_tonnase_lahan] = "TONNASE LAHAN"
    if col_jumlah_grading and col_jumlah_grading != "JUMLAH GRADING":
        rename_map[col_jumlah_grading] = "JUMLAH GRADING"
    if col_stok and col_stok != "STOK GUDANG":
        rename_map[col_stok] = "STOK GUDANG"
    if col_harga_beli and col_harga_beli != "HARGA BELI":
        rename_map[col_harga_beli] = "HARGA BELI"
    if col_harga_modal and col_harga_modal != "HARGA MODAL":
        rename_map[col_harga_modal] = "HARGA MODAL"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "STOK GUDANG" in df.columns:
        df["STOK GUDANG"] = to_number(df["STOK GUDANG"])
    if "TONNASE LAHAN" in df.columns:
        df["TONNASE LAHAN"] = to_number(df["TONNASE LAHAN"])
    if "JUMLAH GRADING" in df.columns:
        df["JUMLAH GRADING"] = to_number(df["JUMLAH GRADING"])
    if "HARGA BELI" in df.columns:
        df["HARGA BELI"] = to_number(df["HARGA BELI"])
    if "HARGA MODAL" in df.columns:
        df["HARGA MODAL"] = to_number(df["HARGA MODAL"])
    if "JENIS" in df.columns:
        df["JENIS"] = df["JENIS"].astype(str).str.strip()
    if "GRADE" in df.columns:
        df["GRADE"] = df["GRADE"].astype(str).str.strip()
    if "TUJUAN" in df.columns:
        df["LOKASI"] = df["TUJUAN"]
        df = df[df["TUJUAN"].apply(is_filled)].reset_index(drop=True)

    if "TANGGAL" in df.columns:
        df["Tanggal_Lengkap"] = pd.to_datetime(df["TANGGAL"], dayfirst=True, errors="coerce")

    return df

@st.cache_data(ttl=300, show_spinner="Memuat Barang Masuk...")
def load_barang_masuk() -> pd.DataFrame:
    # Sheet BARANG_MASUK (spreadsheet Penjualan Lapak) -- struktur posisi kolomnya
    # sama seperti STOK GUDANG lama, tapi HARGA BELI sekarang di posisi tetap R
    # (dulu cuma dicari by nama).
    df = fetch_raw_csv(SHEET_BARANG_MASUK, spreadsheet_id=PENJUALAN_LAPAK_SPREADSHEET_ID)
    if df.empty:
        return df

    all_cols = list(df.columns)

    def _get_col(idx, name_candidates):
        for name in name_candidates:
            matches = [c for c in all_cols if c.strip().lower() == name.lower()]
            if matches:
                return matches[0]
        if idx < len(all_cols):
            return all_cols[idx]
        return None

    col_tanggal        = _get_col(0,  ["tanggal", "tgl", "date", "timestamp"])
    col_invoice        = _get_col(1,  ["invoice", "no invoice", "nomor invoice"])
    col_asal           = _get_col(5,  ["asal barang", "asal"])
    col_penimbang      = _get_col(6,  ["penimbang"])
    col_tujuan         = _get_col(7,  ["tujuan", "kode gudang", "gudang", "lokasi"])
    col_nopol          = _get_col(8,  ["nopol", "no polisi", "plat nomor"])
    col_jenis          = _get_col(10, ["jenis", "jenis tanaman", "nama barang", "produk", "komoditas"])
    col_grade          = _get_col(11, ["grade", "kelas", "mutu"])
    col_tonnase_lahan  = _get_col(12, ["tonnase lahan", "tonase lahan"])
    col_total_grading  = _get_col(13, ["total grading", "jumlah grading", "jumlah_grading"])
    col_harga_modal    = _get_col(17, ["harga modal", "harga beli"])
    col_stok           = _get_col(19, ["stok gudang", "stok"])

    rename_map = {}
    if col_tanggal and col_tanggal != "TANGGAL":
        rename_map[col_tanggal] = "TANGGAL"
    if col_invoice and col_invoice != "INVOICE":
        rename_map[col_invoice] = "INVOICE"
    if col_asal and col_asal != "ASAL BARANG":
        rename_map[col_asal] = "ASAL BARANG"
    if col_penimbang and col_penimbang != "PENIMBANG":
        rename_map[col_penimbang] = "PENIMBANG"
    if col_tujuan and col_tujuan != "TUJUAN":
        rename_map[col_tujuan] = "TUJUAN"
    if col_nopol and col_nopol != "NOPOL":
        rename_map[col_nopol] = "NOPOL"
    if col_jenis and col_jenis != "JENIS":
        rename_map[col_jenis] = "JENIS"
    if col_grade and col_grade != "GRADE":
        rename_map[col_grade] = "GRADE"
    if col_tonnase_lahan and col_tonnase_lahan != "TONNASE LAHAN":
        rename_map[col_tonnase_lahan] = "TONNASE LAHAN"
    if col_total_grading and col_total_grading != "TOTAL GRADING":
        rename_map[col_total_grading] = "TOTAL GRADING"
    if col_harga_modal and col_harga_modal != "HARGA MODAL":
        rename_map[col_harga_modal] = "HARGA MODAL"
    if col_stok and col_stok != "STOK GUDANG":
        rename_map[col_stok] = "STOK GUDANG"
    if rename_map:
        df = df.rename(columns=rename_map)
        df = df.loc[:, ~df.columns.duplicated()]

    for col in ["TONNASE LAHAN", "TOTAL GRADING", "HARGA MODAL", "STOK GUDANG"]:
        if col in df.columns:
            df[col] = to_number(df[col])
    if "JENIS" in df.columns:
        df["JENIS"] = df["JENIS"].astype(str).str.strip()
    if "GRADE" in df.columns:
        df["GRADE"] = df["GRADE"].astype(str).str.strip()
    if "INVOICE" in df.columns:
        # Distandarkan (strip spasi) supaya cocok persis dengan INVOICE di
        # load_stok_lapak_invoice() -- tanpa ini, pencocokan invoice di tabel
        # "Data Lapak (Setelah Moving)" bisa gagal total kalau ada spasi
        # tersembunyi di salah satu sisi (dropdown Invoice ambil nilai dari sini,
        # tidak di-strip, sementara STOK LAPAK sudah di-strip).
        df["INVOICE"] = df["INVOICE"].astype(str).str.strip()
    if "TUJUAN" in df.columns:
        df = df[df["TUJUAN"].apply(is_filled)].reset_index(drop=True)

    if "TANGGAL" in df.columns:
        df["Tanggal_Lengkap"] = pd.to_datetime(df["TANGGAL"], dayfirst=True, errors="coerce")

    return df

@st.cache_data(ttl=300, show_spinner="Memuat Stok Lapak (Rincian per Invoice)...")
def load_stok_lapak_invoice() -> pd.DataFrame:
    # Versi STOK LAPAK khusus untuk Rincian per Invoice: sumbernya dari spreadsheet
    # Penjualan Lapak (bukan spreadsheet utama) dan TIDAK dibatasi 14 hari terakhir
    # seperti load_stok_lapak(), supaya invoice lama tetap bisa ditelusuri.
    df = fetch_raw_csv(SHEET_STOK_LAPAK, spreadsheet_id=PENJUALAN_LAPAK_SPREADSHEET_ID)
    if df.empty:
        return df

    all_cols = list(df.columns)

    def _get_col(idx, name_candidates):
        for name in name_candidates:
            matches = [c for c in all_cols if c.strip().lower() == name.lower()]
            if matches:
                return matches[0]
        if idx < len(all_cols):
            return all_cols[idx]
        return None

    col_tanggal        = _get_col(0,  ["tanggal", "tgl", "date", "timestamp"])
    col_invoice        = _get_col(1,  ["invoice", "no invoice", "nomor invoice"])  # B
    col_tujuan         = _get_col(7,  ["tujuan"])          # H = Moving (Lapak tujuan)
    col_jenis          = _get_col(10, ["jenis", "jenis tanaman", "nama barang", "produk", "komoditas"])
    col_grade          = _get_col(11, ["grade", "kelas", "mutu"])
    col_jumlah_moving  = _get_col(12, ["jumlah moving", "jumlah_moving"])   # M
    col_hpp_lapak      = _get_col(18, ["hpp lapak", "hpp_lapak"])          # S
    col_stok           = _get_col(19, ["stok lapak", "stok"])              # T

    rename_map = {}
    if col_tanggal and col_tanggal != "TANGGAL":
        rename_map[col_tanggal] = "TANGGAL"
    if col_invoice and col_invoice != "INVOICE":
        rename_map[col_invoice] = "INVOICE"
    if col_tujuan and col_tujuan != "TUJUAN":
        rename_map[col_tujuan] = "TUJUAN"
    if col_jenis and col_jenis != "JENIS":
        rename_map[col_jenis] = "JENIS"
    if col_grade and col_grade != "GRADE":
        rename_map[col_grade] = "GRADE"
    if col_jumlah_moving and col_jumlah_moving != "JUMLAH MOVING":
        rename_map[col_jumlah_moving] = "JUMLAH MOVING"
    if col_hpp_lapak and col_hpp_lapak != "HPP LAPAK":
        rename_map[col_hpp_lapak] = "HPP LAPAK"
    if col_stok and col_stok != "STOK LAPAK":
        rename_map[col_stok] = "STOK LAPAK"
    if rename_map:
        df = df.rename(columns=rename_map)
        df = df.loc[:, ~df.columns.duplicated()]

    for col in ["JUMLAH MOVING", "HPP LAPAK", "STOK LAPAK"]:
        if col in df.columns:
            df[col] = to_number(df[col])
    if "JENIS" in df.columns:
        df["JENIS"] = df["JENIS"].astype(str).str.strip()
    if "GRADE" in df.columns:
        df["GRADE"] = df["GRADE"].astype(str).str.strip()
    if "INVOICE" in df.columns:
        df["INVOICE"] = df["INVOICE"].astype(str).str.strip()
    if "TUJUAN" in df.columns:
        df["TUJUAN"] = df["TUJUAN"].astype(str).str.strip()
        df = df[df["TUJUAN"].apply(is_filled)].reset_index(drop=True)

    if "TANGGAL" in df.columns:
        df["Tanggal_Lengkap"] = pd.to_datetime(df["TANGGAL"], dayfirst=True, errors="coerce")

    return df

# LOADERS - GREEN HOUSE (spreadsheet terpisah)
def _gh_find_col(all_cols, candidates):
    for cand in candidates:
        m = [c for c in all_cols if c.strip().upper() == cand.upper()]
        if m:
            return m[0]
    return None

def _gh_to_number(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    def parse_val(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        if x == "" or x.lower() in ("nan", "none", "-", "rp -", "rp-"):
            return np.nan
        x = x.replace("Rp", "").replace("rp", "").strip()
        x = x.replace(",", "")
        x = "".join(ch for ch in x if ch.isdigit() or ch in ".-")
        if x in ("", "-", "."):
            return np.nan
        return x
    return pd.to_numeric(series.map(parse_val), errors="coerce")

def _gh_normalize_rincian(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = drop_placeholder_cols(df)
    all_cols = list(df.columns)

    lokasi_col      = _gh_find_col(all_cols, ["Lokasi"])
    siklus_col      = _gh_find_col(all_cols, ["Siklus"])
    nomor_gh_col    = _gh_find_col(all_cols, ["Nomor GH", "No GH", "No. GH", "GH"])
    subkategori_col = _gh_find_col(all_cols, ["Sub Kategori", "Subkategori", "Sub-Kategori"])
    total_col       = _gh_find_col(all_cols, ["Total"])

    rename_map = {}
    for col, target in [
        (lokasi_col, "Lokasi"), (siklus_col, "Siklus"), (nomor_gh_col, "Nomor GH"),
        (subkategori_col, "Sub Kategori"), (total_col, "Total"),
    ]:
        if col and col != target:
            rename_map[col] = target
    if rename_map:
        df = df.rename(columns=rename_map)

    for c in list(df.columns):
        cu = c.strip().upper()
        if cu == "TOTAL" or "HARGA" in cu:
            df[c] = _gh_to_number(df[c])

    if "Siklus" in df.columns:
        df["Siklus"] = df["Siklus"].astype(str).str.strip()
    if "Lokasi" in df.columns:
        df = df[df["Lokasi"].apply(is_filled)].reset_index(drop=True)

    df = _parse_tanggal(df)

    return df

@st.cache_data(ttl=300, show_spinner="Memuat Data Bahan (Green House)...")
def load_gh_bahan() -> pd.DataFrame:
    df = fetch_raw_csv(SHEET_GH_BAHAN, spreadsheet_id=GH_SPREADSHEET_ID)
    return _gh_normalize_rincian(df)

@st.cache_data(ttl=300, show_spinner="Memuat Data Pemupukan (Green House)...")
def load_gh_pemupukan() -> pd.DataFrame:
    df = fetch_raw_csv(SHEET_GH_PEMUPUKAN, spreadsheet_id=GH_SPREADSHEET_ID)
    return _gh_normalize_rincian(df)

@st.cache_data(ttl=300, show_spinner="Memuat Data Tenaga (Green House)...")
def load_gh_tenaga() -> pd.DataFrame:
    df = fetch_raw_csv(SHEET_GH_TENAGA, spreadsheet_id=GH_SPREADSHEET_ID)
    return _gh_normalize_rincian(df)

@st.cache_data(ttl=300, show_spinner="Memuat Data Lokasi (Green House)...")
def load_gh_lokasi() -> pd.DataFrame:
    df = fetch_raw_csv(SHEET_GH_LOKASI, spreadsheet_id=GH_SPREADSHEET_ID)
    if df is None or df.empty:
        return df
    df = drop_placeholder_cols(df)
    all_cols = list(df.columns)

    lokasi_col    = _gh_find_col(all_cols, ["Lokasi"])
    jumlah_gh_col = _gh_find_col(all_cols, ["Jumlah GH", "Jumlah Green House", "Total GH", "GH"])

    if lokasi_col is None and len(all_cols) > 0:
        lokasi_col = all_cols[0]
    if jumlah_gh_col is None and len(all_cols) > 1:
        jumlah_gh_col = all_cols[1]

    rename_map = {}
    if lokasi_col and lokasi_col != "Lokasi":
        rename_map[lokasi_col] = "Lokasi"
    if jumlah_gh_col and jumlah_gh_col != "Jumlah GH":
        rename_map[jumlah_gh_col] = "Jumlah GH"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Jumlah GH" in df.columns:
        df["Jumlah GH"] = _gh_to_number(df["Jumlah GH"])
    if "Lokasi" in df.columns:
        df = df[df["Lokasi"].apply(is_filled)].reset_index(drop=True)

    return df

@st.cache_data(ttl=300, show_spinner="Memuat Data Tanaman (Green House)...")
def load_gh_tanaman() -> pd.DataFrame:
    df = fetch_raw_csv(SHEET_GH_TANAMAN, spreadsheet_id=GH_SPREADSHEET_ID)
    if df is None or df.empty:
        return df
    df = drop_placeholder_cols(df)
    all_cols = list(df.columns)

    masuk_col  = _gh_find_col(all_cols, ["Kas Masuk", "Pemasukan", "Masuk"])
    keluar_col = _gh_find_col(all_cols, ["Kas Keluar", "Pengeluaran", "Keluar"])
    ket_col    = _gh_find_col(all_cols, ["Keterangan", "Ket"])

    rename_map = {}
    for col, target in [
        (masuk_col, "Kas Masuk"), (keluar_col, "Kas Keluar"), (ket_col, "Keterangan"),
    ]:
        if col and col != target:
            rename_map[col] = target
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Kas Masuk" in df.columns:
        df["Kas Masuk"] = _gh_to_number(df["Kas Masuk"])
    if "Kas Keluar" in df.columns:
        df["Kas Keluar"] = _gh_to_number(df["Kas Keluar"])

    df = _parse_tanggal(df)

    return df

@st.cache_data(ttl=300, show_spinner="Memuat Kas Green House...")
def load_gh_kas() -> pd.DataFrame:
    df = fetch_raw_csv(SHEET_GH_KAS, spreadsheet_id=GH_SPREADSHEET_ID)
    if df is None or df.empty:
        return df
    df = drop_placeholder_cols(df)
    all_cols = list(df.columns)

    kategori_col = _gh_find_col(all_cols, ["Kategori", "Kategori Kas"])
    masuk_col    = _gh_find_col(all_cols, ["Kas Masuk", "Pemasukan", "Masuk"])
    keluar_col   = _gh_find_col(all_cols, ["Kas Keluar", "Pengeluaran", "Keluar"])

    rename_map = {}
    for col, target in [
        (kategori_col, "Kategori"), (masuk_col, "Kas Masuk"), (keluar_col, "Kas Keluar"),
    ]:
        if col and col != target:
            rename_map[col] = target
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Kas Masuk" in df.columns:
        df["Kas Masuk"] = _gh_to_number(df["Kas Masuk"])
    if "Kas Keluar" in df.columns:
        df["Kas Keluar"] = _gh_to_number(df["Kas Keluar"])
    if "Kategori" in df.columns:
        df["Kategori"] = df["Kategori"].astype(str).str.strip()

    df = _parse_tanggal(df)

    return df

# LOAD SEMUA DATA
try:
    df_penjualan_raw      = load_penjualan_lapak()
    df_penjualan_luar_raw = load_penjualan_lapak_luar()
    df_kas_raw            = load_arus_kas()
    df_pengeluaran_raw    = load_pengeluaran_lapak()
    df_piutang_raw        = load_piutang_lapak()
    df_piutang_luar_raw   = load_piutang_lapak_luar()
    df_hutang_petani_raw  = load_hutang_petani()
    df_ekspedisi_raw      = load_ekspedisi()
    df_tanaman_belum      = load_tanaman_belum_panen()
    df_tanaman_sudah      = load_tanaman_sudah_panen()
    df_biaya_bulanan_raw  = load_biaya_berjalan_bulanan()
    df_kerugian_gudang_raw = load_kerugian_gudang()
    df_hutang_pake_tani_raw = load_hutang_pake_tani()
except Exception as e:
    st.error(f"Gagal mengambil data. Detail: {e}")
    st.stop()

try:
    df_gh_bahan_raw     = load_gh_bahan()
    df_gh_pemupukan_raw = load_gh_pemupukan()
    df_gh_tenaga_raw    = load_gh_tenaga()
    df_gh_lokasi_raw    = load_gh_lokasi()
    df_gh_tanaman_raw   = load_gh_tanaman()
    df_gh_kas_raw       = load_gh_kas()
except Exception as e:
    st.warning(f"Gagal mengambil data Green House. Detail: {e}")
    df_gh_bahan_raw     = pd.DataFrame()
    df_gh_pemupukan_raw = pd.DataFrame()
    df_gh_tenaga_raw    = pd.DataFrame()
    df_gh_lokasi_raw    = pd.DataFrame()
    df_gh_tanaman_raw   = pd.DataFrame()
    df_gh_kas_raw       = pd.DataFrame()

try:
    df_stok_lapak_raw  = load_stok_lapak()
    df_stok_gudang_raw = load_stok_gudang()
except Exception as e:
    st.warning(f"Gagal mengambil data Stok Lapak/Stok Gudang. Detail: {e}")
    df_stok_lapak_raw  = pd.DataFrame()
    df_stok_gudang_raw = pd.DataFrame()

try:
    df_barang_masuk_raw      = load_barang_masuk()
    df_stok_lapak_invoice_raw = load_stok_lapak_invoice()
except Exception as e:
    st.warning(f"Gagal mengambil data Barang Masuk/Stok Lapak (Rincian per Invoice). Detail: {e}")
    df_barang_masuk_raw       = pd.DataFrame()
    df_stok_lapak_invoice_raw = pd.DataFrame()

# SIDEBAR - FILTER
st.sidebar.title("⚙️ Filter Data")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("📅 Periode Analisis")
valid_dates = pd.concat([
    df_penjualan_raw["Tanggal_Lengkap"]      if not df_penjualan_raw.empty      and "Tanggal_Lengkap" in df_penjualan_raw.columns      else pd.Series(dtype="datetime64[ns]"),
    df_penjualan_luar_raw["Tanggal_Lengkap"] if not df_penjualan_luar_raw.empty and "Tanggal_Lengkap" in df_penjualan_luar_raw.columns else pd.Series(dtype="datetime64[ns]"),
    df_pengeluaran_raw["Tanggal_Lengkap"]    if not df_pengeluaran_raw.empty    and "Tanggal_Lengkap" in df_pengeluaran_raw.columns    else pd.Series(dtype="datetime64[ns]"),
    df_kas_raw["Tanggal_Kas"]                if not df_kas_raw.empty            and "Tanggal_Kas"     in df_kas_raw.columns            else pd.Series(dtype="datetime64[ns]"),
    df_ekspedisi_raw["Tanggal_Lengkap"]      if not df_ekspedisi_raw.empty      and "Tanggal_Lengkap" in df_ekspedisi_raw.columns      else pd.Series(dtype="datetime64[ns]"),
]).dropna()

date_range = None
if not valid_dates.empty:
    min_d, max_d = valid_dates.min().date(), valid_dates.max().date()

    today = datetime.now().date()
    awal_bulan  = today.replace(day=1)
    akhir_bulan = (pd.Timestamp(today) + pd.offsets.MonthEnd(0)).date()

    default_start = max(min_d, awal_bulan)
    default_end   = min(max_d, akhir_bulan)

    if default_start > default_end:
        default_start, default_end = min_d, max_d

    date_range = st.sidebar.date_input(
        "Rentang Tanggal",
        value=(default_start, default_end),
        min_value=min_d, max_value=max_d
    )
    st.sidebar.caption("Default: bulan berjalan. Ubah rentang di atas untuk melihat periode lain.")

st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout"):
    st.session_state["password_correct"] = False
    st.rerun()

# APLIKASI FILTER TANGGAL
df_penjualan             = df_penjualan_raw.copy()
df_penjualan_luar        = df_penjualan_luar_raw.copy()
df_kas                   = df_kas_raw.copy()
df_pengeluaran           = df_pengeluaran_raw.copy()
df_ekspedisi             = df_ekspedisi_raw.copy()
df_piutang_filtered      = df_piutang_raw.copy()
df_piutang_luar_filtered = df_piutang_luar_raw.copy()
df_tanaman_sudah_filtered = df_tanaman_sudah.copy()
df_biaya_bulanan         = df_biaya_bulanan_raw.copy()
df_gh_kas_biaya          = df_gh_kas_raw.copy()
df_gh_tanaman_biaya      = df_gh_tanaman_raw.copy()

if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
    start_ts = pd.Timestamp(date_range[0])
    end_ts   = pd.Timestamp(date_range[1])

    if not df_penjualan.empty and "Tanggal_Lengkap" in df_penjualan.columns:
        df_penjualan = df_penjualan[(df_penjualan["Tanggal_Lengkap"] >= start_ts) & (df_penjualan["Tanggal_Lengkap"] <= end_ts)]
    if not df_penjualan_luar.empty and "Tanggal_Lengkap" in df_penjualan_luar.columns:
        df_penjualan_luar = df_penjualan_luar[(df_penjualan_luar["Tanggal_Lengkap"] >= start_ts) & (df_penjualan_luar["Tanggal_Lengkap"] <= end_ts)]
    if not df_kas.empty and "Tanggal_Kas" in df_kas.columns:
        df_kas = df_kas[(df_kas["Tanggal_Kas"] >= start_ts) & (df_kas["Tanggal_Kas"] <= end_ts)]
    if not df_pengeluaran.empty and "Tanggal_Lengkap" in df_pengeluaran.columns:
        df_pengeluaran = df_pengeluaran[(df_pengeluaran["Tanggal_Lengkap"] >= start_ts) & (df_pengeluaran["Tanggal_Lengkap"] <= end_ts)]
    if not df_ekspedisi.empty and "Tanggal_Lengkap" in df_ekspedisi.columns:
        _mask_eks = (
            ((df_ekspedisi["Tanggal_Lengkap"] >= start_ts) & (df_ekspedisi["Tanggal_Lengkap"] <= end_ts))
            | df_ekspedisi["Tanggal_Lengkap"].isna()
        )
        df_ekspedisi = df_ekspedisi[_mask_eks]
    if not df_piutang_filtered.empty and "Tanggal_Lengkap" in df_piutang_filtered.columns:
        df_piutang_filtered = df_piutang_filtered[(df_piutang_filtered["Tanggal_Lengkap"] >= start_ts) & (df_piutang_filtered["Tanggal_Lengkap"] <= end_ts)]
    if not df_piutang_luar_filtered.empty and "Tanggal_Lengkap" in df_piutang_luar_filtered.columns:
        df_piutang_luar_filtered = df_piutang_luar_filtered[(df_piutang_luar_filtered["Tanggal_Lengkap"] >= start_ts) & (df_piutang_luar_filtered["Tanggal_Lengkap"] <= end_ts)]
    if not df_gh_kas_biaya.empty and "Tanggal_Lengkap" in df_gh_kas_biaya.columns:
        _mask_gh_kas = (
            ((df_gh_kas_biaya["Tanggal_Lengkap"] >= start_ts) & (df_gh_kas_biaya["Tanggal_Lengkap"] <= end_ts))
            | df_gh_kas_biaya["Tanggal_Lengkap"].isna()
        )
        df_gh_kas_biaya = df_gh_kas_biaya[_mask_gh_kas]
    if not df_gh_tanaman_biaya.empty and "Tanggal_Lengkap" in df_gh_tanaman_biaya.columns:
        _mask_gh_tanaman = (
            ((df_gh_tanaman_biaya["Tanggal_Lengkap"] >= start_ts) & (df_gh_tanaman_biaya["Tanggal_Lengkap"] <= end_ts))
            | df_gh_tanaman_biaya["Tanggal_Lengkap"].isna()
        )
        df_gh_tanaman_biaya = df_gh_tanaman_biaya[_mask_gh_tanaman]
    def _find_tgl_panen_col(cols):
        for c in cols:
            cu = c.strip().upper()
            if "PANEN" in cu and any(k in cu for k in ["TANGGAL", "TGL", "DATE"]):
                return c
        for c in cols:
            if "PANEN" in c.strip().upper():
                return c
        for c in cols:
            if c.strip().upper() in ["TANGGAL", "TGL", "DATE"]:
                return c
        return None

    tgl_col_sudah = _find_tgl_panen_col(list(df_tanaman_sudah_filtered.columns))
    if tgl_col_sudah:
        df_tanaman_sudah_filtered["_tgl_parsed"] = pd.to_datetime(df_tanaman_sudah_filtered[tgl_col_sudah], dayfirst=True, errors="coerce")
        df_tanaman_sudah_filtered = df_tanaman_sudah_filtered[
            (df_tanaman_sudah_filtered["_tgl_parsed"] >= start_ts) &
            (df_tanaman_sudah_filtered["_tgl_parsed"] <= end_ts)
        ].drop(columns=["_tgl_parsed"])

    if not df_biaya_bulanan.empty and "Bulan_Num" in df_biaya_bulanan.columns:
        _bulan_periods = pd.period_range(start=start_ts, end=end_ts, freq="M")
        _bulan_terpilih = set(p.month for p in _bulan_periods)
        _mask_bb = df_biaya_bulanan["Bulan_Num"].isin(_bulan_terpilih) | df_biaya_bulanan["Bulan_Num"].isna()
        df_biaya_bulanan = df_biaya_bulanan[_mask_bb]

# KALKULASI UTAMA
omzet_lapak   = df_penjualan["Total harga"].sum() if not df_penjualan.empty and "Total harga" in df_penjualan.columns else 0
laba_lapak    = df_penjualan["Keuntungan"].sum()  if not df_penjualan.empty and "Keuntungan"  in df_penjualan.columns else 0
tunai_lapak   = df_penjualan["Tunai"].sum()       if not df_penjualan.empty and "Tunai"       in df_penjualan.columns else 0
kredit_lapak  = df_penjualan["Kredit"].sum()      if not df_penjualan.empty and "Kredit"      in df_penjualan.columns else 0

omzet_lapak_luar  = df_penjualan_luar["Total harga"].sum() if not df_penjualan_luar.empty and "Total harga" in df_penjualan_luar.columns else 0
laba_lapak_luar   = df_penjualan_luar["Keuntungan"].sum()  if not df_penjualan_luar.empty and "Keuntungan"  in df_penjualan_luar.columns else 0

omzet_ekspedisi = df_ekspedisi["PENDAPATAN"].sum()  if not df_ekspedisi.empty and "PENDAPATAN"  in df_ekspedisi.columns else 0
biaya_ekspedisi = df_ekspedisi["PENGELUARAN"].sum() if not df_ekspedisi.empty and "PENGELUARAN" in df_ekspedisi.columns else 0
laba_ekspedisi  = omzet_ekspedisi - biaya_ekspedisi

def _find_col_tanaman(df, keywords):
    if df is None or df.empty:
        return None
    return next((c for c in df.columns if any(k in c.strip().lower() for k in keywords)), None)

_omzet_col_tp = _find_col_tanaman(df_tanaman_sudah_filtered, ["omzet", "total harga", "pendapatan"])

omzet_tanaman = to_number(df_tanaman_sudah_filtered[_omzet_col_tp]).sum() if _omzet_col_tp else 0

biaya_berjalan_bulanan = (
    df_biaya_bulanan["BIAYA BERJALAN"].sum()
    if not df_biaya_bulanan.empty and "BIAYA BERJALAN" in df_biaya_bulanan.columns
    else 0
)
laba_tanaman = omzet_tanaman - biaya_berjalan_bulanan

total_omzet = omzet_lapak + omzet_lapak_luar + omzet_ekspedisi + omzet_tanaman
total_laba  = laba_lapak + laba_lapak_luar + laba_ekspedisi + laba_tanaman

# HEADER UTAMA
st.title("📊 Dashboard Ratu Jaya")
st.caption(f"Update Terakhir: {datetime.now().strftime('%d %B %Y, %H:%M')}")
st.divider()

# TABS
tab1, tab2, tab2b, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "💰 Pendapatan",
    "🏪 Analisa Lapak",
    "🏬 Analisa Lapak Luar",
    "🌱 Tanaman",
    "🧾 Piutang",
    "👨‍🌾 Hutang",
    "💸 Arus Kas",
    "🚛 Ekspedisi",
    "🏭 Kerugian Gudang",
    "🔮 Prediksi Harga",
    "🧮 Net Income",
    "🌿 Green House",
])

# TAB 1: PENDAPATAN
with tab1:
    section_heading("🧮 Ringkasan Keseluruhan")
    st.markdown(
        '<div class="hero-row">'
        + hero_card("💰 Total Seluruh Omzet", rp(total_omzet), "hero-blue")
        + hero_card("📈 Total Seluruh Laba",  rp(total_laba),  "hero-green")
        + '</div>',
        unsafe_allow_html=True
    )
    st.write("")

    with st.container(border=True):
        st.markdown('<div class="income-card-title">🏪 Pendapatan Lapak</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Omzet Lapak", rp(omzet_lapak))
        c2.metric("Laba Lapak",  rp(laba_lapak))
        c3.metric("Tunai",       rp(tunai_lapak))
        c4.metric("Kredit",      rp(kredit_lapak))

    with st.container(border=True):
        st.markdown('<div class="income-card-title">🏬 Pendapatan Lapak Luar</div>', unsafe_allow_html=True)
        l1, l2 = st.columns(2)
        l1.metric("Omzet Lapak Luar", rp(omzet_lapak_luar))
        l2.metric("Laba Lapak Luar",  rp(laba_lapak_luar))

    with st.container(border=True):
        st.markdown('<div class="income-card-title">🚛 Pendapatan Ekspedisi</div>', unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        e1.metric("Omzet Ekspedisi", rp(omzet_ekspedisi))
        e2.metric("Pengeluaran",     rp(biaya_ekspedisi))
        e3.metric("Laba Ekspedisi",  rp(laba_ekspedisi))

    with st.container(border=True):
        st.markdown('<div class="income-card-title">🌱 Pendapatan Tanaman Panen</div>', unsafe_allow_html=True)
        tp1, tp2, tp3 = st.columns(3)
        tp1.metric("Omzet Tanaman Panen",    rp(omzet_tanaman))
        tp2.metric("Biaya Berjalan Bulanan", rp(biaya_berjalan_bulanan))
        tp3.metric("Laba Tanaman Panen",     rp(laba_tanaman))
        if not _omzet_col_tp:
            st.caption("⚠️ Kolom Omzet tidak ditemukan di sheet 'TANAMAN PANEN'.")
        else:
            st.caption("Laba = Omzet − Total Biaya Berjalan Bulanan (kolom O-P sheet 'TANAMAN BELUM PANEN'). Omzet & Biaya Berjalan Bulanan mengikuti filter rentang tanggal sidebar (biaya dicocokkan berdasarkan bulan).")
        if not df_biaya_bulanan.empty:
            with st.expander("📅 Rincian Biaya Berjalan per Bulan (sesuai filter)"):
                _tbl_bb = df_biaya_bulanan[["BULAN", "BIAYA BERJALAN"]].copy()
                _tbl_bb["BIAYA BERJALAN"] = df_biaya_bulanan["BIAYA BERJALAN"].apply(rp)
                st.dataframe(_tbl_bb, use_container_width=True, hide_index=True)

    st.divider()
    section_heading("📊 Komposisi Omzet & Laba Keseluruhan")
    pie_col1, pie_col2 = st.columns(2)

    with pie_col1:
        fig_pie_omzet = px.pie(
            names=["Lapak", "Lapak Luar", "Ekspedisi", "Tanaman Panen"],
            values=[omzet_lapak, omzet_lapak_luar, omzet_ekspedisi, omzet_tanaman],
            title="Komposisi Omzet",
            color_discrete_sequence=["#1f77b4", "#17becf", "#2ca02c", "#ff7f0e"],
            hole=0.4
        )
        fig_pie_omzet.update_traces(
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{percent}<br>Rp %{value:,.0f}',
            textfont_size=14
        )
        st.plotly_chart(fig_pie_omzet, use_container_width=True)

    with pie_col2:
        fig_pie_laba = px.pie(
            names=["Laba Lapak", "Laba Lapak Luar", "Laba Ekspedisi", "Laba Tanaman Panen"],
            values=[laba_lapak, laba_lapak_luar, laba_ekspedisi, laba_tanaman],
            title="Komposisi Laba",
            color_discrete_sequence=["#d62728", "#e377c2", "#9467bd", "#8c564b"],
            hole=0.4
        )
        fig_pie_laba.update_traces(
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{percent}<br>Rp %{value:,.0f}',
            textfont_size=14
        )
        st.plotly_chart(fig_pie_laba, use_container_width=True)

    has_trend_lapak = not df_penjualan.empty and "Tanggal_Lengkap" in df_penjualan.columns and "Total harga" in df_penjualan.columns
    has_trend_luar  = not df_penjualan_luar.empty and "Tanggal_Lengkap" in df_penjualan_luar.columns and "Total harga" in df_penjualan_luar.columns

    if has_trend_lapak or has_trend_luar:
        section_heading("📅 Tren Omzet Bulanan")
        dfs_trend = []
        if has_trend_lapak:
            tmp = df_penjualan.copy()
            tmp["Sumber"] = "Lapak"
            dfs_trend.append(tmp)
        if has_trend_luar:
            tmp2 = df_penjualan_luar.copy()
            tmp2["Sumber"] = "Lapak Luar"
            dfs_trend.append(tmp2)
        df_all_penjualan = pd.concat(dfs_trend, ignore_index=True)
        df_all_penjualan["Bulan_Label"] = df_all_penjualan["Tanggal_Lengkap"].dt.to_period("M").astype(str)
        bulanan_trend = df_all_penjualan.groupby(["Bulan_Label", "Sumber"]).agg(
            Omzet=("Total harga", "sum"),
            Laba=("Keuntungan", "sum")
        ).reset_index()

        fig_trend = px.bar(
            bulanan_trend, x="Bulan_Label", y="Omzet", color="Sumber",
            barmode="group",
            title="Tren Omzet Bulanan (Lapak & Lapak Luar)",
            text=bulanan_trend["Omzet"].apply(rp_short),
            color_discrete_map={"Lapak": "#1f77b4", "Lapak Luar": "#17becf"}
        )
        fig_trend.update_traces(textposition="outside", textfont_size=11)
        max_trend = bulanan_trend["Omzet"].max() if not bulanan_trend.empty else 0
        pad_yaxis(fig_trend, max_trend)
        fig_trend.update_layout(height=420, yaxis_tickformat=",")
        st.plotly_chart(fig_trend, use_container_width=True)

# TAB 2: ANALISA LAPAK
with tab2:
    section_heading("🚛 Rincian per Invoice")
    if df_barang_masuk_raw.empty or "INVOICE" not in df_barang_masuk_raw.columns:
        st.info("Kolom 'INVOICE' tidak ditemukan di sheet BARANG_MASUK (atau datanya kosong), jadi Rincian per Invoice belum bisa ditampilkan.")
    else:
        def _invoice_sort_key_rpi(s):
            if pd.isna(s):
                return (-1, "")
            s = str(s)
            digits = "".join(ch for ch in s if ch.isdigit())
            return (int(digits) if digits else -1, s)

        # Pilihan Invoice cuma diambil dari 40 hari terakhir (berdasarkan TANGGAL di
        # sheet BARANG_MASUK) supaya dropdown tidak penuh nomor invoice lama.
        df_bm_opsi_rpi = df_barang_masuk_raw
        if "Tanggal_Lengkap" in df_barang_masuk_raw.columns:
            batas_awal_rpi = pd.Timestamp.now().normalize() - pd.Timedelta(days=40)
            df_bm_opsi_rpi = df_barang_masuk_raw[df_barang_masuk_raw["Tanggal_Lengkap"] >= batas_awal_rpi]

        invoice_opts_rpi = sorted(
            df_bm_opsi_rpi["INVOICE"].dropna().astype(str).unique(),
            key=_invoice_sort_key_rpi, reverse=True
        )
        if not invoice_opts_rpi:
            st.info("Belum ada nomor Invoice di sheet BARANG_MASUK dalam 40 hari terakhir.")
        else:
            sel_invoice_rpi = st.selectbox("Invoice", invoice_opts_rpi, key="tab2_rpi_invoice")
            df_bm = df_barang_masuk_raw[df_barang_masuk_raw["INVOICE"].astype(str) == sel_invoice_rpi].copy().reset_index(drop=True)

            # Tanggal/Nopol/Penimbang/Asal Barang/Gudang dianggap sama untuk 1
            # invoice -- ditampilkan sekali di luar tabel.
            info_fields_rpi = []
            for src, lbl_info in [
                ("TANGGAL", "📅 Tanggal"), ("NOPOL", "🚚 Nopol"), ("PENIMBANG", "⚖️ Penimbang"),
                ("ASAL BARANG", "📦 Asal Barang"), ("TUJUAN", "📍 Gudang"),
            ]:
                if src in df_bm.columns:
                    _val = df_bm[src].iloc[0] if not df_bm.empty else None
                    info_fields_rpi.append((lbl_info, str(_val) if is_filled(_val) else "-"))
            if info_fields_rpi:
                info_cols_rpi = st.columns(len(info_fields_rpi))
                for _col, (lbl_info, val_info) in zip(info_cols_rpi, info_fields_rpi):
                    _col.metric(lbl_info, val_info)

            gudang_invoice_rpi = None
            if "TUJUAN" in df_bm.columns and not df_bm.empty and is_filled(df_bm["TUJUAN"].iloc[0]):
                gudang_invoice_rpi = str(df_bm["TUJUAN"].iloc[0]).strip()

            # Modal Beli (Lahan) = SUM(Tonnase Lahan x Harga Modal) per baris di
            # sheet BARANG_MASUK, untuk invoice terpilih.
            modal_beli_rpi = 0.0
            if "TONNASE LAHAN" in df_bm.columns and "HARGA MODAL" in df_bm.columns:
                modal_beli_rpi = float(
                    (to_number(df_bm["TONNASE LAHAN"]).fillna(0) * to_number(df_bm["HARGA MODAL"]).fillna(0)).sum()
                )

            # ---------- Tabel HIJAU: rekap level Gudang ----------
            df_green_rpi = pd.DataFrame(columns=["JENIS", "GRADE", "Total Grading", "Tonase Lahan", "Stok Gudang", "Terjual Gudang", "Pendapatan Gudang", "Laba Gudang"])
            if "JENIS" in df_bm.columns and "GRADE" in df_bm.columns:
                df_bm_valid_rpi = df_bm[df_bm["JENIS"].apply(is_filled) & df_bm["GRADE"].apply(is_filled)]
                if not df_bm_valid_rpi.empty:
                    agg_green_rpi = {}
                    if "TOTAL GRADING" in df_bm_valid_rpi.columns:
                        agg_green_rpi["Total Grading"] = ("TOTAL GRADING", "sum")
                    if "TONNASE LAHAN" in df_bm_valid_rpi.columns:
                        agg_green_rpi["Tonase Lahan"] = ("TONNASE LAHAN", "sum")
                    if "STOK GUDANG" in df_bm_valid_rpi.columns:
                        agg_green_rpi["Stok Gudang"] = ("STOK GUDANG", "sum")
                    if agg_green_rpi:
                        df_green_rpi = df_bm_valid_rpi.groupby(["JENIS", "GRADE"], as_index=False).agg(**agg_green_rpi)
                    else:
                        df_green_rpi = df_bm_valid_rpi[["JENIS", "GRADE"]].drop_duplicates().reset_index(drop=True)

            # Terjual Gudang / Pendapatan Gudang / Laba Gudang: sheet PENJUALAN,
            # baris dengan Kode Lapak (kolom B) = kode Gudang invoice ini (nilai
            # kolom B bisa berisi kode Gudang kalau terjual langsung di gudang, atau
            # kode Lapak kalau sudah di-moving) + Invoice (kolom F) = invoice
            # terpilih + Jenis/Grade sama.
            if not df_green_rpi.empty and gudang_invoice_rpi and not df_penjualan_raw.empty \
                    and "KODE LAPAK" in df_penjualan_raw.columns \
                    and "JENIS" in df_penjualan_raw.columns and "GRADE" in df_penjualan_raw.columns:
                mask_gudang_rpi = df_penjualan_raw["KODE LAPAK"] == gudang_invoice_rpi
                if "INVOICE" in df_penjualan_raw.columns:
                    mask_gudang_rpi = mask_gudang_rpi & (df_penjualan_raw["INVOICE"].astype(str) == sel_invoice_rpi)
                df_pj_gudang_rpi = df_penjualan_raw[mask_gudang_rpi]
                agg_pj_gudang_rpi = {}
                if "Jumlah (KG)" in df_pj_gudang_rpi.columns:
                    agg_pj_gudang_rpi["Terjual Gudang"] = ("Jumlah (KG)", "sum")
                if "Total harga" in df_pj_gudang_rpi.columns:
                    agg_pj_gudang_rpi["Pendapatan Gudang"] = ("Total harga", "sum")
                if "Keuntungan" in df_pj_gudang_rpi.columns:
                    agg_pj_gudang_rpi["Laba Gudang"] = ("Keuntungan", "sum")
                if agg_pj_gudang_rpi and not df_pj_gudang_rpi.empty:
                    df_pj_gudang_grp_rpi = df_pj_gudang_rpi.groupby(["JENIS", "GRADE"], as_index=False).agg(**agg_pj_gudang_rpi)
                    df_green_rpi = df_green_rpi.merge(df_pj_gudang_grp_rpi, on=["JENIS", "GRADE"], how="left")

            for _c in ["Total Grading", "Tonase Lahan", "Stok Gudang", "Terjual Gudang", "Pendapatan Gudang", "Laba Gudang"]:
                if _c not in df_green_rpi.columns:
                    df_green_rpi[_c] = 0.0
                df_green_rpi[_c] = to_number(df_green_rpi[_c]).fillna(0)
            df_green_rpi = df_green_rpi.sort_values(["JENIS", "GRADE"]).reset_index(drop=True)

            # ---------- Tabel PINK: rekap level Lapak (setelah moving) ----------
            # Baris Stok Lapak diambil langsung dari Invoice (kolom B sheet STOK
            # LAPAK) yang sama dengan invoice terpilih -- bukan lagi dicocokkan
            # tidak langsung lewat Jenis+Grade saja, supaya tidak ikut kebawa data
            # dari invoice lain yang kebetulan Jenis+Grade-nya sama.
            df_pink_rpi = pd.DataFrame(columns=["TUJUAN", "JENIS", "GRADE", "JUMLAH MOVING", "STOK LAPAK", "Terjual", "Pendapatan", "Laba", "Modal", "Total Pendapatan"])
            if not df_stok_lapak_invoice_raw.empty and "INVOICE" in df_stok_lapak_invoice_raw.columns \
                    and "TUJUAN" in df_stok_lapak_invoice_raw.columns:
                df_pink_rpi = df_stok_lapak_invoice_raw[
                    (df_stok_lapak_invoice_raw["INVOICE"].astype(str) == sel_invoice_rpi)
                    & df_stok_lapak_invoice_raw["TUJUAN"].apply(is_filled)
                ].reset_index(drop=True)

            if not df_pink_rpi.empty:
                jkg_p = to_number(df_pink_rpi["JUMLAH MOVING"]).fillna(0) if "JUMLAH MOVING" in df_pink_rpi.columns else pd.Series([0.0] * len(df_pink_rpi))
                hpp_p = to_number(df_pink_rpi["HPP LAPAK"]).fillna(0) if "HPP LAPAK" in df_pink_rpi.columns else pd.Series([0.0] * len(df_pink_rpi))
                stk_p = to_number(df_pink_rpi["STOK LAPAK"]).fillna(0) if "STOK LAPAK" in df_pink_rpi.columns else pd.Series([0.0] * len(df_pink_rpi))
                df_pink_rpi["JUMLAH MOVING"] = jkg_p.values
                df_pink_rpi["STOK LAPAK"] = stk_p.values
                df_pink_rpi["Modal"] = (jkg_p * hpp_p).values

                # Terjual / Pendapatan / Laba (lapak): sheet PENJUALAN, dicocokkan
                # Kode Lapak (kolom B) == Moving/Tujuan lapak (di data stok) +
                # Invoice (kolom F) = invoice terpilih + Jenis + Grade sama.
                if not df_penjualan_raw.empty and "KODE LAPAK" in df_penjualan_raw.columns \
                        and "JENIS" in df_penjualan_raw.columns and "GRADE" in df_penjualan_raw.columns:
                    df_pj_valid_rpi = df_penjualan_raw[df_penjualan_raw["KODE LAPAK"].apply(is_filled)]
                    if "INVOICE" in df_pj_valid_rpi.columns:
                        df_pj_valid_rpi = df_pj_valid_rpi[df_pj_valid_rpi["INVOICE"].astype(str) == sel_invoice_rpi]
                    agg_pj_lapak_rpi = {}
                    if "Jumlah (KG)" in df_pj_valid_rpi.columns:
                        agg_pj_lapak_rpi["Terjual"] = ("Jumlah (KG)", "sum")
                    if "Total harga" in df_pj_valid_rpi.columns:
                        agg_pj_lapak_rpi["Pendapatan"] = ("Total harga", "sum")
                    if "Keuntungan" in df_pj_valid_rpi.columns:
                        agg_pj_lapak_rpi["Laba"] = ("Keuntungan", "sum")
                    if agg_pj_lapak_rpi and not df_pj_valid_rpi.empty:
                        df_pj_lapak_grp_rpi = df_pj_valid_rpi.groupby(["KODE LAPAK", "JENIS", "GRADE"], as_index=False).agg(**agg_pj_lapak_rpi)
                        df_pink_rpi = df_pink_rpi.merge(
                            df_pj_lapak_grp_rpi, left_on=["TUJUAN", "JENIS", "GRADE"],
                            right_on=["KODE LAPAK", "JENIS", "GRADE"], how="left"
                        )
                for _c in ["Terjual", "Pendapatan", "Laba"]:
                    if _c not in df_pink_rpi.columns:
                        df_pink_rpi[_c] = 0.0
                    df_pink_rpi[_c] = to_number(df_pink_rpi[_c]).fillna(0)

                df_pink_rpi = df_pink_rpi.sort_values(["TUJUAN", "JENIS", "GRADE"]).reset_index(drop=True)
                # Total Pendapatan dijumlah per kelompok Moving (Tujuan), tampil
                # sekali per kelompok (gaya sel gabungan) di HTML nanti.
                df_pink_rpi["Total Pendapatan"] = df_pink_rpi.groupby("TUJUAN")["Pendapatan"].transform("sum")

            # Total Pendapatan & Total Laba = gabungan Data Gudang + Data Lapak
            # (setelah moving), dipakai untuk metrik ringkasan di atas tabel.
            total_pendapatan_gudang_rpi = float(df_green_rpi["Pendapatan Gudang"].sum()) if not df_green_rpi.empty else 0.0
            total_pendapatan_lapak_rpi = float(df_pink_rpi["Pendapatan"].sum()) if not df_pink_rpi.empty else 0.0
            total_pendapatan_rpi = total_pendapatan_gudang_rpi + total_pendapatan_lapak_rpi

            total_laba_gudang_rpi = float(df_green_rpi["Laba Gudang"].sum()) if not df_green_rpi.empty else 0.0
            total_laba_lapak_rpi = float(df_pink_rpi["Laba"].sum()) if not df_pink_rpi.empty else 0.0
            total_laba_rpi = total_laba_gudang_rpi + total_laba_lapak_rpi

            total_moving_rpi = float(df_pink_rpi["JUMLAH MOVING"].sum()) if not df_pink_rpi.empty and "JUMLAH MOVING" in df_pink_rpi.columns else 0.0

            st.write("")
            agg_cols_rpi = st.columns(4)
            agg_cols_rpi[0].metric("💰 Modal Beli (Lahan)", rp(modal_beli_rpi))
            agg_cols_rpi[1].metric("📈 Total Pendapatan", rp(total_pendapatan_rpi))
            agg_cols_rpi[2].metric("💵 Total Laba", rp(total_laba_rpi))
            agg_cols_rpi[3].metric("🚚 Total Moving", f"{total_moving_rpi:,.0f} KG")
            st.write("")

            def _esc_rpi(x):
                s = "" if x is None else str(x)
                return (s.replace("&", "&amp;").replace("<", "&lt;")
                         .replace(">", "&gt;").replace('"', "&quot;"))

            # --- HTML tabel HIJAU (Data Gudang) ---
            if df_green_rpi.empty:
                green_html = (
                    '<div class="rpi-panel-title rpi-green">📗 Data Gudang</div>'
                    '<p class="rpi-empty">Tidak ada data Jenis/Grade untuk invoice ini.</p>'
                )
            else:
                rows_g = []
                for _, r in df_green_rpi.iterrows():
                    rows_g.append(
                        "<tr>"
                        f"<td>{_esc_rpi(r['JENIS'])}</td>"
                        f"<td>{_esc_rpi(r['GRADE'])}</td>"
                        f"<td class=\"rpi-num\">{r['Total Grading']:,.1f}</td>"
                        f"<td class=\"rpi-num\">{r['Tonase Lahan']:,.1f}</td>"
                        f"<td class=\"rpi-num\">{r['Stok Gudang']:,.1f}</td>"
                        f"<td class=\"rpi-num\">{r['Terjual Gudang']:,.1f}</td>"
                        f"<td class=\"rpi-num\">{_esc_rpi(rp(r['Pendapatan Gudang']))}</td>"
                        f"<td class=\"rpi-num\">{_esc_rpi(rp(r['Laba Gudang']))}</td>"
                        "</tr>"
                    )
                green_html = (
                    '<div class="rpi-panel-title rpi-green">📗 Data Gudang</div>'
                    '<div class="rpi-wrap"><table class="rpi-table">'
                    "<thead>"
                    '<tr class="rpi-total-row"><td>TOTAL</td><td></td>'
                    f'<td class="rpi-num">{df_green_rpi["Total Grading"].sum():,.1f}</td>'
                    f'<td class="rpi-num">{df_green_rpi["Tonase Lahan"].sum():,.1f}</td>'
                    f'<td class="rpi-num">{df_green_rpi["Stok Gudang"].sum():,.1f}</td>'
                    f'<td class="rpi-num">{df_green_rpi["Terjual Gudang"].sum():,.1f}</td>'
                    f'<td class="rpi-num">{_esc_rpi(rp(df_green_rpi["Pendapatan Gudang"].sum()))}</td>'
                    f'<td class="rpi-num">{_esc_rpi(rp(df_green_rpi["Laba Gudang"].sum()))}</td></tr>'
                    "<tr><th>Jenis</th><th>Grade</th><th>Total Grading</th><th>Tonase Lahan</th><th>Stok Gudang</th>"
                    "<th>Terjual Gudang</th><th>Pendapatan Gudang</th><th>Laba Gudang</th></tr>"
                    "</thead>"
                    f'<tbody>{"".join(rows_g)}</tbody>'
                    "</table></div>"
                )

            # --- HTML tabel PINK (Data Lapak setelah Moving) ---
            if df_pink_rpi.empty:
                pink_html = (
                    '<div class="rpi-panel-title rpi-pink">📕 Data Lapak (Setelah Moving)</div>'
                    '<p class="rpi-empty">Belum ada data STOK LAPAK dengan nomor Invoice yang sama dengan invoice ini.</p>'
                )
            else:
                n_pk = len(df_pink_rpi)
                grp_size_pk = df_pink_rpi.groupby("TUJUAN")["TUJUAN"].transform("size").tolist()
                is_first_pk = (df_pink_rpi["TUJUAN"] != df_pink_rpi["TUJUAN"].shift(1)).tolist()
                is_last_pk = (df_pink_rpi["TUJUAN"] != df_pink_rpi["TUJUAN"].shift(-1)).tolist()
                rows_p = []
                for i in range(n_pk):
                    r = df_pink_rpi.iloc[i]
                    end_cls_pk = " rpi-group-end" if is_last_pk[i] else ""
                    cell_moving = (
                        f'<td class="rpi-merge rpi-group-end" rowspan="{grp_size_pk[i]}">{_esc_rpi(r["TUJUAN"])}</td>'
                        if is_first_pk[i] else ""
                    )
                    cell_total_pend = (
                        f'<td class="rpi-merge rpi-group-end" rowspan="{grp_size_pk[i]}">{_esc_rpi(rp(r["Total Pendapatan"]))}</td>'
                        if is_first_pk[i] else ""
                    )
                    rows_p.append(
                        "<tr>"
                        f"{cell_moving}"
                        f"<td class=\"{end_cls_pk.strip()}\">{_esc_rpi(r['JENIS'])}</td>"
                        f"<td class=\"{end_cls_pk.strip()}\">{_esc_rpi(r['GRADE'])}</td>"
                        f"<td class=\"rpi-num{end_cls_pk}\">{r['JUMLAH MOVING']:,.1f}</td>"
                        f"<td class=\"rpi-num{end_cls_pk}\">{r['STOK LAPAK']:,.1f}</td>"
                        f"<td class=\"rpi-num{end_cls_pk}\">{r['Terjual']:,.1f}</td>"
                        f"<td class=\"rpi-num{end_cls_pk}\">{_esc_rpi(rp(r['Pendapatan']))}</td>"
                        f"<td class=\"rpi-num{end_cls_pk}\">{_esc_rpi(rp(r['Laba']))}</td>"
                        f"{cell_total_pend}"
                        f"<td class=\"rpi-num{end_cls_pk}\">{_esc_rpi(rp(r['Modal']))}</td>"
                        "</tr>"
                    )
                pink_html = (
                    '<div class="rpi-panel-title rpi-pink">📕 Data Lapak (Setelah Moving)</div>'
                    '<div class="rpi-wrap"><table class="rpi-table">'
                    "<thead>"
                    '<tr class="rpi-total-row"><td>TOTAL</td><td></td><td></td>'
                    f'<td class="rpi-num">{df_pink_rpi["JUMLAH MOVING"].sum():,.1f}</td>'
                    f'<td class="rpi-num">{df_pink_rpi["STOK LAPAK"].sum():,.1f}</td>'
                    f'<td class="rpi-num">{df_pink_rpi["Terjual"].sum():,.1f}</td>'
                    f'<td class="rpi-num">{_esc_rpi(rp(df_pink_rpi["Pendapatan"].sum()))}</td>'
                    f'<td class="rpi-num">{_esc_rpi(rp(df_pink_rpi["Laba"].sum()))}</td>'
                    f'<td class="rpi-num">{_esc_rpi(rp(df_pink_rpi["Pendapatan"].sum()))}</td>'
                    f'<td class="rpi-num">{_esc_rpi(rp(df_pink_rpi["Modal"].sum()))}</td></tr>'
                    "<tr><th>Moving</th><th>Jenis</th><th>Grade</th><th>Jumlah (KG)</th><th>Stok (KG)</th>"
                    "<th>Terjual</th><th>Pendapatan</th><th>Laba</th><th>Total Pendapatan</th><th>Modal</th></tr>"
                    "</thead>"
                    f'<tbody>{"".join(rows_p)}</tbody>'
                    "</table></div>"
                )

            rpi_html = f"""<style>
.rpi-row {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }}
.rpi-panel {{ flex: 1; min-width: 380px; }}
.rpi-panel-title {{ font-size: 15px; font-weight: 800; padding: 9px 12px; border-radius: 8px 8px 0 0; color: #fff; }}
.rpi-green {{ background: #2ca02c; }}
.rpi-pink  {{ background: #d6336c; }}
.rpi-empty {{ padding: 14px; color: #888; border: 1px solid #e0e6f0; border-top: none; border-radius: 0 0 8px 8px; margin: 0; }}
.rpi-wrap {{ max-height: 480px; overflow: auto; border: 1px solid #e0e6f0; border-radius: 0 0 8px 8px; }}
.rpi-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.rpi-table th {{ position: sticky; top: 32px; background: #1f3864; color: #fff; padding: 8px 9px; text-align: left; white-space: nowrap; z-index: 1; }}
.rpi-table td {{ padding: 7px 9px; border-bottom: 1px solid #eef1f6; white-space: nowrap; }}
.rpi-table td.rpi-num {{ text-align: right; }}
.rpi-table td.rpi-merge {{ text-align: center; vertical-align: middle; font-weight: 800; color: #1f3864; border-left: 1px solid #e0e6f0; background: #f4f7fc; }}
.rpi-table td.rpi-group-end {{ border-bottom: 2px solid #1f3864; }}
.rpi-table tr.rpi-total-row td {{ position: sticky; top: 0; background: #4472c4; color: #fff; font-weight: 800; z-index: 2; }}
</style>
<div class="rpi-row">
<div class="rpi-panel">{green_html}</div>
<div class="rpi-panel">{pink_html}</div>
</div>"""
            st.markdown(rpi_html, unsafe_allow_html=True)

    st.divider()

    section_heading("📦 Stok Lapak & Gudang")
    st.caption("📌 Stok dihitung dari invoice 14 hari terakhir (khusus GDC: 7 hari terakhir), berdasarkan tanggal di sheet STOK LAPAK / BARANG_MASUK. Dipecah per Invoice, bukan per Grade.")

    def _cutoff_tanggal_stok(tujuan):
        hari = 7 if str(tujuan).strip().upper() == "GDC" else 14
        return pd.Timestamp.now().normalize() - pd.Timedelta(days=hari)

    frames_stok = []
    stok_warnings = []

    if df_stok_lapak_invoice_raw.empty:
        stok_warnings.append("Sheet 'STOK LAPAK' kosong atau tidak ditemukan.")
    elif not all(c in df_stok_lapak_invoice_raw.columns for c in ["TUJUAN", "STOK LAPAK", "INVOICE", "Tanggal_Lengkap"]):
        stok_warnings.append("Kolom 'TUJUAN'/'STOK LAPAK'/'INVOICE'/Tanggal tidak lengkap di sheet STOK LAPAK.")
    else:
        tmp_sl = df_stok_lapak_invoice_raw[df_stok_lapak_invoice_raw["TUJUAN"].apply(is_filled)].copy()
        tmp_sl = tmp_sl[tmp_sl["Tanggal_Lengkap"] >= tmp_sl["TUJUAN"].apply(_cutoff_tanggal_stok)]
        tmp_sl["Tipe"] = "Lapak"
        tmp_sl["Stok (KG)"] = tmp_sl["STOK LAPAK"]
        frames_stok.append(tmp_sl[["Tipe", "TUJUAN", "INVOICE", "Stok (KG)"]])

    if df_barang_masuk_raw.empty:
        stok_warnings.append("Sheet 'BARANG_MASUK' kosong atau tidak ditemukan.")
    elif not all(c in df_barang_masuk_raw.columns for c in ["TUJUAN", "STOK GUDANG", "INVOICE", "Tanggal_Lengkap"]):
        stok_warnings.append("Kolom 'TUJUAN'/'STOK GUDANG'/'INVOICE'/Tanggal tidak lengkap di sheet BARANG_MASUK.")
    else:
        tmp_sg = df_barang_masuk_raw[
            df_barang_masuk_raw["TUJUAN"].apply(is_filled)
            & df_barang_masuk_raw["TUJUAN"].astype(str).str.strip().str.upper().isin(["GDC", "GDM"])
        ].copy()
        tmp_sg = tmp_sg[tmp_sg["Tanggal_Lengkap"] >= tmp_sg["TUJUAN"].apply(_cutoff_tanggal_stok)]
        tmp_sg["Tipe"] = "Gudang"
        tmp_sg["Stok (KG)"] = tmp_sg["STOK GUDANG"]
        frames_stok.append(tmp_sg[["Tipe", "TUJUAN", "INVOICE", "Stok (KG)"]])

    for _msg in stok_warnings:
        st.warning(_msg)

    df_stok_gabungan = pd.concat(frames_stok, ignore_index=True) if frames_stok else pd.DataFrame()

    if df_stok_gabungan.empty:
        st.info("Belum ada data stok (Lapak maupun Gudang GDC/GDM) yang bisa ditampilkan.")
    else:
        df_stok_gabungan["Stok (KG)"] = to_number(df_stok_gabungan["Stok (KG)"])
        df_stok_gabungan["INVOICE"] = df_stok_gabungan["INVOICE"].fillna("-").astype(str).str.strip().replace({"": "-", "nan": "-", "None": "-"})

        st.markdown(
            f'<div class="big-total">📦 Total Stok Keseluruhan: '
            f'{df_stok_gabungan["Stok (KG)"].sum():,.1f} KG</div>',
            unsafe_allow_html=True
        )

        lokasi_opts_stok = sorted(df_stok_gabungan["TUJUAN"].dropna().astype(str).unique())
        sel_lokasi_stok = st.multiselect(
            "🔎 Filter Lokasi (Lapak/Gudang)", lokasi_opts_stok, default=lokasi_opts_stok,
            key="tab2_stok_lokasi_filter"
        )

        if not sel_lokasi_stok:
            st.info("Pilih minimal satu lokasi (lapak/gudang) untuk menampilkan rincian stok.")
        else:
            df_stok_filtered = df_stok_gabungan[df_stok_gabungan["TUJUAN"].astype(str).isin(sel_lokasi_stok)].copy()

            per_lok_invoice = (
                df_stok_filtered.groupby(["Tipe", "TUJUAN", "INVOICE"])["Stok (KG)"]
                .sum().reset_index()
            )

            total_per_lokasi = (
                per_lok_invoice.groupby("TUJUAN")["Stok (KG)"].sum()
                .sort_values(ascending=False)
            )
            urutan_lokasi_stok = total_per_lokasi.index.tolist()

            st.metric("Total Stok (Lokasi Terpilih)", f"{per_lok_invoice['Stok (KG)'].sum():,.1f} KG")

            st.markdown("**📍 Total Stok per Lapak/Gudang**")
            tabel_total_lokasi = (
                per_lok_invoice.groupby(["Tipe", "TUJUAN"])["Stok (KG)"].sum()
                .reset_index()
                .sort_values("Stok (KG)", ascending=False)
                .rename(columns={"TUJUAN": "Lokasi"})
            )
            tabel_total_lokasi["Stok (KG)"] = tabel_total_lokasi["Stok (KG)"].apply(lambda x: f"{x:,.1f} KG")
            st.dataframe(tabel_total_lokasi, use_container_width=True, hide_index=True)

            fig_stok_gab = px.bar(
                per_lok_invoice, x="TUJUAN", y="Stok (KG)", color="INVOICE",
                barmode="stack",
                category_orders={"TUJUAN": urutan_lokasi_stok},
                title="Stok per Lokasi (Lapak/Gudang), dipecah Invoice",
                labels={"TUJUAN": "Lokasi (Lapak/Gudang)"}
            )
            fig_stok_gab.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=520, xaxis_tickangle=-30
            )
            for lok in urutan_lokasi_stok:
                fig_stok_gab.add_annotation(
                    x=lok, y=total_per_lokasi.loc[lok],
                    text=f"<b>{total_per_lokasi.loc[lok]:,.1f} KG</b>",
                    showarrow=False, yshift=14,
                    font=dict(size=12, color="#1f3864")
                )
            pad_yaxis(fig_stok_gab, total_per_lokasi.max() if not total_per_lokasi.empty else 0, pad=0.3)
            st.plotly_chart(fig_stok_gab, use_container_width=True)

            tabel_stok_gab = per_lok_invoice.sort_values(["TUJUAN", "Stok (KG)"], ascending=[True, False]).copy()
            tabel_stok_gab = tabel_stok_gab[["Tipe", "TUJUAN", "INVOICE", "Stok (KG)"]].rename(columns={"TUJUAN": "Lokasi"})
            tabel_stok_gab["Stok (KG)"] = tabel_stok_gab["Stok (KG)"].apply(lambda x: f"{x:,.1f} KG")
            st.dataframe(tabel_stok_gab, use_container_width=True, hide_index=True)

    st.divider()

    if df_penjualan.empty:
        st.info("Data penjualan lapak kosong untuk periode yang dipilih.")
    else:
        KG_COL    = "Jumlah (KG)"
        GRADE_COL = "GRADE"
        JENIS_COL = "JENIS"

        has_kg    = KG_COL    in df_penjualan.columns
        has_grade = GRADE_COL in df_penjualan.columns
        has_jenis = JENIS_COL in df_penjualan.columns

        if has_kg:
            df_ton_all = df_penjualan.copy()
            df_ton_all[KG_COL] = to_number(df_ton_all[KG_COL])
            if "Is_Dibuang" in df_ton_all.columns:
                total_kg_terjual = df_ton_all[df_ton_all["Is_Dibuang"] == False][KG_COL].sum()
                total_kg_dibuang = df_ton_all[df_ton_all["Is_Dibuang"] == True][KG_COL].sum()
            else:
                total_kg_terjual = df_ton_all[KG_COL].sum()
                total_kg_dibuang = 0
            total_kg_semua = df_ton_all[KG_COL].sum()

            ton1, ton2, ton3 = st.columns(3)
            ton1.metric("⚖️ Total Tonnase Terjual", f"{total_kg_terjual:,.1f} KG")
            ton2.metric("🗑️ Total Tonnase Dibuang", f"{total_kg_dibuang:,.1f} KG")
            ton3.metric("📦 Total Tonnase Keseluruhan", f"{total_kg_semua:,.1f} KG")
            st.divider()

        section_heading("🏆 Grade & Jenis Terlaris")
        col_tl1, col_tl2 = st.columns(2)

        with col_tl1:
            if has_grade and has_kg:
                df_g = df_penjualan.copy()
                df_g[KG_COL] = to_number(df_g[KG_COL])
                grade_rank = (
                    df_g[df_g[KG_COL].notna() & df_g[GRADE_COL].apply(is_filled)]
                    .groupby(GRADE_COL)[KG_COL].sum()
                    .sort_values(ascending=False)
                )
                if not grade_rank.empty:
                    st.metric("🥇 Grade Terlaris", str(grade_rank.index[0]),
                              f"{grade_rank.iloc[0]:,.1f} KG")
                else:
                    st.info("Belum ada data grade.")
            else:
                st.info("Kolom GRADE tidak ditemukan.")

        with col_tl2:
            if has_jenis and has_kg:
                df_j = df_penjualan.copy()
                df_j[KG_COL] = to_number(df_j[KG_COL])
                jenis_rank = (
                    df_j[df_j[KG_COL].notna() & df_j[JENIS_COL].apply(is_filled)]
                    .groupby(JENIS_COL)[KG_COL].sum()
                    .sort_values(ascending=False)
                )
                if not jenis_rank.empty:
                    st.metric("🥇 Jenis Terlaris", str(jenis_rank.index[0]),
                              f"{jenis_rank.iloc[0]:,.1f} KG")
                else:
                    st.info("Belum ada data jenis.")
            else:
                st.info("Kolom JENIS tidak ditemukan.")

        st.divider()

        section_heading("📊 Omzet & Profit per Lapak")
        if "KODE LAPAK" in df_penjualan.columns:
            per_lapak = df_penjualan.groupby("KODE LAPAK").agg(
                Omzet=("Total harga", "sum"),
                Profit=("Keuntungan", "sum")
            ).reset_index().sort_values("Omzet", ascending=False)

            per_lapak["Margin_%"] = per_lapak.apply(
                lambda r: (r["Profit"] / r["Omzet"] * 100) if r["Omzet"] > 0 else 0, axis=1
            )

            fig_bar_lapak = go.Figure()
            fig_bar_lapak.add_trace(go.Bar(
                name="Omzet",
                x=per_lapak["KODE LAPAK"], y=per_lapak["Omzet"],
                marker_color="#1f77b4",
                text=[rp_short(v) for v in per_lapak["Omzet"]],
                textposition="outside", textfont=dict(size=13, color="#1f3864")
            ))
            fig_bar_lapak.add_trace(go.Bar(
                name="Profit",
                x=per_lapak["KODE LAPAK"], y=per_lapak["Profit"],
                marker_color="#d62728",
                text=[rp_short(v) for v in per_lapak["Profit"]],
                textposition="outside", textfont=dict(size=13, color="#d62728")
            ))
            fig_bar_lapak.update_layout(
                barmode="group", title="Omzet & Profit per Lapak",
                yaxis_title="Rupiah", xaxis_title="Kode Lapak",
                legend=dict(orientation="h", yanchor="bottom", y=1.02), height=480
            )
            pad_yaxis(fig_bar_lapak, max(per_lapak["Omzet"].max(), per_lapak["Profit"].max()) if not per_lapak.empty else 0)
            st.plotly_chart(fig_bar_lapak, use_container_width=True)

            fig_margin = go.Figure()
            fig_margin.add_trace(go.Bar(
                x=per_lapak["KODE LAPAK"],
                y=per_lapak["Margin_%"],
                marker_color=[
                    "#2ca02c" if v >= 20 else "#ff7f0e" if v >= 10 else "#d62728"
                    for v in per_lapak["Margin_%"]
                ],
                text=[f"{v:.1f}%" for v in per_lapak["Margin_%"]],
                textposition="outside",
                textfont=dict(size=13)
            ))
            fig_margin.update_layout(
                title="% Margin Profit per Lapak (Profit / Omzet × 100%)",
                xaxis_title="Kode Lapak", yaxis_title="Margin (%)",
                showlegend=False, height=380
            )
            pad_yaxis(fig_margin, per_lapak["Margin_%"].max() if not per_lapak.empty else 0)
            st.plotly_chart(fig_margin, use_container_width=True)

            tabel_lapak = per_lapak.copy()
            tabel_lapak["Omzet"]    = per_lapak["Omzet"].apply(rp)
            tabel_lapak["Profit"]   = per_lapak["Profit"].apply(rp)
            tabel_lapak["Margin_%"] = per_lapak["Margin_%"].apply(lambda x: f"{x:.1f}%")
            tabel_lapak = tabel_lapak.rename(columns={"Margin_%": "% Margin Profit"})
            st.dataframe(tabel_lapak[["KODE LAPAK", "Omzet", "Profit", "% Margin Profit"]], use_container_width=True, hide_index=True)
        else:
            st.info("Kolom 'KODE LAPAK' tidak ditemukan di data penjualan.")

        st.divider()

        section_heading("⚖️ Tonnase Terjual vs Dibuang per Lapak")
        if has_kg and "KODE LAPAK" in df_penjualan.columns and "Is_Dibuang" in df_penjualan.columns:
            df_ton_work = df_penjualan.copy()
            df_ton_work[KG_COL] = to_number(df_ton_work[KG_COL])
            ton_grp = df_ton_work.groupby(["KODE LAPAK", "Is_Dibuang"])[KG_COL].sum().reset_index()
            pivot_ton = ton_grp.pivot(index="KODE LAPAK", columns="Is_Dibuang", values=KG_COL).fillna(0).reset_index()
            pivot_ton = pivot_ton.rename(columns={False: "Terjual (KG)", True: "Dibuang (KG)"})
            for cn in ["Terjual (KG)", "Dibuang (KG)"]:
                if cn not in pivot_ton.columns:
                    pivot_ton[cn] = 0

            pivot_ton["Total_KG"] = pivot_ton["Terjual (KG)"] + pivot_ton["Dibuang (KG)"]
            pivot_ton["%_Dibuang"] = pivot_ton.apply(
                lambda r: (r["Dibuang (KG)"] / r["Total_KG"] * 100) if r["Total_KG"] > 0 else 0, axis=1
            )

            fig_ton = go.Figure()
            fig_ton.add_trace(go.Bar(
                name="Terjual (KG)", x=pivot_ton["KODE LAPAK"], y=pivot_ton["Terjual (KG)"],
                marker_color="#2ca02c",
                text=[f"{v:,.1f} kg" for v in pivot_ton["Terjual (KG)"]],
                textposition="outside", textfont=dict(size=12)
            ))
            fig_ton.add_trace(go.Bar(
                name="Dibuang (KG)", x=pivot_ton["KODE LAPAK"], y=pivot_ton["Dibuang (KG)"],
                marker_color="#ff7f0e",
                text=[f"{v:,.1f} kg" for v in pivot_ton["Dibuang (KG)"]],
                textposition="outside", textfont=dict(size=12)
            ))
            fig_ton.update_layout(
                barmode="group", title="Perbandingan Tonnase Terjual vs Dibuang",
                yaxis_title="Kilogram (KG)", xaxis_title="Kode Lapak"
            )
            pad_yaxis(fig_ton, max(pivot_ton["Terjual (KG)"].max(), pivot_ton["Dibuang (KG)"].max()) if not pivot_ton.empty else 0)
            st.plotly_chart(fig_ton, use_container_width=True)

            fig_pct_buang = go.Figure()
            fig_pct_buang.add_trace(go.Bar(
                x=pivot_ton["KODE LAPAK"],
                y=pivot_ton["%_Dibuang"],
                marker_color=[
                    "#d62728" if v >= 20 else "#ff7f0e" if v >= 10 else "#2ca02c"
                    for v in pivot_ton["%_Dibuang"]
                ],
                text=[f"{v:.1f}%" for v in pivot_ton["%_Dibuang"]],
                textposition="outside",
                textfont=dict(size=13)
            ))
            fig_pct_buang.update_layout(
                title="% Dibuang per Lapak (Dibuang / Total KG × 100%)",
                xaxis_title="Kode Lapak", yaxis_title="% Dibuang",
                showlegend=False, height=360
            )
            pad_yaxis(fig_pct_buang, pivot_ton["%_Dibuang"].max() if not pivot_ton.empty else 0)
            st.plotly_chart(fig_pct_buang, use_container_width=True)

            tabel_ton = pivot_ton.copy()
            tabel_ton["Terjual (KG)"]  = pivot_ton["Terjual (KG)"].apply(lambda x: f"{x:,.1f} KG")
            tabel_ton["Dibuang (KG)"]  = pivot_ton["Dibuang (KG)"].apply(lambda x: f"{x:,.1f} KG")
            tabel_ton["Total KG"]       = pivot_ton["Total_KG"].apply(lambda x: f"{x:,.1f} KG")
            tabel_ton["% Dibuang"]      = pivot_ton["%_Dibuang"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(
                tabel_ton[["KODE LAPAK", "Terjual (KG)", "Dibuang (KG)", "Total KG", "% Dibuang"]],
                use_container_width=True, hide_index=True
            )
        else:
            st.info("Kolom berat (KG/Jumlah) tidak ditemukan.")

        st.divider()

        section_heading("💳 Pembayaran Tunai vs Kredit per Lapak")
        if all(c in df_penjualan.columns for c in ["KODE LAPAK", "Tunai", "Kredit"]):
            pay_grp = df_penjualan.groupby("KODE LAPAK").agg(
                Tunai=("Tunai", "sum"), Kredit=("Kredit", "sum")
            ).reset_index().sort_values("Tunai", ascending=False)

            fig_pay = go.Figure()
            fig_pay.add_trace(go.Bar(
                name="Tunai", x=pay_grp["KODE LAPAK"], y=pay_grp["Tunai"],
                marker_color="#2ca02c",
                text=[rp_short(v) for v in pay_grp["Tunai"]],
                textposition="outside", textfont=dict(size=12)
            ))
            fig_pay.add_trace(go.Bar(
                name="Kredit/Piutang", x=pay_grp["KODE LAPAK"], y=pay_grp["Kredit"],
                marker_color="#ff7f0e",
                text=[rp_short(v) for v in pay_grp["Kredit"]],
                textposition="outside", textfont=dict(size=12)
            ))
            fig_pay.update_layout(
                barmode="group", title="Komposisi Pembayaran: Tunai vs Kredit per Lapak",
                yaxis_title="Rupiah", xaxis_title="Kode Lapak"
            )
            pad_yaxis(fig_pay, max(pay_grp["Tunai"].max(), pay_grp["Kredit"].max()) if not pay_grp.empty else 0)
            st.plotly_chart(fig_pay, use_container_width=True)
        else:
            st.info("Kolom 'Tunai' atau 'Kredit' tidak tersedia.")

        st.divider()

        section_heading("💸 Pengeluaran per Lapak (Overhead vs HPP)")

        has_lokasi   = not df_pengeluaran.empty and "LOKASI LAPAK" in df_pengeluaran.columns
        has_nominal  = not df_pengeluaran.empty and "NOMINAL" in df_pengeluaran.columns
        has_kategori = not df_pengeluaran.empty and "JENIS PENGELUARAN" in df_pengeluaran.columns

        if has_lokasi and has_nominal:
            df_peng_plot = df_pengeluaran.copy()
            total_peng_semua = df_peng_plot["NOMINAL"].sum()
            st.markdown(
                f'<div class="big-total">💰 Total Pengeluaran Semua Lapak: {rp(total_peng_semua)}</div>',
                unsafe_allow_html=True
            )

            if has_kategori:
                def _normalize_kategori(x):
                    if not is_filled(x):
                        return "Tidak Berkategori"
                    su = str(x).strip().upper()
                    if "OVERHEAD" in su:
                        return "Overhead"
                    if "HPP" in su:
                        return "HPP"
                    return str(x).strip()

                df_peng_plot["JENIS PENGELUARAN"] = df_peng_plot["JENIS PENGELUARAN"].apply(_normalize_kategori)

                per_lok_kat = df_peng_plot.groupby(["LOKASI LAPAK", "JENIS PENGELUARAN"])["NOMINAL"].sum().reset_index()
                total_per_lokasi = per_lok_kat.groupby("LOKASI LAPAK")["NOMINAL"].sum().sort_values(ascending=False)
                urutan_lokasi = total_per_lokasi.index.tolist()

                def _kategori_key(k):
                    if k == "Overhead": return 0
                    if k == "HPP": return 1
                    if k == "Tidak Berkategori": return 99
                    return 2
                kategori_urut = sorted(per_lok_kat["JENIS PENGELUARAN"].unique().tolist(), key=_kategori_key)

                palet_lain = ["#6c5ce7", "#00838f", "#8e44ad", "#2d6a4f", "#c2185b"]
                idx_lain = 0
                warna_kategori = {}
                for k in kategori_urut:
                    if k == "Overhead":
                        warna_kategori[k] = "#ff7f0e"
                    elif k == "HPP":
                        warna_kategori[k] = "#d62728"
                    elif k == "Tidak Berkategori":
                        warna_kategori[k] = "#95a5a6"
                    else:
                        warna_kategori[k] = palet_lain[idx_lain % len(palet_lain)]
                        idx_lain += 1

                fig_peng = go.Figure()
                for kat in kategori_urut:
                    df_k = (
                        per_lok_kat[per_lok_kat["JENIS PENGELUARAN"] == kat]
                        .set_index("LOKASI LAPAK").reindex(urutan_lokasi)["NOMINAL"]
                        .fillna(0).reset_index()
                    )
                    fig_peng.add_trace(go.Bar(
                        name=kat,
                        x=df_k["LOKASI LAPAK"], y=df_k["NOMINAL"],
                        text=[rp_short(v) if v > 0 else "" for v in df_k["NOMINAL"]],
                        textposition="inside", insidetextanchor="middle",
                        textfont=dict(size=12, color="white"),
                        marker_color=warna_kategori[kat]
                    ))

                max_total = total_per_lokasi.max() if not total_per_lokasi.empty else 0
                fig_peng.update_layout(
                    barmode="stack",
                    title="Pengeluaran per Lapak: Overhead vs HPP",
                    xaxis_title="Lokasi Lapak", yaxis_title="Rupiah",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    yaxis_tickformat=","
                )
                for lok in urutan_lokasi:
                    fig_peng.add_annotation(
                        x=lok, y=total_per_lokasi.loc[lok],
                        text=f"<b>{rp_short(total_per_lokasi.loc[lok])}</b>",
                        showarrow=False, yshift=18,
                        font=dict(size=13, color="#1f3864")
                    )
                pad_yaxis(fig_peng, max_total, pad=0.3)
                st.plotly_chart(fig_peng, use_container_width=True)

                tabel_peng = per_lok_kat.pivot(index="LOKASI LAPAK", columns="JENIS PENGELUARAN", values="NOMINAL").fillna(0)
                tabel_peng = tabel_peng.reindex(urutan_lokasi)
                tabel_peng["Total"] = tabel_peng.sum(axis=1)
                st.dataframe(tabel_peng.apply(lambda col: col.map(rp)), use_container_width=True)
            else:
                st.info(
                    "Kolom kategori pengeluaran belum terdeteksi di sheet PENGELUARAN LAPAK. "
                    "Tambahkan kolom berjudul **'JENIS PENGELUARAN'** (isi tiap baris dengan "
                    "'Overhead' atau 'HPP') agar batang bisa dipecah otomatis dengan warna berbeda. "
                    "Sementara ini ditampilkan total per lapak saja (belum dipecah kategori):"
                )
                per_lokasi_saja = df_peng_plot.groupby("LOKASI LAPAK")["NOMINAL"].sum().sort_values(ascending=False).reset_index()
                fig_peng_simple = go.Figure()
                fig_peng_simple.add_trace(go.Bar(
                    x=per_lokasi_saja["LOKASI LAPAK"], y=per_lokasi_saja["NOMINAL"],
                    text=[rp_short(v) for v in per_lokasi_saja["NOMINAL"]],
                    textposition="outside", textfont=dict(size=12),
                    marker_color="#1f77b4"
                ))
                fig_peng_simple.update_layout(
                    title="Total Pengeluaran per Lapak",
                    xaxis_title="Lokasi Lapak", yaxis_title="Rupiah", showlegend=False
                )
                pad_yaxis(fig_peng_simple, per_lokasi_saja["NOMINAL"].max() if not per_lokasi_saja.empty else 0)
                st.plotly_chart(fig_peng_simple, use_container_width=True)

            st.divider()
            st.markdown("#### 📋 Rincian Pengeluaran per Lokasi")
            lokasi_semua_rincian = (
                df_peng_plot.groupby("LOKASI LAPAK")["NOMINAL"].sum()
                .sort_values(ascending=False).index.tolist()
            )
            sel_lokasi_rincian = st.multiselect(
                "Filter Lokasi Lapak", lokasi_semua_rincian,
                default=lokasi_semua_rincian, key="tab2_peng_rincian_lokasi"
            )
            if sel_lokasi_rincian:
                df_rincian = df_peng_plot[df_peng_plot["LOKASI LAPAK"].isin(sel_lokasi_rincian)].copy()

                kolom_tampil = []
                if "Tanggal_Lengkap" in df_rincian.columns and df_rincian["Tanggal_Lengkap"].notna().any():
                    df_rincian["Tanggal"] = df_rincian["Tanggal_Lengkap"].dt.strftime("%d/%m/%Y")
                    kolom_tampil.append("Tanggal")
                kolom_tampil.append("LOKASI LAPAK")
                if "JENIS PENGELUARAN" in df_rincian.columns:
                    kolom_tampil.append("JENIS PENGELUARAN")
                kolom_tampil.append("NOMINAL")

                kolom_raw_tgl = [c for c in df_rincian.columns if c.strip().upper() in ["TANGGAL", "TGL", "DATE"]]
                kolom_exclude = set(kolom_tampil) | {"Tanggal_Lengkap"} | set(kolom_raw_tgl)
                kolom_tampil += [c for c in df_rincian.columns if c not in kolom_exclude]

                sort_cols = ["LOKASI LAPAK"] + (["Tanggal_Lengkap"] if "Tanggal_Lengkap" in df_rincian.columns else [])
                sort_asc  = [True] + ([False] if "Tanggal_Lengkap" in df_rincian.columns else [])
                df_rincian = df_rincian.sort_values(sort_cols, ascending=sort_asc, na_position="last")

                df_rincian_tampil = df_rincian[kolom_tampil].copy()
                df_rincian_tampil["NOMINAL"] = df_rincian_tampil["NOMINAL"].map(rp)
                st.dataframe(df_rincian_tampil, use_container_width=True, hide_index=True)
                st.caption(f"📌 {len(df_rincian_tampil)} baris pengeluaran · Total: {rp(df_rincian['NOMINAL'].sum())}")
            else:
                st.info("Pilih minimal satu lokasi untuk menampilkan rincian pengeluaran.")
        else:
            missing = []
            if not has_lokasi:  missing.append("'LOKASI LAPAK'")
            if not has_nominal: missing.append("'NOMINAL'")
            st.warning(f"Kolom {' dan '.join(missing)} tidak ditemukan di sheet PENGELUARAN LAPAK.")

        st.divider()

        section_heading("📊 Tonnase Terjual Berdasarkan Grade")
        if has_grade and has_kg:
            df_grade_work = df_penjualan.copy()
            df_grade_work[KG_COL] = to_number(df_grade_work[KG_COL])
            df_grade_valid = df_grade_work[
                df_grade_work[KG_COL].notna() & df_grade_work[GRADE_COL].apply(is_filled)
            ]

            agg_grade = {"Total_KG": (KG_COL, "sum")}
            if "Total harga" in df_grade_valid.columns:
                agg_grade["Omzet"] = ("Total harga", "sum")
            if "Keuntungan" in df_grade_valid.columns:
                agg_grade["Laba"] = ("Keuntungan", "sum")
            grade_grp = df_grade_valid.groupby(GRADE_COL).agg(**agg_grade).reset_index().sort_values("Total_KG", ascending=False)

            g1, g2 = st.columns(2)
            with g1:
                fig_grade_bar = go.Figure()
                fig_grade_bar.add_trace(go.Bar(
                    x=grade_grp[GRADE_COL],
                    y=grade_grp["Total_KG"],
                    text=[f"{v:,.1f} KG" for v in grade_grp["Total_KG"]],
                    textposition="outside",
                    textfont=dict(size=12),
                    marker_color="#1f77b4"
                ))
                fig_grade_bar.update_layout(
                    title="Total KG per Grade",
                    xaxis_title="Grade", yaxis_title="Kilogram (KG)", showlegend=False
                )
                pad_yaxis(fig_grade_bar, grade_grp["Total_KG"].max() if not grade_grp.empty else 0)
                st.plotly_chart(fig_grade_bar, use_container_width=True)

            with g2:
                fig_grade_pie = px.pie(
                    grade_grp, names=GRADE_COL, values="Total_KG",
                    title="Proporsi KG per Grade", hole=0.4
                )
                fig_grade_pie.update_traces(textinfo="label+percent", textfont_size=13)
                st.plotly_chart(fig_grade_pie, use_container_width=True)

            grade_display = grade_grp.copy()
            grade_display["Total_KG"] = grade_display["Total_KG"].apply(lambda x: f"{x:,.1f} KG")
            disp_cols = [GRADE_COL, "Total_KG"]
            if "Omzet" in grade_display.columns:
                grade_display["Omzet"] = grade_grp["Omzet"].apply(rp)
                disp_cols.append("Omzet")
            if "Laba" in grade_display.columns:
                grade_display["Laba"] = grade_grp["Laba"].apply(rp)
                disp_cols.append("Laba")
            st.dataframe(grade_display[disp_cols], use_container_width=True, hide_index=True)
        else:
            st.info("Kolom GRADE atau Jumlah (KG) tidak ditemukan di data penjualan lapak.")

        st.divider()

        section_heading("📊 Tonnase Terjual Berdasarkan Jenis")
        if has_jenis and has_kg:
            df_jenis_work = df_penjualan.copy()
            df_jenis_work[KG_COL] = to_number(df_jenis_work[KG_COL])
            df_jenis_valid = df_jenis_work[
                df_jenis_work[KG_COL].notna() & df_jenis_work[JENIS_COL].apply(is_filled)
            ]

            agg_jenis = {"Total_KG": (KG_COL, "sum")}
            if "Total harga" in df_jenis_valid.columns:
                agg_jenis["Omzet"] = ("Total harga", "sum")
            if "Keuntungan" in df_jenis_valid.columns:
                agg_jenis["Laba"] = ("Keuntungan", "sum")
            jenis_grp = df_jenis_valid.groupby(JENIS_COL).agg(**agg_jenis).reset_index().sort_values("Total_KG", ascending=False)

            j1, j2 = st.columns(2)
            with j1:
                fig_jenis_bar = go.Figure()
                fig_jenis_bar.add_trace(go.Bar(
                    x=jenis_grp[JENIS_COL],
                    y=jenis_grp["Total_KG"],
                    text=[f"{v:,.1f} KG" for v in jenis_grp["Total_KG"]],
                    textposition="outside",
                    textfont=dict(size=12),
                    marker_color="#2ca02c"
                ))
                fig_jenis_bar.update_layout(
                    title="Total KG per Jenis",
                    xaxis_title="Jenis", yaxis_title="Kilogram (KG)", showlegend=False
                )
                pad_yaxis(fig_jenis_bar, jenis_grp["Total_KG"].max() if not jenis_grp.empty else 0)
                st.plotly_chart(fig_jenis_bar, use_container_width=True)

            with j2:
                fig_jenis_pie = px.pie(
                    jenis_grp, names=JENIS_COL, values="Total_KG",
                    title="Proporsi KG per Jenis", hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_jenis_pie.update_traces(textinfo="label+percent", textfont_size=13)
                st.plotly_chart(fig_jenis_pie, use_container_width=True)

            jenis_display = jenis_grp.copy()
            jenis_display["Total_KG"] = jenis_display["Total_KG"].apply(lambda x: f"{x:,.1f} KG")
            disp_cols2 = [JENIS_COL, "Total_KG"]
            if "Omzet" in jenis_display.columns:
                jenis_display["Omzet"] = jenis_grp["Omzet"].apply(rp)
                disp_cols2.append("Omzet")
            if "Laba" in jenis_display.columns:
                jenis_display["Laba"] = jenis_grp["Laba"].apply(rp)
                disp_cols2.append("Laba")
            st.dataframe(jenis_display[disp_cols2], use_container_width=True, hide_index=True)
        else:
            st.info("Kolom JENIS atau Jumlah (KG) tidak ditemukan di data penjualan lapak.")

        st.divider()

        section_heading("🔀 Analisa Gabungan Grade × Jenis")
        if has_grade and has_jenis and has_kg:
            df_gab = df_penjualan.copy()
            df_gab[KG_COL] = to_number(df_gab[KG_COL])
            df_gabungan = df_gab[
                df_gab[KG_COL].notna() &
                df_gab[GRADE_COL].apply(is_filled) &
                df_gab[JENIS_COL].apply(is_filled)
            ].copy()

            if not df_gabungan.empty:
                gabungan_grp = (
                    df_gabungan.groupby([JENIS_COL, GRADE_COL])[KG_COL]
                    .sum().reset_index()
                )
                gabungan_grp.columns = ["Jenis", "Grade", "Total_KG"]

                fig_gabungan = px.bar(
                    gabungan_grp, x="Jenis", y="Total_KG", color="Grade",
                    barmode="group",
                    title="Total KG per Jenis × Grade",
                    text=gabungan_grp["Total_KG"].apply(lambda v: f"{v:,.1f}"),
                    labels={"Total_KG": "Kilogram (KG)", "Jenis": "Jenis", "Grade": "Grade"}
                )
                fig_gabungan.update_traces(textposition="outside", textfont_size=10)
                pad_yaxis(fig_gabungan, gabungan_grp["Total_KG"].max() if not gabungan_grp.empty else 0)
                fig_gabungan.update_layout(
                    xaxis_tickangle=-30,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    height=500
                )
                st.plotly_chart(fig_gabungan, use_container_width=True)

                pivot_gab = gabungan_grp.pivot(index="Jenis", columns="Grade", values="Total_KG").fillna(0)
                pivot_display = pivot_gab.copy()
                for c in pivot_display.columns:
                    pivot_display[c] = pivot_gab[c].apply(lambda x: f"{x:,.1f} KG")
                st.markdown("**Tabel Pivot KG: Jenis (baris) × Grade (kolom)**")
                st.dataframe(pivot_display, use_container_width=True)
            else:
                st.info("Tidak ada data dengan Grade dan Jenis yang terisi.")
        else:
            st.info("Kolom GRADE dan/atau JENIS tidak tersedia untuk analisa gabungan.")

# TAB 2B: ANALISA LAPAK LUAR
with tab2b:
    if df_penjualan_luar.empty:
        st.info("Data penjualan lapak luar kosong untuk periode yang dipilih.")
    else:
        KG_COL_LUAR    = "Jumlah (KG)"
        GRADE_COL_LUAR = "GRADE"
        JENIS_COL_LUAR = "JENIS"
        NAMA_COL_LUAR  = "NAMA PELANGGAN"

        has_kg_luar          = KG_COL_LUAR    in df_penjualan_luar.columns
        has_grade_luar       = GRADE_COL_LUAR in df_penjualan_luar.columns
        has_jenis_luar       = JENIS_COL_LUAR in df_penjualan_luar.columns
        has_nama_luar        = NAMA_COL_LUAR  in df_penjualan_luar.columns
        has_tunai_luar       = "Tunai"       in df_penjualan_luar.columns
        has_kredit_luar      = "Kredit"      in df_penjualan_luar.columns
        has_omzet_laba_luar  = "Total harga" in df_penjualan_luar.columns and "Keuntungan" in df_penjualan_luar.columns

        section_heading("📋 Rincian Data Penjualan Lapak Luar")
        kolom_rincian_ada = [
            src for src in ["TANGGAL", "TANGGAL NOTA BALIK", "INVOICE", NAMA_COL_LUAR, GRADE_COL_LUAR, KG_COL_LUAR]
            if src in df_penjualan_luar.columns
        ]
        if not kolom_rincian_ada and not has_omzet_laba_luar:
            st.info("Kolom rincian (Tanggal/Invoice/Nama Pelanggan/Grade/Tonnase/Omzet/Laba) tidak ditemukan di sheet PENJUALAN LAPAK LUAR.")
        else:
            df_rincian_luar = df_penjualan_luar.copy()

            if has_nama_luar:
                nama_opts_rincian = sorted(df_rincian_luar[NAMA_COL_LUAR].dropna().astype(str).unique())
                sel_nama_rincian = st.multiselect(
                    "🔎 Filter Nama Pelanggan", nama_opts_rincian, default=nama_opts_rincian,
                    key="tab2b_rincian_nama_filter"
                )
                df_rincian_luar = (
                    df_rincian_luar[df_rincian_luar[NAMA_COL_LUAR].astype(str).isin(sel_nama_rincian)]
                    if sel_nama_rincian else df_rincian_luar.iloc[0:0]
                )

            total_omzet_rincian = df_rincian_luar["Total harga"].sum() if "Total harga" in df_rincian_luar.columns else 0
            total_laba_rincian  = df_rincian_luar["Keuntungan"].sum()  if "Keuntungan"  in df_rincian_luar.columns else 0
            rf1, rf2 = st.columns(2)
            rf1.metric("💰 Total Omzet (sesuai filter nama)", rp(total_omzet_rincian))
            rf2.metric("📈 Total Laba (sesuai filter nama)", rp(total_laba_rincian))

            # Total per invoice dihitung setelah filter nama diterapkan, supaya tetap
            # akurat ke baris yang sedang tampil kalau tabelnya geser karena difilter.
            # min_count=1 supaya invoice yang Omzet/Laba-nya kosong semua tetap kosong
            # (bukan malah kebaca 0).
            if "INVOICE" in df_rincian_luar.columns and "Total harga" in df_rincian_luar.columns:
                df_rincian_luar["Total Omzet per Invoice"] = df_rincian_luar.groupby("INVOICE")["Total harga"].transform(lambda s: s.sum(min_count=1))
            if "INVOICE" in df_rincian_luar.columns and "Keuntungan" in df_rincian_luar.columns:
                df_rincian_luar["Total Laba per Invoice"] = df_rincian_luar.groupby("INVOICE")["Keuntungan"].transform(lambda s: s.sum(min_count=1))
            if "INVOICE" in df_rincian_luar.columns and "_RAW_LUAR_HARGA_BELI" in df_rincian_luar.columns:
                df_rincian_luar["Total Harga Beli"] = df_rincian_luar.groupby("INVOICE")["_RAW_LUAR_HARGA_BELI"].transform(lambda s: s.sum(min_count=1))
            if "INVOICE" in df_rincian_luar.columns and "_RAW_LUAR_MODAL" in df_rincian_luar.columns:
                df_rincian_luar["Total Modal"] = df_rincian_luar.groupby("INVOICE")["_RAW_LUAR_MODAL"].transform(lambda s: s.sum(min_count=1))
            if "INVOICE" in df_rincian_luar.columns and "_RAW_LUAR_NOTA_BALIK" in df_rincian_luar.columns:
                df_rincian_luar["Total Nota Balik"] = df_rincian_luar.groupby("INVOICE")["_RAW_LUAR_NOTA_BALIK"].transform(lambda s: s.sum(min_count=1))

            if "INVOICE" in df_rincian_luar.columns:
                def _invoice_sort_key(s):
                    if pd.isna(s):
                        return (-1, "")
                    s = str(s)
                    digits = "".join(ch for ch in s if ch.isdigit())
                    return (int(digits) if digits else -1, s)
                df_rincian_luar["_urutan_invoice"] = df_rincian_luar["INVOICE"].apply(_invoice_sort_key)
                df_rincian_luar = df_rincian_luar.sort_values(
                    "_urutan_invoice", ascending=False, kind="mergesort"
                ).drop(columns=["_urutan_invoice"])

                # Total per Invoice cuma ditampilkan sekali di baris pertama tiap
                # kelompok invoice (gaya sel gabungan seperti contoh gambar), baris
                # grade lain di invoice yang sama dikosongkan.
                is_baris_pertama_invoice = df_rincian_luar["INVOICE"] != df_rincian_luar["INVOICE"].shift(1)
                for _kolom_total_merge in ["Total Omzet per Invoice", "Total Laba per Invoice", "Total Harga Beli", "Total Modal", "Total Nota Balik"]:
                    if _kolom_total_merge in df_rincian_luar.columns:
                        df_rincian_luar.loc[~is_baris_pertama_invoice, _kolom_total_merge] = np.nan

            def _fmt_blank(series, formatter):
                return series.apply(lambda x: formatter(x) if pd.notna(x) else "")

            def _esc(x):
                s = "" if x is None else str(x)
                return (s.replace("&", "&amp;").replace("<", "&lt;")
                         .replace(">", "&gt;").replace('"', "&quot;"))

            n_rows_rl = len(df_rincian_luar)
            if "INVOICE" in df_rincian_luar.columns and n_rows_rl > 0:
                grp_size_rl = df_rincian_luar.groupby("INVOICE")["INVOICE"].transform("size").tolist()
                is_first_rl = (df_rincian_luar["INVOICE"] != df_rincian_luar["INVOICE"].shift(1)).tolist()
                is_last_rl = (df_rincian_luar["INVOICE"] != df_rincian_luar["INVOICE"].shift(-1)).tolist()
            else:
                grp_size_rl = [1] * n_rows_rl
                is_first_rl = [True] * n_rows_rl
                is_last_rl = [True] * n_rows_rl

            # kolom_spec: tiap kolom "text" (tampil apa adanya di setiap baris) atau
            # "merge" (digabung/rowspan mengikuti kelompok Invoice -- rata tengah,
            # tebal, tanpa garis dalam, sama seperti Total Omzet/Laba per Invoice).
            kolom_spec = []
            if "TANGGAL" in df_rincian_luar.columns:
                if "Tanggal_Lengkap" in df_rincian_luar.columns:
                    _tgl_fmt = df_rincian_luar["Tanggal_Lengkap"].dt.strftime("%d/%m/%Y")
                    _vals = _tgl_fmt.where(_tgl_fmt.notna(), df_rincian_luar["TANGGAL"]).fillna("")
                else:
                    _vals = df_rincian_luar["TANGGAL"].fillna("")
                kolom_spec.append({"label": "Tanggal", "kind": "text", "css": "", "vals": _vals.tolist()})
            if "TANGGAL NOTA BALIK" in df_rincian_luar.columns:
                if "Tanggal_Nota_Balik" in df_rincian_luar.columns:
                    _tnb_fmt = df_rincian_luar["Tanggal_Nota_Balik"].dt.strftime("%d/%m/%Y")
                    _vals = _tnb_fmt.where(_tnb_fmt.notna(), df_rincian_luar["TANGGAL NOTA BALIK"]).fillna("")
                else:
                    _vals = df_rincian_luar["TANGGAL NOTA BALIK"].fillna("")
                kolom_spec.append({"label": "Tanggal Nota Balik", "kind": "text", "css": "", "vals": _vals.tolist()})
            if "INVOICE" in df_rincian_luar.columns:
                kolom_spec.append({
                    "label": "Invoice", "kind": "merge", "css": "",
                    "vals": df_rincian_luar["INVOICE"].fillna("").tolist(), "fmt": lambda x: x,
                })
            if NAMA_COL_LUAR in df_rincian_luar.columns:
                kolom_spec.append({"label": "Nama Pelanggan", "kind": "text", "css": "", "vals": df_rincian_luar[NAMA_COL_LUAR].fillna("").tolist()})
            if GRADE_COL_LUAR in df_rincian_luar.columns:
                kolom_spec.append({"label": "Grade", "kind": "text", "css": "", "vals": df_rincian_luar[GRADE_COL_LUAR].fillna("").tolist()})
            if "_RAW_LUAR_TONASE_LAHAN_KG" in df_rincian_luar.columns:
                kolom_spec.append({"label": "Tonase Lahan (KG)", "kind": "text", "css": "rl-num", "vals": _fmt_blank(df_rincian_luar["_RAW_LUAR_TONASE_LAHAN_KG"], lambda x: f"{x:,.1f} KG").tolist()})
            if KG_COL_LUAR in df_rincian_luar.columns:
                kolom_spec.append({"label": "Tonnase Nota Balik", "kind": "text", "css": "rl-num", "vals": _fmt_blank(df_rincian_luar[KG_COL_LUAR], lambda x: f"{x:,.1f} KG").tolist()})
            if "Total Harga Beli" in df_rincian_luar.columns:
                kolom_spec.append({
                    "label": "Total Harga Beli", "kind": "merge", "css": "",
                    "vals": df_rincian_luar["Total Harga Beli"].tolist(),
                    "fmt": lambda x: (rp(x) if pd.notna(x) else ""),
                })
            if "Total Modal" in df_rincian_luar.columns:
                kolom_spec.append({
                    "label": "Total Modal", "kind": "merge", "css": "",
                    "vals": df_rincian_luar["Total Modal"].tolist(),
                    "fmt": lambda x: (rp(x) if pd.notna(x) else ""),
                })
            if "Total Nota Balik" in df_rincian_luar.columns:
                kolom_spec.append({
                    "label": "Total Nota Balik", "kind": "merge", "css": "",
                    "vals": df_rincian_luar["Total Nota Balik"].tolist(),
                    "fmt": lambda x: (rp(x) if pd.notna(x) else ""),
                })
            if "Total Omzet per Invoice" in df_rincian_luar.columns:
                kolom_spec.append({
                    "label": "Total Omzet per Invoice", "kind": "merge", "css": "",
                    "vals": df_rincian_luar["Total Omzet per Invoice"].tolist(),
                    "fmt": lambda x: (rp(x) if pd.notna(x) else ""),
                })
            if "Total Laba per Invoice" in df_rincian_luar.columns:
                kolom_spec.append({
                    "label": "Total Laba per Invoice", "kind": "merge", "css": "",
                    "vals": df_rincian_luar["Total Laba per Invoice"].tolist(),
                    "fmt": lambda x: (rp(x) if pd.notna(x) else ""),
                })

            header_html = "".join(f"<th>{_esc(k['label'])}</th>" for k in kolom_spec)

            body_rows_html = []
            for i in range(n_rows_rl):
                cells = []
                for k in kolom_spec:
                    if k["kind"] == "text":
                        cls = (k["css"] + " rl-group-end").strip() if is_last_rl[i] else k["css"]
                        cells.append(f'<td class="{cls}">{_esc(k["vals"][i])}</td>' if cls else f"<td>{_esc(k['vals'][i])}</td>")
                    elif is_first_rl[i]:
                        raw = k["vals"][i]
                        disp = k["fmt"](raw) if k.get("fmt") else raw
                        cells.append(f'<td class="rl-merge rl-group-end" rowspan="{grp_size_rl[i]}">{_esc(disp)}</td>')
                    # baris bukan-pertama dalam kelompok: sel merge dilewati (sudah tercakup rowspan)
                body_rows_html.append(f"<tr>{''.join(cells)}</tr>")

            rincian_html = f"""<style>
.rl-wrap {{ max-height: 620px; overflow: auto; border: 1px solid #e0e6f0; border-radius: 8px; }}
.rl-table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
.rl-table th {{ position: sticky; top: 0; background: #1f3864; color: #fff; padding: 9px 10px; text-align: left; white-space: nowrap; z-index: 1; }}
.rl-table td {{ padding: 8px 10px; border-bottom: 1px solid #eef1f6; white-space: nowrap; }}
.rl-table tbody tr:hover td {{ background: #f8f9fa; }}
.rl-table td.rl-num {{ text-align: right; }}
.rl-table td.rl-merge {{ text-align: center; vertical-align: middle; font-weight: 800; color: #1f3864; border-left: 1px solid #e0e6f0; background: #f4f7fc; }}
.rl-table td.rl-group-end {{ border-bottom: 2px solid #1f3864; }}
</style>
<div class="rl-wrap"><table class="rl-table">
<thead><tr>{header_html}</tr></thead>
<tbody>{"".join(body_rows_html)}</tbody>
</table></div>"""
            st.markdown(rincian_html, unsafe_allow_html=True)

        st.divider()

        if has_kg_luar:
            df_ton_all_luar = df_penjualan_luar.copy()
            df_ton_all_luar[KG_COL_LUAR] = to_number(df_ton_all_luar[KG_COL_LUAR])
            total_kg_terjual_luar = df_ton_all_luar[KG_COL_LUAR].sum()

            st.metric("⚖️ Total Tonnase Terjual", f"{total_kg_terjual_luar:,.1f} KG")
            st.divider()

        section_heading("📊 Omzet & Profit per Pelanggan")
        if has_nama_luar and has_omzet_laba_luar:
            per_pelanggan = df_penjualan_luar.groupby(NAMA_COL_LUAR).agg(
                Omzet=("Total harga", "sum"),
                Profit=("Keuntungan", "sum")
            ).reset_index().sort_values("Omzet", ascending=False)

            per_pelanggan["Margin_%"] = per_pelanggan.apply(
                lambda r: (r["Profit"] / r["Omzet"] * 100) if r["Omzet"] > 0 else 0, axis=1
            )

            fig_bar_pelanggan = go.Figure()
            fig_bar_pelanggan.add_trace(go.Bar(
                name="Omzet",
                x=per_pelanggan[NAMA_COL_LUAR], y=per_pelanggan["Omzet"],
                marker_color="#1f77b4",
                text=[rp_short(v) for v in per_pelanggan["Omzet"]],
                textposition="outside", textfont=dict(size=13, color="#1f3864")
            ))
            fig_bar_pelanggan.add_trace(go.Bar(
                name="Profit",
                x=per_pelanggan[NAMA_COL_LUAR], y=per_pelanggan["Profit"],
                marker_color="#d62728",
                text=[rp_short(v) for v in per_pelanggan["Profit"]],
                textposition="outside", textfont=dict(size=13, color="#d62728")
            ))
            fig_bar_pelanggan.update_layout(
                barmode="group", title="Omzet & Profit per Nama Pelanggan",
                yaxis_title="Rupiah", xaxis_title="Nama Pelanggan",
                legend=dict(orientation="h", yanchor="bottom", y=1.02), height=480,
                xaxis_tickangle=-30
            )
            pad_yaxis(fig_bar_pelanggan, max(per_pelanggan["Omzet"].max(), per_pelanggan["Profit"].max()) if not per_pelanggan.empty else 0)
            st.plotly_chart(fig_bar_pelanggan, use_container_width=True)

            fig_margin_luar = go.Figure()
            fig_margin_luar.add_trace(go.Bar(
                x=per_pelanggan[NAMA_COL_LUAR],
                y=per_pelanggan["Margin_%"],
                marker_color=[
                    "#2ca02c" if v >= 20 else "#ff7f0e" if v >= 10 else "#d62728"
                    for v in per_pelanggan["Margin_%"]
                ],
                text=[f"{v:.1f}%" for v in per_pelanggan["Margin_%"]],
                textposition="outside",
                textfont=dict(size=13)
            ))
            fig_margin_luar.update_layout(
                title="% Margin Profit per Nama Pelanggan (Profit / Omzet × 100%)",
                xaxis_title="Nama Pelanggan", yaxis_title="Margin (%)",
                showlegend=False, height=380, xaxis_tickangle=-30
            )
            pad_yaxis(fig_margin_luar, per_pelanggan["Margin_%"].max() if not per_pelanggan.empty else 0)
            st.plotly_chart(fig_margin_luar, use_container_width=True)

            tabel_pelanggan = per_pelanggan.copy()
            tabel_pelanggan["Omzet"]    = per_pelanggan["Omzet"].apply(rp)
            tabel_pelanggan["Profit"]   = per_pelanggan["Profit"].apply(rp)
            tabel_pelanggan["Margin_%"] = per_pelanggan["Margin_%"].apply(lambda x: f"{x:.1f}%")
            tabel_pelanggan = tabel_pelanggan.rename(columns={"Margin_%": "% Margin Profit", NAMA_COL_LUAR: "Nama Pelanggan"})
            st.dataframe(tabel_pelanggan[["Nama Pelanggan", "Omzet", "Profit", "% Margin Profit"]], use_container_width=True, hide_index=True)
        else:
            st.info("Kolom 'NAMA PELANGGAN' (kolom D), 'Total harga' (kolom S) dan/atau 'Keuntungan' (kolom T) tidak ditemukan di data penjualan lapak luar.")

        st.divider()

        if has_tunai_luar and has_kredit_luar and has_nama_luar:
            section_heading("💳 Pembayaran Tunai vs Kredit per Pelanggan")
            pay_grp_luar = df_penjualan_luar.groupby(NAMA_COL_LUAR).agg(
                Tunai=("Tunai", "sum"), Kredit=("Kredit", "sum")
            ).reset_index().sort_values("Tunai", ascending=False)

            fig_pay_luar = go.Figure()
            fig_pay_luar.add_trace(go.Bar(
                name="Tunai", x=pay_grp_luar[NAMA_COL_LUAR], y=pay_grp_luar["Tunai"],
                marker_color="#2ca02c",
                text=[rp_short(v) for v in pay_grp_luar["Tunai"]],
                textposition="outside", textfont=dict(size=12)
            ))
            fig_pay_luar.add_trace(go.Bar(
                name="Kredit/Piutang", x=pay_grp_luar[NAMA_COL_LUAR], y=pay_grp_luar["Kredit"],
                marker_color="#ff7f0e",
                text=[rp_short(v) for v in pay_grp_luar["Kredit"]],
                textposition="outside", textfont=dict(size=12)
            ))
            fig_pay_luar.update_layout(
                barmode="group", title="Komposisi Pembayaran: Tunai vs Kredit per Nama Pelanggan",
                yaxis_title="Rupiah", xaxis_title="Nama Pelanggan", xaxis_tickangle=-30
            )
            pad_yaxis(fig_pay_luar, max(pay_grp_luar["Tunai"].max(), pay_grp_luar["Kredit"].max()) if not pay_grp_luar.empty else 0)
            st.plotly_chart(fig_pay_luar, use_container_width=True)
            st.divider()

        if has_jenis_luar and has_kg_luar:
            section_heading("📊 Tonnase Terjual Berdasarkan Jenis")
            df_jenis_work_luar = df_penjualan_luar.copy()
            df_jenis_work_luar[KG_COL_LUAR] = to_number(df_jenis_work_luar[KG_COL_LUAR])
            df_jenis_valid_luar = df_jenis_work_luar[
                df_jenis_work_luar[KG_COL_LUAR].notna() & df_jenis_work_luar[JENIS_COL_LUAR].apply(is_filled)
            ]

            agg_jenis_luar = {"Total_KG": (KG_COL_LUAR, "sum")}
            if "Total harga" in df_jenis_valid_luar.columns:
                agg_jenis_luar["Omzet"] = ("Total harga", "sum")
            if "Keuntungan" in df_jenis_valid_luar.columns:
                agg_jenis_luar["Laba"] = ("Keuntungan", "sum")
            jenis_grp_luar = df_jenis_valid_luar.groupby(JENIS_COL_LUAR).agg(**agg_jenis_luar).reset_index().sort_values("Total_KG", ascending=False)

            jl1, jl2 = st.columns(2)
            with jl1:
                fig_jenis_bar_luar = go.Figure()
                fig_jenis_bar_luar.add_trace(go.Bar(
                    x=jenis_grp_luar[JENIS_COL_LUAR],
                    y=jenis_grp_luar["Total_KG"],
                    text=[f"{v:,.1f} KG" for v in jenis_grp_luar["Total_KG"]],
                    textposition="outside",
                    textfont=dict(size=12),
                    marker_color="#2ca02c"
                ))
                fig_jenis_bar_luar.update_layout(
                    title="Total KG per Jenis",
                    xaxis_title="Jenis", yaxis_title="Kilogram (KG)", showlegend=False
                )
                pad_yaxis(fig_jenis_bar_luar, jenis_grp_luar["Total_KG"].max() if not jenis_grp_luar.empty else 0)
                st.plotly_chart(fig_jenis_bar_luar, use_container_width=True)

            with jl2:
                fig_jenis_pie_luar = px.pie(
                    jenis_grp_luar, names=JENIS_COL_LUAR, values="Total_KG",
                    title="Proporsi KG per Jenis", hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_jenis_pie_luar.update_traces(textinfo="label+percent", textfont_size=13)
                st.plotly_chart(fig_jenis_pie_luar, use_container_width=True)

            jenis_display_luar = jenis_grp_luar.copy()
            jenis_display_luar["Total_KG"] = jenis_display_luar["Total_KG"].apply(lambda x: f"{x:,.1f} KG")
            disp_cols2_luar = [JENIS_COL_LUAR, "Total_KG"]
            if "Omzet" in jenis_display_luar.columns:
                jenis_display_luar["Omzet"] = jenis_grp_luar["Omzet"].apply(rp)
                disp_cols2_luar.append("Omzet")
            if "Laba" in jenis_display_luar.columns:
                jenis_display_luar["Laba"] = jenis_grp_luar["Laba"].apply(rp)
                disp_cols2_luar.append("Laba")
            st.dataframe(jenis_display_luar[disp_cols2_luar], use_container_width=True, hide_index=True)
            st.divider()

        if has_grade_luar and has_jenis_luar and has_kg_luar:
            section_heading("🔀 Analisa Gabungan Grade × Jenis")
            df_gab_luar = df_penjualan_luar.copy()
            df_gab_luar[KG_COL_LUAR] = to_number(df_gab_luar[KG_COL_LUAR])
            df_gabungan_luar = df_gab_luar[
                df_gab_luar[KG_COL_LUAR].notna() &
                df_gab_luar[GRADE_COL_LUAR].apply(is_filled) &
                df_gab_luar[JENIS_COL_LUAR].apply(is_filled)
            ].copy()

            if not df_gabungan_luar.empty:
                gabungan_grp_luar = (
                    df_gabungan_luar.groupby([JENIS_COL_LUAR, GRADE_COL_LUAR])[KG_COL_LUAR]
                    .sum().reset_index()
                )
                gabungan_grp_luar.columns = ["Jenis", "Grade", "Total_KG"]

                fig_gabungan_luar = px.bar(
                    gabungan_grp_luar, x="Jenis", y="Total_KG", color="Grade",
                    barmode="group",
                    title="Total KG per Jenis × Grade",
                    text=gabungan_grp_luar["Total_KG"].apply(lambda v: f"{v:,.1f}"),
                    labels={"Total_KG": "Kilogram (KG)", "Jenis": "Jenis", "Grade": "Grade"}
                )
                fig_gabungan_luar.update_traces(textposition="outside", textfont_size=10)
                pad_yaxis(fig_gabungan_luar, gabungan_grp_luar["Total_KG"].max() if not gabungan_grp_luar.empty else 0)
                fig_gabungan_luar.update_layout(
                    xaxis_tickangle=-30,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    height=500
                )
                st.plotly_chart(fig_gabungan_luar, use_container_width=True)

                pivot_gab_luar = gabungan_grp_luar.pivot(index="Jenis", columns="Grade", values="Total_KG").fillna(0)
                pivot_display_luar = pivot_gab_luar.copy()
                for c in pivot_display_luar.columns:
                    pivot_display_luar[c] = pivot_gab_luar[c].apply(lambda x: f"{x:,.1f} KG")
                st.markdown("**Tabel Pivot KG: Jenis (baris) × Grade (kolom)**")
                st.dataframe(pivot_display_luar, use_container_width=True)
            else:
                st.info("Tidak ada data dengan Grade dan Jenis yang terisi.")

# TAB 3: TANAMAN
with tab3:
    st.markdown("### 🌱 Status Lahan Tanaman")

    biaya_berjalan_col = next(
        (c for c in df_tanaman_belum.columns if "biaya berjalan" in c.strip().lower()),
        None
    ) if not df_tanaman_belum.empty else None

    if biaya_berjalan_col:
        total_biaya_berjalan = to_number(df_tanaman_belum[biaya_berjalan_col]).sum()
    else:
        biaya_keywords = ["BIAYA", "MODAL", "BERJALAN"]
        biaya_cols_belum = [c for c in df_tanaman_belum.columns if not df_tanaman_belum.empty and any(k in str(c).upper() for k in biaya_keywords)] if not df_tanaman_belum.empty else []
        biaya_cols_sudah = [c for c in df_tanaman_sudah.columns if not df_tanaman_sudah.empty and any(k in str(c).upper() for k in biaya_keywords)] if not df_tanaman_sudah.empty else []
        def _sum_cols(df: pd.DataFrame, cols) -> float:
            total = 0.0
            for c in cols:
                total += to_number(df[c]).sum()
            return total
        total_biaya_berjalan = _sum_cols(df_tanaman_belum, biaya_cols_belum) + _sum_cols(df_tanaman_sudah, biaya_cols_sudah)

    m1, m2, m3 = st.columns(3)
    m1.metric("💰 Total Biaya Berjalan", rp(total_biaya_berjalan))
    m2.metric("🌿 Cok AB per KG",        f"Rp {HARGA_COK_AB:,}/KG")
    m3.metric("🌾 Cok RUT per KG",       f"Rp {HARGA_COK_RUT:,}/KG")

    st.divider()
    tanaman_tab1, tanaman_tab2 = st.tabs(["🌾 Belum Panen", "✅ Sudah Panen"])

    with tanaman_tab1:
        if not df_tanaman_belum.empty:
            luas_col      = next((c for c in df_tanaman_belum.columns if "luas" in c.strip().lower()), None)
            est_kg_col    = next((c for c in df_tanaman_belum.columns if "estimasi" in c.strip().lower() and "kg" in c.strip().lower()), None)
            est_ab_col    = next((c for c in df_tanaman_belum.columns if "estimasi" in c.strip().lower() and "ab" in c.strip().lower()), None)
            est_rut_col   = next((c for c in df_tanaman_belum.columns if "estimasi" in c.strip().lower() and "rut" in c.strip().lower()), None)

            summary_cols = st.columns(4)
            if luas_col:
                total_luas = to_number(df_tanaman_belum[luas_col]).sum()
                summary_cols[0].metric("🌍 Total Luas", f"{total_luas:,.2f} Ha")
            else:
                summary_cols[0].info("Kolom Luas (Ha) tidak ditemukan")

            if est_kg_col:
                total_est_kg = to_number(df_tanaman_belum[est_kg_col]).sum()
                summary_cols[1].metric("⚖️ Total Estimasi KG", f"{total_est_kg:,.0f} KG")
            else:
                summary_cols[1].info("Kolom Estimasi KG tidak ditemukan")

            if est_ab_col:
                total_est_ab = to_number(df_tanaman_belum[est_ab_col]).sum()
                summary_cols[2].metric("🌿 Total Estimasi AB", rp(total_est_ab))
            else:
                summary_cols[2].info("Kolom Estimasi AB tidak ditemukan")

            if est_rut_col:
                total_est_rut = to_number(df_tanaman_belum[est_rut_col]).sum()
                summary_cols[3].metric("🌾 Total Estimasi RUT", rp(total_est_rut))
            else:
                summary_cols[3].info("Kolom Estimasi RUT tidak ditemukan")

            st.markdown(f"**Total Lahan Tanam (Belum Panen): {len(df_tanaman_belum)} baris**")
            st.dataframe(format_money_table(df_tanaman_belum), use_container_width=True, hide_index=True)
        else:
            st.info("Data sheet 'TANAMAN BELUM PANEN' kosong atau tidak ditemukan.")

    with tanaman_tab2:
        if not df_tanaman_sudah.empty:
            omzet_col_sudah = next((c for c in df_tanaman_sudah.columns if "omzet" in c.strip().lower() or "total harga" in c.strip().lower() or "pendapatan" in c.strip().lower()), None)
            laba_col_sudah  = next((c for c in df_tanaman_sudah.columns if "laba" in c.strip().lower() or "keuntungan" in c.strip().lower() or "profit" in c.strip().lower()), None)
            luas_col_sudah  = next((c for c in df_tanaman_sudah.columns if "luas" in c.strip().lower()), None)

            if omzet_col_sudah:
                df_tanaman_sudah[omzet_col_sudah] = to_number(df_tanaman_sudah[omzet_col_sudah])
                if omzet_col_sudah in df_tanaman_sudah_filtered.columns:
                    df_tanaman_sudah_filtered[omzet_col_sudah] = to_number(df_tanaman_sudah_filtered[omzet_col_sudah])
            if laba_col_sudah:
                df_tanaman_sudah[laba_col_sudah] = to_number(df_tanaman_sudah[laba_col_sudah])
                if laba_col_sudah in df_tanaman_sudah_filtered.columns:
                    df_tanaman_sudah_filtered[laba_col_sudah] = to_number(df_tanaman_sudah_filtered[laba_col_sudah])
            if luas_col_sudah:
                df_tanaman_sudah[luas_col_sudah] = to_number(df_tanaman_sudah[luas_col_sudah])
                if luas_col_sudah in df_tanaman_sudah_filtered.columns:
                    df_tanaman_sudah_filtered[luas_col_sudah] = to_number(df_tanaman_sudah_filtered[luas_col_sudah])

            panen_s1, panen_s2, panen_s3 = st.columns(3)
            if omzet_col_sudah:
                total_omzet_panen = df_tanaman_sudah_filtered[omzet_col_sudah].sum() if omzet_col_sudah in df_tanaman_sudah_filtered.columns else df_tanaman_sudah[omzet_col_sudah].sum()
                panen_s1.metric("💰 Total Omzet Panen (Filter)", rp(total_omzet_panen))
            else:
                panen_s1.info("Kolom Omzet tidak ditemukan")

            if laba_col_sudah:
                total_laba_panen = df_tanaman_sudah_filtered[laba_col_sudah].sum() if laba_col_sudah in df_tanaman_sudah_filtered.columns else df_tanaman_sudah[laba_col_sudah].sum()
                panen_s2.metric("📈 Total Laba Panen (Filter)", rp(total_laba_panen))
            else:
                panen_s2.info("Kolom Laba tidak ditemukan")

            if luas_col_sudah:
                total_luas_panen = df_tanaman_sudah[luas_col_sudah].sum()
                panen_s3.metric("🌍 Total Luas Panen", f"{total_luas_panen:,.2f} Ha")
            else:
                panen_s3.info("Kolom Luas (Ha) tidak ditemukan")

            st.caption("Omzet & Laba mengikuti filter rentang tanggal sidebar (kolom Tanggal Panen). Total Luas Panen selalu menampilkan keseluruhan data (tidak difilter). Tabel di bawah menampilkan seluruh data (tidak difilter).")
            st.markdown(f"**Total Lahan Panen: {len(df_tanaman_sudah)} baris**")
            st.dataframe(format_money_table(df_tanaman_sudah), use_container_width=True, hide_index=True)
        else:
            st.info("Data sheet 'TANAMAN PANEN' kosong atau tidak ditemukan.")

# TAB 4: PIUTANG
with tab4:
    st.markdown("### 🧾 Piutang")
    st.caption("Sisa Hutang = total keseluruhan (tidak difilter tanggal).")

    sisa_lapak      = df_piutang_raw["Sisa Hutang"].sum()      if not df_piutang_raw.empty      and "Sisa Hutang" in df_piutang_raw.columns      else 0
    sisa_lapak_luar = df_piutang_luar_raw["Sisa Hutang"].sum() if not df_piutang_luar_raw.empty and "Sisa Hutang" in df_piutang_luar_raw.columns else 0
    total_sisa_semua = sisa_lapak + sisa_lapak_luar

    st.markdown(
        f'<div class="big-total">⚠️ Total Seluruh Sisa Piutang: {rp(total_sisa_semua)}</div>',
        unsafe_allow_html=True
    )
    col_sisa1, col_sisa2 = st.columns(2)
    col_sisa1.metric("Sisa Piutang Pelanggan Lapak", rp(sisa_lapak))
    col_sisa2.metric("Sisa Piutang Lapak Luar",      rp(sisa_lapak_luar))

    st.divider()

    piutang_tab1, piutang_tab2 = st.tabs(["🏪 Piutang Pelanggan Lapak", "🏬 Piutang Lapak Luar"])

    def _render_piutang_tab(df_raw, df_filtered, label_suffix=""):
        if df_raw.empty:
            st.info(f"Data Piutang {label_suffix} kosong.")
            return

        total_hutang_t = df_raw["Hutang"].sum()      if "Hutang"      in df_raw.columns else 0
        total_bayar_t  = df_raw["Payment"].sum()     if "Payment"     in df_raw.columns else 0
        total_sisa_t   = df_raw["Sisa Hutang"].sum() if "Sisa Hutang" in df_raw.columns else 0

        p1, p2, p3 = st.columns(3)
        p1.metric("📋 Total Piutang", rp(total_hutang_t))
        p2.metric("✅ Terbayar",       rp(total_bayar_t))
        p3.metric("⚠️ Sisa Piutang",  rp(total_sisa_t))

        st.divider()

        kode_col = "KODE" if "KODE" in df_raw.columns else None
        if kode_col and "Sisa Hutang" in df_raw.columns:
            sisa_grp = df_raw.groupby(kode_col)["Sisa Hutang"].sum().reset_index().sort_values("Sisa Hutang", ascending=False)
            fig_sisa = go.Figure()
            fig_sisa.add_trace(go.Bar(
                x=sisa_grp[kode_col], y=sisa_grp["Sisa Hutang"],
                text=[rp_short(v) for v in sisa_grp["Sisa Hutang"]],
                textposition="outside", textfont=dict(size=13, color="#d62728"),
                marker_color="#d62728"
            ))
            fig_sisa.update_layout(
                title=f"Sisa Hutang per Lapak {label_suffix}",
                xaxis_title="Kode Lapak", yaxis_title="Rupiah",
                xaxis_tickangle=-30, showlegend=False
            )
            pad_yaxis(fig_sisa, sisa_grp["Sisa Hutang"].max() if not sisa_grp.empty else 0)
            st.plotly_chart(fig_sisa, use_container_width=True)

    with piutang_tab1:
        _render_piutang_tab(df_piutang_raw, df_piutang_filtered, "Pelanggan Lapak")

        if not df_piutang_raw.empty:
            nama_col_pl = next(
                (c for c in df_piutang_raw.columns if c.strip().upper() in ["NAMA", "NAMA PELANGGAN", "CUSTOMER", "PELANGGAN"]),
                None
            )
            kode_col_pl = "KODE" if "KODE" in df_piutang_raw.columns else None

            if nama_col_pl and "Sisa Hutang" in df_piutang_raw.columns:
                st.divider()
                st.markdown("#### 👤 Daftar Piutang per Nama Pelanggan")

                if kode_col_pl:
                    lapak_opts_pl = sorted(df_piutang_raw[kode_col_pl].dropna().astype(str).unique())
                    sel_lapak_pl = st.multiselect(
                        "Filter Lapak (KODE)", lapak_opts_pl, default=lapak_opts_pl, key="tab4_piutang_lapak_filter"
                    )
                    df_piutang_nama = (
                        df_piutang_raw[df_piutang_raw[kode_col_pl].astype(str).isin(sel_lapak_pl)]
                        if sel_lapak_pl else df_piutang_raw.iloc[0:0]
                    )
                else:
                    df_piutang_nama = df_piutang_raw

                if df_piutang_nama.empty:
                    st.info("Pilih minimal satu lapak untuk menampilkan daftar pelanggan.")
                else:
                    group_cols_pl = [nama_col_pl] + ([kode_col_pl] if kode_col_pl else [])
                    agg_nama_pl = {"Sisa_Hutang": ("Sisa Hutang", "sum")}
                    if "Hutang" in df_piutang_nama.columns:
                        agg_nama_pl["Total_Hutang"] = ("Hutang", "sum")
                    if "Payment" in df_piutang_nama.columns:
                        agg_nama_pl["Total_Terbayar"] = ("Payment", "sum")

                    per_nama_pl = (
                        df_piutang_nama.groupby(group_cols_pl)
                        .agg(**agg_nama_pl)
                        .reset_index()
                        .sort_values("Sisa_Hutang", ascending=False)
                    )

                    per_nama_pl = per_nama_pl[per_nama_pl["Sisa_Hutang"].fillna(0) > 0].reset_index(drop=True)

                    if per_nama_pl.empty:
                        st.success("🎉 Semua pelanggan pada lapak terpilih sudah lunas (sisa piutang 0).")
                    else:
                        top_nama_pl = per_nama_pl.head(25)
                        fig_nama_pl = go.Figure()
                        fig_nama_pl.add_trace(go.Bar(
                            x=top_nama_pl[nama_col_pl], y=top_nama_pl["Sisa_Hutang"],
                            text=[rp_short(v) for v in top_nama_pl["Sisa_Hutang"]],
                            textposition="outside", textfont=dict(size=12, color="#d62728"),
                            marker_color="#d62728"
                        ))
                        fig_nama_pl.update_layout(
                            title="Sisa Piutang per Nama Pelanggan (Top 25, hanya yang masih ada sisa)",
                            xaxis_title="Nama Pelanggan", yaxis_title="Rupiah",
                            xaxis_tickangle=-30, showlegend=False, height=450
                        )
                        pad_yaxis(fig_nama_pl, top_nama_pl["Sisa_Hutang"].max() if not top_nama_pl.empty else 0)
                        st.plotly_chart(fig_nama_pl, use_container_width=True)

                        tabel_nama_pl = per_nama_pl.copy()
                        rename_pl = {nama_col_pl: "Nama Pelanggan", "Sisa_Hutang": "Sisa Piutang"}
                        if kode_col_pl:
                            rename_pl[kode_col_pl] = "Kode Lapak"
                        tabel_nama_pl["Sisa_Hutang"] = per_nama_pl["Sisa_Hutang"].apply(rp)
                        if "Total_Hutang" in tabel_nama_pl.columns:
                            tabel_nama_pl["Total_Hutang"] = per_nama_pl["Total_Hutang"].apply(rp)
                            rename_pl["Total_Hutang"] = "Total Piutang"
                        if "Total_Terbayar" in tabel_nama_pl.columns:
                            tabel_nama_pl["Total_Terbayar"] = per_nama_pl["Total_Terbayar"].apply(rp)
                            rename_pl["Total_Terbayar"] = "Total Terbayar"
                        tabel_nama_pl = tabel_nama_pl.rename(columns=rename_pl)

                        ordered_pl = ["Nama Pelanggan"]
                        if "Kode Lapak" in tabel_nama_pl.columns:      ordered_pl.append("Kode Lapak")
                        if "Total Piutang" in tabel_nama_pl.columns:   ordered_pl.append("Total Piutang")
                        if "Total Terbayar" in tabel_nama_pl.columns:  ordered_pl.append("Total Terbayar")
                        ordered_pl.append("Sisa Piutang")

                        st.markdown(f"**Total: {len(tabel_nama_pl)} pelanggan masih memiliki sisa piutang**")
                        st.dataframe(tabel_nama_pl[ordered_pl], use_container_width=True, hide_index=True)
            elif not nama_col_pl:
                st.info("Kolom 'NAMA' / 'NAMA PELANGGAN' tidak ditemukan di sheet PIUTANG LAPAK, sehingga daftar per nama pelanggan tidak bisa ditampilkan.")

    with piutang_tab2:
        _render_piutang_tab(df_piutang_luar_raw, df_piutang_luar_filtered, "Lapak Luar")

        if not df_piutang_luar_raw.empty:
            nama_col_pll = next(
                (c for c in df_piutang_luar_raw.columns if c.strip().upper() in ["NAMA", "NAMA PELANGGAN", "CUSTOMER", "PELANGGAN"]),
                None
            )
            if nama_col_pll and "Sisa Hutang" in df_piutang_luar_raw.columns:
                st.divider()
                st.markdown("#### 👤 Total Piutang per Nama Pelanggan")

                agg_nama = {"Sisa_Hutang": ("Sisa Hutang", "sum")}
                if "Hutang" in df_piutang_luar_raw.columns:
                    agg_nama["Total_Hutang"] = ("Hutang", "sum")
                if "Payment" in df_piutang_luar_raw.columns:
                    agg_nama["Total_Terbayar"] = ("Payment", "sum")

                per_nama = (
                    df_piutang_luar_raw.groupby(nama_col_pll)
                    .agg(**agg_nama)
                    .reset_index()
                    .sort_values("Sisa_Hutang", ascending=False)
                )

                per_nama = per_nama[per_nama["Sisa_Hutang"].fillna(0) > 0].reset_index(drop=True)

                fig_per_nama = go.Figure()
                fig_per_nama.add_trace(go.Bar(
                    x=per_nama[nama_col_pll], y=per_nama["Sisa_Hutang"],
                    text=[rp_short(v) for v in per_nama["Sisa_Hutang"]],
                    textposition="outside", textfont=dict(size=12, color="#d62728"),
                    marker_color="#d62728"
                ))
                fig_per_nama.update_layout(
                    title="Sisa Piutang per Nama Pelanggan (Lapak Luar)",
                    xaxis_title="Nama Pelanggan", yaxis_title="Rupiah",
                    xaxis_tickangle=-30, showlegend=False
                )
                pad_yaxis(fig_per_nama, per_nama["Sisa_Hutang"].max() if not per_nama.empty else 0)
                st.plotly_chart(fig_per_nama, use_container_width=True)

                tabel_nama = per_nama.copy()
                rename_nama = {nama_col_pll: "Nama Pelanggan", "Sisa_Hutang": "Sisa Piutang"}
                tabel_nama["Sisa_Hutang"] = per_nama["Sisa_Hutang"].apply(rp)
                if "Total_Hutang" in tabel_nama.columns:
                    tabel_nama["Total_Hutang"] = per_nama["Total_Hutang"].apply(rp)
                    rename_nama["Total_Hutang"] = "Total Piutang"
                if "Total_Terbayar" in tabel_nama.columns:
                    tabel_nama["Total_Terbayar"] = per_nama["Total_Terbayar"].apply(rp)
                    rename_nama["Total_Terbayar"] = "Total Terbayar"

                tabel_nama = tabel_nama.rename(columns=rename_nama)
                ordered_cols = ["Nama Pelanggan"]
                if "Total Piutang" in tabel_nama.columns:  ordered_cols.append("Total Piutang")
                if "Total Terbayar" in tabel_nama.columns: ordered_cols.append("Total Terbayar")
                ordered_cols.append("Sisa Piutang")
                st.dataframe(tabel_nama[ordered_cols], use_container_width=True, hide_index=True)

                st.divider()
                st.markdown("#### 📋 Rincian Piutang per Nama Pelanggan (Lapak Luar)")
                nama_opts_rincian_pll = sorted(df_piutang_luar_raw[nama_col_pll].dropna().astype(str).unique())
                sel_nama_rincian_pll = st.multiselect(
                    "Filter Nama Pelanggan", nama_opts_rincian_pll,
                    default=nama_opts_rincian_pll, key="tab4_piutang_luar_rincian_nama"
                )
                if sel_nama_rincian_pll:
                    df_rincian_pll = df_piutang_luar_raw[df_piutang_luar_raw[nama_col_pll].astype(str).isin(sel_nama_rincian_pll)].copy()

                    kolom_tampil_pll = []
                    if "Tanggal_Lengkap" in df_rincian_pll.columns and df_rincian_pll["Tanggal_Lengkap"].notna().any():
                        df_rincian_pll["Tanggal"] = df_rincian_pll["Tanggal_Lengkap"].dt.strftime("%d/%m/%Y")
                        kolom_tampil_pll.append("Tanggal")
                    if "KODE" in df_rincian_pll.columns:
                        kolom_tampil_pll.append("KODE")
                    kolom_tampil_pll.append(nama_col_pll)
                    for _c in ["Hutang", "Payment"]:
                        if _c in df_rincian_pll.columns:
                            kolom_tampil_pll.append(_c)

                    kolom_raw_tgl_pll = [c for c in df_rincian_pll.columns if c.strip().upper() in ["TANGGAL", "TGL", "DATE"]]
                    kolom_exclude_pll = set(kolom_tampil_pll) | {"Tanggal_Lengkap", "Sisa Hutang"} | set(kolom_raw_tgl_pll)
                    kolom_tampil_pll += [c for c in df_rincian_pll.columns if c not in kolom_exclude_pll and not str(c).startswith("_col_")]

                    sort_cols_pll = ["Tanggal_Lengkap"] if "Tanggal_Lengkap" in df_rincian_pll.columns else [nama_col_pll]
                    sort_asc_pll  = [False] if "Tanggal_Lengkap" in df_rincian_pll.columns else [True]
                    df_rincian_pll = df_rincian_pll.sort_values(sort_cols_pll, ascending=sort_asc_pll, na_position="last")

                    df_rincian_pll_tampil = df_rincian_pll[kolom_tampil_pll].copy()
                    for _c in ["Hutang", "Payment"]:
                        if _c in df_rincian_pll_tampil.columns:
                            df_rincian_pll_tampil[_c] = df_rincian_pll_tampil[_c].map(rp)
                    st.dataframe(df_rincian_pll_tampil, use_container_width=True, hide_index=True)

                    total_sisa_rincian_pll = df_rincian_pll["Sisa Hutang"].sum() if "Sisa Hutang" in df_rincian_pll.columns else 0
                    st.caption(f"📌 {len(df_rincian_pll_tampil)} baris transaksi · Total Sisa Piutang: {rp(total_sisa_rincian_pll)}")
                else:
                    st.info("Pilih minimal satu nama pelanggan untuk menampilkan rincian transaksinya.")
            elif not nama_col_pll:
                st.info("Kolom 'NAMA' / 'NAMA PELANGGAN' tidak ditemukan di sheet PIUTANG LAPAK LUAR.")

# TAB 5: HUTANG
def _render_hutang_section(df_raw, judul_total):
    cols_h = list(df_raw.columns)
    hutang_col  = next((c for c in cols_h if c.strip().upper() in ["HUTANG", "JUMLAH HUTANG", "TOTAL HUTANG", "PINJAMAN"]), None)
    payment_col = next((c for c in cols_h if c.strip().upper() in ["PAYMENT", "BAYAR", "TERBAYAR", "PEMBAYARAN", "ANGSURAN"]), None)
    sisa_col    = next((c for c in cols_h if any(k in c.strip().upper() for k in ["SISA", "OUTSTANDING"])), None)

    if sisa_col:
        total_sisa = to_number(df_raw[sisa_col]).sum()
    elif hutang_col and payment_col:
        total_sisa = to_number(df_raw[hutang_col]).sum() - to_number(df_raw[payment_col]).sum()
    elif hutang_col:
        total_sisa = to_number(df_raw[hutang_col]).sum()
    else:
        hutang_cols_all  = [c for c in cols_h if "HUTANG" in c.upper() and "SISA" not in c.upper()]
        payment_cols_all = [c for c in cols_h if any(k in c.upper() for k in ["PAYMENT", "BAYAR", "TERBAYAR"])]
        total_sisa = (
            sum(to_number(df_raw[c]).sum() for c in hutang_cols_all)
            - sum(to_number(df_raw[c]).sum() for c in payment_cols_all)
        )

    st.markdown(
        f'<div class="big-total">{judul_total}: {rp(total_sisa)}</div>',
        unsafe_allow_html=True
    )

    if hutang_col or sisa_col:
        h1, h2, h3 = st.columns(3)
        if hutang_col:
            h1.metric("📋 Total Hutang",   rp(to_number(df_raw[hutang_col]).sum()))
        if payment_col:
            h2.metric("✅ Total Terbayar", rp(to_number(df_raw[payment_col]).sum()))
        h3.metric("⚠️ Sisa Hutang",        rp(total_sisa))

    st.divider()
    st.markdown(f"**Total Data: {len(df_raw)} baris**")
    st.dataframe(format_money_table(df_raw), use_container_width=True, hide_index=True)

with tab5:
    st.markdown("### 👨‍🌾 Hutang")

    hutang_tab1, hutang_tab2 = st.tabs(["👨‍🌾 Hutang Petani", "🧑‍🌾 Hutang Pak'e Tani"])

    with hutang_tab1:
        if df_hutang_petani_raw.empty:
            st.info("Data Hutang Petani kosong atau sheet tidak ditemukan.")
        else:
            _render_hutang_section(df_hutang_petani_raw, "💳 Total Sisa Hutang Petani")

    with hutang_tab2:
        if df_hutang_pake_tani_raw.empty:
            st.info("Data sheet \"HUTANG PAK'E TANI\" kosong atau tidak ditemukan.")
        else:
            total_hutang_pt   = df_hutang_pake_tani_raw["HUTANG"].sum()   if "HUTANG"   in df_hutang_pake_tani_raw.columns else 0
            total_terbayar_pt = df_hutang_pake_tani_raw["TERBAYAR"].sum() if "TERBAYAR" in df_hutang_pake_tani_raw.columns else 0
            total_sisa_pt     = total_hutang_pt - total_terbayar_pt

            st.markdown(
                f'<div class="big-total">💳 Total Sisa Hutang Pak\'e Tani: {rp(total_sisa_pt)}</div>',
                unsafe_allow_html=True
            )

            pt1, pt2, pt3 = st.columns(3)
            pt1.metric("📋 Total Hutang",   rp(total_hutang_pt))
            pt2.metric("✅ Total Terbayar", rp(total_terbayar_pt))
            pt3.metric("⚠️ Sisa Hutang",    rp(total_sisa_pt))

            st.divider()
            st.markdown(f"**Total Data: {len(df_hutang_pake_tani_raw)} baris**")
            st.dataframe(format_money_table(df_hutang_pake_tani_raw), use_container_width=True, hide_index=True)

# TAB 6: ARUS KAS
with tab6:
    st.markdown("### 💸 Laporan Arus Kas")
    if df_kas.empty:
        st.info("Data Arus Kas kosong untuk periode yang dipilih.")
    else:
        masuk_kas  = df_kas["KAS MASUK"].sum()  if "KAS MASUK"  in df_kas.columns else 0
        keluar_kas = df_kas["KAS KELUAR"].sum() if "KAS KELUAR" in df_kas.columns else 0
        saldo_kas  = df_kas["SALDO"].dropna().iloc[-1] if "SALDO" in df_kas.columns and not df_kas["SALDO"].dropna().empty else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🏦 Saldo Terakhir",   rp(saldo_kas))
        k2.metric("🟩 Total Kas Masuk",  rp(masuk_kas))
        k3.metric("🟥 Total Kas Keluar", rp(keluar_kas))
        k4.metric("📊 Selisih Bersih",   rp(masuk_kas - keluar_kas))

        st.divider()

        section_heading("🔎 Total Arus Kas per Jenis (Filter)")
        if "JENIS" in df_kas.columns:
            jenis_opts_top = sorted(df_kas["JENIS"].dropna().unique())
            sel_jenis_top = st.multiselect(
                "Pilih Jenis untuk ditampilkan totalnya", jenis_opts_top, default=jenis_opts_top, key="tab6_jenis_top"
            )
            df_kas_jenis_top = df_kas[df_kas["JENIS"].isin(sel_jenis_top)] if sel_jenis_top else df_kas.iloc[0:0]
            masuk_top  = df_kas_jenis_top["KAS MASUK"].sum()  if "KAS MASUK"  in df_kas_jenis_top.columns else 0
            keluar_top = df_kas_jenis_top["KAS KELUAR"].sum() if "KAS KELUAR" in df_kas_jenis_top.columns else 0
            t1, t2, t3 = st.columns(3)
            t1.metric("Kas Masuk (Jenis Terpilih)",  rp(masuk_top))
            t2.metric("Kas Keluar (Jenis Terpilih)", rp(keluar_top))
            t3.metric("Selisih",                      rp(masuk_top - keluar_top))
        else:
            st.info("Kolom 'JENIS' tidak tersedia di sheet Arus Kas.")

        st.divider()

        if "Tanggal_Kas" in df_kas.columns:
            df_kas_plot = df_kas.copy()
            df_kas_plot["Bulan_Label"] = df_kas_plot["Tanggal_Kas"].dt.to_period("M").astype(str)
            kas_bulanan = df_kas_plot.groupby("Bulan_Label").agg(
                Masuk=("KAS MASUK", "sum"), Keluar=("KAS KELUAR", "sum")
            ).reset_index()

            fig_kas = go.Figure()
            fig_kas.add_trace(go.Bar(
                name="Kas Masuk", x=kas_bulanan["Bulan_Label"], y=kas_bulanan["Masuk"],
                marker_color="#2ca02c",
                text=[rp_short(v) for v in kas_bulanan["Masuk"]],
                textposition="outside", textfont=dict(size=11)
            ))
            fig_kas.add_trace(go.Bar(
                name="Kas Keluar", x=kas_bulanan["Bulan_Label"], y=kas_bulanan["Keluar"],
                marker_color="#d62728",
                text=[rp_short(v) for v in kas_bulanan["Keluar"]],
                textposition="outside", textfont=dict(size=11)
            ))
            fig_kas.update_layout(
                barmode="group", title="Kas Masuk vs Kas Keluar per Bulan",
                yaxis_title="Rupiah", xaxis_title="Bulan"
            )
            pad_yaxis(fig_kas, max(kas_bulanan["Masuk"].max(), kas_bulanan["Keluar"].max()) if not kas_bulanan.empty else 0)
            st.plotly_chart(fig_kas, use_container_width=True)

        st.divider()
        st.markdown("#### 🔍 Rincian Transaksi per Jenis")
        if "JENIS" in df_kas.columns:
            jenis_kas_all = sorted(df_kas["JENIS"].dropna().unique())
            sel_jenis_kas6 = st.multiselect("Filter Jenis", jenis_kas_all, default=jenis_kas_all, key="tab6_jenis")
            df_kas_show = df_kas[df_kas["JENIS"].isin(sel_jenis_kas6)] if sel_jenis_kas6 else df_kas.copy()
        else:
            df_kas_show = df_kas.copy()
        st.dataframe(format_money_table(df_kas_show.drop(columns=["Tanggal_Kas"], errors="ignore"), extra_keywords=["KAS", "SALDO", "MASUK", "KELUAR"]), use_container_width=True, hide_index=True)

        st.markdown("#### 📊 Ringkasan per Jenis")
        if "JENIS" in df_kas.columns:
            grp_jenis_kas = df_kas.groupby("JENIS").agg(
                Masuk=("KAS MASUK", "sum"), Keluar=("KAS KELUAR", "sum")
            ).reset_index()
            grp_jenis_kas["Selisih"] = grp_jenis_kas["Masuk"] - grp_jenis_kas["Keluar"]
            grp_jenis_kas = grp_jenis_kas.sort_values("Selisih", ascending=False)

            fig_jenis = go.Figure()
            fig_jenis.add_trace(go.Bar(
                name="Kas Masuk", x=grp_jenis_kas["JENIS"], y=grp_jenis_kas["Masuk"],
                marker_color="#2ca02c",
                text=[rp_short(v) for v in grp_jenis_kas["Masuk"]],
                textposition="outside", textfont=dict(size=10)
            ))
            fig_jenis.add_trace(go.Bar(
                name="Kas Keluar", x=grp_jenis_kas["JENIS"], y=grp_jenis_kas["Keluar"],
                marker_color="#d62728",
                text=[rp_short(v) for v in grp_jenis_kas["Keluar"]],
                textposition="outside", textfont=dict(size=10)
            ))
            fig_jenis.update_layout(
                barmode="group", title="Kas Masuk vs Keluar per Jenis",
                yaxis_title="Rupiah", xaxis_title="Jenis", xaxis_tickangle=-35
            )
            pad_yaxis(fig_jenis, max(grp_jenis_kas["Masuk"].max(), grp_jenis_kas["Keluar"].max()) if not grp_jenis_kas.empty else 0)
            st.plotly_chart(fig_jenis, use_container_width=True)

            grp_display = grp_jenis_kas.copy()
            grp_display["Masuk"]   = grp_display["Masuk"].apply(rp)
            grp_display["Keluar"]  = grp_display["Keluar"].apply(rp)
            grp_display["Selisih"] = grp_jenis_kas["Selisih"].apply(rp)
            st.dataframe(grp_display, use_container_width=True, hide_index=True)

# TAB 7: EKSPEDISI
with tab7:
    st.markdown("### 🚛 Operasional Ekspedisi")

    if df_ekspedisi.empty:
        st.info("Data Ekspedisi kosong untuk periode yang dipilih.")
    else:
        pend_eks  = df_ekspedisi["PENDAPATAN"].sum()  if "PENDAPATAN"  in df_ekspedisi.columns else 0
        pengl_eks = df_ekspedisi["PENGELUARAN"].sum() if "PENGELUARAN" in df_ekspedisi.columns else 0
        laba_eks  = pend_eks - pengl_eks

        e1, e2, e3 = st.columns(3)
        e1.metric("💰 Total Pendapatan",  rp(pend_eks))
        e2.metric("💸 Total Pengeluaran", rp(pengl_eks))
        e3.metric("📈 Laba Bersih",       rp(laba_eks),
                  delta=f"{(laba_eks/pend_eks*100):.1f}%" if pend_eks > 0 else None)

        st.divider()

        if not df_ekspedisi_raw.empty and "Tanggal_Lengkap" in df_ekspedisi_raw.columns:
            df_eks_plot = df_ekspedisi_raw.copy()
            df_eks_plot["Bulan_Label"] = df_eks_plot["Tanggal_Lengkap"].dt.to_period("M").astype(str)
            eks_bulanan = df_eks_plot.groupby("Bulan_Label").agg(
                Pendapatan=("PENDAPATAN", "sum"), Pengeluaran=("PENGELUARAN", "sum")
            ).reset_index().sort_values("Bulan_Label")
            eks_bulanan["Laba"] = eks_bulanan["Pendapatan"] - eks_bulanan["Pengeluaran"]

            fig_eks = go.Figure()
            fig_eks.add_trace(go.Bar(
                name="Pendapatan", x=eks_bulanan["Bulan_Label"], y=eks_bulanan["Pendapatan"],
                marker_color="#1f77b4",
                text=[rp_short(v) for v in eks_bulanan["Pendapatan"]],
                textposition="outside", textfont=dict(size=11)
            ))
            fig_eks.add_trace(go.Bar(
                name="Pengeluaran", x=eks_bulanan["Bulan_Label"], y=eks_bulanan["Pengeluaran"],
                marker_color="#d62728",
                text=[rp_short(v) for v in eks_bulanan["Pengeluaran"]],
                textposition="outside", textfont=dict(size=11)
            ))
            fig_eks.add_trace(go.Scatter(
                name="Laba Bersih", x=eks_bulanan["Bulan_Label"], y=eks_bulanan["Laba"],
                mode="lines+markers", line=dict(color="#ff7f0e", width=2), marker=dict(size=8)
            ))
            fig_eks.update_layout(
                barmode="group",
                title="Pendapatan vs Pengeluaran Ekspedisi per Bulan (Seluruh Data, Tidak Difilter)",
                yaxis_title="Rupiah", xaxis_title="Bulan"
            )
            pad_yaxis(fig_eks, max(eks_bulanan["Pendapatan"].max(), eks_bulanan["Pengeluaran"].max()) if not eks_bulanan.empty else 0)
            st.plotly_chart(fig_eks, use_container_width=True)
            st.caption("Grafik ini selalu menampilkan semua bulan, tidak mengikuti filter rentang tanggal di sidebar.")

        st.divider()

        nama_col_eks = next((c for c in df_ekspedisi.columns if c.strip().upper() == "NAMA"), None)
        if nama_col_eks:
            st.markdown("#### 👤 Kinerja per Nama (Driver/Pengemudi)")
            grp_nama = df_ekspedisi.groupby(nama_col_eks).agg(
                Pendapatan=("PENDAPATAN", "sum"), Pengeluaran=("PENGELUARAN", "sum")
            ).reset_index()
            grp_nama["Laba"] = grp_nama["Pendapatan"] - grp_nama["Pengeluaran"]
            grp_nama = grp_nama.sort_values("Pendapatan", ascending=False)

            fig_nama = go.Figure()
            fig_nama.add_trace(go.Bar(
                name="Pendapatan", x=grp_nama[nama_col_eks], y=grp_nama["Pendapatan"],
                marker_color="#1f77b4",
                text=[rp_short(v) for v in grp_nama["Pendapatan"]],
                textposition="outside", textfont=dict(size=11)
            ))
            fig_nama.add_trace(go.Bar(
                name="Pengeluaran", x=grp_nama[nama_col_eks], y=grp_nama["Pengeluaran"],
                marker_color="#d62728",
                text=[rp_short(v) for v in grp_nama["Pengeluaran"]],
                textposition="outside", textfont=dict(size=11)
            ))
            fig_nama.update_layout(
                barmode="group", title="Pendapatan vs Pengeluaran per Driver",
                yaxis_title="Rupiah", xaxis_title="Nama Driver"
            )
            pad_yaxis(fig_nama, max(grp_nama["Pendapatan"].max(), grp_nama["Pengeluaran"].max()) if not grp_nama.empty else 0)
            st.plotly_chart(fig_nama, use_container_width=True)

            grp_display_eks = grp_nama.copy()
            grp_display_eks["Pendapatan"]  = grp_nama["Pendapatan"].apply(rp)
            grp_display_eks["Pengeluaran"] = grp_nama["Pengeluaran"].apply(rp)
            grp_display_eks["Laba"]        = grp_nama["Laba"].apply(rp)
            st.dataframe(grp_display_eks, use_container_width=True, hide_index=True)

        st.divider()

        ket_col_eks = next((c for c in df_ekspedisi.columns if c.strip().upper() == "KETERANGAN"), None)
        if ket_col_eks:
            st.markdown("#### 📋 Distribusi per Keterangan/Jenis Transaksi")
            grp_ket = df_ekspedisi.groupby(ket_col_eks).agg(
                Pendapatan=("PENDAPATAN", "sum"),
                Pengeluaran=("PENGELUARAN", "sum"),
                Jumlah_Transaksi=(ket_col_eks, "count")
            ).reset_index().sort_values("Pendapatan", ascending=False)

            col_ket1, col_ket2 = st.columns(2)
            with col_ket1:
                top_ket_pend = grp_ket[grp_ket["Pendapatan"] > 0].head(8)
                if not top_ket_pend.empty:
                    fig_pie_ket = px.pie(
                        top_ket_pend, names=ket_col_eks, values="Pendapatan",
                        title="Komposisi Pendapatan per Keterangan (Top 8)", hole=0.35
                    )
                    fig_pie_ket.update_traces(textinfo="label+percent")
                    st.plotly_chart(fig_pie_ket, use_container_width=True)

            with col_ket2:
                top_ket_peng = grp_ket[grp_ket["Pengeluaran"] > 0].head(8)
                if not top_ket_peng.empty:
                    fig_pie_peng = px.pie(
                        top_ket_peng, names=ket_col_eks, values="Pengeluaran",
                        title="Komposisi Pengeluaran per Keterangan (Top 8)",
                        hole=0.35, color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    fig_pie_peng.update_traces(textinfo="label+percent")
                    st.plotly_chart(fig_pie_peng, use_container_width=True)

        st.divider()
        st.markdown("#### 📄 Detail Semua Transaksi Ekspedisi")
        st.dataframe(
            format_money_table(
                df_ekspedisi.drop(columns=["Tanggal_Lengkap"], errors="ignore"),
                extra_keywords=["PENDAPATAN", "PENGELUARAN"]
            ),
            use_container_width=True, hide_index=True
        )

# TAB 8: KERUGIAN GUDANG
with tab8:
    st.markdown("### 🏭 Kerugian Gudang")

    if df_kerugian_gudang_raw.empty:
        st.info("Data sheet 'KERUGIAN GUDANG' kosong atau tidak ditemukan.")
    else:
        total_hutang_kg   = df_kerugian_gudang_raw["HUTANG"].sum()   if "HUTANG"   in df_kerugian_gudang_raw.columns else 0
        total_terbayar_kg = df_kerugian_gudang_raw["TERBAYAR"].sum() if "TERBAYAR" in df_kerugian_gudang_raw.columns else 0
        sisa_kerugian_kg  = total_hutang_kg - total_terbayar_kg

        st.markdown(
            f'<div class="big-total">⚠️ Sisa Kerugian Belum Terbayar: {rp(sisa_kerugian_kg)}</div>',
            unsafe_allow_html=True
        )

        kg1, kg2, kg3 = st.columns(3)
        kg1.metric("📋 Total Hutang (Kerugian)", rp(total_hutang_kg))
        kg2.metric("✅ Total Terbayar",          rp(total_terbayar_kg))
        kg3.metric("⚠️ Sisa",                    rp(sisa_kerugian_kg))

        st.markdown(f"**Total Data: {len(df_kerugian_gudang_raw)} baris**")
        st.dataframe(format_money_table(df_kerugian_gudang_raw), use_container_width=True, hide_index=True)

# TAB 9: PREDIKSI HARGA
PRED_SPREADSHEET_ID = "1izW66Dv1H7XINUHJonwtJWNUo7BO9CjRkYU-LxEe6N0"

@st.cache_data(ttl=300, show_spinner="Memuat data harga harian...")
def load_data_harga_prediksi() -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{PRED_SPREADSHEET_ID}/gviz/tq?tqx=out:csv"
    try:
        df = pd.read_csv(url, dtype=str)
    except Exception as e:
        st.error(f"Gagal membaca spreadsheet harga harian: {e}")
        return pd.DataFrame()
    df.columns = [str(c).strip().upper() for c in df.columns]
    if "TANGGAL" not in df.columns or "HARGA" not in df.columns:
        st.error("Spreadsheet harga harian harus memiliki kolom 'TANGGAL' dan 'HARGA'.")
        return pd.DataFrame()
    df["TANGGAL"] = pd.to_datetime(df["TANGGAL"], format="%m/%d/%Y", errors="coerce")
    mask_nat = df["TANGGAL"].isna()
    if mask_nat.any():
        raw = pd.read_csv(url, dtype=str)
        raw.columns = [str(c).strip().upper() for c in raw.columns]
        df.loc[mask_nat, "TANGGAL"] = pd.to_datetime(raw.loc[mask_nat, "TANGGAL"], dayfirst=False, errors="coerce")
    df = df[df["TANGGAL"].notna()].reset_index(drop=True)
    df["HARGA"] = to_number(df["HARGA"])
    if "MBG" in df.columns:
        df["MBG"] = pd.to_numeric(df["MBG"], errors="coerce")
    return df.sort_values("TANGGAL").reset_index(drop=True)

with tab9:
    st.markdown("### 🔮 Analisis & Prediksi Tren Harga")

    if not HAS_LGBM and not HAS_PROPHET:
        st.error(
            "Library prediksi belum terpasang di server. Jalankan perintah berikut lalu restart aplikasi:\n\n"
            "```\npip install lightgbm prophet openpyxl\n```"
        )
    else:
        if not HAS_LGBM:
            st.warning("Library `lightgbm` belum terpasang — hanya model Prophet yang tersedia.")
        if not HAS_PROPHET:
            st.warning("Library `prophet` belum terpasang — hanya model LightGBM yang tersedia.")

        with st.container(border=True):
            st.markdown('<div class="income-card-title">🎛️ Pengaturan Prediksi</div>', unsafe_allow_html=True)

            model_opts = []
            if HAS_LGBM:    model_opts.append("LightGBM (Tree-Based)")
            if HAS_PROPHET: model_opts.append("Prophet (Seasonal Trend)")

            pc1, pc2 = st.columns(2)
            chosen_model = pc1.selectbox("Pilih Model Prediksi", model_opts, key="pred_model")
            forecast_horizon = pc2.slider("Durasi Hari Prediksi", min_value=7, max_value=180, value=116, step=1, key="pred_horizon")

            pc3, pc4, pc5 = st.columns(3)
            use_mbg = pc3.checkbox(
                "Gunakan Variabel Pengaruh MBG", value=False, key="pred_mbg",
                help="Jika aktif, model memakai kolom 'MBG' pada data Anda (0 = tidak ada pengaruh, 1 = ada pengaruh) sebagai regressor tambahan."
            )
            show_hist_toggle = pc4.checkbox("Tampilkan Data Historis", value=True, key="pred_show_hist")
            show_pred_toggle = pc5.checkbox("Tampilkan Hasil Prediksi", value=True, key="pred_show_pred")

        future_mbg_value = 1

        dfp_raw = load_data_harga_prediksi()

        if dfp_raw.empty:
            st.info("Data harga harian belum bisa dimuat dari spreadsheet. Periksa akses spreadsheet lalu klik 🔄 Refresh Data.")
        else:
            try:
                dfp_raw = dfp_raw.sort_values("TANGGAL").reset_index(drop=True)

                dfp_known = dfp_raw[dfp_raw["HARGA"].notna()].reset_index(drop=True)
                dfp_future_input = dfp_raw[dfp_raw["HARGA"].isna()].reset_index(drop=True)

                if len(dfp_future_input) == 0:
                    st.info("💡 Tidak ada baris berisi tanggal tanpa harga di spreadsheet. Tanggal prediksi digenerate otomatis sesuai durasi di pengaturan.")
                    last_date = dfp_known["TANGGAL"].max()
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon)
                    dfp_future_input = pd.DataFrame({"TANGGAL": future_dates})
                    dfp_future_input["MBG"] = future_mbg_value
                else:
                    if len(dfp_future_input) >= forecast_horizon:
                        dfp_future_input = dfp_future_input.iloc[:forecast_horizon].reset_index(drop=True)
                        st.success(f"✅ Mengambil {forecast_horizon} hari teratas dari baris tanpa harga di spreadsheet.")
                    else:
                        deficit = forecast_horizon - len(dfp_future_input)
                        last_date = dfp_future_input["TANGGAL"].max() if len(dfp_future_input) > 0 else dfp_known["TANGGAL"].max()

                        extra_dates = pd.date_range(start=last_date + timedelta(days=1), periods=deficit)
                        dfp_extra = pd.DataFrame({"TANGGAL": extra_dates})
                        dfp_extra["MBG"] = future_mbg_value

                        dfp_future_input = pd.concat([dfp_future_input, dfp_extra], ignore_index=True)
                        st.warning(f"⚠️ Baris tanpa harga di spreadsheet kurang. {len(dfp_future_input) - deficit} hari diambil dari spreadsheet, {deficit} hari digenerate otomatis.")

                if use_mbg:
                    if "MBG" not in dfp_raw.columns:
                        st.error(
                            "Kolom 'MBG' tidak ditemukan pada spreadsheet harga harian. Tambahkan kolom 'MBG' "
                            "berisi 0 (tidak ada pengaruh) atau 1 (ada pengaruh) pada setiap baris data, "
                            "atau matikan opsi 'Gunakan Variabel Pengaruh MBG'."
                        )
                        st.stop()
                    dfp_known["MBG"] = pd.to_numeric(dfp_known["MBG"], errors="coerce").fillna(0).clip(0, 1).round().astype(int)
                    dfp_future_input["MBG"] = pd.to_numeric(dfp_future_input["MBG"], errors="coerce").fillna(0).clip(0, 1).round().astype(int)
                else:
                    dfp_known["MBG"] = 0
                    dfp_future_input["MBG"] = 0

                dfp_known["TAHUN"] = dfp_known["TANGGAL"].dt.year
                dfp_known["HARI_KE"] = dfp_known["TANGGAL"].dt.dayofyear

                pm1, pm2, pm3 = st.columns(3)
                with pm1:
                    st.metric("Total Data Historis", f"{len(dfp_known)} Hari")
                with pm2:
                    st.metric("Harga Terakhir", f"{dfp_known['HARGA'].iloc[-1]:.2f}")
                with pm3:
                    st.metric("Target Jangka Waktu", f"{forecast_horizon} Hari")

                if use_mbg:
                    st.caption("✅ Variabel MBG **disertakan** sebagai regressor tambahan pada model berdasarkan data baris masa depan.")
                else:
                    st.caption("ℹ️ Variabel MBG **tidak** disertakan dalam model (opsi nonaktif).")

                with st.spinner("Sedang melatih model..."):
                    if "LightGBM" in chosen_model:
                        dfp_forecast = run_lightgbm(dfp_known, dfp_future_input, use_mbg)
                    else:
                        dfp_forecast = run_prophet(dfp_known, dfp_future_input, use_mbg)

                pred_tab_tren, pred_tab_yoy, pred_tab_data = st.tabs([
                    "📈 Grafik Tren Utama",
                    "📅 Perbandingan Tiap Tahun (YoY)",
                    "📋 Tabel Data Hasil"
                ])

                with pred_tab_tren:
                    st.subheader("Visualisasi Runtun Waktu Harga")
                    fig_main = go.Figure()

                    if show_hist_toggle:
                        fig_main.add_trace(go.Scatter(
                            x=dfp_known["TANGGAL"], y=dfp_known["HARGA"],
                            mode="lines", name="Data Historis", line=dict(color="#2b5c8f", width=2)
                        ))

                    if show_pred_toggle:
                        fig_main.add_trace(go.Scatter(
                            x=dfp_forecast["TANGGAL"], y=dfp_forecast["PREDIKSI"],
                            mode="lines+markers", name="Hasil Prediksi", line=dict(color="#e67e22", width=2.5, dash="dash")
                        ))

                    fig_main.update_layout(
                        hovermode="x unified",
                        xaxis_title="Tanggal", yaxis_title="Harga (IDR)",
                        margin=dict(l=40, r=40, t=20, b=40),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_main, use_container_width=True)

                with pred_tab_yoy:
                    st.subheader("Analisis Musiman Berdasarkan Periode Tahunan (Overlay Tahun ke Tahun)")

                    dfp_yoy = dfp_known.copy()
                    dfp_yoy["TANGGAL_SINTETIS"] = pd.to_datetime(2000 * 1000 + dfp_yoy["HARI_KE"], format="%Y%j")
                    dfp_yoy["TANGGAL_LENGKAP"] = dfp_yoy["TANGGAL"].dt.strftime("%d %B %Y")
                    dfp_yoy["RETURN"] = dfp_yoy["HARGA"].pct_change() * 100

                    fig_yoy = px.line(
                        dfp_yoy, x="TANGGAL_SINTETIS", y="HARGA", color="TAHUN",
                        labels={
                            "TANGGAL_SINTETIS": "Periode (Bulan-Tanggal)",
                            "HARGA": "Harga (IDR)",
                            "TAHUN": "Tahun",
                            "TANGGAL_LENGKAP": "Tanggal Lengkap",
                            "RETURN": "Return Harian (%)"
                        },
                        hover_data={"TANGGAL_LENGKAP": True, "TANGGAL_SINTETIS": False, "RETURN": ":.2f"},
                        color_discrete_sequence=px.colors.qualitative.Safe
                    )
                    fig_yoy.update_xaxes(tickformat="%d %b", dtick="M1")
                    fig_yoy.update_layout(hovermode="closest", margin=dict(l=40, r=40, t=20, b=40))
                    st.plotly_chart(fig_yoy, use_container_width=True)

                    st.markdown("##### 📉 Return Harian (%) Antar Tahun")
                    st.caption("Persentase perubahan harga dari hari sebelumnya — makin lebar lonjakannya, makin tinggi volatilitas di periode itu.")
                    fig_return = px.line(
                        dfp_yoy, x="TANGGAL_SINTETIS", y="RETURN", color="TAHUN",
                        labels={
                            "TANGGAL_SINTETIS": "Periode (Bulan-Tanggal)",
                            "RETURN": "Return Harian (%)",
                            "TAHUN": "Tahun",
                            "TANGGAL_LENGKAP": "Tanggal Lengkap"
                        },
                        hover_data={"TANGGAL_LENGKAP": True, "TANGGAL_SINTETIS": False},
                        color_discrete_sequence=px.colors.qualitative.Safe
                    )
                    fig_return.add_hline(y=0, line_dash="dot", line_color="gray")
                    fig_return.update_xaxes(tickformat="%d %b", dtick="M1")
                    fig_return.update_layout(hovermode="closest", margin=dict(l=40, r=40, t=20, b=40))
                    st.plotly_chart(fig_return, use_container_width=True)

                    st.markdown("##### 📊 Ringkasan Volatilitas Return per Tahun")
                    dfp_ret_summary = (
                        dfp_yoy.groupby("TAHUN")["RETURN"]
                        .agg(**{"Rata-rata Return (%)": "mean", "Volatilitas / Std Return (%)": "std"})
                        .round(2)
                        .reset_index()
                        .rename(columns={"TAHUN": "Tahun"})
                    )
                    st.dataframe(dfp_ret_summary, use_container_width=True, hide_index=True)

                with pred_tab_data:
                    st.subheader(f"Data Hasil Prediksi {forecast_horizon} Hari Kedepan")
                    dfp_display = dfp_forecast.copy()
                    dfp_display["TANGGAL"] = pd.to_datetime(dfp_display["TANGGAL"]).dt.strftime("%Y-%m-%d")
                    dfp_display["PREDIKSI"] = dfp_display["PREDIKSI"].apply(lambda x: f"{x:.2f}")
                    st.dataframe(dfp_display, use_container_width=True)

                    csv_pred = dfp_forecast.copy()
                    csv_pred["TANGGAL"] = pd.to_datetime(csv_pred["TANGGAL"]).dt.strftime("%Y-%m-%d")
                    st.download_button(
                        "⬇️ Unduh Hasil Prediksi (CSV)",
                        data=csv_pred.to_csv(index=False).encode("utf-8"),
                        file_name="hasil_prediksi_harga.csv",
                        mime="text/csv",
                        key="pred_download"
                    )

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data prediksi: {str(e)}")

# TAB 10: NET INCOME
with tab10:
    st.markdown("### 🧮 Net Income (Laba Bersih)")
    st.caption(
        "Net Income dihitung dari **Total Laba** (Lapak + Lapak Luar + Ekspedisi + Tanaman Panen) "
        "dikurangi **Pengeluaran Arus Kas** dari kategori Kantor, Beban, Angsuran Mobil, Gaji Kantor, "
        "dan Barang Kantor. Mengikuti filter rentang tanggal di sidebar."
    )

    URUTAN_PENCOCOKAN = [
        ("Angsuran Mobil", ["ANGSURAN MOBIL", "CICILAN MOBIL"]),
        ("Gaji Kantor",    ["GAJI KANTOR"]),
        ("Barang Kantor",  ["BARANG KANTOR", "PERLENGKAPAN KANTOR", "ATK"]),
        ("Beban",          ["BEBAN"]),
        ("Kantor",         ["KANTOR"]),
    ]
    URUTAN_TAMPIL = ["Kantor", "Beban", "Angsuran Mobil", "Gaji Kantor", "Barang Kantor"]

    ada_kas = (not df_kas_raw.empty) and ("JENIS" in df_kas_raw.columns) and ("KAS KELUAR" in df_kas_raw.columns)

    if ada_kas:
        semua_jenis = sorted(df_kas_raw["JENIS"].dropna().unique().tolist())
        pemetaan_otomatis = {}
        for nama, keywords in URUTAN_PENCOCOKAN:
            for jv in semua_jenis:
                if jv in pemetaan_otomatis:
                    continue
                if any(kw in jv for kw in keywords):
                    pemetaan_otomatis[jv] = nama
        deteksi_otomatis = {nama: [jv for jv, n in pemetaan_otomatis.items() if n == nama] for nama in URUTAN_TAMPIL}
    else:
        semua_jenis = []
        deteksi_otomatis = {nama: [] for nama in URUTAN_TAMPIL}

    pilihan_kategori = deteksi_otomatis

    if not df_kas.empty and "JENIS" in df_kas.columns and "KAS KELUAR" in df_kas.columns:
        pengeluaran_per_kategori = {
            nama: (df_kas[df_kas["JENIS"].isin(pilihan_kategori[nama])]["KAS KELUAR"].sum() if pilihan_kategori[nama] else 0.0)
            for nama in URUTAN_TAMPIL
        }
        jenis_terpilih_semua = [jv for nama in URUTAN_TAMPIL for jv in pilihan_kategori[nama]]
        df_kas_deduksi = df_kas[df_kas["JENIS"].isin(jenis_terpilih_semua)].copy() if jenis_terpilih_semua else df_kas.iloc[0:0].copy()
    else:
        pengeluaran_per_kategori = {nama: 0.0 for nama in URUTAN_TAMPIL}
        df_kas_deduksi = pd.DataFrame()

    total_pengeluaran_kategori = sum(pengeluaran_per_kategori.values())
    laba_bersih = total_laba - total_pengeluaran_kategori
    margin_bersih = (laba_bersih / total_omzet * 100) if total_omzet > 0 else None

    kelas_hero_net = "hero-green" if laba_bersih >= 0 else "hero-red"
    st.markdown(
        '<div class="hero-row">'
        + hero_card("📈 Total Laba", rp(total_laba), "hero-blue")
        + hero_card("💸 Total Deduksi Arus Kas", rp(total_pengeluaran_kategori), "hero-orange")
        + hero_card("🧮 Net Income", rp(laba_bersih), kelas_hero_net)
        + '</div>',
        unsafe_allow_html=True
    )
    if margin_bersih is not None:
        st.caption(f"📐 Net Margin (Net Income ÷ Total Omzet): **{margin_bersih:.1f}%**")

    if laba_bersih >= 0:
        st.success(f"✅ Net Income periode ini **positif**: {rp(laba_bersih)}. Total laba masih mampu menutup seluruh pengeluaran kategori kantor.")
    else:
        st.error(f"⚠️ Net Income periode ini **negatif**: {rp(laba_bersih)}. Pengeluaran kategori kantor melebihi total laba yang dihasilkan pada periode ini.")

    if total_pengeluaran_kategori > 0:
        kategori_terbesar = max(pengeluaran_per_kategori, key=pengeluaran_per_kategori.get)
        pct_terbesar = pengeluaran_per_kategori[kategori_terbesar] / total_pengeluaran_kategori * 100
        st.caption(f"💡 Kategori deduksi terbesar: **{kategori_terbesar}** — {rp(pengeluaran_per_kategori[kategori_terbesar])} ({pct_terbesar:.0f}% dari total deduksi).")

    st.write("")
    st.divider()

    section_heading("🌊 Alur Perhitungan: Total Laba → Net Income")

    labels_wf   = ["Total Laba"] + URUTAN_TAMPIL + ["Net Income"]
    measures_wf = ["absolute"] + ["relative"] * len(URUTAN_TAMPIL) + ["total"]
    values_wf   = [total_laba] + [-pengeluaran_per_kategori[n] for n in URUTAN_TAMPIL] + [0]
    text_wf     = [rp_short(total_laba)] + [f"-{rp_short(pengeluaran_per_kategori[n])}" for n in URUTAN_TAMPIL] + [rp_short(laba_bersih)]

    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures_wf,
        x=labels_wf,
        y=values_wf,
        text=text_wf,
        textposition="outside",
        textfont=dict(size=13),
        connector=dict(line=dict(color="#b0b8c7", width=1.2)),
        increasing=dict(marker=dict(color="#2ca02c")),
        decreasing=dict(marker=dict(color="#d62728")),
        totals=dict(marker=dict(color="#1f3864")),
    ))

    running = total_laba
    running_series = [running]
    for n in URUTAN_TAMPIL:
        running = running - pengeluaran_per_kategori[n]
        running_series.append(running)
    wf_low  = min(0, min(running_series))
    wf_high = max(max(running_series), total_laba)
    wf_span = max(wf_high - wf_low, 1)
    fig_wf.update_yaxes(range=[wf_low - wf_span * 0.15, wf_high + wf_span * 0.2])
    fig_wf.update_traces(cliponaxis=False)
    fig_wf.update_layout(
        title="Bridge Chart Net Income",
        yaxis_title="Rupiah", showlegend=False, height=480,
        yaxis_tickformat=","
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    st.divider()

    section_heading("📊 Rincian Pengeluaran per Kategori")
    tabel_kat = pd.DataFrame({
        "Kategori": URUTAN_TAMPIL,
        "Jumlah JENIS Terpilih": [len(pilihan_kategori[n]) for n in URUTAN_TAMPIL],
        "Total Pengeluaran": [rp(pengeluaran_per_kategori[n]) for n in URUTAN_TAMPIL],
    })
    st.dataframe(tabel_kat, use_container_width=True, hide_index=True)

    st.divider()

    jenis_ke_kategori = {jv: nama for nama in URUTAN_TAMPIL for jv in pilihan_kategori[nama]}

    section_heading("📄 Detail Transaksi Arus Kas (Kategori Terpilih)")
    if df_kas_deduksi.empty:
        st.info("Tidak ada transaksi arus kas yang cocok dengan kategori terpilih pada periode ini.")
    else:
        df_detail = df_kas_deduksi.copy()
        df_detail["Kategori Net Income"] = df_detail["JENIS"].map(jenis_ke_kategori)
        csv_detail = df_detail.drop(columns=["Tanggal_Kas"], errors="ignore").copy()

        st.markdown(f"**Total Transaksi: {len(df_detail)} baris**")
        cols_show = [c for c in df_detail.columns if c != "Tanggal_Kas"]
        st.dataframe(
            format_money_table(df_detail[cols_show], extra_keywords=["KAS", "SALDO", "MASUK", "KELUAR"]),
            use_container_width=True, hide_index=True
        )

        st.download_button(
            "⬇️ Unduh Detail Transaksi (CSV)",
            data=csv_detail.to_csv(index=False).encode("utf-8"),
            file_name="detail_deduksi_net_income.csv",
            mime="text/csv",
            key="net_income_download"
        )

# TAB 11: GREEN HOUSE
with tab11:
    st.markdown("### 🌿 Green House")

    total_gh_bahan       = df_gh_bahan_raw["Total"].sum()      if not df_gh_bahan_raw.empty      and "Total" in df_gh_bahan_raw.columns      else 0
    total_gh_pemupukan   = df_gh_pemupukan_raw["Total"].sum()  if not df_gh_pemupukan_raw.empty  and "Total" in df_gh_pemupukan_raw.columns  else 0
    total_gh_tenaga      = df_gh_tenaga_raw["Total"].sum()     if not df_gh_tenaga_raw.empty     and "Total" in df_gh_tenaga_raw.columns     else 0
    total_gh_tanaman_biaya = df_gh_tanaman_biaya["Kas Keluar"].sum() if not df_gh_tanaman_biaya.empty and "Kas Keluar" in df_gh_tanaman_biaya.columns else 0

    if not df_gh_kas_biaya.empty and "Kategori" in df_gh_kas_biaya.columns and "Kas Keluar" in df_gh_kas_biaya.columns:
        mask_bangunan_gh = df_gh_kas_biaya["Kategori"].astype(str).str.strip().str.lower() == "bangunan"
        biaya_bangunan_gh = df_gh_kas_biaya.loc[mask_bangunan_gh, "Kas Keluar"].sum()
    else:
        biaya_bangunan_gh = 0

    total_biaya_berjalan_gh = biaya_bangunan_gh + total_gh_tanaman_biaya

    kas_masuk_gh_total  = df_gh_kas_raw["Kas Masuk"].sum()  if not df_gh_kas_raw.empty and "Kas Masuk"  in df_gh_kas_raw.columns else 0
    kas_keluar_gh_total = df_gh_kas_raw["Kas Keluar"].sum() if not df_gh_kas_raw.empty and "Kas Keluar" in df_gh_kas_raw.columns else 0
    saldo_kas_gh        = kas_masuk_gh_total - kas_keluar_gh_total
    kelas_hero_kas_gh   = "hero-green" if saldo_kas_gh >= 0 else "hero-red"

    st.markdown(
        '<div class="hero-row">'
        + hero_card("🌿 Total Biaya Berjalan", rp(total_biaya_berjalan_gh), "hero-blue")
        + hero_card("💰 Kas Green House", rp(saldo_kas_gh), kelas_hero_kas_gh)
        + '</div>',
        unsafe_allow_html=True
    )
    st.write("")

    with st.container(border=True):
        st.markdown('<div class="income-card-title">📦 Rincian Total per Kategori</div>', unsafe_allow_html=True)
        gc1, gc2, gc3 = st.columns(3)
        gc1.metric("Bahan", rp(total_gh_bahan))
        gc2.metric("Pemupukan", rp(total_gh_pemupukan))
        gc3.metric("Tenaga", rp(total_gh_tenaga))

    st.divider()

    section_heading("🏡 Jumlah GH per Lokasi")

    sumber_ket_lokasi = None
    if not df_gh_lokasi_raw.empty and "Lokasi" in df_gh_lokasi_raw.columns and "Jumlah GH" in df_gh_lokasi_raw.columns:
        tabel_gh_lokasi = df_gh_lokasi_raw[["Lokasi", "Jumlah GH"]].copy()
    else:
        frames_for_count = [
            d[["Lokasi", "Nomor GH"]] for d in [df_gh_bahan_raw, df_gh_pemupukan_raw, df_gh_tenaga_raw]
            if not d.empty and "Lokasi" in d.columns and "Nomor GH" in d.columns
        ]
        if frames_for_count:
            df_gab_gh = pd.concat(frames_for_count, ignore_index=True)
            tabel_gh_lokasi = (
                df_gab_gh.groupby("Lokasi")["Nomor GH"].nunique()
                .reset_index().rename(columns={"Nomor GH": "Jumlah GH"})
            )
            sumber_ket_lokasi = "⚠️ Sheet LOKASI tidak terbaca sesuai format yang diharapkan — jumlah GH di atas dihitung otomatis dari Nomor GH unik pada data Bahan/Pemupukan/Tenaga."
        else:
            tabel_gh_lokasi = pd.DataFrame(columns=["Lokasi", "Jumlah GH"])

    if not tabel_gh_lokasi.empty:
        total_unit_gh = tabel_gh_lokasi["Jumlah GH"].sum()
        st.metric("Total Unit GH (Seluruh Lokasi)", f"{total_unit_gh:,.0f} GH")
        if sumber_ket_lokasi:
            st.caption(sumber_ket_lokasi)
        st.dataframe(tabel_gh_lokasi, use_container_width=True, hide_index=True)
    else:
        st.info("Data jumlah GH per lokasi tidak ditemukan.")

    st.divider()

    section_heading("🔍 Rincian Pengeluaran Green House")

    kategori_opts = ["Bahan", "Pemupukan", "Tenaga"]
    sel_kategori = st.multiselect("Kategori", kategori_opts, default=kategori_opts, key="gh_filter_kategori")

    def _gh_collect_unique(col_name, *dfs):
        vals = set()
        for d in dfs:
            if d is not None and not d.empty and col_name in d.columns:
                vals.update(d[col_name].dropna().astype(str).unique().tolist())
        return sorted(vals)

    lokasi_opts      = _gh_collect_unique("Lokasi", df_gh_bahan_raw, df_gh_pemupukan_raw, df_gh_tenaga_raw)
    siklus_opts      = _gh_collect_unique("Siklus", df_gh_bahan_raw, df_gh_pemupukan_raw, df_gh_tenaga_raw)
    nomor_gh_opts    = _gh_collect_unique("Nomor GH", df_gh_bahan_raw, df_gh_pemupukan_raw, df_gh_tenaga_raw)
    subkategori_opts = _gh_collect_unique("Sub Kategori", df_gh_bahan_raw)

    gf1, gf2, gf3 = st.columns(3)
    sel_lokasi   = gf1.multiselect("Lokasi", lokasi_opts, default=lokasi_opts, key="gh_filter_lokasi")
    sel_siklus   = gf2.multiselect("Siklus", siklus_opts, default=siklus_opts, key="gh_filter_siklus")
    sel_nomor_gh = gf3.multiselect("Nomor GH", nomor_gh_opts, default=nomor_gh_opts, key="gh_filter_nomor_gh")

    sel_subkategori = subkategori_opts
    if "Bahan" in sel_kategori and subkategori_opts:
        sel_subkategori = st.multiselect(
            "Sub Kategori (khusus kategori Bahan)", subkategori_opts, default=subkategori_opts,
            key="gh_filter_subkategori"
        )

    frames = []
    if "Bahan" in sel_kategori and not df_gh_bahan_raw.empty:
        tmp = df_gh_bahan_raw.copy(); tmp["Kategori"] = "Bahan"; frames.append(tmp)
    if "Pemupukan" in sel_kategori and not df_gh_pemupukan_raw.empty:
        tmp = df_gh_pemupukan_raw.copy(); tmp["Kategori"] = "Pemupukan"; frames.append(tmp)
    if "Tenaga" in sel_kategori and not df_gh_tenaga_raw.empty:
        tmp = df_gh_tenaga_raw.copy(); tmp["Kategori"] = "Tenaga"; frames.append(tmp)

    df_gh_gabungan = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    df_gh_filtered = df_gh_gabungan.copy()
    if not df_gh_filtered.empty:
        if "Lokasi" in df_gh_filtered.columns:
            df_gh_filtered = df_gh_filtered[df_gh_filtered["Lokasi"].astype(str).isin(sel_lokasi)]
        if "Siklus" in df_gh_filtered.columns:
            df_gh_filtered = df_gh_filtered[df_gh_filtered["Siklus"].astype(str).isin(sel_siklus)]
        if "Nomor GH" in df_gh_filtered.columns:
            df_gh_filtered = df_gh_filtered[df_gh_filtered["Nomor GH"].astype(str).isin(sel_nomor_gh)]
        if "Sub Kategori" in df_gh_filtered.columns and "Kategori" in df_gh_filtered.columns:
            mask_bahan = df_gh_filtered["Kategori"] == "Bahan"
            mask_keep = (~mask_bahan) | (df_gh_filtered["Sub Kategori"].astype(str).isin(sel_subkategori))
            df_gh_filtered = df_gh_filtered[mask_keep]

    st.divider()

    total_gh_filtered = df_gh_filtered["Total"].sum() if not df_gh_filtered.empty and "Total" in df_gh_filtered.columns else 0
    st.metric("💰 Total Pengeluaran (Sesuai Filter)", rp(total_gh_filtered))

    if df_gh_filtered.empty:
        st.info("Tidak ada data yang cocok dengan kombinasi filter yang dipilih.")
    else:
        cols_priority = [c for c in ["Kategori", "Lokasi", "Siklus", "Nomor GH", "Sub Kategori"] if c in df_gh_filtered.columns]
        cols_rest = [c for c in df_gh_filtered.columns if c not in cols_priority]
        df_gh_display = df_gh_filtered[cols_priority + cols_rest].dropna(axis=1, how="all")
        st.markdown(f"**Total Baris: {len(df_gh_filtered)}**")
        st.dataframe(format_money_table(df_gh_display), use_container_width=True, hide_index=True)

    st.divider()
    section_heading("💸 Rincian Kas Green House")

    if df_gh_kas_raw.empty:
        st.info("Data Kas Green House (sheet GREEN HOUSE KAS) kosong atau tidak ditemukan.")
    else:
        rk1, rk2, rk3 = st.columns(3)
        rk1.metric("🟩 Total Kas Masuk", rp(kas_masuk_gh_total))
        rk2.metric("🟥 Total Kas Keluar", rp(kas_keluar_gh_total))
        rk3.metric("💰 Saldo Kas", rp(saldo_kas_gh))

        if "Kategori" in df_gh_kas_raw.columns:
            st.markdown("**📊 Ringkasan per Kategori**")
            ringkasan_kategori_gh = df_gh_kas_raw.groupby("Kategori").agg(
                Masuk=("Kas Masuk", "sum"), Keluar=("Kas Keluar", "sum")
            ).reset_index()
            ringkasan_kategori_gh["Selisih"] = ringkasan_kategori_gh["Masuk"] - ringkasan_kategori_gh["Keluar"]
            ringkasan_kategori_gh = ringkasan_kategori_gh.sort_values("Selisih", ascending=False).rename(
                columns={"Masuk": "Kas Masuk", "Keluar": "Kas Keluar"}
            )
            ringkasan_display_gh = ringkasan_kategori_gh.copy()
            for c in ["Kas Masuk", "Kas Keluar", "Selisih"]:
                ringkasan_display_gh[c] = ringkasan_display_gh[c].apply(rp)
            st.dataframe(ringkasan_display_gh, use_container_width=True, hide_index=True)

        st.markdown("**🔍 Rincian Transaksi**")
        st.dataframe(
            format_money_table(df_gh_kas_raw.drop(columns=["Tanggal_Lengkap"], errors="ignore"), extra_keywords=["KAS", "MASUK", "KELUAR"]),
            use_container_width=True, hide_index=True
        )
