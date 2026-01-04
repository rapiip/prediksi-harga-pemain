import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Football Player Valuation Tool",
    page_icon="⚽",
    layout="wide"
)

# ==========================================
# 2. LOAD DATA & MODEL
# ==========================================
@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv('data_pemain_siap_pakai.csv')
    return df

@st.cache_resource
def load_model():
    # Load model machine learning
    model = joblib.load('model_rf_market_value.pkl')
    return model

try:
    df = load_data()
    model = load_model()
except FileNotFoundError:
    st.error("File tidak ditemukan! Pastikan 'data_pemain_siap_pakai.csv' dan 'model_rf_market_value.pkl' ada di satu folder.")
    st.stop()

# ==========================================
# 3. SIDEBAR (PENCARIAN PEMAIN)
# ==========================================
st.sidebar.header("🔍 Cari Pemain")

# Filter Liga (Opsional)
liga_pilihan = st.sidebar.selectbox("Pilih Liga:", df['league_name'].unique())
df_liga = df[df['league_name'] == liga_pilihan]

# Dropdown Nama Pemain
nama_pemain = st.sidebar.selectbox("Pilih Nama Pemain:", df_liga['short_name'].unique())

# Ambil data baris pemain tersebut
player_data = df[df['short_name'] == nama_pemain].iloc[0]

# ==========================================
# 4. PROSES PREDIKSI & LOGIKA STATUS
# ==========================================
# Definisikan kolom fitur X (HARUS SAMA PERSIS dengan saat training)
feature_cols = [
    'age', 'pace', 'movement_acceleration', 'movement_sprint_speed',
    'movement_agility', 'movement_reactions', 'movement_balance',
    'physic', 'power_stamina', 'power_strength', 'mentality_composure',
    'shooting', 'attacking_finishing', 'power_shot_power', 'power_long_shots',
    'passing', 'attacking_short_passing', 'mentality_vision', 'attacking_crossing',
    'dribbling', 'skill_dribbling', 'skill_ball_control',
    'defending', 'defending_standing_tackle', 'mentality_interceptions'
]

# Siapkan data input
input_data = pd.DataFrame([player_data[feature_cols]])

# Lakukan Prediksi
predicted_value = model.predict(input_data)[0]
actual_value = player_data['value_eur']

# --- LOGIKA BARU: MENGHITUNG PERSENTASE SELISIH ---
diff = predicted_value - actual_value
persentase_diff = (diff / actual_value) * 100 if actual_value > 0 else 0

# Ambang Batas Toleransi (Threshold) = 10%
threshold = 10.0

if persentase_diff > threshold:
    status = "Undervalued (Layak Beli)"
    warna_status = "green"
    pesan_tambahan = "Statistik pemain ini jauh di atas harganya. Potensi keuntungan tinggi!"
elif persentase_diff < -threshold:
    status = "Overvalued (Terlalu Mahal)"
    warna_status = "red"
    pesan_tambahan = "Performa statistik tidak sebanding dengan harganya yang tinggi."
else:
    status = "Fair Value (Harga Wajar)"
    warna_status = "blue"
    pesan_tambahan = "Harga pasar saat ini sudah sesuai dengan kualitas statistik pemain."

# ==========================================
# 5. TAMPILAN UTAMA (DASHBOARD)
# ==========================================
st.title(f"⚽ Analisis Harga: {player_data['short_name']}")
st.markdown(f"**Klub:** {player_data['club_name']} | **Umur:** {player_data['age']} Tahun")

# Tampilkan Angka Kunci
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Harga Asli (Market)", value=f"€ {actual_value:,.0f}")

with col2:
    st.metric(label="Harga Prediksi (Fair Value)", value=f"€ {predicted_value:,.0f}", 
              delta=f"{diff:,.0f} ({persentase_diff:.1f}%)")

with col3:
    st.markdown(f"### Status:")
    # Menampilkan status dengan warna dinamis
    st.markdown(f":{warna_status}[**{status}**]")

st.info(pesan_tambahan) # Menampilkan pesan penjelasan status

st.divider()

# Tampilkan Penjelasan Visual
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Atribut Penentu Harga")
    # Tampilkan stats utama pemain ini
    stats_to_show = input_data.T.reset_index()
    stats_to_show.columns = ['Atribut', 'Nilai']
    stats_to_show = stats_to_show.sort_values(by='Nilai', ascending=False).head(10)
    
    st.bar_chart(stats_to_show.set_index('Atribut'))

with col_right:
    st.markdown("### ℹ️ Panduan Status")
    st.caption("""
    **Logika Analisis:**
    * **Fair Value:** Selisih harga prediksi & asli < 10%. (Dianggap wajar).
    * **Undervalued:** Prediksi lebih tinggi > 10% dari harga asli.
    * **Overvalued:** Prediksi lebih rendah > 10% dari harga asli.
    """)