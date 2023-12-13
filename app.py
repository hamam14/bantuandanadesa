import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# Load the data
@st.cache
def load_data():
    df = pd.read_csv("https://gist.githubusercontent.com/hamam14/a2e95006a59388a5beee27530a92badf/raw/9efcaa5f78cb99d8ef19149b1a01eda8d501835c/penerimabantuandesa.csv")
    return df

df = load_data()

# Preprocess the data
fitur = ['ANGGOTA KELUARGA', 'JUMLAH PENGELUARAN per BULAN ( RP )', 'JUMLAH PENDAPATAN SEBULAN ( RP)', 'LUAS ASET LAHAN YANG DIMILIKI ( M2)', 'LUAS RUMAH TINGGAL (M2)']
scaler = MinMaxScaler()
df[fitur] = scaler.fit_transform(df[fitur])

# Model training
X = df[['ANGGOTA KELUARGA', 'JUMLAH PENGELUARAN per BULAN ( RP )', 'JUMLAH PENDAPATAN SEBULAN ( RP)', 'LUAS ASET LAHAN YANG DIMILIKI ( M2)', 'LUAS RUMAH TINGGAL (M2)']]
y = df['MENERIMA BANTUAN ']
model = KNeighborsClassifier(n_neighbors=10)
model.fit(X, y)

# Streamlit app
st.title('Prediction of Aid Recipient')

st.sidebar.header('Input Data')

# Create input fields for user input
anggota_keluarga = st.sidebar.number_input('Jumlah Anggota Keluarga', min_value=1)
pengeluaran_per_bulan = st.sidebar.number_input('Jumlah Pengeluaran per Bulan (RP)', min_value=0)
pendapatan_sebulan = st.sidebar.number_input('Jumlah Pendapatan per Bulan (RP)', min_value=0)
luas_aset = st.sidebar.number_input('Luas Aset Lahan yang Dimiliki (M2)', min_value=0)
luas_rumah = st.sidebar.number_input('Luas Rumah Tinggal (M2)', min_value=0)

# Predict function
def predict():
    input_data = scaler.transform([[anggota_keluarga, pengeluaran_per_bulan, pendapatan_sebulan, luas_aset, luas_rumah]])
    prediction = model.predict(input_data)
    return prediction[0]

if st.sidebar.button('Predict'):
    prediction = predict()
    if prediction == 1:
        st.write('Hasil Prediksi: Menerima Bantuan')
    else:
        st.write('Hasil Prediksi: Tidak Menerima Bantuan')
