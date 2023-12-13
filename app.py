import streamlit as st
import pandas as pd
import joblib

# Load the data
@st.cache
def load_data():
    df = pd.read_csv("https://gist.githubusercontent.com/hamam14/a2e95006a59388a5beee27530a92badf/raw/9efcaa5f78cb99d8ef19149b1a01eda8d501835c/penerimabantuandesa.csv")
    return df

df = load_data()

# Manual MinMaxScaler
def manual_minmax_scaling(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

# Define min and max values for each feature for manual scaling
min_max_values = {
    'ANGGOTA KELUARGA': (df['ANGGOTA KELUARGA'].min(), df['ANGGOTA KELUARGA'].max()),
    'JUMLAH PENGELUARAN per BULAN ( RP )': (df['JUMLAH PENGELUARAN per BULAN ( RP )'].min(), df['JUMLAH PENGELUARAN per BULAN ( RP )'].max()),
    'JUMLAH PENDAPATAN SEBULAN ( RP)': (df['JUMLAH PENDAPATAN SEBULAN ( RP)'].min(), df['JUMLAH PENDAPATAN SEBULAN ( RP)'].max()),
    'LUAS ASET LAHAN YANG DIMILIKI ( M2)': (df['LUAS ASET LAHAN YANG DIMILIKI ( M2)'].min(), df['LUAS ASET LAHAN YANG DIMILIKI ( M2)'].max()),
    'LUAS RUMAH TINGGAL (M2)': (df['LUAS RUMAH TINGGAL (M2)'].min(), df['LUAS RUMAH TINGGAL (M2)'].max())
}

# Model training
X = df[['ANGGOTA KELUARGA', 'JUMLAH PENGELUARAN per BULAN ( RP )', 'JUMLAH PENDAPATAN SEBULAN ( RP)', 'LUAS ASET LAHAN YANG DIMILIKI ( M2)', 'LUAS RUMAH TINGGAL (M2)']]
y = df['MENERIMA BANTUAN ']
model = KNeighborsClassifier(n_neighbors=10)
model.fit(X, y)

# Save the KNN model
joblib.dump(model, 'knn_model.joblib')

# Streamlit app
st.title('Prediction of Aid Recipient')

# Create input fields for user input
anggota_keluarga = st.number_input('Jumlah Anggota Keluarga', min_value=1)
pengeluaran_per_bulan = st.number_input('Jumlah Pengeluaran per Bulan (RP)', min_value=0)
pendapatan_sebulan = st.number_input('Jumlah Pendapatan per Bulan (RP)', min_value=0)
luas_aset = st.number_input('Luas Aset Lahan yang Dimiliki (M2)', min_value=0)
luas_rumah = st.number_input('Luas Rumah Tinggal (M2)', min_value=0)

# Predict function using manual scaling
def predict():
    scaled_input = [
        manual_minmax_scaling(anggota_keluarga, *min_max_values['ANGGOTA KELUARGA']),
        manual_minmax_scaling(pengeluaran_per_bulan, *min_max_values['JUMLAH PENGELUARAN per BULAN ( RP )']),
        manual_minmax_scaling(pendapatan_sebulan, *min_max_values['JUMLAH PENDAPATAN SEBULAN ( RP)']),
        manual_minmax_scaling(luas_aset, *min_max_values['LUAS ASET LAHAN YANG DIMILIKI ( M2)']),
        manual_minmax_scaling(luas_rumah, *min_max_values['LUAS RUMAH TINGGAL (M2)'])
    ]
    prediction = model.predict([scaled_input])
    return prediction[0]

if st.button('Predict'):
    prediction = predict()
    if prediction == 1:
        st.write('Hasil Prediksi: Menerima Bantuan')
    else:
        st.write('Hasil Prediksi: Tidak Menerima Bantuan')
