import streamlit as st
import pandas as pd

# Pastikan model sudah ada di session state
if 'model' not in st.session_state:
    st.warning("Model belum tersedia. Silakan jalankan evaluasi model terlebih dahulu di halaman Homepage.")
else:
    st.subheader("Input Manual Karakteristik Pribadi")

    model = st.session_state['model']
    scaler = st.session_state['scaler']
    feature_names = st.session_state['feature_names']
    le_target = st.session_state['le_target']

    # Form input
    input_dict = {}
    input_dict['Time_spent_Alone'] = st.number_input("Berapa jam anda menghabiskan waktu sendirian per hari?", 0, 20, step=1)
    input_dict['Stage_fear'] = 1 if st.radio("Apakah anda kurang percaya diri tampil di depan umum?", ["Ya", "Tidak"]) == "Ya" else 0
    input_dict['Social_event_attendance'] = st.number_input("Seberapa sering anda menghadiri acara sosial? (0–20)", 0, 20, step=1)
    input_dict['Going_outside'] = st.number_input("Seberapa sering anda keluar rumah? (0–7)", 0, 7, step=1)
    input_dict['Drained_after_socializing'] = 1 if st.radio("Apakah anda merasa lelah setelah bersosialisasi?", ["Ya", "Tidak"]) == "Ya" else 0
    input_dict['Friends_circle_size'] = st.number_input("Berapa banyak teman dekat yang anda miliki?", 0, 30, step=1)
    input_dict['Post_frequency'] = st.number_input("Seberapa sering anda memposting di media sosial? (0–15)", 0, 15, step=1)

    # Tombol prediksi
    if st.button("Cek Hasil Prediksi"):
        input_df = pd.DataFrame([input_dict])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        if le_target:
            personality_label = le_target.inverse_transform([prediction])[0]
        else:
            personality_label = prediction

        st.success(f"Hasil Prediksi Kepribadian Anda = {personality_label}")