import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.header("📊 Evaluasi Dataset & Naive Bayes")

# --- Path Dataset
DATASET_PATH = 'personality_dataset.csv'

# --- Load Data
try:
    data = pd.read_csv(DATASET_PATH)
    st.subheader("Data Awal (5 Baris Pertama)")
    st.dataframe(data.head())
except FileNotFoundError:
    st.error("❌ Dataset tidak ditemukan! Pastikan file berada di path yang sesuai.")
    st.stop()

# --- Imputasi Nilai Kosong
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

data[numerical_cols] = SimpleImputer(strategy='mean').fit_transform(data[numerical_cols])
data[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(data[categorical_cols])

# --- Label Encoding
le_target = None
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    if col == 'Personality':
        le_target = le

st.subheader("Data Setelah Encoding")
st.dataframe(data.head())

# --- Preprocessing Fitur
X = data.drop('Personality', axis=1)
y = data['Personality']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Model Training
model = BernoulliNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Evaluasi
st.subheader("Evaluasi Model Naive Bayes")
st.write("✅ Akurasi:", accuracy_score(y_test, y_pred))
st.text("📄 Classification Report:\n" + classification_report(y_test, y_pred))
st.write("🧩 Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

# --- Simpan ke Session State
st.session_state['model'] = model
st.session_state['scaler'] = scaler
st.session_state['le_target'] = le_target
st.session_state['feature_names'] = X.columns
