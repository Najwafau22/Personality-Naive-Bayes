import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# === CSS Styling Sidebar Tanpa Lingkaran ===
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #008080; /* Teal */
        color: black;
    }
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        color: white;
        margin-bottom: 25px;
    }
    .menu-item {
        font-size: 18px;
        padding: 10px 0;
        color: white;
        cursor: pointer;
    }
    .menu-item:hover {
        color: #ffc107;
    }
    .selected {
        color: #ff4b4b;
        font-weight: bold;
    }
</style>

""", unsafe_allow_html=True)

# === Sidebar Title ===
st.sidebar.markdown('<div class="sidebar-title">📝 MENU</div>', unsafe_allow_html=True)

# === Inisialisasi session_state
if "page" not in st.session_state:
    st.session_state.page = "Homepage"

# === Fungsi tombol menu sidebar
def menu_button(label, icon, page_name):
    selected = st.session_state.page == page_name
    css_class = "menu-item selected" if selected else "menu-item"
    clicked = st.sidebar.button(f"{icon} {label}", key=page_name)
    if clicked:
        st.session_state.page = page_name

# === Daftar Menu Sidebar ===
menu_button("Homepage", "🧠", "Homepage")
menu_button("Data", "📊", "Data")
menu_button("Cek Disini!", "📑", "Cek Disini!")

# === Jalankan file sesuai menu yang dipilih ===
if st.session_state.page == "Homepage":
    exec(open("homepage.py", encoding="utf-8").read())
elif st.session_state.page == "Data":
    exec(open("data.py", encoding="utf-8").read())
elif st.session_state.page == "Cek Disini!":
    exec(open("cekdisini.py", encoding="utf-8").read())