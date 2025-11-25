import streamlit as st
import numpy as np
import pickle
import re
import plotly.express as px
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(page_title="📩 Spam Detection", layout="wide")
st.markdown("<h1 style='text-align: center;'>📩 Deteksi Pesan Spam atau Ham</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Made by: <b>Hilman Singgih Wicaksana, M.Kom.</b></p>", unsafe_allow_html=True)
st.markdown("---")

# =======================
# LOAD MODEL & ASSETS
# =======================
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('models/LSTM Model_With Optimization.h5')
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('models/encoder.pickle', 'rb') as handle:
        encoder = pickle.load(handle)
    word2vec_model = Word2Vec.load('models/word2vec.model')
    return model, tokenizer, encoder, word2vec_model

model, tokenizer, encoder, word2vec_model = load_assets()

# =======================
# PREPROCESSING
# =======================
max_sequence_length = 64
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))
tokenizer_regex = RegexpTokenizer(r'[a-z]+')

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = tokenizer_regex.tokenize(text)
    stemmed = [stemmer.stem(word) for word in tokens]
    filtered = [word for word in stemmed if word not in stop_words and word != '']
    return ' '.join(filtered)

def predict(text):
    cleaned = preprocess(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_sequence_length, padding='post', truncating='post')
    predicted_proba = model.predict(padded)[0]
    all_labels = encoder.categories_[0].tolist()
    pred_idx = np.argmax(predicted_proba)
    predicted_label = all_labels[pred_idx]
    other_label = all_labels[1 - pred_idx]
    confidence_score = predicted_proba[pred_idx]
    return predicted_label, confidence_score, other_label

# =======================
# TABS LAYOUT
# =======================
tab1, tab2 = st.tabs(["📥 Input Kalimat", "📄 Input File"])

# =======================
# TAB 1 - INPUT KALIMAT
# =======================
with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📝 Masukkan Pesan SMS")
        with st.form(key='predict_form'):
            user_input = st.text_area("Teks pesan:", height=180, placeholder="Ketikkan kalimat di sini...")
            submitted = st.form_submit_button("🔍 Predict")

    with col2:
        st.markdown("### 📊 Hasil Prediksi")
        if submitted and user_input.strip() != "":
            label, confidence, other_label = predict(user_input)
            if label.lower() == "ham":
                st.markdown(f"<div style='background: green; border-left: 5px solid #28a745; padding: 0.5em;'>"
            f"✅ Pesan ini dikategorikan sebagai <b>{label.upper()}</b>"
            "</div>", unsafe_allow_html=True)
                label_color = '#28a745'
            else:
                st.markdown(f"<div style='background: #b22222; border-left: 5px solid #dc3545; padding: 0.5em;'>"
            f"⚠️ Pesan ini dikategorikan sebagai <b>{label.upper()}</b>"
            "</div>", unsafe_allow_html=True)
                label_color = '#dc3545'

            fig_data = pd.DataFrame({
                'Label': [label, other_label],
                'Confidence': [confidence, 1 - confidence]
            })
            fig = px.pie(
                fig_data,
                values='Confidence',
                names='Label',
                hole=0.6,
                color='Label',
                color_discrete_map={label: label_color, other_label: '#eeeeee'}
            )
            fig.update_traces(textinfo='none', hoverinfo='label+percent')
            fig.update_layout(
                showlegend=True,
                annotations=[dict(text=f"{confidence * 100:.1f}%", x=0.5, y=0.5, font_size=24, showarrow=False)],
                title_text="Confidence Score", title_x=0.5
            )
            st.plotly_chart(fig, width="stretch")

# =======================
# TAB 2 - INPUT FILE
# =======================
with tab2:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📄 Upload File")
        with st.form(key='csv_form'):
            uploaded_file = st.file_uploader("Unggah file di sini", type=["csv"])
            csv_submitted = st.form_submit_button("🔍 Predict")

    with col2:
        st.markdown("### 📊 Hasil Prediksi")
        if csv_submitted and uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'cleaned' not in df.columns:
                st.error("❌ Kolom `cleaned` tidak ditemukan dalam file.")
            else:
                sequences = tokenizer.texts_to_sequences(df['cleaned'].astype(str))
                padded = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
                predictions = model.predict(padded)
                pred_indices = np.argmax(predictions, axis=1)
                confidence_scores = np.max(predictions, axis=1)
                labels = encoder.categories_[0][pred_indices]

                df['Prediksi'] = labels
                df['Confidence'] = (confidence_scores * 100).round(2).astype(str) + " %"
                df_display = df.rename(columns={'cleaned': 'Pesan'})
                st.dataframe(df_display[['Pesan', 'Prediksi', 'Confidence']])

                csv_download = df_display.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Hasil Prediksi", data=csv_download, file_name="hasil_prediksi.csv", mime='text/csv')

    # =======================
    # 📊 DISTRIBUSI SPAM-HAM
    # =======================
    if csv_submitted and uploaded_file is not None and 'cleaned' in df.columns:
        count_data = df['Prediksi'].value_counts().reset_index()
        count_data.columns = ['Label', 'Jumlah']
        fig_bar = px.bar(
            count_data,
            x="Label",
            y="Jumlah",
            text="Jumlah",
            color="Label",
            height=600,
            color_discrete_map={"ham": "#28a745", "spam": "#dc3545"},
            title="Distribusi Prediksi Spam vs Ham"
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(yaxis_title="Jumlah", xaxis_title="Kategori", showlegend=False)
        st.plotly_chart(fig_bar, width="stretch")

# =======================
# FOOTER
# =======================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size:14px;'>"
    "Copryight © 2025 <b>Hilman Singgih Wicaksana</b>"
    "</p>", unsafe_allow_html=True
)
