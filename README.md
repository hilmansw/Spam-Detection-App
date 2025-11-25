# 📬 Spam Detection App

This project is built to detect **spam messages** using a **Long Short-Term Memory (LSTM)** model combined with **Word2Vec** as the word embedding technique. The model has been optimized using **Grid Search**, achieving a best accuracy of **95.65%**.

---

## 📊 Model Performance Summary

Comparison of model performance before and after optimization:

| Metrics   | Before Optimization | After Optimization |
| --------- | ------------------- | ------------------ |
| Precision | 0.8939              | 0.9365             |
| Recall    | 0.9833              | 0.9833             |
| F1-score  | 0.9365              | 0.9593             |
| Accuracy  | 0.9304              | 0.9565             |

---

## 📥 Clone the Repository

To download this project to your local machine:

```bash
git clone https://github.com/hilmansw/Spam-Detection-App.git
```

Navigate into the project directory:

```bash
cd Spam-Detection-App
```

---

## 🧪 (Optional but Recommended) Create a Virtual Environment

### **Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

### **macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 📦 Install Dependencies

Once the virtual environment is activated, install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Application

If the application uses Streamlit:

```bash
streamlit run app.py
```

If it uses a standard Python script:

```bash
python app.py
```

---

## 📚 Technologies Used

- Python 3.x
- TensorFlow / Keras
- Word2Vec (Gensim)
- LSTM Model
- Grid Search Optimization
- Streamlit (optional UI)

---

## 📝 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this project.
