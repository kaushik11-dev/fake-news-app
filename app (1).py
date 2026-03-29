import streamlit as st
import joblib
import re
import base64  # ✅ new import for background image

# -----------------------
# Load model and vectorizer
# -----------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")
    tfidf = joblib.load("tfidf.joblib")
    return model, tfidf

model, tfidf = load_model()

# -----------------------
# Add background image
# -----------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    .stApp {{
        background-color: rgba(255, 255, 255, 0.88);
        border-radius: 16px;
        padding: 25px;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# Apply your local background image
add_bg_from_local("d1.jpg")

# -----------------------
# Text cleaning function
# -----------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------
# Streamlit app UI
# -----------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.markdown("<h4 style='color:Blue;'>Enter a news title and text to check if it's <b>Real</b> or <b>Fake</b></h4>", unsafe_allow_html=True)

title = st.text_input("📰 Enter News Title:")
text = st.text_area("🗞️ Enter News Text:")

if st.button("🔍 Check News"):
    full_text = title + " " + text
    clean = clean_text(full_text)

    X_tfidf = tfidf.transform([clean])
    prediction = model.predict(X_tfidf)[0]

    if prediction == 0:
        st.success("✅ This looks like **REAL** news.")
    else:
        st.error("🚨 This looks like **FAKE** news.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Developed by our team ⚡</p>", unsafe_allow_html=True)



