import streamlit as st
import pandas as pd
import requests
import zipfile
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Spam Detector", page_icon="🚫")

# --- APP HEADER ---
st.title("🚫 Spam Message Classifier")
st.markdown("Enter a message below to see if it's **Spam** or **Ham (Legitimate)**.")

# --- DATA & MODEL CACHING ---
# We use @st.cache_data so the model doesn't retrain every time you click a button
@st.cache_resource
def load_and_train_model():
    # Load Data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open('SMSSpamCollection'), sep='\t', names=['label', 'message'])
    
    # Preprocess
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Vectorize
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(df['message'])
    y = df['label_num']
    
    # Train
    model = MultinomialNB()
    model.fit(X, y)
    
    return tfidf, model

tfidf, model = load_and_train_model()

# --- USER INTERFACE ---
user_input = st.text_area("Paste your message here:", placeholder="e.g., Congratulations! You've won a prize...")

if st.button("Analyze Message"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Prediction logic
        data = tfidf.transform([user_input])
        prediction = model.predict(data)
        result = "SPAM" if prediction[0] == 1 else "HAM (Legitimate)"
        
        # Display Result
        if result == "SPAM":
            st.error(f"Prediction: **{result}**")
            st.info("This message looks suspicious. It likely contains common spam triggers.")
        else:
            st.success(f"Prediction: **{result}**")
            st.info("This message looks safe!")

# --- SIDEBAR INFO ---
st.sidebar.header("About")
st.sidebar.info("This app uses a Multinomial Naive Bayes classifier trained on the UCI SMS Spam Collection dataset.")