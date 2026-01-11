import pandas as pd
import requests
import zipfile
import io
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# --- STEP 1: Load the Dataset ---
# Downloading the dataset directly from UCI Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
# The file is tab-separated (tsv) and has no header
df = pd.read_csv(z.open('SMSSpamCollection'), sep='\t', names=['label', 'message'])

print(f"Dataset Loaded: {df.shape[0]} messages.")

# --- STEP 2: Preprocessing ---
# Convert labels to binary (0 for ham, 1 for spam)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# --- STEP 3: Vectorization (Turning text into numbers) ---
# We use TF-IDF to penalize common words like 'the' and 'is'
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['message'])
y = df['label_num']

# --- STEP 4: Split & Train ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# --- STEP 5: Evaluation ---
y_pred = model.predict(X_test)
print("\n--- Model Performance ---")
print(classification_report(y_test, y_pred))

# --- STEP 6: Test it yourself! ---
def predict_message(msg):
    # Transform the input using the same TF-IDF used for training
    msg_transformed = tfidf.transform([msg])
    prediction = model.predict(msg_transformed)
    return "SPAM" if prediction[0] == 1 else "HAM (Legit)"

print("\n--- Custom Testing ---")
test_msgs = [
    "Hey, are we still meeting for coffee at 5?", 
    "WINNER! You have won a $1000 Walmart gift card. Click here to claim now!"
]

for m in test_msgs:
    print(f"Message: {m} \nResult: {predict_message(m)}\n")