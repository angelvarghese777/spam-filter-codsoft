import pandas as pd

print("script started.")

df = pd.read_csv("spam.csv", encoding="latin-1")

print("loaded the data.")

df = df.iloc[:, [0, 1]]
df.columns = ["label", "message"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

print(df.head())

print(df["label"].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# using countvectorizer because naive bayes works better with it
vectorizer = CountVectorizer(stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# using Multinomial naive bayes model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))



feature_names = vectorizer.get_feature_names_out()

# to see top spam words. optional but it helps to understand the naive bayes intuition or idea.
spam_word_probs = model.feature_log_prob_[1]

top_spam_words = sorted(
    zip(spam_word_probs, feature_names),
    reverse=True
)[:20]

print("\nTop Spam Words (Naive Bayes):")

for prob, word in top_spam_words:
    print(word, prob)

#test samples for custom testing


sample_messages = [
    "Congratulations! You have won a free iPhone. Click here now!",
    "URGENT! Claim your cash prize immediately.",
    "Win $5000 today. Call now.",
    "Free entry into our prize draw."
]

sample_vec = vectorizer.transform(sample_messages)

predictions = model.predict(sample_vec)
probabilities = model.predict_proba(sample_vec)

for msg, prob, pred in zip(sample_messages, probabilities, predictions):
    print("\nMessage:", msg)
    print("Ham probability:", prob[0])
    print("Spam probability:", prob[1])
    print("Prediction:", "SPAM" if pred == 1 else "HAM")