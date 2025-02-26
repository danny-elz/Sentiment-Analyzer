import matplotlib.pyplot as plt
import pandas as pd
import re
from dateutil import parser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


with open('stopwords.txt', 'r') as f:
    stopwords = [line.strip() for line in f]

df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)       # remove @ mentions
    text = re.sub(r'#\w+', '', text)       # remove hashtags
    text = re.sub(r'[^a-z\s]', '', text)   # keep only letters and whitespace
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords]
    return ' '.join(tokens)


df['clean_text'] = df['text'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])

df['sentiment'] = df['target'].apply(lambda x: 1 if x == 4 else 0)
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Test Accuracy: {accuracy:.4f}")


def parse_date(d):
    try:
        return parser.parse(d)
    except:
        return pd.NaT

df['parsed_date'] = df['date'].apply(parse_date)

start_date = pd.to_datetime('2009-01-01')
end_date = pd.to_datetime('2010-12-31')
mask = (df['parsed_date'] >= start_date) & (df['parsed_date'] <= end_date)
df_filtered = df[mask].copy()

df_filtered['predicted_sentiment'] = model.predict(vectorizer.transform(df_filtered['clean_text']))

df_filtered['date_only'] = df_filtered['parsed_date'].dt.date
trend = df_filtered.groupby('date_only')['predicted_sentiment'].mean()

plt.figure(figsize=(12, 6))
trend.plot()
plt.title("Average Sentiment Over Time (2009–2010)")
plt.xlabel("Date")
plt.ylabel("Avg Sentiment (0=Negative, 1=Positive)")
plt.show()
