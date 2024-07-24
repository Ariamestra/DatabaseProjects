# DatabaseProjects
Sure, here is the code formatted for a README file:

```markdown
# Occupations Analysis and Clustering

This project provides an analysis and clustering of occupations and job families using Natural Language Processing (NLP) and Machine Learning techniques. The dataset used in this project is available at [Occupations Dataset](https://raw.githubusercontent.com/Ariamestra/Occupations/main/Occupations/occupations.csv).

## Installation

To install the required packages, run the following command:
```bash
!pip install nltk pandas scikit-learn matplotlib seaborn wordcloud
```

## Importing Libraries

```python
import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('punkt')
```

## Load Dataset

```python
data_URL = 'https://raw.githubusercontent.com/Ariamestra/Occupations/main/Occupations/occupations.csv'
df = pd.read_csv(data_URL)
print(f"Shape: {df.shape}")
df.head()
```

## Data Description

```python
print("Describe the occupations:")
print(df['Occupation'].describe())
print('-' * 50)
print("Describe the job family:")
print(df['Job Family'].describe())
```

## Visualize Top Occupations

```python
occupation_counts = df['Occupation'].value_counts()

top_n = 20
top_occupations = occupation_counts.nlargest(top_n)

plt.figure(figsize=(12, 8))
top_occupations.plot(kind='bar')
plt.title('Top 20 Occupations')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()  
plt.show()
```

## Visualize Job Family Distribution

```python
job_family_counts = df['Job Family'].value_counts()

plt.figure(figsize=(12, 10))
plt.pie(job_family_counts, labels=job_family_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Job Family Distribution')
plt.axis('equal')  
plt.show()
```

## Train-Test Split

```python
X = df['Occupation']
y = df['Job Family']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Vectorization using TF-IDF

```python
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

## Train a Random Forest Classifier

```python
clf = RandomForestClassifier()
clf.fit(X_train_tfidf, y_train)

y_pred = clf.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

## Preprocess Job Titles

```python
stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['Occupation'] = df['Occupation'].apply(preprocess)
```

## Vectorize Job Titles

```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Occupation'])
```

## Determine Optimal Number of Clusters

```python
wcss = []
for i in range(1, 21):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 21), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

optimal_clusters = 10
```

## KMeans Clustering

```python
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

for i in range(optimal_clusters):
    print(f"Cluster {i}:")
    print(df[df['Cluster'] == i]['Occupation'].values)
    print()

df.to_csv('clustered_job_families.csv', index=False)
```

## Conclusion

This project demonstrated how to perform NLP and machine learning on a dataset of occupations and job families. The analysis included visualization of top occupations, job family distribution, classification using Random Forest, and clustering using KMeans. The results were saved to a CSV file for further analysis.
```