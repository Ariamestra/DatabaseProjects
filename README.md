# Occupations Analysis and Clustering

This project provides an analysis and clustering of occupations and job families using Natural Language Processing (NLP) and Machine Learning techniques. The dataset used in this project is available at [Occupations Dataset](https://raw.githubusercontent.com/Ariamestra/Occupations/main/Occupations/occupations.csv).

## Installation
To install the required packages, run the following command:
```markdown
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


## Visualize Dataset


## Train a Random Forest Classifier



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
