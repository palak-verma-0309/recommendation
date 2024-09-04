import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

file_path = 'job_descriptions.csv'
df = pd.read_csv(file_path)

df['Job Posting Date'] = pd.to_datetime(df['Job Posting Date'])
df['Job Posting Age'] = (datetime.now() - df['Job Posting Date']).dt.days

df.fillna('', inplace=True)
text_columns = ['skills', 'Experience', 'Qualifications', 'Job Title']
for col in text_columns:
    df[col] = df[col].astype(str)

df = df[df['Job Title'].str.strip() != '']

def preprocess_text(text):
    words = text.split()
    words = [word for word in words if word.lower() not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

df['Job Title'] = df['Job Title'].apply(preprocess_text)

# Initialize vectorizers which is ignorning english stop words
skills_vectorizer = TfidfVectorizer(stop_words='english')
experience_vectorizer = TfidfVectorizer(stop_words='english')
qualifications_vectorizer = TfidfVectorizer(stop_words='english')
job_title_vectorizer = TfidfVectorizer(stop_words='english')

#convert to sparse matrix
skills_matrix = skills_vectorizer.fit_transform(df['skills'])
experience_matrix = experience_vectorizer.fit_transform(df['Experience'])
qualifications_matrix = qualifications_vectorizer.fit_transform(df['Qualifications'])
try:
    job_title_matrix = job_title_vectorizer.fit_transform(df['Job Title'])
except Exception as e:
    print(f"Error during vectorization: {e}")

# Combine matrices
combined_matrix = hstack([skills_matrix, experience_matrix, qualifications_matrix, job_title_matrix])

job_idx = 0 
similarity_scores = cosine_similarity(combined_matrix[job_idx], combined_matrix).flatten()
similar_jobs_idx = np.argsort(-similarity_scores)[1:6]

recommended_jobs = df.iloc[similar_jobs_idx][['Job Id', 'Job Title', 'skills', 'Company', 'location']]
print(recommended_jobs)



