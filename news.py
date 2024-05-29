import requests
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Fetch news articles using NewsAPI
def fetch_news(api_key):
    url = f'https://newsapi.org/v2/top-headlines?country=us&category=general&apiKey={api_key}'
    response = requests.get(url)
    articles = response.json()['articles']
    articles_data = [[article['title'], article['description'], article['content']] for article in articles]
    return pd.DataFrame(articles_data, columns=['title', 'description', 'content'])

# Step 2: Preprocess the text data
def preprocess_text(text):
    if text:
        text = re.sub(r'http\S+', '', text)  # remove URLs
        text = re.sub(r'\W', ' ', text)      # remove special characters
        text = text.lower()                  # convert to lowercase
        text = ' '.join([word for word in text.split() if word not in stop_words])
    else:
        text = ''
    return text

# Step 3: Load the 20 Newsgroups dataset and train the model
def train_model():
    newsgroups = fetch_20newsgroups(subset='all', categories=None, shuffle=True, random_state=42)
    newsgroups_data = [preprocess_text(text) for text in newsgroups.data]
    newsgroups_df = pd.DataFrame({'text': newsgroups_data, 'category': newsgroups.target})

    X = newsgroups_df['text']
    y = newsgroups_df['category']

    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=newsgroups.target_names, yticklabels=newsgroups.target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return model, vectorizer, newsgroups.target_names

# Step 4: Predict categories for new articles
def categorize_articles(articles_df, model, vectorizer, target_names):
    articles_df['cleaned_content'] = articles_df['content'].apply(preprocess_text)
    new_articles_vectorized = vectorizer.transform(articles_df['cleaned_content'])
    new_articles_pred = model.predict(new_articles_vectorized)
    articles_df['predicted_category'] = new_articles_pred
    articles_df['predicted_category_name'] = articles_df['predicted_category'].apply(lambda x: target_names[x])
    return articles_df

# Main script
if __name__ == "__main__":
    # Set your News API key
    api_key = 'f2669470edb64750964ebaacbe6224f5'
    
    # Download NLTK data
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    # Fetch and preprocess news articles
    articles_df = fetch_news(api_key)

    # Train the model on 20 Newsgroups dataset
    model, vectorizer, target_names = train_model()

    # Categorize the new articles
    categorized_articles_df = categorize_articles(articles_df, model, vectorizer, target_names)

    # Write the DataFrame with categorized articles to a CSV file
    categorized_articles_df.to_csv('News/news_categories.csv', index=False)

    # Display the DataFrame with categorized articles
    print(categorized_articles_df[['title', 'predicted_category_name']])
