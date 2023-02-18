# Sentiment-Analysis-for-IMDB-Movie-Reviews

# About
The dataset contains a collection of 50,000 reviews from the IMDB Website with an equal number of positive and negative reviews. The task is to predict the polarity (positive or negative) of a given review(text).

in this project i applied a lot of concepts 
1. Loading and exploration of Data
2. Data preprocessing
     1-removing html tags
     2-taking only words 
     3-lowercase
     4-tokenization
     5-stop_words removal
     6-lemmatization
3. Vectorizing Text(reviews)
   Splitting the data set into train and test(70–30)
   BOW (Bag Of Words)
   TF-IDF

4. Building ML Classifiers
  Naive Bayes with reviews BOW encoded
  Naive Bayes with reviews TF-IDF encoded
  Logistic Regression with reviews TF-IDF encoded (apply L1 regulariztion)
  Logistic Regression with reviews TF-IDF encoded (apply L2 regulariztion)

5. Summary & comparing the models

+------------+------------------------+----------+
| Vectorizer |         Model          | Accuracy |
+------------+------------------------+----------+
|    BOW     |      Naive Bayes       |  85.1%   |
|   TFIDF    |      Naive Bayes       |  85.3%   |
|   TFIDF    | Logistic Regression-L1 |  88.0%   |
|   TFIDF    | Logistic Regression-L2 |  89.0%   |
+------------+------------------------+----------+


6. Save the Model dumping using pickle

7. Model deployment Using Streamlit
  link: https://abdelrahmanragab38-sentiment-analysis-sentiment-analysis-dkw0ki.streamlit.app/

