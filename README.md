# Sentiment-Analysis-for-IMDB-Movie-Reviews

# About
The dataset contains a collection of 50,000 reviews from the IMDB Website with an equal number of positive and negative reviews. The task is to predict the polarity (positive or negative) of a given review(text).

in this project i applied a lot of concepts 
# 1. Loading and exploration of Data
# 2. Data preprocessing
     1-removing html tags
     2-taking only words 
     3-lowercase
     4-tokenization
     5-stop_words removal
     6-lemmatization
# 3. Vectorizing Text(reviews)
   Splitting the data set into train and test(70–30)
   BOW (Bag Of Words)
   TF-IDF

# 4. Building ML Classifiers
  1-Naive Bayes with reviews BOW encoded
  
  2-Naive Bayes with reviews TF-IDF encoded
  
  3-Logistic Regression with reviews TF-IDF encoded (apply L1 regulariztion)
  
  4-Logistic Regression with reviews TF-IDF encoded (apply L2 regulariztion)

# 5. Summary & comparing the models

![Screenshot (12)](https://user-images.githubusercontent.com/49238901/219855218-68978218-354d-4d93-88fd-092c802a0f87.png)



# 6. Save the Model dumping using pickle

# 7. Model deployment Using Streamlit

     link: https://abdelrahmanragab38-sentiment-analysis-sentiment-analysis-dkw0ki.streamlit.app/

