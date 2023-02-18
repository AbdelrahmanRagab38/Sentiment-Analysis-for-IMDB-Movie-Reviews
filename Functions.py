import nltk
nltk.download('punkt') 
nltk.download('stopwords')
nltk.download('wordnet')
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
stop_words = stopwords.words('english') # defining stop_words
stop_words.remove('not') # removing not from the stop_words list as it contains value in negative movies
lemmatizer = WordNetLemmatizer()



def data_preprocessing(review):
    
  # data cleaning
    review = re.sub(re.compile('<.*?>'), '', review) #removing html tags
    review =  re.sub('[^A-Za-z0-9]+', ' ', review) #taking only words
  
  # lowercase
    review = review.lower()
  
  # tokenization
    tokens = nltk.word_tokenize(review) # converts review to tokens
  
  # stop_words removal
    review = [word for word in tokens if word not in stop_words] #removing stop words
  
  # lemmatization
    review = [lemmatizer.lemmatize(word) for word in review]
  #We are using lemmatization and not stemming because 
  #while testing results with both, lemmatization gives slightly better results compared to stemming.
    
  # join words in preprocessed review
    review = ' '.join(review)
  
    return review



def predict_sent(review ,vectorizer, model):
    review=data_preprocessing(review)
    review_vec=vectorizer.transform([review])
    prediction =model.predict(review_vec)
    
    if(prediction==1):
        prediction = 'That is a positive review'
    else:
        prediction = 'That is a Negative review'
    print(prediction)
    return prediction
    
    
