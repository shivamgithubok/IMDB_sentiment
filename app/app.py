from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Ensure required NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')

# Initialize required objects
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Function to preprocess text
def preprocess_text(text):
    """
    Preprocesses the input text:
    1. Converts to lowercase.
    2. Removes HTML tags.
    3. Removes punctuation.
    4. Removes stopwords.
    5. Removes numeric characters.
    6. Applies stemming.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()

    # Remove punctuation
    text = ''.join(ch for ch in text if ch not in string.punctuation)

    # Remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    text = ' '.join(filtered_words)

    # Remove numeric characters
    text = ''.join([char for char in text if not char.isdigit()])

    # Apply stemming
    words = word_tokenize(text)
    stemmed_words = [ps.stem(word) for word in words]
    return ' '.join(stemmed_words)

# Load vectorizer and model
vectorizer_path = r'c:\Users\Asus\python\Project\IMDB_50_k\vectorizer_sent.pkl'
model_path = r'c:\Users\Asus\python\Project\IMDB_50_k\model2_sent.pkl'

try:
    with open(vectorizer_path, 'rb') as vec_file:
        tfidf = pickle.load(vec_file)

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    st.success("Model and vectorizer loaded successfully.")
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Streamlit app
st.title("Sentiment Analysis of IMDB Movie Reviews")

# Input text
input_sms = st.text_input("Enter the review")
if input_sms:
    # Preprocess the input text
    transformed_sms = preprocess_text(input_sms)
    st.write("Transformed Text:", transformed_sms)

    try:
        # Vectorize the processed text
        vector_input = tfidf.transform([transformed_sms])
        st.write("Vectorized Input Shape:", vector_input.shape)

        # Reshape if the model expects a 3D input
        if len(model.input_shape) == 3:  # Check if 3D input is required
            vector_input = vector_input.toarray()  # Convert sparse to dense
            vector_input = vector_input.reshape(1, 1, vector_input.shape[1])  # Reshape to (1, 1, features)
            st.write("Reshaped Input Shape for Model:", vector_input.shape)

        # Predict sentiment
        result = model.predict(vector_input)[0]
        st.write("Prediction Result:", result)

        # Display result
        if result == 1:
            st.header("Good Review")
        else:
            st.header("Bad Review")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
