import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('./fine_tuned_model')
tokenizer = DistilBertTokenizerFast.from_pretrained('./fine_tuned_model')

# Function for prediction
def predict_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted label
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    
    # Map label to sentiment
    return "Positive sentiment" if predicted_label == 1 else "Negative sentiment"

# Streamlit UI
st.title("Sentiment Analysis: Spam Detection")
st.markdown("Enter a message below, and the model will predict if it's 'Positive sentiment' or 'Negative sentiment'.")

# Input from the user
user_input = st.text_area("Enter the message", "")

# Display the prediction when button is clicked
if st.button('Predict'):
    if user_input:
        prediction = predict_sentiment(user_input)
        st.write(f"Prediction: {prediction}")
    else:
        st.write("Please enter a message.")
