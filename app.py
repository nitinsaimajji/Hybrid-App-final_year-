import tensorflow_text as text
import streamlit as st
import pandas as pd
import numpy as np
import re
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from keras.models import load_model
from spellchecker import SpellChecker
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import string
import re


# Encode labels
label_mapping = {
    'bully-Spam': 0,
    'not_bully-Spam': 1,
    'bully-Ham': 2,
    'not_bully-Ham': 3
}


# Define function for text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Apply stemming
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

# Load first model
cyber_loaded_model = TFBertForSequenceClassification.from_pretrained('cyber_bert_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Load second model
model = tf.keras.models.load_model("bert_new_sms_model")
dnn_model = load_model("final_dnn.h5")
bilstm_model=load_model("final_bilstm.h5")

# Create Streamlit app
st.title("Sheilding Aganist SMS spam and Cyber-Bullying")

# Sidebar with options
option = st.sidebar.selectbox(
    'Choose detection type:',
    ('Bully Detection', 'Spam Detection', 'All','DNN','bilstm'))




spam_accuracy = 0.93
spam_precision = 0.93
spam_recall = 0.93
spam_f1 = 0.93

bully_accuracy = 0.85
bully_precision = 0.87
bully_recall = 0.85
bully_f1 = 0.85

st.sidebar.write("\n")
st.sidebar.write("\n")

st.sidebar.write("### Spam Detection Scores:")
st.sidebar.write(f"- Accuracy: {spam_accuracy}")
st.sidebar.write(f"- Precision: {spam_precision}")
st.sidebar.write(f"- Recall: {spam_recall}")
st.sidebar.write(f"- F1-score: {spam_f1}")

st.sidebar.write("\n")
st.sidebar.write("\n")

st.sidebar.write("### Cyber-bullying Detection Scores:")
st.sidebar.write(f"- Accuracy: {bully_accuracy}")
st.sidebar.write(f"- Precision: {bully_precision}")
st.sidebar.write(f"- Recall: {bully_recall}")
st.sidebar.write(f"- F1-score: {bully_f1}")



# Text input box
text_input = st.text_area("Enter your text here:")

# Button to trigger predictions
if st.button('Enter'):
    if option == 'Bully Detection' or option == 'All':
        # Preprocess text for bully detection
        preprocessed_text = preprocess_text(text_input)
        encoded_text = tokenizer(preprocessed_text, padding=True, truncation=True, return_tensors='tf')
        X_padded = pad_sequences(encoded_text['input_ids'], padding='post')
        
        # Predict with first model
        predictions_bully = cyber_loaded_model.predict({'input_ids': X_padded, 'attention_mask': encoded_text['attention_mask']})
        predicted_label_first_model = np.argmax(predictions_bully.logits)
        st.write("#### Prediction from Bully Detection Model:")
        if predicted_label_first_model == 0:
            st.sidebar.write("\n")
            st.sidebar.write("\n")
            st.write("#### Bully")
            k="Bully"
        else:
            st.sidebar.write("\n")
            st.sidebar.write("\n")
            st.write("#### Not Bully")
            k="Not-Bully"

    if option == 'Spam Detection' or option == 'All':
        # Predict with second model
        predicted_label_second_model = np.round(model.predict([text_input]))
        st.write("#### Prediction from Spam Detection Model:")
        if predicted_label_second_model == 0:
            st.sidebar.write("\n")
            st.sidebar.write("\n")
            st.write("#### Ham")
            m="Ham"
        else:
            st.sidebar.write("\n")
            st.sidebar.write("\n")
            st.write("#### Spam")
            m="Spam"
    
    if option== 'DNN':
        tokenizer = Tokenizer()
        train_data = pd.read_csv('final_train.csv')
        tokenizer.fit_on_texts(train_data['text'])
        new_text_sequence = tokenizer.texts_to_sequences([text_input])
        new_text_padded = pad_sequences(new_text_sequence, padding='post', maxlen=100)

        # Make predictions on new text
        new_text_prediction = dnn_model.predict(new_text_padded)
        predicted_label = np.argmax(new_text_prediction)

        # Map predicted label back to original label
        reverse_label_mapping = {v: k for k, v in label_mapping.items()}
        predicted_label_text = reverse_label_mapping[predicted_label]

        st.write("#### Prediction from DNN Model:", predicted_label_text)
    if option== 'bilstm':
        tokenizer = Tokenizer()
        train_data = pd.read_csv('final_train.csv')
        tokenizer.fit_on_texts(train_data['text'])
        new_text_sequence = tokenizer.texts_to_sequences([text_input])
        new_text_padded = pad_sequences(new_text_sequence, padding='post', maxlen=100)

        # Make predictions on new text
        new_text_prediction = bilstm_model.predict(new_text_padded)
        predicted_label = np.argmax(new_text_prediction)

        # Map predicted label back to original label
        reverse_label_mapping = {v: k for k, v in label_mapping.items()}
        predicted_label_text = reverse_label_mapping[predicted_label]

        st.write("#### Predicted label from bilstm Model:", predicted_label_text)
            
