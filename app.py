import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import keras_nlp
from keras_nlp.models import DistilBertClassifier, DistilBertBackbone, DistilBertPreprocessor, DistilBertTokenizer

MODEL_SAVE_PATH = '/home/inductive-anks/Predict-AI-Generated-Text/Detect-AI-Generated-Text/models/trained_model.keras'

# Ensure that all the necessary classes are included in custom_objects
custom_objects = {
    'DistilBertClassifier': DistilBertClassifier,
    'DistilBertBackbone': DistilBertBackbone,
    'DistilBertPreprocessor': DistilBertPreprocessor,
    'DistilBertTokenizer': DistilBertTokenizer
}

classifier = load_model(MODEL_SAVE_PATH, custom_objects=custom_objects)

# Streamlit app title
st.title("AI-Generated Text Detection")

# Text input from user
user_input = st.text_area("Enter the text you want to analyze:")

# If the user provides input, make a prediction
if st.button("Analyze Text"):
    if user_input.strip() == "":
        st.warning("Please enter some text for analysis.")
    else:
        # Convert input text to a pandas Series
        input_data = pd.Series([user_input])

        # Make predictions directly without preprocessing
        prediction_probs = classifier.predict(input_data)

        # Extract the probability of AI-generated text
        ai_generated_prob = prediction_probs[0][1]

        # Display the result with likelihood interpretation
        st.write(f"**Probability of the text being AI-generated:** {ai_generated_prob:.2%}")

        if ai_generated_prob > 0.75:
            st.warning("The model suggests that this text is **highly likely** AI-generated.")
        elif ai_generated_prob > 0.5:
            st.warning("The model suggests that this text is **likely** AI-generated.")
        elif ai_generated_prob > 0.25:
            st.info("The model suggests that this text is **somewhat likely** AI-generated.")
        else:
            st.success("The model suggests that this text is **unlikely** to be AI-generated.")

# Footer
st.write("---")
st.write("Powered by DistilBERT and Streamlit.")
