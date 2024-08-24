import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import keras_nlp
from keras_nlp.models import DistilBertClassifier, DistilBertBackbone, DistilBertPreprocessor, DistilBertTokenizer

MODEL_SAVE_PATH = '/home/inductive-anks/kaggle/Detect-AI-Generated-Text/models/my_model_test_kaggle.keras'

custom_objects = {
    'DistilBertClassifier': DistilBertClassifier,
    'DistilBertBackbone': DistilBertBackbone,
    'DistilBertPreprocessor': DistilBertPreprocessor,
    'DistilBertTokenizer': DistilBertTokenizer
}

classifier = load_model(MODEL_SAVE_PATH, custom_objects=custom_objects)

st.title("AI-Generated Text Detection")

user_input = st.text_area("Enter the text you want to analyze:")

if st.button("Analyze Text"):
    if user_input.strip() == "":
        st.warning("Please enter some text for analysis.")
    else:
    
        input_data = pd.Series([user_input])

        prediction_probs = classifier.predict(input_data)

        ai_generated_prob = prediction_probs[0][1]

        st.write(f"**Probability of the text being AI-generated:** {ai_generated_prob:.2%}")


        st.warning(f"The model suggests that this text is {ai_generated_prob:.2%} likely to be AI-generated.")
        # if ai_generated_prob > 0.75:
        #     st.warning("The model suggests that this text is **highly likely** AI-generated.")
        # elif ai_generated_prob > 0.5:
        #     st.warning("The model suggests that this text is **likely** AI-generated.")
        # elif ai_generated_prob > 0.25:
        #     st.info("The model suggests that this text is **somewhat likely** AI-generated.")
        # else:
        #     st.success("The model suggests that this text is **unlikely** to be AI-generated.")

# Footer
st.write("---")
st.write("Powered by DistilBERT and Streamlit.")
