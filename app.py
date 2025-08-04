import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection App")
st.subheader("Paste a news article below and check if it's Fake or Real.")


user_input = st.text_area("Enter News Article Text Here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
       
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 1:
            st.success("✅ This news article is likely REAL.")
        else:
            st.error("❌ This news article is likely FAKE.")