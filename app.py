import streamlit as st
import joblib
from streamlit_mic_recorder import speech_to_text
import speech_recognition as sr
import base64
import pyttsx3

model=joblib.load("cpgrams_model.pkl")
vectorizer=joblib.load("cpgrams_tfidf.pkl")
label_encoder=joblib.load("cpgrams_label_encoder.pkl")

st.title("CPGRAMS VOICE ASSISTANT")
st.write("Enter your complaint or use your voice.")

complaint=st.text_area("Enter your complaint:")
spoken_text=None
spoken_text=speech_to_text(
 start_prompt=" Start Recording",
 stop_prompt="Stop Recording",
 language="en",
 use_container_width=True
)
if spoken_text:
  complaint=spoken_text
  st.write("You said:", spoken_text)



if st.button("Predict Category"):
  if not complaint:
      st.warning("Please enter a complaint.")
  else:
      complaint_tfidf=vectorizer.transform([complaint])
      prediction=model.predict(complaint_tfidf)
      category=prediction[0]   
      st.success(f"Predicted Category: {category}")

      engine=pyttsx3.init()
      response=f"Your complaint has been classified as {category}."
      st.write(response)
      engine.say(response)
      engine.runAndWait()  