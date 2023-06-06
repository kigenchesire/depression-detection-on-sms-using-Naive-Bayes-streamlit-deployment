import streamlit as st
import pickle
import joblib,os
import pandas as pd

nb = pd.read_csv("final.csv")

# Load the trained model
#model = pickle.load(open('model_pkl', 'rb'))

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.pipeline import Pipeline
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
print(nb.head())
cv = CountVectorizer(stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
cv.fit_transform(nb['Text '].values.astype('U'))




def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model
model = load_prediction_models("CNB_pkl2")
# pipeline = Pipeline(('count', CountVectorizer()),
#                     ('model',  model))


def prediction(text):
    
    
    text = [text]
    text = cv.transform(text)
    #vectorizer = CountVectorizer(stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
    #vectorizer.fit(text)
# summarize encoded vector



    
    prediction = model.predict(text)   #

    # set prediction whole number integer
    #prediction = int(prediction)
    if prediction == 1:
        pred = 'This text shows signs of depression.'
    elif prediction == 0:
        pred = "This text is neutral and doesn't indicate depression."
    else:
       pred = 'Unable to classify the text.'
    return pred

def main():
    st.title('''Depression Detection in Text Messages''')
    st.write("Enter a text message below to check for signs of depression.")
    text = st.text_input("Enter your text here", "")
    
    if st.button("Predict"): 
        result = prediction(text) 
        st.write(text)
        st.success(result)
       
        

if __name__ == '__main__':
    main()
























# # Define the Streamlit app
# def main():
#     st.title("Depression Detection in Text Messages")
#     st.write("Enter a text message below to check for signs of depression.")

#     # Text input for user
#     user_input = st.text_input("Enter your text here", "")

#     # Check if the user has entered any text
#     if user_input:
#         # Preprocess the user input (if required)
#         # ...

#         # Make predictions using the loaded model
#         prediction = model.predict([user_input])[0]

#         # Display the prediction
#         if prediction == 'depressive':
#             st.write("This text shows signs of depression.")
#         elif prediction == 'neutral':
#             st.write("This text is neutral and doesn't indicate depression.")
#         else:
#             st.write("Unable to classify the text.")

# # Run the app
# if __name__ == '__main__':
#     main()
