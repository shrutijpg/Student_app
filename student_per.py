import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


uri = "mongodb+srv://shrutibh1001:shruti1234@cluster0.h7nm7qw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['student'] #Creating the database
collection = db["student_pred"] # inside same db we are also creating the collection


def load_model():
  with open("student_lr_final_mode.pkl",'rb') as file:
    model,scaler,le = pickle.load(file)#while storing we created 3 things it will return the same 3 things
    return model,scaler,le


# As user will input based on previous data and we have done the transformations on our data

def preprocessing_input_data(data, scaler,le):
  data["Extracurricular Activities"] = le.transform([data["Extracurricular Activities"]])[0]
  df = pd.DataFrame([data])
  df_transformed = scaler.transform(df)
  return df_transformed

#Predict Function
def predict_data(data):
  model,scaler,le = load_model()
  processed_data = preprocessing_input_data(data,scaler,le)
  prediction = model.predict(processed_data)
  return prediction

#where we can use these functions 
#wewillbe using this functions when someone will enter the data through UI
#How we can create the UI
def main():
  st.title("Student performance prediction")
  st.write("enter your data to get a prediction for your performance")
  hour_sutdied = st.number_input("Hours Studied",min_value = 1, max_value = 10 , value = 5)
  prvious_score = st.number_input("Previous Score",min_value = 40, max_value = 100 , value = 70)
  extra = st.selectbox("Extracurricular Activities" , ['Yes',"No"])
  sleeping_hour = st.number_input("Sleeping Hours",min_value = 4, max_value = 10 , value = 7)
  number_of_paper_solved = st.number_input("number of question paper solved",min_value = 0, max_value = 10 , value = 5)
    
  if st.button("predict your score"):

    user_data = {
        "Hours Studied":hour_sutdied,
        "Previous Scores":prvious_score,
        "Extracurricular Activities":extra,
        "Sleep Hours":sleeping_hour,
        "Sample Question Papers Practiced":number_of_paper_solved
    }
    
    prediction = predict_data(user_data)
    st.success(f"your prediciotn result is {prediction}")
    user_data['prediction'] = round(float(prediction[0]),2)
    user_data ={key: int(value) if isinstance(value,np.integer) else float(value) if isinstance(value,np.floating) else value for key , value in user_data.items()}
    collection.insert_one(user_data)
    
    

if __name__ == "__main__":
  main()
