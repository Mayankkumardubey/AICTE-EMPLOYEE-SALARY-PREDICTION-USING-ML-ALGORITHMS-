import streamlit as st
import pandas as pd
import joblib

#load the trained model
model=joblib.load("best_model.pkl")
st.set_page_config(page_title="Employee Salary Prediction",page_icon=":money_with_wings:",layout="centered")
st.title("Employee Salary Prediction App")
st.markdown("Predict whether an employee earns less, equal to  or more than 50K based on input features")

#sidebar inputs
st.sidebar.header("Input Employee Details")
#Replace the field with dataset's actual input.
age=st.sidebar.slider("Age",18,65,30)
education=st.sidebar.selectbox("Education Level",["Bachelors","Masters","PhD","HS-grad","Assoc","Some-college"])
occupation=st.sidebar.selectbox("Job Role",["Tech -support","Craft-repair","Other-service","Sales","Exec-managerial","Prof-specialty","Handlers-cleaners","Machine-op-inspct","Adm-clerical","Farming-fishing","Transport-moving","Priv-house-serv","Protective-serv","Armed-Forces"])
hours_per_week=st.sidebar.slider("Hours Worked Per Week",1,80,40)
experience=st.sidebar.slider("Years of Experience",0,40,5)
gender=st.sidebar.selectbox("Gender Role",["Male", "Female"])
native_country=st.sidebar.selectbox("Residence",["United-States", "Others"])
marital_status=st.sidebar.selectbox("Marital Status",["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
#build input dataframes.
input_df=pd.DataFrame({
    "age":[age],
    "education":[education],
    "occupation":[occupation],
    "hours-per-week":[hours_per_week],
    "experience":[experience],
    "gender":[gender],
    "native-country":[native_country],
    "marital-status":[marital_status]
})
st.write("!! Input Data !!")
st.write(input_df)
#predict button
if st.button('Predict Salary Class'):
   st.snow()
   prediction=model.predict(input_df)
   st.write("!! Predicted Salary Class !!")
   st.write(prediction)
   st.success(f"Predicted Salary Class: {prediction[0]}")
#batch
st.markdown("-----------")
st.markdown("!! Batch Prediction !!")
uploaded_file=st.file_uploader("Upload a CSV file for batch prediction",type=["csv"])
if uploaded_file is not None:
    batch_data=pd.read_csv(uploaded_file)
    st.write("Uploaded Data Previewd",batch_data.head())
    batch_preds=model.predict(batch_data)
    batch_data["predicted_salary_class"]=batch_preds
    st.write("Predictions")
    st.write(batch_data.head())
    csv=batch_data.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions CSV",csv,file_name="predicted_classes.csv",mime='text/csv')
    