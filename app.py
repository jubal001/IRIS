
import streamlit as st 
import pandas as pd
import joblib


model = joblib.load("rfiris.pkl")



st.title(" IRIS FLOWER CLASSIFICATION APPLICATION")


st.write("Predict the species of an Iris Flower Using a Random Forest Model")


form = st.form("iris form")

form.subheader("Enter Flower Measurement")

sepal_length = form.number_input(

		"sepallength (cm)",
		min_value= 4.0,
		max_value= 8.0,
		value = 5.1	

	)

petal_length = form.number_input(

		"petallength (cm)",
		min_value= 1.0,
		max_value= 7.0,
		value = 5.0

	)

sepal_width = form.number_input(

		"sepalwidth (cm)",
		min_value= 1.0,
		max_value= 4.5,
		value = 4.0

	)

petal_width = form.number_input(

		"petalwidth (cm)",
		min_value= 0.1,
		max_value= 2.5,
		value = 0.2

	)


submit_button = form.form_submit_button("Predict")


if submit_button:
	input_data = pd.DataFrame({

		"sepallength (cm)": [sepal_length],
		"sepalwidth (cm)" : [sepal_width],
		"petallength (cm)" : [petal_length],
		"petalwidth (cm)" : [petal_width]

		})
	prediction = model.predict(input_data)


	st.subheader("Prediction Result")

	st.success(f" Predicted species: {prediction[0]}")


