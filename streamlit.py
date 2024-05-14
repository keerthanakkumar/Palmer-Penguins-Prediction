import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

st.title('Palmer Penguins Perdictor')
image = Image.open('./palmer_penguins.png')
st.image(image)
st.write("""
         The palmerpenguins are three penguin species observed on three islands in the Palmer Archipelago, Antarctica.
         There are three islands Biscoe, Dream, Torgersen with three species Adelie, Chinstrap, Gentoo.
         
Here you can enter the required inputs and this website will predict the specie of the Penguin.
""")


def user_input_features():
    
    st.markdown('Enter the name of the island and sex of the penguin')
    island = st.selectbox('Island',('Biscoe','Dream','Torgersen'))
    sex = st.selectbox('Sex',('male','female'))
    
    st.subheader('Choose the measurement')
    images = Image.open('./culmen_depth.png')
    st.image(images)
    
    bill_length_mm = st.slider('Bill length (mm)', 32.1,59.6,43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1,21.5,17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0,231.0,201.0)
    body_mass_g = st.slider('Body mass (g)', 2700.0,6300.0,4207.0)
    data = {'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')
st.write(df)

# Reads in saved classification model
load_clf = joblib.load('saved_model')

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
p_s =(penguins_species[prediction])
st.write('The predicted specie is ',p_s[0])

st.subheader('Prediction Probability')
fd = pd.DataFrame(prediction_proba, columns=['Adelie','Chinstrap','Gentoo'])
st.write(fd)
