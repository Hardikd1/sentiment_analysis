import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
model = joblib.load('glass_model.pkl')

# Define the glass types based on the problem description
glass_type_dict = {
    1: 'Building Windows (Float Processed)',
    2: 'Building Windows (Non-Float Processed)',
    3: 'Vehicle Windows (Float Processed)',
    4: 'Vehicle Windows (Non-Float Processed)', # Note: This type is not in the dataset
    5: 'Containers',
    6: 'Tableware',
    7: 'Headlamps'
}


# --- Streamlit User Interface ---
st.set_page_config(page_title="Glass Type Predictor", page_icon="🔮", layout="wide")

# App title and description
st.title('🔮 Glass Type Predictor ML App')
st.write(
    "This application uses a Random Forest model to predict the type of glass "
    "based on its chemical composition. Use the sliders in the sidebar to input "
    "the chemical properties."
)

# Sidebar for user inputs
st.sidebar.header('Input Glass Features')

def user_input_features():
    """Creates sidebar sliders for user input and returns a DataFrame."""
    ri = st.sidebar.slider('Refractive Index (RI)', 1.511, 1.534, 1.518)
    na = st.sidebar.slider('Sodium (Na)', 10.73, 17.38, 13.40)
    mg = st.sidebar.slider('Magnesium (Mg)', 0.0, 4.49, 2.68)
    al = st.sidebar.slider('Aluminum (Al)', 0.29, 3.50, 1.44)
    si = st.sidebar.slider('Silicon (Si)', 69.81, 75.41, 72.65)
    k = st.sidebar.slider('Potassium (K)', 0.0, 6.21, 0.49)
    ca = st.sidebar.slider('Calcium (Ca)', 5.43, 16.19, 8.95)
    ba = st.sidebar.slider('Barium (Ba)', 0.0, 3.15, 0.17)
    fe = st.sidebar.slider('Iron (Fe)', 0.0, 0.51, 0.05)

    data = {
        'RI': ri, 'Na': na, 'Mg': mg, 'Al': al,
        'Si': si, 'K': k, 'Ca': ca, 'Ba': ba, 'Fe': fe
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user inputs
st.subheader('Your Input Parameters')
st.dataframe(input_df, hide_index=True)

# Prediction button and logic
if st.button('Predict Glass Type', type="primary"):
    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Get the predicted glass type name
    predicted_glass_name = glass_type_dict.get(prediction[0], "Unknown Type")

    # Display the result
    st.subheader('Prediction Result')
    st.success(f"The predicted glass type is: **Type {prediction[0]} - {predicted_glass_name}**")

    st.subheader('Prediction Probability')
    # Create a clean DataFrame for probabilities
    proba_df = pd.DataFrame(
        prediction_proba,
        columns=[glass_type_dict.get(i, f"Type {i}") for i in model.classes_],
        index=['Probability']
    )
    st.dataframe(proba_df)