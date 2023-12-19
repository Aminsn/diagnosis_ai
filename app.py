import streamlit as st
import pandas as pd
from transformers import pipeline
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import matplotlib.cm as cm


# Load the pre-trained classifier
clf = joblib.load('doctor_classifier.joblib')
mma_data = pd.read_csv('MMZ_cleaned.csv')
mma_data = mma_data.query('Classificatie_MMZ != 3 and Classificatie_MMZ != 8')

# Initialize the QA pipeline globally to avoid reloading the model on each function call
qa_pipeline = pipeline("question-answering")

# Function to extract feature using a question-answering pipeline
def extract_feature(question, paragraph):
    result = qa_pipeline(question=question, context=paragraph)
    return result['answer']

# Define the classification scheme (if not already defined)
classes_plt = {
    '0': 'Decreased MMZ',
    '1': 'Normal',
    '2': 'Mild B12 deficiency',
    '4': 'B12 deficiency',
    '5': 'kidney problem or B12 deficiency',
    '6': 'B12 and kidney problem',
    '7': 'Nitrous oxide use'
}

mma_data['Label'] = mma_data['Classificatie_MMZ'].astype(str).map(classes_plt)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(mma_data['Label'])

# Main Streamlit app
st.title('MA Elevation Diagnosis AI')

# User input for the flower description
description = st.text_area("Describe the subject's conditions:", 'The patient has eGFR of 80 and MMZ value of 1000.', height=150)

# When the user clicks the 'Classify Iris' button
if st.button('Diagnose'):
    with st.spinner('Extracting features from description...'):
        try:
            # Extract features using the QA pipeline
            egfr = extract_feature("What is the eGFR value?", description)
            mmz = extract_feature("What is the MMZ value?", description)

            # Convert extracted features to float and reshape for prediction
            features = [float(egfr.split()[0]), float(mmz.split()[0])]
            
            # Predict the Iris species from the features
            probabilities = clf.predict_proba([features])[0]

            # Predict the Iris species from the features
            prediction_index = clf.predict([features])
            predicted_class_name = label_encoder.classes_[prediction_index]


            # Display the predicted Iris species
            st.success(f"Subject's condition is {predicted_class_name[0]}")

            # Plot the probabilities
            # Prepare the colormap
            cmap = cm.get_cmap('viridis')
            norm = plt.Normalize(vmin=probabilities.min(), vmax=probabilities.max())
            colors = cmap(norm(probabilities))
            
            # Custom font settings
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = 'Times New Roman'
            plt.rcParams['font.size'] = 12
            
            # Plotting
            plt.figure(figsize=(10, 6))
            bars = plt.barh(label_encoder.classes_, probabilities, color=colors, align='center')
            plt.xlabel('Probability', fontsize=14, fontweight='bold')
            plt.xlim(0, 1)
            plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label='Probability')
            
            # Add value labels to each bar
            for bar, prob in zip(bars, probabilities):
                plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                         f'{prob:.2f}', 
                         va='center', ha='left', color='black', fontsize=12)

            st.pyplot(plt)

        except Exception as e:
            st.error(f'An error occurred during feature extraction or prediction: {e}')
