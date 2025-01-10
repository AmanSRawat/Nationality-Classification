from flask_bootstrap import Bootstrap
from flask import Flask, request, render_template, url_for
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Initialize Flask app
gender_app = Flask(__name__)
Bootstrap(gender_app)

# Define the class names and their corresponding nationalities
class_name_to_nationality = {
    'arabic': 'Arabic',
    'chinese': 'Chinese',
    'czech': 'Czech',
    'english': 'English',
    'french': 'French',
    'german': 'German',
    'greece': 'Greek',
    'india': 'Indian',
    'indonesia': 'Indonesian',
    'italian': 'Italian',
    'japanese': 'Japanese',
    'korean': 'Korean',
    'polish': 'Polish',
    'scottish': 'Scottish',
    'spanish': 'Spanish',
    'vietname': 'Vietnamese'
}

# Load the CountVectorizer and VotingClassifier model
cv = joblib.load('cot_vectorizer.pkl')  # Ensure you have saved this after training
voting_clf = joblib.load('vot_classifier.pkl')  # Ensure this is the trained model

@gender_app.route('/')
def index():
    return render_template('index.html')

@gender_app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input name from the form
        name_query = request.form['name_query']
        data = [name_query]
        
        # Transform the input using the saved CountVectorizer
        vct = cv.transform(data).toarray()
        
        # Make a prediction using the VotingClassifier
        predicted_label = voting_clf.predict(vct)[0]  # Predicted label like "india", "french"
        
        # Map the predicted label to nationality
        predicted_nationality = class_name_to_nationality.get(predicted_label, "Unknown")

    # Pass the nationality and name to the results template
    return render_template(
        'results.html', 
        prediction=predicted_nationality,  # Display the nationality
        name=name_query.upper()  # Display the name input in uppercase
    )

if __name__ == '__main__':
    gender_app.run(debug=True)
