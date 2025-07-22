import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv('glass.csv')

# Separate features (X) and target (y)
X = df.drop('Type', axis=1)
y = df['Type']

# Initialize and train the RandomForestClassifier on the entire dataset
# Using the best parameters found during experimentation
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model to a file
joblib.dump(model, 'glass_classifier_model.joblib')

print("Model trained and saved successfully as glass_classifier_model.joblib")