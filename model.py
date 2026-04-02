import pandas as pd
from sklearn.neural_network import MLPClassifier
import joblib

# Load dataset
data = pd.read_csv("data.csv")

# Features and target
X = data[['experience', 'skills_count', 'education_level',
          'projects', 'certifications', 'employment_gap']]
y = data['fake']

# Create ANN model
model = MLPClassifier(hidden_layer_sizes=(6,4), max_iter=1000, random_state=42)

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained successfully!")