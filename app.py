import pandas as pd
from flask import Flask, render_template, request
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Initialize the Flask app
app = Flask(__name__)

# Load the dataset
df = pd.read_excel("Naan mudhalvan Dataset (1).xlsx")

# Check the column names to confirm target variable
print(df.columns)

# Feature columns (replace 'Diabetes_012' with the actual target column name you want to predict)
X = df.drop(columns=['Diabetes_012'])  # Features (replace 'Diabetes_012' with the target column name)
y = df['Diabetes_012']  # Target variable (replace 'Diabetes_012' with the correct column name)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Save the trained model and scaler using pickle
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the form
    user_input = [float(request.form[key]) for key in X.columns]  # Convert input to float
    
    # Scale the user input using the same scaler
    scaled_input = scaler.transform([user_input])

    # Load the saved model
    model = pickle.load(open('model.pkl', 'rb'))

    # Make the prediction
    prediction = model.predict(scaled_input)
    
    # Render the result on the result page
    if prediction == 1:
        result = "The patient is likely to have the disease."
    else:
        result = "The patient is not likely to have the disease."

    return render_template('result.html', result=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
