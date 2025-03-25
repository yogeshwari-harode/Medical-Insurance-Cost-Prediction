from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Initialize Flask app
app = Flask(__name__)

# Load dataset
insurance_dataset = pd.read_csv('insurance.csv')

# Encode categorical variables with proper handling to avoid FutureWarnings
insurance_dataset['sex'] = insurance_dataset['sex'].replace({'male': 0, 'female': 1})
insurance_dataset['smoker'] = insurance_dataset['smoker'].replace({'yes': 0, 'no': 1})
insurance_dataset['region'] = insurance_dataset['region'].replace({'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3})

# Split data
X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

# Feature Engineering: Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)  # Degree 2 to capture non-linearities
X_poly = poly.fit_transform(X)

# Feature Scaling: Standardization of features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Define the models and parameter grids for hyperparameter tuning
models = {
    'XGBoost': XGBRegressor(random_state=42),
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

param_grids = {
    'XGBoost': {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 6, 10]
    },
    'RandomForest': {
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    },
    'GradientBoosting': {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 6, 10]
    }
}

# Hyperparameter tuning with GridSearchCV
best_models = {}
for model_name, model in models.items():
    print(f"Performing GridSearch for {model_name}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, Y_train)
    best_models[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")

# Evaluate the models
model_r2_scores = {}
for model_name, model in best_models.items():
    predictions = model.predict(X_test)
    r2 = r2_score(Y_test, predictions)
    model_r2_scores[model_name] = r2
    print(f"{model_name} R2 Score: {r2:.2f}")

# Select the best model based on R2 score
best_model_name = max(model_r2_scores, key=model_r2_scores.get)
best_model = best_models[best_model_name]
print(f"Best Model Selected: {best_model_name}")

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])

        # Prepare input as a DataFrame to retain feature names
        input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], columns=X.columns)

        # Transform the features (polynomial + scaling)
        input_data_poly = poly.transform(input_data)  # Polynomial transformation
        input_data_scaled = scaler.transform(input_data_poly)  # Scaling

        # Predict using the best selected model
        prediction = best_model.predict(input_data_scaled)[0]

        print(f"Input Data: {input_data.values}, Predicted Charge: {prediction:.2f}")

        return render_template('result.html', prediction=prediction)
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Error in processing the request. Please try again."

if __name__ == '__main__':
    print("Running Flask app on http://127.0.0.1:5000/")
    app.run(debug=True)

