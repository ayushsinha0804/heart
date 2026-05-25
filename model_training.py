import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def map_binary(X):
    # placeholder; actual binary_features will be bound inside train_model
    # but the transformer will call this function with the DataFrame slice
    # We'll implement a generic mapping that converts values starting with
    # 'yes' or 'male' to 1, otherwise 0 for any column present.
    X = X.copy()
    for col in X.columns:
        X[col] = X[col].astype(str).str.strip().map(lambda x: 1 if str(x).lower().startswith('yes') or str(x).lower().startswith('male') else 0)
    return X

def train_model():
    # Use local dataset as requested for real-world fidelity
    # switch to heart_disease_data2.csv (supports categorical fields)
    file_path = 'heart_disease_data2.csv'

    print(f"Loading local data from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find local {file_path}")

    heart_data = pd.read_csv(file_path)

    # If dataset uses 'HeartDisease' with Yes/No, map to numeric target
    if 'HeartDisease' in heart_data.columns:
        heart_data['target'] = heart_data['HeartDisease'].astype(str).str.strip().map(lambda x: 1 if x.lower().startswith('yes') else 0)
        heart_data.drop(columns=['HeartDisease'], inplace=True)

    # We'll build a sklearn Pipeline that handles preprocessing so the same
    # transformations can be applied in the app.
    # Define expected raw columns
    raw_features = [
        'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth',
        'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race',
        'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime',
        'Asthma', 'KidneyDisease', 'SkinCancer'
    ]

    # Ensure numeric columns are numeric
    numeric_features = [c for c in ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime'] if c in heart_data.columns]
    for c in numeric_features:
        heart_data[c] = pd.to_numeric(heart_data[c], errors='coerce').fillna(0)

    # Binary features (Yes/No) and Sex
    binary_features = [c for c in ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Sex'] if c in heart_data.columns]

    # Categorical features to one-hot
    categorical_features = [c for c in ['AgeCategory', 'Race', 'GenHealth', 'Diabetic'] if c in heart_data.columns]

    # Create transformers
    # map_binary is defined at module level to allow pickling
    binary_transformer = FunctionTransformer(map_binary)

    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('bin', binary_transformer, binary_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='drop'
    )

    # Features and target
    if 'target' not in heart_data.columns:
        raise KeyError("No target column found in dataset after preprocessing")

    X = heart_data[[c for c in raw_features if c in heart_data.columns]]
    Y = heart_data['target']

    # Build pipeline: preprocessor -> scaler -> classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=7, random_state=3))
    ])

    # Fit pipeline
    pipeline.fit(X, Y)

    # Train/test split on raw X so pipeline receives the correct raw columns
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=3)

    print("Training pipeline (preprocessor + scaler + classifier)...")
    pipeline.fit(X_train, Y_train)

    # Evaluation using pipeline (it will preprocess internally)
    train_pred = pipeline.predict(X_train)
    train_acc = accuracy_score(Y_train, train_pred)
    print(f'Accuracy on Training data : {train_acc:.2f}')

    test_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(Y_test, test_pred)
    print(f'Accuracy on Test data : {test_acc:.2f}')

    print("\nClinical Validation - Classification Report:")
    print(classification_report(Y_test, test_pred))

    # Save the pipeline and feature list
    print("\nSaving pipeline artifact and feature list...")
    os.makedirs('models', exist_ok=True)
    with open('models/pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    # Save the expected raw feature order so the app can build inputs
    feature_columns = [c for c in raw_features if c in heart_data.columns]
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)

    print("Pipeline and feature list saved to models/ directory.")

if __name__ == "__main__":
    train_model()
