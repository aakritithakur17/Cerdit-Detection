import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv(r"creditcard.csv")

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression(max_iter=1000000)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# Only one input for all features (comma-separated)
input_string = st.text_input("Input All Features")

if st.button("Submit"):
    try:
        # Convert comma-separated string to numeric list
        input_list = [float(x.strip()) for x in input_string.split(",")]

        # Convert to NumPy array and reshape for model
        features = np.array(input_list).reshape(1, -1)

        # Validate feature count
        if features.shape[1] != X.shape[1]:
            st.error(f"Error: Expected {X.shape[1]} features, but got {features.shape[1]}.")
        elif np.isnan(features).any() or np.isinf(features).any():
            st.error("Please enter valid numeric values for all features.")
        else:
            prediction = model.predict(features)
            if prediction[0] == 0:
                st.success("Legitimate transaction")
            else:
                st.error("Fraudulent transaction")
    except ValueError as e:
        st.error(f"Error: {e}. Please enter valid numeric values separated by commas.")
