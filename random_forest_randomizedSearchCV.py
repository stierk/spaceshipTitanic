import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
import joblib
#Label encodings
from sklearn.preprocessing import LabelEncoder 
#Linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#neural network
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Function to preprocess the data
def preprocess_data(df, label_encoder=None):
    dropped_rows = df[df.isnull().any(axis=1)]
    df = df.dropna()
    df.drop(['Name', 'PassengerId'], axis=1, inplace=True)
    df[['Cabin_Deck', 'Cabin_Num', 'Cabin_Side']] = df['Cabin'].str.split('/', expand=True)
    df['Cabin_Num'] = df['Cabin_Num'].astype(int)
    df.drop('Cabin', axis=1, inplace=True)
    
    if label_encoder is None:
        label_encoder = LabelEncoder()
        df['HomePlanet'] = label_encoder.fit_transform(df['HomePlanet'])
        df['Cabin_Deck'] = label_encoder.fit_transform(df['Cabin_Deck'])
        df['Cabin_Side'] = label_encoder.fit_transform(df['Cabin_Side'])
        df['Destination'] = label_encoder.fit_transform(df['Destination'])
    else:

        # Add new labels to the label encoder for HomePlanet
        new_labels = set(df['HomePlanet']) - set(label_encoder.classes_)
        if new_labels:
            label_encoder.classes_ = np.append(label_encoder.classes_, list(new_labels))
        df['HomePlanet'] = label_encoder.transform(df['HomePlanet'])

        # Add new labels to the label encoder for Cabin_Deck
        new_labels = set(df['Cabin_Deck']) - set(label_encoder.classes_)
        if new_labels:
            label_encoder.classes_ = np.append(label_encoder.classes_, list(new_labels))
        df['Cabin_Deck'] = label_encoder.transform(df['Cabin_Deck'])

        # Add new labels to the label encoder for Cabin_Side
        new_labels = set(df['Cabin_Side']) - set(label_encoder.classes_)
        if new_labels:
            label_encoder.classes_ = np.append(label_encoder.classes_, list(new_labels))
        df['Cabin_Side'] = label_encoder.transform(df['Cabin_Side'])

        # Add new labels to the label encoder for Destination
        new_labels = set(df['Destination']) - set(label_encoder.classes_)
        if new_labels:
            label_encoder.classes_ = np.append(label_encoder.classes_, list(new_labels))
        df['Destination'] = label_encoder.transform(df['Destination'])
    
    df['VIP'] = df['VIP'].astype(str).str.strip().fillna('False').map({"True": 1, "False": 0}).astype(int)
    df['CryoSleep'] = df['CryoSleep'].astype(str).str.strip().map({'True': 1, 'False': 0}).astype(int)
    
    return df, label_encoder, dropped_rows

# Read the data
fileName = "data/train.csv"
df = pd.read_csv(fileName)

# Preprocess the training data
df, label_encoder, _ = preprocess_data(df)

# Split the data into features and target variable
X = df.drop('Transported', axis=1)
y = df['Transported']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


######################################
# Hyperparameter Tuning 
######################################

# Check if the model already exists
model_path_random = 'model/best_rf_model_randomSearchCV.pkl'
params_path_random = 'model/params_rf_model_randomSearchCV.txt'
if os.path.exists(model_path_random):
    # Load the model
    best_rf_model_randomSearchCV = joblib.load(model_path_random)
    print("Loaded existing RandomizedSearchCV model.")
else:
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=3)]
    # Number of features to consider at every split
    max_features = ['log2', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=3)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }

    # Initialize the random forest model
    rf_model = RandomForestClassifier()

    # Perform RandomizedSearchCV
    rf_random = RandomizedSearchCV(estimator=rf_model, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
    rf_random.fit(X_train, y_train)

    # Print the best parameters
    best_params_random = rf_random.best_params_
    print(f"Best parameters found: {best_params_random}")

    # Save the best parameters to a text file
    with open(params_path_random, 'w') as f:
        f.write(str(best_params_random))

    # Use the best model
    best_rf_model_randomSearchCV = rf_random.best_estimator_

    # Save the model
    os.makedirs('model', exist_ok=True)
    joblib.dump(best_rf_model_randomSearchCV, model_path_random)
    print("Saved RandomizedSearchCV model.")

# Check if the GridSearchCV model already exists
model_path_grid = 'model/best_rf_model_gridSearchCV.pkl'
params_path_grid = 'model/params_rf_model_gridSearchCV.txt'
if os.path.exists(model_path_grid):
    # Load the model
    best_rf_model_gridSearchCV = joblib.load(model_path_grid)
    print("Loaded existing GridSearchCV model.")
else:
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=1400, stop=1800, num=5)]
    # Number of features to consider at every split
    max_features = ['log2', 'sqrt']
    # Maximum number of levels in tree
    max_depth = []#[int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [int(x) for x in np.linspace(start=8, stop=12, num=2)]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    param_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }

    grid_search = GridSearchCV(estimator=best_rf_model_randomSearchCV, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Print the best parameters from GridSearchCV
    best_params_grid = grid_search.best_params_
    print(f"Best parameters found by GridSearchCV: {best_params_grid}")

    # Save the best parameters to a text file
    with open(params_path_grid, 'w') as f:
        f.write(str(best_params_grid))

    # Use the best model from GridSearchCV
    best_rf_model_gridSearchCV = grid_search.best_estimator_

    # Save the model
    joblib.dump(best_rf_model_gridSearchCV, model_path_grid)
    print("Saved GridSearchCV model.")

# Make predictions
y_pred_rf = best_rf_model_gridSearchCV.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
accuracy_rf = np.mean(y_pred_rf == y_test)

print(f"Mean Squared Error: Random Forest {mse_rf}")
print(f"R-squared: Random Forest {r2_rf}")
print(f"Accuracy: Random Forest {accuracy_rf}")

######################################
# Predicting on Test Data and Saving Results
######################################

# Load the test data
test_fileName = "data/test.csv"
test_df = pd.read_csv(test_fileName)

# Save PassengerId for the final output
passenger_ids = test_df['PassengerId']

# Preprocess the test data
test_df, _, dropped_rows = preprocess_data(test_df, label_encoder)

# Remove PassengerIds which are both in test_df and dropped_rows. PassengerIds in 
# dropped_rows have missing values and will be added later
passenger_ids = passenger_ids[~passenger_ids.isin(dropped_rows['PassengerId'])]

# Predict the Transported column using the Random Forest model
predictions = best_rf_model_gridSearchCV.predict(test_df)

# Create a DataFrame with PassengerId and the predicted Transported column
output_df = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': predictions})

# Add dropped rows with Transported = 0
dropped_rows['Transported'] = 0
output_df = pd.concat([output_df, dropped_rows[['PassengerId', 'Transported']]], ignore_index=True)

# Translate 1 to 'True' and 0 to 'False'
output_df['Transported'] = output_df['Transported'].map({1: 'True', 0: 'False'})

# Save the result as a CSV file
output_df.to_csv('data/prediction.csv', index=False)

