import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
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
# Hyperparameter Tuning n_estimators
######################################
optimal_n_estimators = 2000 #-> tuning would say 48 but more seems better eventhough validation
# says otherwise???
"""

# Define the range of n_estimators. 10^1.7 ~50
#n_estimators_range = np.logspace(1, 1.7, num=50, dtype=int)
n_estimators_range = np.arange(10, 50, 1)

# Initialize lists to store accuracies for each split
all_train_accuracies = []
all_val_accuracies = []

# Perform the calculations four times with different splits
for i in range(30):
    print(f"Iteration {i+1}")
    # Split the training data into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=i)
    
    # Initialize lists to store accuracies for this split
    train_accuracies = []
    val_accuracies = []
    
    # Iterate over the range of n_estimators
    for n_estimators in n_estimators_range:
        print(f"Training with n_estimators={n_estimators}")
        # Initialize the random forest model
        rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
        
        # Train the model
        rf_model.fit(X_train_split, y_train_split)
        
        # Calculate accuracy on the training set
        train_accuracy = rf_model.score(X_train_split, y_train_split)
        train_accuracies.append(train_accuracy)
        
        # Calculate accuracy on the validation set
        val_accuracy = rf_model.score(X_val, y_val)
        val_accuracies.append(val_accuracy)
    
    # Store the accuracies for this split
    all_train_accuracies.append(train_accuracies)
    all_val_accuracies.append(val_accuracies)

# Calculate the mean accuracies across all splits
mean_train_accuracies = np.mean(all_train_accuracies, axis=0)
mean_val_accuracies = np.mean(all_val_accuracies, axis=0)

# Plot the accuracies
plt.figure(figsize=(10, 6))
sns.lineplot(x=n_estimators_range, y=mean_train_accuracies, label='Training Accuracy')
sns.lineplot(x=n_estimators_range, y=mean_val_accuracies, label='Validation Accuracy')
plt.xscale('log')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy vs. Number of Estimators')
plt.legend()

# Mark the optimal number of estimators
optimal_n_estimators = n_estimators_range[np.argmax(mean_val_accuracies)]
plt.axvline(optimal_n_estimators, color='r', linestyle='--', label=f'Optimal n_estimators: {optimal_n_estimators}')
plt.legend()
plt.show()

# Print the optimal number of estimators
print(f"Optimal number of estimators: {optimal_n_estimators}")

"""


######################################
# Hyperparameter Tuning max_features
######################################
# StackOvervlflow article: https://stackoverflow.com/questions/36107820/how-to-tune-parameters-in-random-forest-using-scikit-learn

optimal_max_features = 4

"""
# Define the range of max_features
max_features_range = np.arange(1, X_train.shape[1] + 1)

# Initialize lists to store accuracies for each split
all_train_accuracies = []
all_val_accuracies = []

# Perform the calculations four times with different splits
for i in range(5):
    print(f"Iteration {i+1}")
    # Split the training data into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=i)
    
    # Initialize lists to store accuracies for this split
    train_accuracies = []
    val_accuracies = []
    
    # Iterate over the range of max_features
    for max_features in max_features_range:
        print(f"Training with max_features={max_features}")
        # Initialize the random forest model
        rf_model = RandomForestClassifier(n_estimators=optimal_n_estimators, max_features=max_features, random_state=0)
        
        # Train the model
        rf_model.fit(X_train_split, y_train_split)
        
        # Calculate accuracy on the training set
        train_accuracy = rf_model.score(X_train_split, y_train_split)
        train_accuracies.append(train_accuracy)
        
        # Calculate accuracy on the validation set
        val_accuracy = rf_model.score(X_val, y_val)
        val_accuracies.append(val_accuracy)
    
    # Store the accuracies for this split
    all_train_accuracies.append(train_accuracies)
    all_val_accuracies.append(val_accuracies)

# Calculate the mean accuracies across all splits
mean_train_accuracies = np.mean(all_train_accuracies, axis=0)
mean_val_accuracies = np.mean(all_val_accuracies, axis=0)

# Plot the accuracies
plt.figure(figsize=(10, 6))
sns.lineplot(x=max_features_range, y=mean_train_accuracies, label='Training Accuracy')
sns.lineplot(x=max_features_range, y=mean_val_accuracies, label='Validation Accuracy')
plt.xlabel('Number of Max Features')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy vs. Number of Max Features')
plt.legend()

# Mark the optimal number of max_features
optimal_max_features = max_features_range[np.argmax(mean_val_accuracies)]
plt.axvline(optimal_max_features, color='r', linestyle='--', label=f'Optimal max_features: {optimal_max_features}')
plt.legend()
plt.show()

# Print the optimal number of max_features
print(f"Optimal number of max_features: {optimal_max_features}")
"""

######################################
# Hyperparameter Tuning criterion
######################################

# Define the range of criterion
criterion_range = ['gini', 'entropy', 'log_loss']

# Initialize lists to store accuracies for each split
all_train_accuracies = []
all_val_accuracies = []

# Perform the calculations with different splits
for i in range(5):
    print(f"Iteration {i+1}")
    # Split the training data into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=i)
    
    # Initialize lists to store accuracies for this split
    train_accuracies = []
    val_accuracies = []
    
    # Iterate over the range of criterion
    for criterion in criterion_range:
        print(f"Training with criterion={criterion}")
        # Initialize the random forest model
        rf_model = RandomForestClassifier(n_estimators=optimal_n_estimators, max_features=optimal_max_features, criterion=criterion, random_state=0)
        
        # Train the model
        rf_model.fit(X_train_split, y_train_split)
        
        # Calculate accuracy on the training set
        train_accuracy = rf_model.score(X_train_split, y_train_split)
        train_accuracies.append(train_accuracy)
        
        # Calculate accuracy on the validation set
        val_accuracy = rf_model.score(X_val, y_val)
        val_accuracies.append(val_accuracy)
    
    # Store the accuracies for this split
    all_train_accuracies.append(train_accuracies)
    all_val_accuracies.append(val_accuracies)

# Calculate the mean accuracies across all splits
mean_train_accuracies = np.mean(all_train_accuracies, axis=0)
mean_val_accuracies = np.mean(all_val_accuracies, axis=0)

# Plot the accuracies
plt.figure(figsize=(10, 6))
sns.lineplot(x=criterion_range, y=mean_train_accuracies, label='Training Accuracy')
sns.lineplot(x=criterion_range, y=mean_val_accuracies, label='Validation Accuracy')
plt.xlabel('Criterion')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy vs. Criterion')
plt.legend()

# Mark the optimal criterion
optimal_criterion = criterion_range[np.argmax(mean_val_accuracies)]
plt.axvline(optimal_criterion, color='r', linestyle='--', label=f'Optimal criterion: {optimal_criterion}')
plt.legend()
plt.show()

# Print the optimal criterion
print(f"Optimal criterion: {optimal_criterion}")

######################################
# Training Random Forest Model
######################################

# Initialize the random forest model with the optimal criterion
rf_model = RandomForestClassifier(n_estimators=optimal_n_estimators, max_features=optimal_max_features, criterion=optimal_criterion, random_state=0)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

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
predictions = rf_model.predict(test_df)

# Create a DataFrame with PassengerId and the predicted Transported column
output_df = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': predictions})

# Add dropped rows with Transported = 0
dropped_rows['Transported'] = 0
output_df = pd.concat([output_df, dropped_rows[['PassengerId', 'Transported']]], ignore_index=True)

# Translate 1 to 'True' and 0 to 'False'
output_df['Transported'] = output_df['Transported'].map({1: 'True', 0: 'False'})

# Save the result as a CSV file
output_df.to_csv('data/prediction.csv', index=False)

