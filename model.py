import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # Import numpy for numerical operations
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting data into training and testing sets
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Import StandardScaler for feature scaling and LabelEncoder for encoding target labels
from sklearn.svm import SVC  # Import SVC for support vector classifier
from sklearn import metrics  # Import metrics from sklearn for model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score  # Import specific metrics for model evaluation
# import category_encoders as ce  # Import category_encoders for encoding categorical features
# import optuna
# Ignore warnings
import warnings  # Import warnings to manage warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Define the path to the CSV file containing the data
path = r'.\input\Cancer_Data.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(path)

# Set option to display all columns
pd.set_option('display.max_columns', None)

# Display the DataFrame to view the loaded data
# peint(df)

df_copy = df.copy()

# Drop columns 'Unnamed: 32' and 'id' from the DataFrame
df_copy.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

# Map 'diagnosis' column values from 'B' and 'M' to 0 and 1 respectively
df_copy['diagnosis'] = df_copy['diagnosis'].map({'B': 0, 'M': 1})

# Iterate over each column in the DataFrame
for col in df_copy.columns:
    # Check if the data type of the column is numerical (int64 or float64)
    if df_copy[col].dtype in ['int64', 'float64']:
        # Fill missing values with the mean of the column
        df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    # Check if the data type of the column is object (categorical)
    elif df_copy[col].dtype == 'object':
        # Fill missing values with the mode (most frequent value) of the column
        df_copy[col] = df_copy[col].fillna(df_copy[col].mode().iloc[0])

# Drop duplicate rows from the DataFrame
df_copy.drop_duplicates(inplace=True)

# Assign the 'diagnosis' column as the target variable
targets = df_copy['diagnosis']

# Create inputs by dropping the 'diagnosis' column from the DataFrame
inputs = df_copy.drop('diagnosis', axis=1)

# Import necessary libraries
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# Define the SVC model with probability estimation enabled
svc_model_def = SVC(probability=True)


def train_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    """
    Train and evaluate the given model on the training and testing data.

    Parameters:
    model_name (str): Name of the model for display purposes.
    model : Machine learning model object.
    X_train : Features of the training data.
    y_train : Target labels of the training data.
    X_test : Features of the testing data.
    y_test : Target labels of the testing data.

    Returns:
    float, float: Gini coefficients calculated from the model's predictions on training and testing data.
    """

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict labels and probabilities on the testing data
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # Predict labels and probabilities on the training data
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]

    # Calculate ROC AUC and Gini coefficient for testing data
    roc_test_prob = roc_auc_score(y_test, y_test_prob)
    gini_test_prob = roc_test_prob * 2 - 1

    # Calculate ROC AUC and Gini coefficient for training data
    roc_train_prob = roc_auc_score(y_train, y_train_prob)
    gini_train_prob = roc_train_prob * 2 - 1

    # Calculate confusion matrix and classification report for testing data
    confusion_matrix_test_result = confusion_matrix(y_test, y_test_pred)
    classification_report_test_result = classification_report(y_test, y_test_pred)

    # Calculate confusion matrix and classification report for training data
    confusion_matrix_train_result = confusion_matrix(y_train, y_train_pred)
    classification_report_train_result = classification_report(y_train, y_train_pred)

    # Print model performance metrics
    print(f'Model Performance for {model_name}')
    print('Gini prob for testing data is', gini_test_prob * 100)
    print('Gini prob for training data is', gini_train_prob * 100)
    print('Classification Report for Testing Data:')
    print(classification_report_test_result)
    print('Confusion Matrix for Testing Data:')
    print(confusion_matrix_test_result)
    print('Classification Report for Training Data:')
    print(classification_report_train_result)
    print('Confusion Matrix for Training Data:')
    print(confusion_matrix_train_result)

    return gini_train_prob, gini_test_prob

# Assuming svc_model_def, x_train, y_train, x_test, y_test are defined
gini_df = pd.DataFrame(columns=['Model', 'Gini_train_prob', 'Gini_test_prob'])
gini_train_prob, gini_test_prob = train_and_evaluate_model('svc', svc_model_def, x_train, y_train, x_test, y_test)

# Add the result to the DataFrame using concat and sort it
new_row = pd.DataFrame([{'Model': 'svc', 'Gini_train_prob': gini_train_prob, 'Gini_test_prob': gini_test_prob}])
gini_df = pd.concat([gini_df, new_row], ignore_index=True)
gini_df_sorted = gini_df.sort_values(by='Gini_test_prob', ascending=False)

print(gini_df_sorted)