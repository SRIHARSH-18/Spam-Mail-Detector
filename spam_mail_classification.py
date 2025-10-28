# Imports and Settings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Ensuring non-interactive backend for matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# Setting a default style for plots
sns.set(style="whitegrid")

# For reproducibility
np.random.seed(42)
# Data Loading
# We load the dataset from the provided CSV path. Note that the file encoding is ascii and delimiter is a comma.
data_path = '/content/emails.csv'
df = pd.read_csv(data_path, encoding='ascii', delimiter=',')

# Let's take a peek at the dataframe structure
print('Dataset shape:', df.shape)
print('Columns:', df.columns.tolist())
# Data Exploration
print('First five rows of the data:')
print(df.head())

print('\nDataset Info:')
df.info()

print('\nSummary Statistics:')
print(df.describe())

# Data Cleaning and Preprocessing
# Dropping duplicates if any
initial_shape = df.shape
df.drop_duplicates(inplace=True)
print(f'Removed {initial_shape[0] - df.shape[0]} duplicate rows.')

# Checking for missing values
missing_values = df.isnull().sum()
print('Missing values in each column:\n', missing_values[missing_values > 0])

# In this dataset, 'Email No.' is an identifier so we drop it from features later but keep it for reference
# Also, it turns out that there is no date column to convert

# Additional preprocessing could be added here if necessary

# Exploratory Data Analysis
# Let's generate several plots to visualize the relationships and distributions in the data

# We first create a numeric dataframe (excluding the identifier column 'Email No.')
numeric_df = df.select_dtypes(include=[np.number])

# Correlation Heatmap: Only if at least 4 numeric columns are present
if numeric_df.shape[1] >= 4:
    plt.figure(figsize=(12,10))
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Numeric Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

# Pair Plot for a small random sample of the numeric features to avoid overcrowding
sample_columns = numeric_df.columns.tolist()[:5]  # taking first 5 numeric columns for brevity
sns.pairplot(df[sample_columns].sample(n=min(200, df.shape[0])), diag_kind='hist')
plt.suptitle('Pair Plot of Sampled Numeric Features', y=1.02)
plt.savefig('pairplot.png')
plt.close()

# Histograms for selected features
plt.figure(figsize=(10,6))
for col in sample_columns:
    sns.histplot(df[col], kde=True, label=col, element='step', stat='density', alpha=0.6)
plt.legend()
plt.title('Histograms of Selected Features')
plt.savefig('histograms.png')
plt.close()

# Pie Chart / Count Plot for the target variable 'Prediction'
plt.figure(figsize=(6,4))
sns.countplot(x='Prediction', data=df, palette='pastel')
plt.title('Count of Predictions')
plt.savefig('prediction_count.png')
plt.close()

# A grouping bar plot might be interesting if we had a categorical variable; here we simply note the class imbalance if any
print('Value counts for target variable (Prediction):')
print(df['Prediction'].value_counts())


# Predictor Model Building
# We assume the 'Prediction' column is our binary target variable.
# We'll remove the identifier column 'Email No.' and use the remaining numeric features as predictors.

# Prepare the features and target
features = df.drop(columns=['Email No.', 'Prediction'])
target = df['Prediction']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42, stratify=target)

# Initialize and train a Logistic Regression model (a good starting point even if it might be overwhelmed by the number of features)
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Prediction Accuracy Score:', accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# ROC Curve and AUC
y_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.close()

# A brief note of dry humor: if the accuracy is lower than your coffee-making skills, well, it might be time to try a different model or add some feature engineering.


# Save Model
import joblib

# Save the model to a file
joblib.dump(model, "Spam_mail_Classifier.joblib")

print("Model saved successfully as Spam_mail_Classifier.joblib")