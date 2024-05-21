# 1. Data Collection and Understanding

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient

# Authenticate using DefaultAzureCredential
credential = DefaultAzureCredential()
client = LogsQueryClient(credential)

# Define your workspace ID and log query
workspace_id = 'your-workspace-id'
query = "AzureActivity | where ActivityStatus == 'Failed' | project TimeGenerated, ActivityStatus, CallerIpAddress"

# Query the logs
response = client.query_workspace(workspace_id, query)

# Convert the response to a pandas DataFrame
columns = [col.name for col in response.tables[0].columns]
logs = pd.DataFrame(response.tables[0].rows, columns=columns)
print(logs.head())


# 2. Data Preprocessing

# Convert TimeGenerated to datetime
logs['TimeGenerated'] = pd.to_datetime(logs['TimeGenerated'])

# Drop rows with missing values
logs.dropna(inplace=True)

# Feature engineering (e.g., extracting the hour from the timestamp)
logs['hour'] = logs['TimeGenerated'].dt.hour

# Select features for the model
features = logs[['hour', 'ActivityStatus']]

# Convert categorical feature to numeric
features = pd.get_dummies(features, columns=['ActivityStatus'], drop_first=True)


# 3. Data Splitting

from sklearn.model_selection import train_test_split

# Assuming you have a target variable 'target'
# X = features
# y = logs['target']
# For unsupervised learning, we only use X
X = features

# Split the data into training and testing sets
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)


# 4. Model Selection

from sklearn.ensemble import IsolationForest

# Initialize the model
model = IsolationForest(contamination=0.01, random_state=42)

# 5.Model Training

# Train the model
model.fit(X_train)


# 6. Model Evaluation

# Predict anomalies
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Convert predictions from {-1, 1} to {0, 1}
y_pred_train = [0 if x == 1 else 1 for x in y_pred_train]
y_pred_test = [0 if x == 1 else 1 for x in y_pred_test]

# Calculate performance metrics
from sklearn.metrics import classification_report, confusion_matrix

print("Training Data Evaluation:")
print(confusion_matrix(X_train, y_pred_train))
print(classification_report(X_train, y_pred_train))

print("Testing Data Evaluation:")
print(confusion_matrix(X_test, y_pred_test))
print(classification_report(X_test, y_pred_test))


# 7. Model Tuning

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'contamination': [0.01, 0.02, 0.05, 0.1]
}

# Initialize Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=5)

# Fit the model
grid_search.fit(X_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")


# 8. Deployment and Monitoring
from azureml.core import Workspace

# Create a workspace
ws = Workspace.create(name='myworkspace',
                      subscription_id='yoursubscriptionid',
                      resource_group='yourresourcegroup',
                      create_resource_group=True,
                      location='eastus')

from azureml.core import Model

# Register the model
model_path = 'path/to/your/model'
model = Model.register(workspace=ws, model_path=model_path, model_name='threat-detection-model')

from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig

# Define the inference configuration
inference_config = InferenceConfig(entry_script='score.py', environment=myenv)

# Define the deployment configuration
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy the model
service = Model.deploy(workspace=ws,
                       name='threat-detection-service',
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deployment_config)
service.wait_for_deployment(show_output=True)
print(service.state)
