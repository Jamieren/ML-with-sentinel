		import pandas as pd
		from azure.identity import DefaultAzureCredential
		from azure.monitor.query import LogsQueryClient
		from sklearn.ensemble import IsolationForest
		
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
		
		# Train the Isolation Forest model
		model = IsolationForest(contamination=0.01)
		model.fit(features)
		
		# Predict anomalies
		logs['anomaly_score'] = model.decision_function(features)
		logs['anomaly'] = model.predict(features)
		
		# Filter the anomalies
		anomalies = logs[logs['anomaly'] == -1]

    		print(anomalies)
