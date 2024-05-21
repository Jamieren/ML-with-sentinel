# Threat Hunting with Machine Learning using Python and Microsoft Sentinel

This repository provides a step-by-step guide on using machine learning (ML) for threat hunting by interacting with Microsoft Sentinel logs. The implementation is done in Python, leveraging various libraries for data processing and ML model training.

## Methodology

1. **Data Collection and Understanding**
   - Collect logs from Microsoft Sentinel or any other logging system.
   - Understand the structure and content of the logs.

2. **Data Preprocessing**
   - Clean the data by handling missing values, removing duplicates, and normalizing data.
   - Feature engineering to extract relevant features from the logs.

3. **Data Splitting**
   - Split the data into training and testing sets to evaluate the performance of the model.

4. **Model Selection**
   - Choose appropriate machine learning models based on the problem type (e.g., anomaly detection, classification).

5. **Model Training**
   - Train the model using the training data.

6. **Model Evaluation**
   - Evaluate the model using the testing data to assess its performance.

7. **Model Tuning**
   - Tune the model hyperparameters to improve performance.

8. **Deployment and Monitoring**
   - Deploy the model for real-time threat detection and continuously monitor its performance.

## Step-by-Step Implementation

### Prerequisites

- **Azure Subscription**: Ensure you have an active Azure subscription.
- **Microsoft Sentinel**: Set up Microsoft Sentinel in your Azure environment.
- **Python Environment**: Have Python installed along with necessary libraries.

### Installation

Install the required Python libraries:

```bash
pip install azure-identity azure-monitor-query pandas scikit-learn

### Implementation

For the complete step-by-step implementation, please refer to the Implementation.py file.
