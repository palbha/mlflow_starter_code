# MLflow Started Code

Welcome to the **MLflow Started Code** repository! This repository provides a hands-on example of how to use MLflow for tracking experiments, comparing models, and managing machine learning workflows. 

## Project Overview

This project demonstrates how to:
- Create and manage MLflow experiments.
- Train and evaluate multiple machine learning models (Decision Tree and Random Forest).
- Log metrics, parameters, feature importances, and predictions.
- Save and load models using MLflow.

## Repository Structure

- **`run_experiments.py`**: Script to train models, log metrics, and save artifacts.
- **`requirements.txt`**: Dependencies for the project.
- **`results/`**: Directory where logs, model artifacts, and predictions will be saved.

![MLflow Logo](https://mlflow.org/img/mlflow-black.svg)  
*Source: [MLflow Documentation](https://mlflow.org)*

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/palbha/mlflow_started_code.git
cd mlflow_started_code
```


### 2. Install Dependencies
Make sure you have Python 3.6+ installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```
### 3.Usage
Running the Experiment Script
Execute the run_experiments.py script to start the MLflow experiment
```bash
python run_experiments.py 
```
Once the file ran completely fine, take a look at the mlflow UI to see the results from your experiments 
```bash
mlflow server --host 127.0.0.1 --port 8080
```

Open your browser & go to http://127.0.0.1:8080/ & You can see th experiments

![image](https://github.com/user-attachments/assets/409ad22f-bed4-4843-824a-a400ad994461)

Click on any experiments & take a look at artifacts to analyse & see the output further

![image](https://github.com/user-attachments/assets/4ca6fa09-09c9-4072-a5b1-eee5313698d2)

One can also download the details of each run to create their own custom graphs & share results with stakeholders 

![image](https://github.com/user-attachments/assets/1ddab6bb-7d5f-42a2-8220-f1b58484f679)


