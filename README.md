# Thyroid Prediction Project

## Problem Statement
Thyroid is a small, butterfly-shaped gland in the front of your neck. It makes hormones that control the way the body uses energy. These hormones affect nearly every organ in your body and control many of your body's most important functions.

To diagnose thyroid diseases, your health care provider may use a medical history, physical exam, and thyroid tests. In some cases, your provider may also do a biopsy.

Treatment depends on the problem, how severe it is, and what your symptoms are. Possible treatments may include medicines, radioiodine therapy, or thyroid surgery.

Challenge is to diagnose the disease at early as possible.

```bash
dataset: https://archive.ics.uci.edu/dataset/102/thyroid+disease
```
## Solution Proposed
In this project, solution is focused on prediction of thyroid disease while checking on different test data and evaluation of doctors. Dataset diagnosis column consists of 32 outputs suggesting doctor's diagnosis. For this project as 9172 datapoints are there hence converting column into "yes" and "No". Different experiments are donr to get the higher accuracy.

--------

## Tech Stack
    1.Python
    2.MongoDB
    3.Machine learning algorithms
    4.Docker
    5.apache airflow

## Infrastruture required
    1.Git Actions
    2.AWS S3
    3.AWS EC2
    4.AWS ECR
    5.apache airflow

--------

## How to Run
Before we run the project, make sure that you are having MongoDB Account since we are using MongoDB for data storage. You also need AWS account to access the service like S3, ECR and EC2 instances.

## Training Pipeline

![image]([./config/workspace/Architecture/Training_pipeline.png](https://github.com/Arun02DS/Thyroid_detection/blob/main/Architecture/Project_Architechture.png?raw=true))

## Project Architechture

![image1](./config/workspace/Architecture/Project_Architechture.png)

### Step 1 - Clone repository

```bash
git clone https://github.com/Arun02DS/Thyroid_detection.git
```

### Step 2 - create environment
```bash
conda create -n venv python=3.7.6 -y
```
```bash
conda activate venv
```
### Step 3 - Install the requirements

```bash
pip install -r requirements.txt
```

### Step 4 - Run train.py file

To check the code on your local machine for training the model.

```bash
python train.py
```

### Step 5 - Run main.py file

To check the code on your local machine for batch prediction.

```bash
python main.py
```
### step 6 - environment variable
    - AWS_ACCESS_KEY_ID = ${AWS_ACCESS_KEY_ID}
    - AWS_SECRET_ACCESS_KEY= ${AWS_SECRET_ACCESS_KEY}
    - AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}
    - MONGO_DB_URL = ${MONGO_DB_URL}
    - BUCKET_NAME = ${BUCKET_NAME}
    - AWS_ECR_LOGIN_URI= ${AWS_ECR_LOGIN_URI}
    - ECR_REPOSITORY_NAME= ${ECR_REPOSITORY_NAME}
