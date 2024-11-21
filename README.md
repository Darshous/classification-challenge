
# Spam Detector

## Overview

This project focuses on developing a spam detection system using machine learning techniques. The dataset utilized is sourced from the [UCI Machine Learning Library](https://archive.ics.uci.edu/dataset/94/spambase) and contains features derived from email content. The system implements two classification models—logistic regression and random forest—to evaluate their performance and select the most effective approach.

## Dataset

The data for this project can be retrieved from the following link:

[Spam Data](https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv)

### Features

The dataset includes:
- Word frequencies
- Capitalization patterns
- Various numerical features extracted from emails

The target column indicates whether the email is spam (1) or not (0).

## Project Steps

1. **Data Import and Cleaning**
   - Import the data using Pandas.
   - Examine and clean the dataset to handle missing values or outliers.

2. **Exploratory Data Analysis (EDA)**
   - Visualize feature distributions and relationships.
   - Identify key attributes that may influence classification.

3. **Feature Engineering**
   - Scale features for uniformity using standardization techniques.
   - Select relevant features based on their importance.

4. **Model Development**
   - Build and train logistic regression and random forest models.
   - Evaluate models using accuracy, precision, recall, and F1-score.

5. **Results**
   - Compare the performance of both models.
   - Select the most effective model for deployment.

## Example Code

Below is a sample snippet demonstrating the initial steps of the project:

```python
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the data
url = "https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv"
data = pd.read_csv(url)

# Preview the dataset
print(data.head())
```

## Requirements

The project requires the following Python libraries:
- pandas
- scikit-learn

You can install these packages using `pip`:

```bash
pip install pandas scikit-learn
```

## Results

After training both models, the Random Forest model achieved a higher accuracy and is selected as the preferred model. Full evaluation metrics and charts are included in the `spam_detector.ipynb` notebook.

## Usage

To use this spam detector, run the `spam_detector.ipynb` notebook in a Jupyter environment. The notebook contains all code and instructions for reproducing the results.
