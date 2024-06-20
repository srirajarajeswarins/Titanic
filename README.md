The dataset used in this project can be found on [Kaggle's Titanic dataset page](https://www.kaggle.com/c/titanic/data).

## Feature Engineering

1. **Handling Missing Values**:
    - Filled missing values in the 'Age' column with the median age.
    - Filled missing values in the 'Embarked' column with the most frequent value (mode).
    - Replaced missing values in the 'Cabin' column with 'Unknown'.
    - Dropped the 'Cabin' column due to high cardinality.

2. **Converting Categorical Variables**:
    - Converted 'Sex' into a binary variable (0 for female, 1 for male).
    - One-hot encoded the 'Embarked' column.

3. **Creating New Features**:
    - Created a 'FamilySize' feature by combining 'SibSp' and 'Parch'.

## Models Used

- **Logistic Regression**
- **Random Forest**
- **Gradient Boosting**
- **CatBoost**

  ## Usage

1. Ensure you have the Titanic dataset CSV file in the project directory.
2. Run the Python script:
    ```bash
    python titanic.py
    ```
To run the catBoost model, do install catBoost classifier using the below command
pip install catboost

## Results

After running the script, you will see the accuracy scores for each classifier and a bar chart comparing their performances.
