# E-Commerce Sales Analysis

## **Overview**
This project aims to analyze E-Commerce sales data to identify customers who are likely to make repeat purchases. By leveraging machine learning techniques, this analysis supports decision-making for enhancing customer retention strategies.

---

## **Project Workflow**
1. **Data Preprocessing**
    - Handled missing values using mean imputation for numeric columns.
    - Encoded categorical features using Label Encoding.
    - Standardized numerical features for consistency.
    - Removed irrelevant features (e.g., datetime columns).

2. **Handling Class Imbalance**
    - Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset by increasing samples of the minority class.

3. **Model Training and Evaluation**
    - Trained and evaluated three machine learning models:
      - **Logistic Regression** (Baseline model)
      - **Random Forest** (Ensemble model)
      - **XGBoost** (Boosting model)
    
4. **Performance Metrics**
    - Accuracy, Precision, Recall, and F1-score were used to evaluate model performance.

---

## **Results**
### Logistic Regression
- **Accuracy**: 0.59
- **Observation**:
  - High precision for class 0 (did not buy again), but poor recall for class 1 (buy again).
  - Indicates difficulty in identifying potential repeat buyers.

### Random Forest
- **Accuracy**: 0.90
- **Observation**:
  - Best overall accuracy and balanced performance.
  - Slight improvement in detecting repeat buyers (class 1) compared to Logistic Regression.

### XGBoost
- **Accuracy**: 0.88
- **Observation**:
  - High accuracy for class 0, but very poor recall for class 1.
  - Struggles to handle class imbalance effectively.

---

## **Insights and Recommendations**
1. **Current Findings**:
    - Random Forest performed best among the tested models.
    - SMOTE improved class balance but did not fully address recall for the minority class.
    - Data imbalance and feature engineering remain critical areas for improvement.

2. **Future Steps**:
    - Incorporate additional features or refine existing ones to improve model prediction.
    - Experiment with other models like CatBoost or LightGBM, which are optimized for imbalanced datasets.
    - Use cost-sensitive learning techniques or class weighting to improve recall for class 1.

3. **Business Applications**:
    - Use the model to identify and target high-potential customers with tailored marketing strategies.
    - Continuously monitor misclassifications to refine the model over time.

---

## **How to Run**
1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute the notebook:
   ```bash
   jupyter notebook E_commers_Sales.ipynb
   ```

4. Analyze results and modify as needed.

---

## **Dependencies**
The following libraries were used in this project:
- Python 3.10+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost

---

## **Acknowledgments**
- The data preprocessing and modeling pipeline are inspired by best practices in machine learning for class imbalance.
- Thanks to the open-source community for providing the libraries and tools used in this project.

---


Python version: 3.10.12
LabelEncoder (from sklearn): 1.6.0
LogisticRegression (from sklearn): 1.6.0
RandomForestClassifier (from sklearn): 1.6.0
SMOTE (from imblearn): 0.12.4
SimpleImputer (from sklearn): 1.6.0
XGBClassifier (from xgboost): 2.1.3
accuracy_score (from sklearn): 1.6.0
confusion_matrix (from sklearn): 1.6.0
files (from google): Built-in Google Colab module
google (colab): Built-in Google Colab module
imblearn: 0.12.4
matplotlib: 3.8.0
numpy: 1.26.4
os: Built-in Python module
pandas: 2.2.2
seaborn: 0.13.2
sklearn: 1.6.0
train_test_split (from sklearn): 1.6.0
xgboost: 2.1.3
zipfile: Built-in Python module
