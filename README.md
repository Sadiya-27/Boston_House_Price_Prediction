# Boston_House_Price_Prediction

# 🏡 Boston House Price Prediction using XGBoost Regressor

This project is an interactive and educational notebook that demonstrates how to predict house prices using machine learning models, with a focus on the **XGBoost Regressor**. The notebook includes both code and theoretical explanations, making it perfect for learners and practitioners.

---

## 📘 Project Description

The Boston Housing dataset is used to build and evaluate regression models that predict house prices based on features like crime rate, number of rooms, property tax, etc. The notebook guides users through:

- Exploratory Data Analysis (EDA)
- Model building using various regressors
- Performance comparison using evaluation metrics
- Visualizations for insights
- Predicting house prices for new data points
- Theoretical background for XGBoost

---

## 🛠️ Technologies & Libraries

- **Jupyter Notebook**
- **Streamlit** (for deploying the model as a web app)
- Python 3.x
- **Libraries**:
  - `pandas`, `numpy` – data handling
  - `matplotlib`, `seaborn` – visualization
  - `scikit-learn` – model building & metrics
  - `xgboost` – main regressor

---

## 🏡 Dataset: Boston Housing

This project uses the **Boston Housing Dataset**, a classic dataset for regression tasks. It contains information collected by the U.S Census Service concerning housing in the area of Boston, Massachusetts. The goal is to predict the **median value of owner-occupied homes** (`PRICE`) based on various features.

---

### 📋 Features Description

| Feature    | Description |
|------------|-------------|
| **CRIM**   | Per capita crime rate by town |
| **ZN**     | Proportion of residential land zoned for lots over 25,000 sq. ft. |
| **INDUS**  | Proportion of non-retail business acres per town |
| **CHAS**   | Charles River dummy variable (1 if tract bounds river; 0 otherwise) |
| **NOX**    | Nitric oxide concentration (parts per 10 million) |
| **RM**     | Average number of rooms per dwelling |
| **AGE**    | Proportion of owner-occupied units built before 1940 |
| **DIS**    | Weighted distance to five Boston employment centers |
| **RAD**    | Index of accessibility to radial highways |
| **TAX**    | Property tax rate per $10,000 |
| **PTRATIO**| Pupil-teacher ratio by town |
| **B**      | 1000(Bk - 0.63)^2, where Bk is the proportion of Black residents |
| **LSTAT**  | Percentage of lower status of the population |
| **PRICE**  | *Target Variable* — Median value of owner-occupied homes (in $1000s) |

---

### 🧾 Sample Data (First 5 Rows)

| CRIM     | ZN  | INDUS | CHAS | NOX  | RM    | AGE  | DIS   | RAD | TAX | PTRATIO | B      | LSTAT | PRICE |
|----------|-----|--------|------|------|-------|------|--------|-----|-----|----------|--------|--------|--------|
| 0.00632  | 18  | 2.31   | 0    | 0.538| 6.575 | 65.2 | 4.0900 | 1   | 296 | 15.3     | 396.90 | 4.98   | 24.0   |
| 0.02731  | 0   | 7.07   | 0    | 0.469| 6.421 | 78.9 | 4.9671 | 2   | 242 | 17.8     | 396.90 | 9.14   | 21.6   |
| 0.02729  | 0   | 7.07   | 0    | 0.469| 7.185 | 61.1 | 4.9671 | 2   | 242 | 17.8     | 392.83 | 4.03   | 34.7   |
| 0.03237  | 0   | 2.18   | 0    | 0.458| 6.998 | 45.8 | 6.0622 | 3   | 222 | 18.7     | 394.63 | 2.94   | 33.4   |
| 0.06905  | 0   | 2.18   | 0    | 0.458| 7.147 | 54.2 | 6.0622 | 3   | 222 | 18.7     | 396.90 | 5.33   | 36.2   |

---

### 🎯 Objective

The primary objective is to build a machine learning model using **XGBoost Regressor** that can accurately predict housing prices based on the above features.

---

## 📊 Models Implemented

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor ✅ (best performing)

Each model is evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

---

## 📌 Highlights

- ✅ **Theoretical explanations** for evaluation metrics and XGBoost algorithm
- 📈 **Visualizations**: Bar charts, scatter plots (Actual vs Predicted)
- 💡 **New data prediction** using XGBoost
- 🧠 **Taylor expansion and formula** for XGBoost explained in simple terms

---

## XGBoost Regressor

## 🌟 What is XGBoost Regressor?
- XGBoost (Extreme Gradient Boosting) is an advanced, efficient implementation of gradient boosting for supervised learning tasks like regression and classification.
- The XGBoost Regressor is specifically designed to predict continuous numerical values (e.g., house prices), and it is known for its:
  - High performance
  - Speed
  - Scalability
  - Accuracy

## ⚙️ How Does It Work?
XGBoost builds a series of decision trees sequentially, where:
- Each new tree tries to correct the errors made by the previous trees.
- Instead of predicting the final target directly, each tree predicts the residuals (errors) of the previous predictions.
- The final prediction is the sum of the predictions from all trees.
- This is called gradient boosting, because the model learns by minimizing a loss function using gradient descent.

## 📈 Key Features
Feature	Description
| Feature                   | Description                                                         |
| ------------------------- | ------------------------------------------------------------------- |
| **Boosting Algorithm**    | Combines many weak learners (small trees) to create a strong model. |
| **Regularization**        | Prevents overfitting by using L1 (Lasso) and L2 (Ridge) penalties.  |
| **Handling Missing Data** | Smart handling of missing values during training.                   |
| **Parallel Processing**   | Fast training using multiple CPU cores.                             |
| **Tree Pruning**          | Uses a depth-first approach and prunes trees for optimal size.      |
| **Custom Loss Functions** | You can define your own objective functions.                        |

## 🧪 Why Use XGBoost for Regression?
- Outperforms many other models (e.g., Decision Trees, Random Forests) on structured/tabular data.
- Works well even with nonlinear relationships between features and target.
- Offers strong tools to tune performance using hyperparameters like max_depth, learning_rate, n_estimators, etc.

## How It Works
![image](https://github.com/user-attachments/assets/74bbbe55-ed70-40ee-9aa4-5b27bbacfcd7)

Here’s a more detailed look at how XGBoost works:
1. Initial Prediction:
   XGBoost starts by making a simple prediction on the training data, often using the average of the target variable.
Error Calculation: It then calculates the residuals, which are the differences between the predicted values and the actual values in the training data. Essentially, these residuals represent the errors in the initial prediction.

2. Building the First Decision Tree:
   XGBoost builds the first decision tree in the ensemble. This tree focuses on learning these residuals, aiming to minimize the overall error. To do this, the algorithm finds the best split points in the features that will reduce the errors the most.
   
3. Subsequent Trees and Error Correction:
   Here’s where the magic of gradient boosting happens. XGBoost doesn’t discard the previous tree. Instead, it uses the residuals again, but this time for the predictions made by the entire ensemble so far (including the first tree). The new tree specifically targets these remaining errors, further improving the model’s accuracy.
   
4. Minimizing Loss Function:
   Throughout the process, XGBoost optimizes a loss function. This function mathematically measures how well the model’s predictions match the actual values. By minimizing the loss function, XGBoost ensures the ensemble is on the right track to make accurate predictions.
   
5. Stopping Criteria:
   XGBoost adds trees until a certain stopping criteria is met. These criteria could be a maximum number of trees, a minimum improvement in the loss function, or reaching a certain level of accuracy.

---
## 📘 Example: Understanding How XGBoost Works (Simplified)

Let’s consider a toy dataset with one feature (`RM` - number of rooms) and corresponding house prices.

### Dataset

| RM (x) | Price (y) |
|--------|-----------|
| 4      | 200       |
| 5      | 250       |
| 6      | 300       |
| 7      | 350       |

---

### Step 1: Initial Prediction

Start with the mean of `y` as the initial prediction:

**Initial Prediction (f₀):**
y_mean = (200 + 250 + 300 + 350) / 4 = 275


| x | Actual y | f₀(x) | Residual = y - f₀(x) |
|---|----------|-------|----------------------|
| 4 | 200      | 275   | -75                  |
| 5 | 250      | 275   | -25                  |
| 6 | 300      | 275   | +25                  |
| 7 | 350      | 275   | +75                  |

---

### Step 2: Train Tree on Residuals

**Tree 1 Logic (simple):**
- If RM ≤ 5 → output = -50  
- If RM > 5 → output = +50

---

### Step 3: Update Predictions

**Learning rate (η) = 0.1**

**Updated Prediction (f₁):**
f₁(x) = f₀(x) + η * Tree₁(x)

| x | f₀(x) | Tree₁(x) | f₁(x) |
|---|-------|----------|--------|
| 4 | 275   | -50      | 270    |
| 5 | 275   | -50      | 270    |
| 6 | 275   | +50      | 280    |
| 7 | 275   | +50      | 280    |

---

### Step 4: Compute New Residuals

| x | y | f₁(x) | New Residual |
|---|---|--------|--------------|
| 4 | 200 | 270  | -70          |
| 5 | 250 | 270  | -20          |
| 6 | 300 | 280  | +20          |
| 7 | 350 | 280  | +70          |

Repeat this process with new trees and residuals to refine the predictions further.

---

### 🔁 Final Model (after n trees):

f(x) = f₀(x) + η * tree₁(x) + η * tree₂(x) + ... + η * treeₙ(x)

This example demonstrates the core idea behind **gradient boosting** used in **XGBoost**.

---

## 🧪 How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/boston-house-price-xgboost.git
   cd boston-house-price-xgboost
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the notebook**:
   ```bash
   jupyter notebook Boston_House_Price_Prediction.ipynb
   ```

4. **Run Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```
---

> ⭐ If you found this project helpful, give it a star!

