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
- Python 3.x
- **Libraries**:
  - `pandas`, `numpy` – data handling
  - `matplotlib`, `seaborn` – visualization
  - `scikit-learn` – model building & metrics
  - `xgboost` – main regressor

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

## 🧪 How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/boston-house-price-xgboost.git
   cd boston-house-price-xgboost

