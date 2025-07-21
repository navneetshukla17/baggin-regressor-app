# 💊 Bagging Regressor App – Medical Cost Prediction

This Streamlit web app demonstrates the use of **Bagging Regression** with different base estimators to predict **medical insurance charges** using the **Medical Cost Personal Dataset**. It allows users to interactively explore how different parameters affect model performance.

---

## 🚀 Features

- 📁 Uses the [Medical Cost Personal Dataset](https://www.kaggle.com/mirichoi0218/insurance)
- 🧠 Implements Bagging Regressor with:
  - Decision Tree
  - Support Vector Regressor (SVR)
  - K-Nearest Neighbors Regressor (KNN)
- 🎛️ Customizable parameters:
  - Number of estimators
  - Bootstrap sampling
  - Row & Column sampling
- 📊 Displays performance metrics:
  - R² Score
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
- 📈 Scatter plot of actual vs predicted charges
- 🎨 Clean and interactive UI with dynamic result highlighting

---

## 🛠️ Technologies Used

- Python 3
- Streamlit
- scikit-learn
- pandas
- matplotlib
- seaborn

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
2. Install Dependencies
If you're using a virtual environment:
pip install -r requirements.txt
Or manually install:
pip install streamlit scikit-learn pandas seaborn matplotlib
3. Run the App
streamlit run app.py
🧾 Dataset Info
📄 File: insurance.csv
Features include:
Age
BMI
Children
Smoker (yes/no)
Region
Sex
Target: charges (medical insurance cost)
📷 Preview
✨ Future Improvements
Add prediction form for user input
Include feature importance visualization
Add GridSearchCV for optimal hyperparameter tuning
🙋‍♂️ Author
Navneet Shukla
📧 LinkedIn
🔗 GitHub
📄 License
This project is licensed under the MIT License.
