# Neccessary imports
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

# Set style
plt.style.use('fast')

# Load dataset
df = pd.read_csv('insurance.csv')

# Encode categorical columns
df_encoded = df.copy()
label_cols = ['sex', 'smoker', 'region']
for col in label_cols:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

# Separate input & output
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Streamlit UI
st.sidebar.markdown('# 💊 Bagging Regressor: Medical Cost')

# Base estimator selection
estimator_name = st.sidebar.selectbox(
    'Select base model',
    ('Decision Tree', 'SVM', 'KNN')
)

# Set estimator
if estimator_name == 'Decision Tree':
    estimator = DecisionTreeRegressor()
elif estimator_name == 'SVM':
    estimator = SVR()
else:
    estimator = KNeighborsRegressor()

# n_estimators input
n_estimators = int(st.sidebar.number_input('Number of base models', value=10, min_value=1))

# Max samples
max_samples = st.sidebar.slider('Max Row Samples', 1, len(X_train), 550, step=10)

# Max features
max_features = st.sidebar.slider('Max Column Features', 1, X_train.shape[1], 3, step=1)

# Bootstrap
bootstrap = st.sidebar.radio(
    'Bootstrap Samples (Rows)',
    ('True', 'False')
) == 'True'

# Bootstrap Features
bootstrap_features = st.sidebar.radio(
    'Bootstrap Features (Columns)',
    ('True', 'False'),
    key=123
) == 'True'

# Train & evaluate
if st.sidebar.button('Run Algorithm'):
    # Bagging Regressor
    bagging_model = BaggingRegressor(
        estimator=estimator,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        bootstrap_features=bootstrap_features,
        random_state=42,
        n_jobs=-1
    )

    bagging_model.fit(X_train, y_train)
    y_pred = bagging_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))


    st.markdown("### 📊 <span style='font-size: 26px;'>Model Performance</span>", unsafe_allow_html=True)
    st.markdown(f"""
    <ul style='list-style-type: disc; font-size: 18px; line-height: 2;'>
      <li><strong>R² Score:</strong> <span style='background-color: #222; color: #4CAF50; padding: 4px 10px; border-radius: 8px;'>{r2:.2%}</span></li>
      <li><strong>Mean Absolute Error (MAE):</strong> <span style='background-color: #222; color: #4CAF50; padding: 4px 10px; border-radius: 8px;'>${mae:,.2f}</span></li>
      <li><strong>Root Mean Squared Error (RMSE):</strong> <span style='background-color: #222; color: #4CAF50; padding: 4px 10px; border-radius: 8px;'>${rmse:,.2f}</span></li>
    </ul>
    """, unsafe_allow_html=True)

    # Scatter plot: Actual vs Predicted
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, edgecolors='black', alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel('Actual Charges')
    ax.set_ylabel('Predicted Charges')
    ax.set_title('Actual vs Predicted Charges')
    st.pyplot(fig)

# Visualize dataset
st.subheader('📌 Training Data Overview')
fig_data, ax_data = plt.subplots()
sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', palette='Set1', ax=ax_data)
st.pyplot(fig_data)
