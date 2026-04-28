#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


# In[2]:


import os
os.getcwd()


# In[3]:


#uploading datasets
#production event data
df_erp = pd.read_csv('D:\\IVY PRO SCHOOL\\data science internship\\datasiet\\Apex Dataset\\datatype\\erp.csv')
df_mes1 = pd.read_csv('D:\\IVY PRO SCHOOL\\data science internship\\datasiet\\Apex Dataset\\datatype\\mes1.csv')
df_mes2 = pd.read_csv('D:\\IVY PRO SCHOOL\\data science internship\\datasiet\\Apex Dataset\\datatype\\mes2.csv')
df_qms1 = pd.read_csv('D:\\IVY PRO SCHOOL\\data science internship\\datasiet\\Apex Dataset\\datatype\\qms1.csv')
df_qms2 = pd.read_csv('D:\\IVY PRO SCHOOL\\data science internship\\datasiet\\Apex Dataset\\datatype\\qms2.csv')
df_scada = pd.read_csv('D:\\IVY PRO SCHOOL\\data science internship\\datasiet\\Apex Dataset\\datatype\\scada.csv')
#Reference data
df_mm= pd.read_csv('D:\\IVY PRO SCHOOL\\data science internship\\datasiet\\Apex Dataset\\datatype\\mm.csv')
df_om = pd.read_csv('D:\\IVY PRO SCHOOL\\data science internship\\datasiet\\Apex Dataset\\datatype\\om.csv')


# In[4]:


#data information
datasets = {
    "ERP": df_erp,
    "MES Part 1": df_mes1,
    "MES Part 2": df_mes2,
    "QMS Part 3": df_qms1,
    "QMS Part 4": df_qms2,
    "SCADA": df_scada,
    "Machine Master": df_mm,
    "Operator Master": df_om
}

for name, df in datasets.items():
    print(f"{name} shape: {df.shape}")


# In[5]:


#Task1:CONCATENATION
df_all = pd.concat([df_erp, df_mes1, df_mes2, df_qms1, df_qms2, df_scada], ignore_index=True)


# In[6]:


print("Shape of integrated datasets:", df_all.shape)
df_all.head()


# In[7]:


#structure of integrated dataset
print(df_all.info())
print(df_all.describe())


# In[8]:


#task1:
df_all = df_all.merge(df_mm, on='machine_id', how='left')


# In[9]:


#task1
df_all = df_all.merge(df_om, on='operator_id', how='left')


# In[10]:


print("shape of final integrated dataset :", df_all.shape)
df_all.head()


# In[11]:


df_all.shape


# In[12]:


df_all.info()


# In[13]:


df_all.describe()


# In[14]:


missing_values = df_all.isnull().sum().sort_values(ascending=False)
missing_values


# In[17]:


missing_count = df_all.isnull().sum().sort_values(ascending=False)
missing_percentage = (df_all.isnull().sum() / len(df_all)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing %': missing_percentage
})

missing_df = missing_df.sort_values(by='Missing %', ascending=False)
missing_df


# In[15]:


# Classify Columns
# PURPOSE: Separate numerical and categorical columns

numerical_cols = df_all.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df_all.select_dtypes(include=['object']).columns

print("Numerical Columns:", len(numerical_cols))
print("Categorical Columns:", len(categorical_cols))


# In[16]:


# PURPOSE: Fill missing values in numerical columns with median

for col in numerical_cols:
    df_all[col].fillna(df_all[col].median(), inplace=True)


# In[17]:


# PURPOSE: Fill missing values in categorical columns

for col in categorical_cols:
    df_all[col].fillna(df_all[col].mode()[0], inplace=True)


# In[18]:


# PURPOSE: Ensure no missing values remain

df_all.isnull().sum().sum()


# In[19]:


missing_count = df_all.isnull().sum().sort_values(ascending=False)
missing_count


# In[20]:


#count of duplicate values
duplicates = df_all.duplicated().sum()
print("Number of duplicate rows:", duplicates)


# In[21]:


#drop duplicate
df_all.drop_duplicates(inplace=True)


# In[22]:


duplicates = df_all.duplicated().sum()
print("Number of duplicate rows:", duplicates)


# In[23]:


#count unique values
df_all.nunique()


# In[24]:


#drop columns
df_all.drop(['machine_id', 'operator_id','source_system','batch_id'], axis=1, inplace=True)


# In[25]:


df_all['prod_date'] = pd.to_datetime(df_all['prod_date'], format='%d-%m-%Y')
df_all['month'] = df_all['prod_date'].dt.month
df_all['day'] = df_all['prod_date'].dt.day
df_all['shift_day'] = df_all['prod_date'].dt.dayofweek


# In[26]:


df_all.head(10)


# In[27]:


#rename month, day, shiftday column
df_all.rename(columns={
    'month': 'prod_month',
    'day': 'prod_day',
    'shift_day': 'prod_shift_day',
})


# In[28]:


#drop prod date column
df_all.drop(['prod_date'], axis=1, inplace=True)


# In[29]:


#drop joining date
df_all.drop(['joining_date'], axis=1, inplace=True)


# In[30]:


#calculate machine age
import datetime
current_year = datetime.datetime.now().year
df_all['commissioned_date'] = pd.to_datetime(df_all['commissioned_date'], format='%d-%m-%Y')
df_all['machine_age'] = current_year - df_all['commissioned_date'].dt.year


# In[31]:


df_all.nunique()


# In[32]:


#drop home location
df_all.drop(['home_location'], axis=1, inplace=True)


# In[33]:


df_all.nunique()


# In[34]:


#Ctask 2: lassify Columns
#o Divide all columns into:
#▪ Numerical variables (numbers, measurements)
#▪ Categorical variables (labels, categories)
num_cols = df_all.select_dtypes(include=['int64', 'float64']).columns


cat_cols = df_all.select_dtypes(include=['object']).columns

print("Numerical Columns:\n", num_cols)
print("\nCategorical Columns:\n", cat_cols)


# In[35]:


#Task3:UNIVARATE ANALYSIS OF defect_flag column
import matplotlib.pyplot as plt
import seaborn as sns

grouped_defect = df_all.groupby('defect_flag').size()

grouped_defect.plot(kind='bar', color=['green', 'red'])


# In[36]:


#task3: Univariate analysis of nemeric columns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Select columns
cols = [
    'speed_rpm',
    'pressure_bar',
    'vibration_mm_s',
    'oil_temp_c',
    'torque_nm',
    'downtime_min',
    'scrap_pct'
]


df_clean = df_all[cols].copy()
df_clean = df_clean.fillna(df_clean.median())


scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean)


scaled_df = pd.DataFrame(scaled_data, columns=cols)


plt.figure()
scaled_df.boxplot()

plt.title('Fixed Boxplot for Numeric Features')
plt.xticks(rotation=45)
plt.ylabel('Scaled Values')

plt.show()


# In[37]:


#univariate analysis for measuring the distribution
import seaborn as sns
import matplotlib.pyplot as plt

cols = [
    'speed_rpm',
    'pressure_bar',
    'vibration_mm_s',
    'oil_temp_c',
    'torque_nm',
    'downtime_min',
    'scrap_pct'
]

plt.figure(figsize=(12,8))

for i, col in enumerate(cols, 1):
    plt.subplot(3,3,i)
    sns.histplot(df_all[col], kde=True)
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()


# In[38]:


#drawing boxplots to perform bivariate analysis on the  numeric columns againts target variables
import matplotlib.pyplot as plt

# Define numerical columns (exclude target)
num_cols = [
    'planned_units','good_units','downtime_min','changeover_min',
    'overtime_min','speed_rpm','pressure_bar','vibration_mm_s',
    'oil_temp_c','torque_nm','ambient_temp_c','ambient_humidity_pct',
    'actual_cycle_time_sec','scrap_pct','rework_min',
    'rated_speed_rpm','rated_pressure_bar','experience_years',
    'training_hours_last_90d'
]

# Plot boxplots
plt.figure(figsize=(15,12))

for i, col in enumerate(num_cols, 1):
    plt.subplot(6,4,i)
    df_all.boxplot(column=col, by='defect_flag')
    plt.title(col)
    plt.suptitle('')  # remove automatic title

plt.tight_layout()
plt.show()


# In[39]:


#code for anova test
# Define numerical columns
from scipy.stats import f_oneway
num_cols = [
    'planned_units','good_units','downtime_min','changeover_min',
    'overtime_min','speed_rpm','pressure_bar','vibration_mm_s',
    'oil_temp_c','torque_nm','ambient_temp_c','ambient_humidity_pct',
    'actual_cycle_time_sec','scrap_pct','rework_min',
    'rated_speed_rpm','rated_pressure_bar','experience_years',
    'training_hours_last_90d'
]

results = []

for col in num_cols:
    group0 = df_all[df_all['defect_flag'] == 0][col]
    group1 = df_all[df_all['defect_flag'] == 1][col]

    # Perform ANOVA
    f_stat, p_value = f_oneway(group0, group1)

    results.append((col, f_stat, p_value))

# Convert to DataFrame
import pandas as pd

anova_df = pd.DataFrame(results, columns=['Feature', 'F-Statistic', 'p-value'])

# Sort by p-value
anova_df = anova_df.sort_values(by='p-value')

print(anova_df)


# In[40]:


#Task3:The following numeric columns are highly correlated with the target (prediction) variable.
important_features = anova_df[anova_df['p-value'] < 0.05]
print(important_features)


# In[41]:


#task3: drawing grouped bars for performing bivariate analysis on the categorial columns
import matplotlib.pyplot as plt
import seaborn as sns

# Define categorical columns
cat_cols = [
    'shift_code',
    'product_code',
    'line_id',
    'area',
    'machine_type',
    'sensor_pack',
    'vendor_type',
    'role',
    'criticality',
    'skill_level'
]

plt.figure(figsize=(15,12))

for i, col in enumerate(cat_cols, 1):
    plt.subplot(5,2,i)

    # Countplot grouped by target
    sns.countplot(x=col, hue='defect_flag', data=df_all)

    plt.title(col)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# In[42]:


#Task3:chi square test for categorical columns
import pandas as pd
from scipy.stats import chi2_contingency

# Define categorical columns
cat_cols = [
    'shift_code',
    'product_code',
    'line_id',
    'area',
    'machine_type',
    'sensor_pack',
    'vendor_type',
    'role',
    'criticality',
    'skill_level'
]

results = []

for col in cat_cols:
    # Create contingency table
    contingency_table = pd.crosstab(df_all[col], df_all['defect_flag'])

    # Perform Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    results.append((col, chi2, p))

# Convert to DataFrame
chi_df = pd.DataFrame(results, columns=['Feature', 'Chi2 Statistic', 'p-value'])

# Sort by p-value
chi_df = chi_df.sort_values(by='p-value')

print(chi_df)


# In[43]:


#Task3:The following categorical columns are highly correlated with the target (prediction) variable.
important_cat = chi_df[chi_df['p-value'] < 0.05]

print(important_cat)


# In[44]:


#one hot encoding
df_all = pd.get_dummies(df_all, columns=[
    'shift_code',
    'product_code',
    'line_id',
    'area',
    'machine_type',
    'sensor_pack',
    'vendor_type',
    'role'
], drop_first=True)


# In[45]:


#label encoder
le = LabelEncoder()
df_all['criticality'] = le.fit_transform(df_all['criticality'])
df_all['skill_level'] = le.fit_transform(df_all['skill_level'])


# In[46]:


df_all.shape


# In[61]:


from sklearn.model_selection import train_test_split


# ── 1. Define selected features ───────────────────────────────────────────────
numerical_cols = [
    'rework_min',
    'scrap_pct',
    'good_units',
    'ambient_temp_c',
    'ambient_humidity_pct',
    'oil_temp_c',
    'training_hours_last_90d',
    'downtime_min',
    'pressure_bar',
    'machine_age'
]

categorical_cols = ['shift_code', 'role']

target = 'defect_flag'


# Get the new encoded column names
encoded_cat_cols = [c for c in df_all.columns
                    if c.startswith('shift_code_') or c.startswith('role_')]

# ── 3. Combine all features ───────────────────────────────────────────────────
feature_cols = numerical_cols

X = df_all[feature_cols]
y = df_all[target]

# ── 4. Train-test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # preserves class balance in both splits
)

# Step 3: Check shape
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)



# In[62]:


from sklearn.linear_model import LogisticRegression

# Initialize model
model = LogisticRegression(max_iter=1000)

# Train
model.fit(X_train, y_train)


# In[63]:


y_pred = model.predict(X_test)


# In[64]:


#training accuracy
from sklearn.metrics import accuracy_score

y_train_pred = model.predict(X_train)

train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)


# In[65]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Test_Accuracy:", accuracy)


# In[66]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[67]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[ ]:


#As per the above classification report and accuracy score it can be said that the performance of the model is very good.
#The difference betwen actual and predicted values is very less.
#As 91% recall of positive defect_flag the model can 9 out of 10 defect products.


# In[68]:


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# In[69]:


import pandas as pd

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print(coef_df)


# In[56]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm


# In[57]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted Non-Defect", "Predicted Defect"],
    yticklabels=["Actual Non-Defect", "Actual Defect"]
)

plt.title("Confusion Matrix – Logistic Regression")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()


# In[70]:


actual_vs_pred = pd.DataFrame({
    "Actual_Defect": y_test.values,
    "Predicted_Defect": y_pred,
    "Predicted_Probability": y_prob
})

actual_vs_pred.head(10)


# In[71]:


import pickle
import streamlit as st
import numpy as np

# Save the trained model to a .pkl file
with open('defect_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the model
with open("defect_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🔍 Defect Prediction System")

st.write("Enter production details to predict defect")

# Input fields
rework_min = st.number_input("Rework Time (min)", 0.0)
scrap_pct = st.number_input("Scrap Percentage", 0.0)
good_units = st.number_input("Good Units", 0.0)
ambient_temp = st.number_input("Ambient Temperature", 0.0)
humidity = st.number_input("Humidity (%)", 0.0)
oil_temp = st.number_input("Oil Temperature", 0.0)
training_hours = st.number_input("Training Hours", 0.0)
downtime = st.number_input("Downtime (min)", 0.0)
pressure = st.number_input("Pressure (bar)", 0.0)
machine_age = st.number_input("Machine Age", 0.0)

# Prediction button
if st.button("Predict"):

    features = np.array([[rework_min, scrap_pct, good_units,
                          ambient_temp, humidity, oil_temp,
                          training_hours, downtime, pressure,
                          machine_age]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f" Defect Likely (Probability: {probability:.2f})")
    else:
        st.success(f" No Defect (Probability: {probability:.2f})")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




