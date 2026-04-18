#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import shap
from lime.lime_tabular import LimeTabularExplainer


# In[2]:


DATA_PATH = "pimadiabetes1.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    zero_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    for c in zero_cols:
        df[c] = df[c].replace(0,np.nan)

    df.fillna(df.median(), inplace=True)

    X = df.drop("Outcome",axis=1)
    y = df["Outcome"]

    return df,X,y


# In[ ]:


@st.cache_resource
def train_models(X,y):
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y
    )

    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss"
    )

    lr.fit(X_train,y_train)
    rf.fit(X_train,y_train)
    xgb.fit(X_train,y_train)

    return lr,rf,xgb,X_train,X_test,y_train,y_test


# In[ ]:


def counterfactual_improved(input_df,model):
    base_pred = model.predict(input_df)[0]

    for g_change in range(5,80,5):
        for b_change in range(1,15,1):
            new_df = input_df.copy()

            new_df["Glucose"] = max(0, input_df["Glucose"].iloc[0] - g_change)
            new_df["BMI"] = max(0, input_df["BMI"].iloc[0] - b_change)

            new_pred = model.predict(new_df)[0]

            if new_pred != base_pred:
                return g_change, b_change

    return None, None


# In[ ]:


st.title("Explainable AI Diabetes Prediction")

df,X,y = load_data()
lr,rf,xgb,X_train,X_test,y_train,y_test = train_models(X,y)

model = xgb  # Use best model


# In[ ]:


st.subheader("Model Comparison")

models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "XGBoost": xgb
}

results = []

for name, m in models.items():
    y_pred = m.predict(X_test)
    y_prob = m.predict_proba(X_test)[:,1]

    results.append([
        name,
        round(accuracy_score(y_test,y_pred),3),
        round(roc_auc_score(y_test,y_prob),3)
    ])

results_df = pd.DataFrame(results, columns=["Model","Accuracy","ROC-AUC"])
st.table(results_df)


# In[ ]:


st.subheader("Model Performance")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

st.write("Accuracy:",round(accuracy_score(y_test,y_pred),3))
st.write("ROC-AUC:",round(roc_auc_score(y_test,y_prob),3))

st.text(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)

fig,ax = plt.subplots(figsize=(4,3))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)

st.pyplot(fig)
plt.close()


# In[ ]:


st.subheader("Enter Patient Data")

defaults = X.median()

input_df = pd.DataFrame([{
"Pregnancies":st.number_input("Pregnancies",0,20,int(defaults["Pregnancies"])),
"Glucose":st.number_input("Glucose",0,300,int(defaults["Glucose"])),
"BloodPressure":st.number_input("BloodPressure",0,200,int(defaults["BloodPressure"])),
"SkinThickness":st.number_input("SkinThickness",0,100,int(defaults["SkinThickness"])),
"Insulin":st.number_input("Insulin",0,900,int(defaults["Insulin"])),
"BMI":st.number_input("BMI",0.0,80.0,float(defaults["BMI"])),
"DiabetesPedigreeFunction":st.number_input("DPF",0.0,5.0,float(defaults["DiabetesPedigreeFunction"])),
"Age":st.number_input("Age",0,120,int(defaults["Age"]))
}])


# In[ ]:


st.subheader("Prediction")

pred_prob = model.predict_proba(input_df)[0][1]
pred = model.predict(input_df)[0]

if pred==1:
    st.error(f"High Risk of Diabetes ({pred_prob*100:.2f}%)")
else:
    st.success(f"Low Risk of Diabetes ({pred_prob*100:.2f}%)")


# In[ ]:


st.subheader("What-if Simulation")

feature = st.selectbox("Select Feature", X.columns)

values = np.linspace(X[feature].min(), X[feature].max(), 50)
probs = []

for v in values:
    temp = input_df.copy()
    temp[feature] = v
    probs.append(model.predict_proba(temp)[0][1])

plt.figure()
plt.plot(values, probs)
plt.xlabel(feature)
plt.ylabel("Diabetes Probability")

st.pyplot(plt)
plt.close()


# In[ ]:


st.subheader("SHAP Feature Importance")

explainer = shap.Explainer(model,X_train)
shap_values = explainer(X_train)

plt.figure(figsize=(6,3))
shap.plots.beeswarm(shap_values,show=False)

st.pyplot(plt.gcf())
plt.close()


# In[ ]:


st.subheader("SHAP Explanation for Patient")

input_shap = explainer(input_df)

plt.figure(figsize=(6,3))
shap.plots.waterfall(input_shap[0],show=False)

st.pyplot(plt.gcf())
plt.close()


# In[ ]:


st.subheader("LIME Explanation")

lime_explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=list(X_train.columns),
    class_names=["No Diabetes","Diabetes"],
    mode="classification"
)

exp = lime_explainer.explain_instance(
    input_df.iloc[0].values,
    model.predict_proba,
    num_features=6
)

lime_df = pd.DataFrame(exp.as_list(),columns=["Feature","Impact"])
st.table(lime_df)


# In[ ]:


st.subheader("Counterfactual Explanation")

if pred==1:
    g_change, b_change = counterfactual_improved(input_df,model)

    if g_change:
        st.success(
            f"If Glucose reduces by {g_change} and BMI reduces by {b_change}, risk may become LOW."
        )
    else:
        st.warning("No improvement found.")
else:
    st.info("Shown only for high-risk patients.")


# In[ ]:


st.subheader("Health Recommendations")

if input_df["BMI"][0] > 30:
    st.warning("High BMI: Consider exercise and weight management.")

if input_df["Glucose"][0] > 140:
    st.warning("High Glucose: Reduce sugar intake and monitor diet.")

if input_df["BloodPressure"][0] > 90:
    st.warning("High Blood Pressure: Reduce salt and manage stress.")

if pred==0:
    st.success("Maintain your current healthy lifestyle!")

