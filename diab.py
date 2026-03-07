"""
Explainable AI Diabetes Predictor
XGBoost + SHAP + LIME + Counterfactual Explanations
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

import shap
from lime.lime_tabular import LimeTabularExplainer


# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "pimadiabetes1.csv"
RANDOM_STATE = 42


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    # ensure numeric
    df = df.apply(pd.to_numeric)

    zero_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

    for c in zero_cols:
        if c in df.columns:
            df[c] = df[c].replace(0,np.nan)

    df.fillna(df.median(), inplace=True)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    return df,X,y


# -----------------------------
# TRAIN MODEL
# -----------------------------
@st.cache_resource
def train_model(X,y):

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

    model.fit(X_train,y_train)

    return model,X_train,X_test,y_train,y_test


# -----------------------------
# COUNTERFACTUAL FUNCTION
# -----------------------------
def counterfactual(input_df,model,max_change=50):

    base_pred = model.predict(input_df)[0]

    for feature in ["Glucose","BMI","Age"]:

        if feature not in input_df.columns:
            continue

        value = input_df[feature].iloc[0]

        for change in range(1,max_change):

            new_df = input_df.copy()

            new_df[feature] = max(0,value-change)

            new_pred = model.predict(new_df)[0]

            if new_pred != base_pred:

                return feature,value,value-change

    return None,None,None


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Diabetes Predictor",layout="wide")

st.title("Explainable AI for Diabetes Prediction")


df,X,y = load_data()

model,X_train,X_test,y_train,y_test = train_model(X,y)


# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
st.subheader("Model Performance")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

st.write("Accuracy:",round(accuracy_score(y_test,y_pred),3))
st.write("ROC-AUC:",round(roc_auc_score(y_test,y_proba),3))

st.text(classification_report(y_test,y_pred))


cm = confusion_matrix(y_test,y_pred)

fig,ax = plt.subplots()
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
st.pyplot(fig)
plt.close()


# -----------------------------
# USER INPUT
# -----------------------------
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


# -----------------------------
# PREDICTION
# -----------------------------
st.subheader("Prediction")

pred_proba = model.predict_proba(input_df)[0][1]
pred = model.predict(input_df)[0]

if pred==1:
    st.error(f"High Risk of Diabetes ({pred_proba*100:.2f}%)")
else:
    st.success(f"Low Risk of Diabetes ({pred_proba*100:.2f}%)")


# -----------------------------
# SHAP GLOBAL IMPORTANCE
# -----------------------------
# -----------------------------
# SHAP GLOBAL + LOCAL EXPLANATION
# -----------------------------
st.subheader("SHAP Explainability")

try:

    # Create SHAP explainer
    explainer = shap.Explainer(model.predict_proba, X_train)

    shap_values = explainer(X_train)

    # -----------------------------
    # Global SHAP (Beeswarm)
    # -----------------------------
    st.markdown("### SHAP Global Feature Importance (Beeswarm Plot)")

    fig_bee, ax = plt.subplots()

    shap.plots.beeswarm(shap_values[:,:,1], show=False)

    st.pyplot(fig_bee)

    plt.close()


    # -----------------------------
    # Local SHAP (Waterfall)
    # -----------------------------
    st.markdown("### SHAP Explanation for This Patient")

    input_shap = explainer(input_df)

    fig_wat, ax = plt.subplots()

    shap.plots.waterfall(input_shap[0,:,1], show=False)

    st.pyplot(fig_wat)

    plt.close()

except Exception as e:

    st.warning(f"SHAP failed: {e}")


# -----------------------------
# LIME EXPLANATION
# -----------------------------
st.subheader("LIME Explanation")

try:

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

except Exception as e:

    st.warning(f"LIME failed: {e}")


# -----------------------------
# COUNTERFACTUAL
# -----------------------------
st.subheader("Counterfactual Explanation")

if pred==1:

    feature,old,new = counterfactual(input_df,model)

    if feature:

        st.success(
            f"If **{feature}** changes from **{old} → {new}**, prediction may become **No Diabetes**"
        )

    else:

        st.warning("No simple counterfactual found.")

else:

    st.info("Counterfactual suggestions appear only for high-risk patients.")
