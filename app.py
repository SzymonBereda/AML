import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(page_title="AML Fraud Detection Dashboard", layout="wide")
st.title("AML Fraud Detection Dashboard")
st.markdown("System wykrywania fraudu na danych transakcyjnych PaySim — XGBoost + SHAP")

# --- wczytanie modelu i danych ---
@st.cache_resource
def load_model():
    return joblib.load('fraud_model.pkl')

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

@st.cache_data
def load_data():
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')
    return X_test, y_test

model = load_model()
explainer = get_explainer(model)
X_test, y_test = load_data()

# --- predykcje ---
y_pred_proba = model.predict_proba(X_test)[:, 1]
X_test['fraud_probability'] = y_pred_proba
X_test['actual_fraud'] = y_test.values

# --- panel boczny: filtr progu ryzyka ---
st.sidebar.header("Ustawienia")
threshold = st.sidebar.slider("Próg ryzyka (prawdopodobieństwo fraudu)", 0.0, 1.0, 0.5, 0.01)

flagged = X_test[X_test['fraud_probability'] >= threshold].sort_values(
    'fraud_probability', ascending=False
)

# --- metryki na górze ---
col1, col2, col3 = st.columns(3)
col1.metric("Oflagowane transakcje", len(flagged))
col2.metric("Faktyczny fraud wśród oflagowanych", int(flagged['actual_fraud'].sum()))
precision = flagged['actual_fraud'].mean() * 100 if len(flagged) > 0 else 0
col3.metric("Precyzja", f"{precision:.1f}%")

# --- tabela oflagowanych transakcji ---
st.subheader("Oflagowane transakcje")
st.dataframe(
    flagged[['amount', 'fraud_probability', 'actual_fraud']].head(50),
    width='stretch'
)

# --- wyjaśnienie SHAP dla wybranej transakcji ---
st.subheader("Uzasadnienie decyzji (SHAP)")
if len(flagged) > 0:
    idx = st.selectbox("Wybierz transakcję do wyjaśnienia (indeks)", flagged.index[:20])

    row = X_test.drop(['fraud_probability', 'actual_fraud'], axis=1).loc[[idx]]
    shap_values = explainer.shap_values(row)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=row.iloc[0],
            feature_names=row.columns.tolist()
        ),
        show=False
    )
    st.pyplot(fig)
else:
    st.info("Brak transakcji powyżej wybranego progu ryzyka.")
