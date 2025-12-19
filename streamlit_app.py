import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Ultimate Penguin Predictor", layout="wide", page_icon="🐧")

# ----------------------------
# Theme toggle
# ----------------------------
mode = st.sidebar.radio("Theme", ["Light", "Dark"])
if mode == "Dark":
    st.markdown(
        """<style>
            body {background-color: #0e1117; color: white;}
            .stButton>button {background-color: #0a84ff; color: white;}
        </style>""", unsafe_allow_html=True
    )

# ----------------------------
# Title
# ----------------------------
st.title("🐧 Ultimate Penguin Species Predictor")
st.info("Interactive app to predict penguin species, save results, and visualize data.")

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv"
    return pd.read_csv(url)

df = load_data()

# ----------------------------
# History Storage
# ----------------------------
if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=["species_predicted", "Adelie", "Chinstrap", "Gentoo"])

# ----------------------------
# Tabs Layout
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Input Features", "Prediction", "Dashboard"])

# ----------------------------
# Tab 1: Data Overview
# ----------------------------
with tab1:
    st.subheader("Dataset")
    st.dataframe(df)

    st.subheader("Scatter Plot")
    fig = px.scatter(df, x="bill_length_mm", y="body_mass_g",
                     color="species", size="flipper_length_mm",
                     hover_data=["sex", "island"])
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Tab 2: Input Features (Cards)
# ----------------------------
with tab2:
    st.subheader("Enter Penguin Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        island = st.selectbox("Island", ("Biscoe", "Dream", "Torgersen"),
                              help="Select the island where the penguin was observed")
        bill_length_mm = st.slider("Bill length (mm)",
                                   float(df.bill_length_mm.min()), float(df.bill_length_mm.max()),
                                   float(df.bill_length_mm.mean()), help="Length of the bill in millimeters")
    
    with col2:
        bill_depth_mm = st.slider("Bill depth (mm)",
                                  float(df.bill_depth_mm.min()), float(df.bill_depth_mm.max()),
                                  float(df.bill_depth_mm.mean()), help="Depth of the bill in millimeters")
        flipper_length_mm = st.slider("Flipper length (mm)",
                                      float(df.flipper_length_mm.min()), float(df.flipper_length_mm.max()),
                                      float(df.flipper_length_mm.mean()), help="Length of the flipper")
    
    with col3:
        body_mass_g = st.slider("Body mass (g)",
                                float(df.body_mass_g.min()), float(df.body_mass_g.max()),
                                float(df.body_mass_g.mean()), help="Body weight in grams")
        sex = st.selectbox("Sex", ("male", "female"), help="Gender of the penguin")
        age_category = st.radio("Age Category", ("Juvenile", "Adult"), help="Approximate age category")
    
    input_data = pd.DataFrame({
        "island": [island],
        "bill_length_mm": [bill_length_mm],
        "bill_depth_mm": [bill_depth_mm],
        "flipper_length_mm": [flipper_length_mm],
        "body_mass_g": [body_mass_g],
        "sex": [sex],
        "age_category": [age_category]
    })
    
    st.write("Your Input Penguin")
    st.dataframe(input_data)

# ----------------------------
# Tab 3: Prediction
# ----------------------------
with tab3:
    df_combined = pd.concat([input_data, df.drop("species", axis=1)], axis=0)
    df_encoded = pd.get_dummies(df_combined, columns=["island", "sex", "age_category"])
    
    X_input = df_encoded.iloc[:1, :]
    X_train = df_encoded.iloc[1:, :]
    
    target_map = {"Adelie":0, "Chinstrap":1, "Gentoo":2}
    y = df.species.map(target_map)
    
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y)
    
    pred_class = clf.predict(X_input)[0]
    pred_proba = clf.predict_proba(X_input)[0]
    
    species = np.array(["Adelie", "Chinstrap", "Gentoo"])
    
    st.subheader("Predicted Species")
    st.success(f"🟢 {species[pred_class]}")
    
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame([pred_proba], columns=species)
    for sp, prob in zip(species, pred_proba):
        st.write(f"{sp}: {prob:.2f}")
        st.progress(int(prob*100))
    
    # Save prediction to history
    new_row = pd.DataFrame({
        "species_predicted": [species[pred_class]],
        "Adelie": [pred_proba[0]],
        "Chinstrap": [pred_proba[1]],
        "Gentoo": [pred_proba[2]]
    })
    st.session_state['history'] = pd.concat([st.session_state['history'], new_row], ignore_index=True)
    
    # Download button
    csv = st.session_state['history'].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Prediction History", data=csv, file_name="penguin_predictions.csv", mime="text/csv")

# ----------------------------
# Tab 4: Dashboard
# ----------------------------
with tab4:
    st.subheader("Prediction History")
    st.dataframe(st.session_state['history'])
    
    st.subheader("Species Distribution")
    fig2 = px.histogram(st.session_state['history'], x="species_predicted", title="Predicted Species Count")
    st.plotly_chart(fig2, use_container_width=True)
