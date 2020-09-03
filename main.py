from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('churn_fr_lgbm')

st.set_option('deprecation.showfileUploaderEncoding', False)

from PIL import Image
image_office = Image.open('office.jpg')
st.image(image_office,use_column_width=True)
add_selectbox = st.sidebar.selectbox("Comment voulez-vous faire votre prédiction?",("Online", "Fichier"))
st.sidebar.info("Ce modèle a été créé dans le but de prédire si un(e) employé(e) quittera l'entreprise")
st.sidebar.info("Résultat: 0 (reste); 1 (démissionne)")
st.sidebar.success('https://www.pycaret.org')
st.title("Prédire le départ d'un(e) employé(e)")


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions



def run():

    if add_selectbox == 'Online':
        satisfaction_level=st.number_input('Niveau de satisfaction' , min_value=0.1, max_value=1.0, value=0.1)
        last_evaluation =st.number_input('Note dernière évaluation',min_value=0.1, max_value=1.0, value=0.1)
        number_project = st.number_input('Nombre de projets', min_value=0, max_value=50, value=5)
        average_monthly_hours = st.number_input("Moyenne d'heures mensuelle", min_value=96, max_value=400, value=100)
        time_spend_company = st.number_input('Ancienneté en années', min_value=1, max_value=10, value=3)
        Work_accident = st.number_input('Accident de travail',  min_value=0, max_value=50, value=0)
        promotion_last_5years = st.number_input('Promotion 5 dernières années',  min_value=0, max_value=50, value=0)
        department= st.selectbox('Département', ['comptabilite', 'technique', 'IT', 'support', 'R&D', 'ventes',
       'management', 'marketing', 'product_mng', 'rh'])
        salary = st.selectbox('Salaire', ['bas', 'normal','eleve'])
        output=""
        input_dict={'niveau_satisfaction':satisfaction_level,'derniere_evaluation':last_evaluation,'nb_projet':number_project, 'heures_par_mois': average_monthly_hours,
        'anciennete':time_spend_company,'accident_travail': Work_accident,'promotion_5ans':promotion_last_5years,'departement':department,'salaire' : salary}
        input_df = pd.DataFrame([input_dict])
        if st.button("Prédiction"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('Le résultat est {}'.format(output))
    if add_selectbox == 'Fichier':
        file_upload = st.file_uploader("Chargez votre fichier csv pour la prediction ", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

run()


