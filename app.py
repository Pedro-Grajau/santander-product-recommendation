import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import xgboost as xgb
import pickle
from rec_sys import load_test_dataset

st.set_page_config(page_title="Santander Recommendation", page_icon="🏦", layout="centered")
header = st.container()

pickled_model = pickle.load(open('xgb_model.pkl', 'rb'))

with header:
    image = Image.open('santander-banner.png')
    st.image(image, caption='Santander Banking')
    st.header("Santander Recommendation System")
    
    st.markdown('''
    Na Figura abaixo é ilustrado o diagrama de um dos processos de aprendizado de máquina utilizados para este trabalho que é modelo criado a partir do XGBoost. Como é visto na figura, a primeira etapa é a separação dos dados em dados de produtos adquiridos pelo cliente no mês e os dados do cliente propriamente dito. Esse processo de categorização é grande importância visto que as características do cliente é essencial para a recomendação.. 
    
                    ''')

    image = Image.open('diagrama_xgboost.png')
    st.image(image, caption='Model Architecture')


df_test, df_products, label_encoder = load_test_dataset()
X_test = df_test.sample(1)

st.markdown("Aperte no botão abaixo para selecionar um novo usuário do banco")

if st.button("Novo Usuário"):
    X_test = df_test.sample(1)

st.dataframe(X_test.T, use_container_width=True)

if st.button('Realizar Previsão'):
    final_test = xgb.DMatrix(X_test)
    preds = pickled_model.predict(final_test)
    top_t_products = [
        label_encoder.inverse_transform(
            np.argsort(pred, axis=0)[::-1][:7]
        ) for pred in preds
    ]
    st.write("Recomendações:")
    st.write(list(top_t_products))
    # for i, product in enumerate(top_t_products):
    #     st.markdown(product)