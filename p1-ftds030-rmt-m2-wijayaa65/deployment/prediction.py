import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from PIL import Image

#load
with open('list_kat.txt', 'r') as file_1:
  list_kat = json.load(file_1)

with open('list_num.txt', 'r') as file_2:
  list_num = json.load(file_2)

with open('random_search_SVM.pkl', 'rb') as file_5:
  pip_svm = pickle.load(file_5)

def run():
    #membuat judul
    st.title('Mushroom clasiffication')
    #membuat sub header
    st.subheader('Milestone 2')
    image = Image.open('humanjamur.png')
    st.image(image)
    with st.form('Mushroom_prediciton'):
        cap_diameter = st.number_input("Cap Diameter", min_value=0.0)
        cap_shape = st.radio("Cap Shape", ['b', 'c', 'x', 'f', 's', 'p', 'o'], index=3,captions=['bell', 'conical', 'convex', 'flat','sunken', 'spherical', 'others'],horizontal=True)
        cap_surface = st.radio("Cap Surface", ['i', 'g', 'y', 's', 'h', 'l', 'k', 't', 'w', 'e'], index=3,captions=['fibrous', 'grooves', 'scaly', 'smooth','shiny', 'leathery', 'silky', 'sticky','wrinkled', 'fleshy'],horizontal=True)
        cap_color = st.radio("Cap Color", ['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k'], index=6,captions=['brown', 'buff', 'gray', 'green', 'pink','purple', 'red', 'white', 'yellow', 'blue','orange', 'black'],horizontal=True)
        stem_height = st.number_input("Stem Height", min_value=0.0)
        stem_width = st.number_input("Stem Width", min_value=0.0)
        stem_root = st.radio("Stem Root", ['b', 's', 'c', 'u', 'e', 'z', 'r'], index=4,captions=[ 'bulbous', 'swollen', 'club', 'cup', 'equal','rhizomorphs', 'rooted'],horizontal=True)
        stem_surface = st.radio("Stem Surface", ['i', 'g', 'y', 's', 'h', 'l', 'k', 't', 'w', 'e', 'f'], index=10,captions=['fibrous', 'grooves', 'scaly', 'smooth','shiny', 'leathery', 'silky', 'sticky','wrinkled', 'fleshy','none'],horizontal=True)
        habitat = st.radio("Habitat", ['g', 'l', 'm', 'p', 'h', 'u', 'w', 'd'], index=3,captions=['grasses', 'leaves', 'meadows', 'paths', 'heaths','urban', 'waste', 'woods'],horizontal=True)
        season = st.radio("Season", ['s', 'u', 'a', 'w'], index=1,captions=['spring', 'summer', 'autumn', 'winter'],horizontal=True)
        
        #bikin submit button form
        submitted = st.form_submit_button('Predict')
  
    data_inf = {
    'cap-diameter':cap_diameter,
    'cap-shape':cap_shape,
    'cap-surface':cap_surface,
    'cap-color':cap_color,
    'stem-height':stem_height, 
    'stem-width':stem_width, 
    'stem-root':stem_root, 
    'stem-surface':stem_surface, 
    'habitat':habitat,
    'season':season,
        }
    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)


    if submitted:
        #predict using SVM
        y_pred_inf = pip_svm.predict(data_inf)

        if y_pred_inf == 0:
          st.write('## Mushroom Class: Edible')
        else:
           st.write('## Mushroom Class: Poisnous')

if __name__ == '__main__':
   run()