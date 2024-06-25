import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix
import json
import pickle

def run():
    #membuat judul
    st.title('Mushroom classification')
    #membuat sub header
    st.subheader('Milestone 2')
    #menambahkan deskripsi
    st.write('Page is made by Andryan wijaya')
    #tambahkan gambar
    image = Image.open('dataset-cover.jpg')
    st.image(image, caption = 'Mushroom')

    #show dataframe
    st.write('# Data')
    data = pd.read_csv('secondary_data.csv',delimiter=';')
    st.dataframe(data)

    st.write('# Boxplot Cap diameter')
    fig = plt.figure(figsize=(15,5))
    data['cap-diameter'].plot.box()
    st.pyplot(fig)
    st.write(data['cap-diameter'].describe())

    st.write('# Bar Cap shape')
    fig = plt.figure(figsize=(15,5))
    data['cap-shape'].value_counts().plot.bar()
    st.pyplot(fig)
    st.write(data['cap-shape'].describe())

    st.write('# Bar Cap surface')
    fig = plt.figure(figsize=(15,5))
    data['cap-surface'].value_counts().plot.bar()
    st.pyplot(fig)
    st.write(data['cap-surface'].describe())

    st.write('# Bar habitat')
    fig = plt.figure(figsize=(15,5))
    data['habitat'].value_counts().plot.bar()
    st.pyplot(fig)
    st.write(data['habitat'].describe())

    st.write('# Bar season')
    fig = plt.figure(figsize=(15,5))
    data['season'].value_counts().plot(kind='bar')
    st.pyplot(fig)
    st.write( data['season'].describe())

    st.write('# Bar class')
    fig = plt.figure(figsize=(15,5))
    data['class'].value_counts().plot(kind='bar')
    st.pyplot(fig)
    st.write(data['class'].describe())

    st.write('# Bar Cap color')
    fig = plt.figure(figsize=(15,5))
    data['cap-color'].value_counts().plot.bar()
    st.pyplot(fig)
    st.write(data['cap-color'].describe())

    st.write('# Boxplot stem height')
    fig = plt.figure(figsize=(15,5))
    data['stem-height'].plot.box()
    st.pyplot(fig)
    st.write(data['stem-height'].describe())

    st.write('# Boxplot stem width')
    fig = plt.figure(figsize=(15,5))
    data['stem-width'].plot.box()
    st.pyplot(fig)
    st.write(data['stem-width'].describe())
    
    st.write('# Bar setm root')
    fig = plt.figure(figsize=(15,5))
    data['stem-root'].value_counts().plot.bar()
    st.pyplot(fig)
    st.write(data['stem-root'].describe())

    st.write('# Bar stem surface')
    fig = plt.figure(figsize=(15,5))
    data['stem-surface'].value_counts().plot.bar()
    st.pyplot(fig)
    st.write(data['stem-surface'].describe())


if __name__ == '__main__':
    run()