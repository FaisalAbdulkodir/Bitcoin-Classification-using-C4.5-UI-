from pkg_resources import load_entry_point
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
import pydotplus
import matplotlib.pyplot as plt
import seaborn as sns
from six import StringIO
import matplotlib.image as mpimg
from sklearn import tree\

st.title('C4.5 for Bitcoin Price')

nama_dataset = st.sidebar.selectbox(
    'Dataset',
    ('BitcoinPriceRaw','-')
)
st.write(f"## Dataset {nama_dataset}")

# model = st.sidebar.selectbox(
#     'Model',
#     ('Decision Tree C4.5', '-')
# )


def pilih_dataset(nama):
    data = None
    if nama == 'BitcoinPriceRaw':
        data = pd.read_excel('4.xlsx')
        st.write(data.head(31))
    elif nama == '-':
        st.write('NONE')
    else:
        st.write('sadasda')
    return data


data = pilih_dataset(nama_dataset)
# st.write('Kelas ', len(np.unique(y)))

# st.subheader('Perbandingan Jumlah Naik dan Turun')


# Preprocessing
df = pd.read_excel('4.xlsx')
df_features = df.drop(['Tanggal', 'Perubahan'], axis=1)
df_target = df['Perubahan']

#Encode
cols = ['Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah', 'Vol']
df_features[cols] = df_features[cols].astype('category')

for col in cols:
    df_features[col] = df_features[col].cat.codes

#standarScale
scale = StandardScaler().fit(df_features).transform(df_features.astype(float))
cols = list(df_features.columns)

df_features_scale = pd.DataFrame(scale, columns=cols)

#split train and test
X_train, X_test, y_train, y_test = train_test_split(df_features_scale, df_target, test_size=0.35, random_state=10)

#undersampling
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
# print("Before undersampling: ", Counter(y_train))
undersample = RandomUnderSampler(sampling_strategy='majority')
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)
# print("After undersampling: ", Counter(y_train_under))

#min max scaler
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1))
X_scaled = sc.fit_transform(df_features)

#load model
load_model = pickle.load(open('modelDT.pkl', 'rb'))

#prediksi
prediksi = load_model.predict(X_test)

#akurasi
akurasi = load_model.score(X_test, y_test)

#plot
import graphviz as graphviz
clf = load_model
dot_data  = tree.export_graphviz(clf, out_file=None)



#####
st.subheader('Hasil Klasifikasi')
st.write(prediksi)

st.subheader('Akurasi')
st.write(akurasi)

st.subheader('Hasil Plot')
st.graphviz_chart(dot_data)