import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from footerframe import footer
image = Image.open('homeimage.png')
st.set_page_config(
   page_title="Custoplus",
 page_icon="pie-chart.png")



st.image(image)

st.title('CustoPlus')
st.subheader('We help make your customers stay 😉')
footer()

uploaded_file = st.file_uploader("Upload Your Customer Data", type=['csv', 'xlsx'])


def plot(churn):
    list_num = []
    list_str = []
    df.drop(options,axis='columns',inplace=True)
    for column in df:
        if (df[column].dtype == 'int64' or df[column].dtype == 'float64'):
            list_num.append(column)
        else:
            list_str.append(column)
    df2 = df
    if (df[churn].dtype == 'object'):
        df2[churn].replace({'Yes': 1, 'No': 0}, inplace=True)

    if(len(list_num) != 0):
        st.subheader('Histograms for your number data values')
        plt_col = st.selectbox('Histograms', list_num)
        fig, ax = plt.subplots()
        ax.hist([df2[df2[churn] == 0][plt_col], df2[df2[churn] == 1][plt_col]], color=['lightgreen', 'red'],
                label=['Customers who stay', 'Customers who leave'])
        ax.set_title(plt_col + ' Vs ' + churn)
        plt.xlabel(plt_col)
        plt.legend()
        st.pyplot(fig)

    if(len(list_str) != 0):
        st.subheader('Pie chart for the Text  Columns')
        plt_pie = st.selectbox('Lets Pie', list_str)
        fig1, ax1 = plt.subplots()
        ax1.pie(df2[df2[churn] == 1][plt_pie].value_counts().values,
                labels=df2[df2[churn] == 1][plt_pie].value_counts().index)
        ax1.set_title(plt_pie + ' Vs ' + churn)
        plt.legend()
        st.pyplot(fig1)


def homelayer1():
    df.drop(options, axis='columns', inplace=True)
    if st.checkbox('Show  sample data'):
        if (df.shape[0] > 20):
            st.write(df.sample(10))
        else:
            st.write(df.sample())
            st.error("Insufficient data")

def homelayer2():
    if (st.checkbox("Show Corelation")):
        try:
            st.write(df1.corr(method='pearson')[churn])
        except:
            st.write("Set the correct exit status column")
        else:
            st.info(
                "positive correlation represents directly proportional while negative corelation represents inverse relationship")
            st.info(
                "For example + 0.19 means 19% of direct relationship of this parameter with our output(churn or exit here) -0,23 means 23 % of inverse relationship")
    if(st.checkbox("See the data statistically")):
        if(st.checkbox("Those who leave")):
            st.write(df1[df1[churn]==1].describe())
        if(st.checkbox("Those who Stay")):
            st.write(df1[df1[churn]==0].describe())
        st.info("See the maximum age of your customers or maybe the mean or average money they invest in buying your products.You can also watch the tenures and how it differs..")


def predict(df3, opt):
    df3.drop(opt, axis='columns', inplace=True)
    for column in df3:
        if (df3[column].dtype == 'int64' or df3[column].dtype == 'float64'):
            continue
        else:
            df3.drop(column, axis='columns', inplace=True)
    X=df3.drop(churn,axis='columns')
    y=df3[churn]
    st.write(X)

    col1 = X.columns
    ans = []
    for c in col1:
        x = st.number_input(c)
        ans.append(x)
    #st.write(ans)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.25, random_state = 42)

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)

    # log = LogisticRegression()
    # log.fit(X_train, y_train)

    # y_pred = log.predict(X_test)
    if(st.button("Predict")):
        if(sum(ans)==0):
            st.e
        y_pred = knn.predict(X_test)
        y_result=knn.predict([ans])
        if(y_result[0] == 0):
            st.success("Custoplus predicts that the customer will STAY with an accuracy of"+str(metrics.accuracy_score(y_test, y_pred)*100)+"%")
        else:
            st.error("Custoplus predicts that the customer will Leave with an accuracy of"+str(metrics.accuracy_score(y_test, y_pred)*100)+"%")





if uploaded_file is not None:
    extension = uploaded_file.name.split('.')[1]
    if extension.upper() == 'CSV':
        df = pd.read_csv(uploaded_file)
    elif extension.upper() == 'XLSX':
        df = pd.read_excel(uploaded_file)
    df = df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('Unknown'))
    churn = st.selectbox('Which Column Shows customer exit status', df.columns)
    options = st.multiselect(
        'Select Columns to drop(Columns like phone number,customer id do not affect customer churn)',
        df.columns)

    st.sidebar.title("Custoplus")
    page = st.sidebar.radio("See what we have for you", ('Home', 'Graphy', 'Predict'))
    st.sidebar.success("Make sure the Dataset has more of integer values and the churn or exit status column must have values like yes / no or 0 / 1 for smooth functioning of the web app")
    st.sidebar.info("Custoplus is designed to help analyze your customer database and understand why they are leaving or staying.Upload your database and choose which column represents churn.Our automated engine will modify the database and help you understand.First select the correlation and then go to the graph for visualization.Atlast give an unknown data and measure whether the person willl stay or not")
    
    if (page == 'Home'):
         homelayer1()
    df1 = df.replace({'Yes': 1, 'No': 0,'yes': 1, 'no': 0})
    if (page == 'Home'):
        homelayer2()

    if (page == 'Graphy'):
        plot(churn)
    if (page == 'Predict'):
        predict(df1,options)
else:
    #st.info("Reach me @ [LinkedIn](https://www.kaggle.com/blastchar/telco-customer-churn?select=WA_Fn-UseC_-Telco-Customer-Churn.csv)")
    st.warning("Don't have a dataset?No worries🥰Test the app after downloading the famous churn datasets from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn?select=WA_Fn-UseC_-Telco-Customer-Churn.csv)")