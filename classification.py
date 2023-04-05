import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, auc

import matplotlib.pyplot as plt



def calculate_metrics(predictions, labels):
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    # Precision
    precision = precision_score(labels, predictions)
    # Recall
    recall = recall_score(labels, predictions)
    # ROC curve
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    # AUC score
    auc_score = auc(fpr, tpr)
    return cm, precision, recall, fpr, tpr, auc_score


def display_confusion_matrix(cm):
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
    plt.xlabel('Predictions')
    plt.ylabel('True labels')
    st.pyplot(fig)


# Set the page title
st.set_page_config(page_title='Threshold Impact', page_icon=':bar_chart:', layout='wide')

# Upload the data
st.sidebar.header('Upload Data')
file = st.sidebar.file_uploader('Choose a CSV file', type='csv')
if file is not None:
    data = pd.read_csv(file)
    st.sidebar.success('Data uploaded successfully!')
else:
    st.sidebar.warning('Please upload a CSV file.')

# Choose the model
st.sidebar.header('Choose Model')
model = st.sidebar.selectbox('Select the model', ['Naive', 'Probabilities'])

# Select the threshold
st.sidebar.header('Select Threshold')
threshold = st.sidebar.slider('Threshold', 0.0, 1.0, 0.5, 0.01)

# Display the results
st.header('Threshold Impact')
if file is not None:
    # Get the predictions and labels
    if model == 'Naive':
        predictions = np.zeros_like(data['label'])
    else:
        predictions = data['prob'] >= threshold
    labels = data['label']

    # Calculate the metrics
    cm, precision, recall, fpr, tpr, auc_score = calculate_metrics(predictions, labels)

    # Display the results
    st.subheader('Confusion Matrix')
    display_confusion_matrix(cm)
    st.subheader('Precision')
    st.write(precision)
    st.subheader('Recall')
    st.write(recall)
    st.subheader('ROC Curve')
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)


    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)
    st.subheader('AUC Score')
    st.write(auc_score)


