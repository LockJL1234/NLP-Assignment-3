# Spam Ham Email Classification

- Feature Extraction: TF-IDF
- Algorithm: Logistic Regression
- SMOTE for balancing dataset
- LIME Explainable AI
- GridSearchCV for Hyperparameter Tuning and Cross Validation

---

This project is developed using JupyterLab and Spyder for Python.

JupyterLab Notebook is accessed through Anaconda as well as Spyder IDE. The streamlit webapp python file is executed in the Anaconda command prompt.

This project contains a jupyter notebook file, pickle files of the classification model and feature extraction model, datasets that was used for this project, and the streamlit python file. The jupyter notebook file contains mainly the training of the model that includes the data exploratory analysis of dataset, data preprocessing of text data, model training, model evaluation, and LIME explainable AI. After the model has finishing training, the model is exported as a pickle file and employed into the web application. 

The web application is developed using Streamlit framework that is in python programming language. Spyder was used as the IDE for developing the Streamlit web appplication which is a spam ham email classification web application.

---

To run the web application python file, use the command below with **anaconda command prompt** under the right environment. Ensure that the command prompt is in the right directory where the web application python file or specified the directory of the web application python file. 
``` streamlit run spam_ham_email_classification_app.py ```
