# -*- coding: utf-8 -*-
"""
NLP Assignment 3 (Group Assignment)
Spam Ham Email Classification Application using Streamlit

0125118 Lock Jun Lin
0125112 Joseph Chang Mun Kit
0129219 Victor Chua Min Chun
0125942 Kenny Lee Yuan Hong
"""

#streamlit version 0.79.0
# import streamlit library for streamlit module to create website application
import streamlit as st
import streamlit.components.v1 as components

#libraries for output (print) input output stream of code
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from contextlib import contextmanager  #contextlib2 version 0.6.0.post1
from threading import current_thread

# utilities
import re  #regex version 2021.3.17
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)   #pandas version 1.2.3
import matplotlib.pyplot as plt # data visualization library   # matplotlib version 3.3.4
# %matplotlib inline
import seaborn as sns # interactive visualization library built on top on matplotlib   #seaborn version 0.11.1
sns.set(style="darkgrid")
sns.set(font_scale=1.5)
import string
import io 
import sys 

#nltk version 3.5
# nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#scikit-learn version 0.24.1
# sklearn for model training
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# sklearn evaluation metrics
from sklearn.metrics import classification_report,accuracy_score, plot_confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

# pickleshare version 0.7.5
# pickle library for exporting model into pkl file
import pickle

# lime version 0.2.0.1
# lime explainable AI for explaining result of prediction by machine learning model
from lime.lime_text import LimeTextExplainer

# %%

# define a CleanText class to perform data cleaning on entered email text by user
@st.cache()
class CleanText(BaseEstimator, TransformerMixin):
    # replacing the shorten word "re" with "reply"
    def replace_re(self, input_text):
        return re.sub(r'\bre\b', "reply", input_text)

    # raplacing the shorten word "fw" with "forward"
    def replace_fw(self, input_text):
        return re.sub(r'\bfw\b', "forward", input_text)

    # remove all punctuation from email text data
    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        # Every punctuation symbol will be replaced by a space
        trantab = str.maketrans(punct, len(punct)*' ')
        return input_text.translate(trantab)

    # remove all digit from email text data
    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)

    # change all letter in email text data to lowercase
    def to_lower(self, input_text):
        return input_text.lower()

    # remove of leading/trailing whitespace
    def remove_space(self, input_text):
        return input_text.strip()

    # remove extra spaces
    def remove_extra_space(self, input_text):
        return re.sub(r'\s+', ' ', input_text, flags=re.I)

    # remove special character
    def remove_special_char(self, input_text):
        return re.sub(r'\W', ' ', input_text)

    # remove all stopwords such as to, i, me, etc. from email text data
    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split()
        clean_words = [word for word in words if (
            word not in stopwords_list or word in whitelist) and len(word) > 1]
        return " ".join(clean_words)

    # remove word "subject" as it appears too often and does not provide any meaning to the text
    def remove_unwanted_words(self, input_text):
        word_remove = ["subject"]
        words = input_text.split()
        word_removed_sent = [
            word for word in words if (word not in word_remove)]
        return " ".join(word_removed_sent)

    # perform lemmatization on each word in email text data to return each word back to its root word without changing the meaning of the word
    def word_lemmatization(self, input_text):
        lemmatizer = WordNetLemmatizer()
        words = input_text.split()
        lemmed_words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmed_words)

    def fit(self, X, y=None, **fit_params):
        return self

    # apply each function in CleanText class
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(
            self.replace_re).apply(self.replace_fw).apply(self.remove_space).apply(self.remove_extra_space).apply(
            self.remove_special_char).apply(self.remove_stopwords).apply(self.remove_unwanted_words).apply(
            self.word_lemmatization)
        return clean_X

#define function to print out code text
@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with io.StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write

@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield

# %%

# load the model and vectorizer from respective file in given filename/directory
lr_model = pickle.load(open('../lr_model.pkl', 'rb'))
tfidf_vector = pickle.load(open('../tfidf_vectorizer.pkl', 'rb'))

# importing the cleaned spam ham dataset
df = pd.read_csv("../cleaned_spam_ham_dataset.csv")

#spliting dataset into test data and train data with 20% of data as test data
X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df.label, test_size=0.2, random_state=37)

# feature extraction of training data with TF-IDF loaded from pickle file
X_train_tfidf = tfidf_vector.fit_transform(X_train["clean_text"])
X_test_tfidf = tfidf_vector.transform(X_test["clean_text"])

# perform class prediction on the feature extracted test dataset for evaluation
lr_predict_test = lr_model.predict(X_test_tfidf)

# predict probabilities 
lr_probs = lr_model.predict_proba(X_test_tfidf)

#define object of CleantText class
ct = CleanText()

#define class name to classify text data from selected row and object for Lime
class_names = ['ham', 'spam']
explainer = LimeTextExplainer(class_names = class_names)

#make pipeline of tfidf and logistic regression model
logreg_tfidf_pipe = make_pipeline(tfidf_vector, lr_model)

# %%

#website title
st.title("Spam Ham Email Classification Application")

st.sidebar.title('Spam Ham Email Classification Application')
st.sidebar.subheader('Action')

#sidebar in website for user to choose different webpage
app_mode = st.sidebar.selectbox('Choose Option', [
                                'About App', 'Classification Model Evaluation Metrics', "Spam Ham Email Classification"])

# %%

if app_mode == "About App":
    st.sidebar.markdown('---')
    # display image for application
    st.image('spam_app_pic.jpg')
    
    st.subheader("About Spam Ham Email Classification Application")
    # introduction of application content
    st.markdown('''
                This is a **Spam Ham Email Classification Application**.\n 
                
                The application takes in email that is either entered manually by the user into the system or an email text file that is uploaded by the user and 
                classify the email using the classification model. 
                
                The classification model is trained using Jupyter Notebook and is exported into a **pickle file** to be loaded later by this application. The feature
                extraction algorithm is also exported into a pickle file. As such, the training of the classification model will not done in the application.
                
                **Logistic Regression** algorithm which is a supervised machine learning algorithm is used as the algorithm for the model. The feature extraction
                algorithm that is used is **TF-IDF** algorithm. 
                
                **Hyperparameter tuning** and **cross-validation** using **GridSearchCV** is also applied to determine the best parameter for Logistic Regression 
                algorithm and TF-IDF algorithm. 
                
                - Best Parameter for Logistic Regression algorithm: **penalty = 'l2', C = 1.0**
                - Best Parameter for TF-IDF algorithm: **min_df = 2, max_df = 0.5, ngram_range = (1, 1)**
                
                The dataset that was used to train the classification model, spam ham dataset was retrieved from kaggle.
                
                The email text that is entered by the user into the system for classification will be pre-process before perform feature extraction on the email 
                text.
                
                #### The text pre-processing technique used are the following:-
                    
                1. Remove punctuations mark.
                2. Remove all digit or number.
                3. Remove stopwords (list of words that do not have any meaning such as a, about, above, etc.).
                4. Change multiple spaces into single space.
                5. Remove all special characters. (ex: star symbol)
                6. Perform lemmatization on the text data to revert the words to its root words.
                7. Replacing shorten words such as "re" and "fw" to "reply" and "forward" respective.
                8. Remove unwanted words such as "subject" that does not provide any meaning to the text.
                9. Lowercase all character in email text data.
                ''')

# %%

elif app_mode == "Classification Model Evaluation Metrics":
    st.markdown("## Classification Model Evaluation Metrics With Graph")
    st.markdown("### Choose the option on the sidebar to display the selected item.")
    st.sidebar.markdown('---')
    
    #following item will be displayed when user checks the check boxes located at the side bar
    if st.sidebar.checkbox("Confusion Matrix"):
        st.subheader("Confusion Matrix")
        #display number of row in cleaned dataset and in test set splitted from cleaned dataset
        st.markdown("No. of observation (rows) in cleaned training dataset: %d" %df.shape[0])
        st.markdown("No. of observation (rows) in test set (after spltting): %d" %X_test.size)
        
        #Plotting a confusion matrix graph to to evaluate the logistic regression model with TF-IDF vectorizer
        fig, ax = plt.subplots(figsize=(13, 6))
        disp = plot_confusion_matrix(lr_model, X_test_tfidf, y_test, cmap=plt.cm.Blues, normalize=None, ax=ax)
        disp.ax_.set_title("Confusion Matrix Without Normalization (Logistic Regression with TF-IDF)")
        #remove white grid lines from the graph
        plt.grid(False)
        st.pyplot(fig)
    
    if st.sidebar.checkbox("Evaluation Metrics"):
        st.subheader("Evaluation Metrics")
        #output the classification report for the spam ham classification model
        with st_stdout("code"):
             print(classification_report(y_test, lr_predict_test))
             
        #calculate the evaluation metrics of the TF-IDF logistic regression model
        lr_tfidf_accuracy = accuracy_score(y_test, lr_predict_test)
        lr_tfidf_precision = precision_score(y_test, lr_predict_test, average='weighted')
        lr_tfidf_recall = recall_score(y_test, lr_predict_test, average='weighted')
        lr_tfidf_f1 = f1_score(y_test, lr_predict_test, average='weighted')
        st.markdown("""
                    #### TFIDF LOGISTIC REGRESSION MODEL
                    
                    - Accuracy score: %.3f%%
                    - Precision score: %.3f%%
                    - Recall score: %.3f%%
                    - F1 score: %.3f%%
                    """ %(lr_tfidf_accuracy*100, lr_tfidf_precision*100,lr_tfidf_recall*100, lr_tfidf_f1*100))
    
    if st.sidebar.checkbox("Precision-Recall Curve Graph"):
        st.subheader("Precision-Recall Curve Graph")
        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]
        # predict class values
        # yhat = best_logreg.predict(X_test_tfidf)
        lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs, pos_label='spam')
        lr_f1 = f1_score(y_test, lr_predict_test, pos_label='spam')
        lr_auc = auc(lr_recall, lr_precision)
        # summarize scores
        st.markdown('Logistic Regression Model with TF-IDF: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
        
        # plot the precision-recall curves
        no_skill = len(y_test[y_test=="spam"]) / len(y_test)
        fig, ax = plt.subplots(figsize=(14, 7))
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
        plt.title("Precision-Recall Curve")
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        # show the plot
        st.pyplot(fig)
    st.sidebar.markdown('---')  
    
# %%

elif app_mode == "Spam Ham Email Classification":
    st.markdown("""
                ## User enter email text or upload email text file to classify if the email is spam or ham email.
                
                _User can choose either to enter email text into the text input field below **OR** upload email text file._
                """)
    
    try:            
        #receive email text input from user
        input_email = st.text_input("Input email text here :")
        upload_email = st.file_uploader("Upload Email Text File Here.")

        #if user did not input email text, then uploaded email text file will be the input
        if input_email == "":
            if upload_email is not None:
                input_email = str(upload_email.read(), "utf-8")

    #error message if an error has occur            
    except ValueError:
        st.error("An error has occured. Please enter a valid input.")
     
    # spam ham email classification will run when user has entered email text data
    if input_email != "":
        #convert user input into Series for text cleaning
        #text cleaning of entered email by user and display result
        input_email_clean = ct.transform(pd.Series([input_email]))
        #display input of email text by user 
        st.markdown("""
                    #### User email text input (before text pre-processing / cleaning):-
                    
                    %s
                    
                    #### User email text input (after text pre-processing / cleaning):-
                    
                    %s""" %(input_email, input_email_clean[0]))
        
        st.subheader("Classification Result of Email Text Entered by User")
        #use lime to explain the prediction made by the logistic model and display the results
        exp = explainer.explain_instance(input_email_clean[0], logreg_tfidf_pipe.predict_proba, num_features = 10)
        #display the probability of both class for email classification of email entered by user
        st.write(f"Probability(spam) = {logreg_tfidf_pipe.predict_proba(input_email_clean)[0,1] *100 :.2f}%")
        st.write(f"Probability(ham) = {logreg_tfidf_pipe.predict_proba(input_email_clean)[0,0] *100 :.2f}%")
        #Display lime as html component and plot lime chart
        exp_html = components.html(exp.as_html(), height=400, width = 800)
        fig = exp.as_pyplot_figure()
        st.pyplot(fig)                 
