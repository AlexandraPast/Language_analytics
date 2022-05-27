# system tools
import os
import sys
sys.path.append(os.path.join("..", "..", "CDS-LANG"))
import argparse
import numpy as np

# data munging tools
import pandas as pd
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# handle warnings
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
    
##-----------------------------------------------------------------------------##
##-----------------------------------------------------------------------------##

def main():
    #Create the parser
    my_parser = argparse.ArgumentParser(description = 'Text classification using TfidfVectorizer() and Logistic regression')
    
    #Plot name and report name to be specified
    my_parser.add_argument('File',
                           metavar = 'filename',
                           type = str,
                           help = 'input the filename only')
    
    my_parser.add_argument('-Dir',
                           metavar = 'directory',
                           type = str,
                           default = "in/",
                           help = 'input the path to the folder containing the data file (only if the file is not inside "in" folder')
    
    my_parser.add_argument('-Perf_plot',
                           metavar = 'performance plot',
                           type = str,
                           default = "performance_plot_logReg",
                           help = 'input your desired name for performance plot')
    
    my_parser.add_argument('-Report',
                           metavar = 'classification report name',
                           type = str,
                           default = "class_rep_logReg",
                           help = 'input your desired name for the classification report')
    
    #Execute parse_args()
    args = my_parser.parse_args()
    
    print('***************************************************************************')
    #print values of arguments for the user to check
    vals = vars(args)
    for k, v in vals.items():
        print(f'{k:<4}: {v}')
     
    print('***************************************************************************')
    
    #load data
    filename = os.path.join(args.Dir + args.File)
    data = pd.read_csv(filename, sep = ",")

    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")
    print("Data sample")
    print(data.sample(10)) #prints out sample of data to check if loaded correctly

    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")
    print("Data shape")
    print(data.shape) #checking the shape

    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")
    print("Label count")
    print(data["label"].value_counts()) #prints label counts for user to check

    data_balanced = clf.balance(data, 1000) #balancing the data
    
    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")
    print("Data shape - balanced") 
    print(data_balanced.shape) #check shape again

    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")
    print("Label count - balanced")
    print(data_balanced["label"].value_counts()) #checking labels again

    #new variables called X and y, taking the data out of the dataframe so that we can mess around with them
    X = data_balanced["text"]
    y = data_balanced["label"]

    #splitting data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X,           
                                                        y,          
                                                        test_size=0.2,   
                                                        random_state=42) 

    #creates vectorizer 
    vectorizer = TfidfVectorizer(ngram_range = (1,2),     
                                 lowercase =  True,       
                                 max_df = 0.95,           
                                 min_df = 0.05,           
                                 max_features = 100)      

    #turning data into vectors
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)
    #get feature names
    feature_names = vectorizer.get_feature_names()

    #fit the classifier to the data
    classifier = LogisticRegression(random_state=42).fit(X_train_feats, y_train)


    #most informative features
    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")
    print("Most informative features")
    clf.show_features(vectorizer, y_train, classifier, n=20)

    #make predictions
    y_pred = classifier.predict(X_test_feats)

    #assessing model performance
    clf.plot_cm(y_test, y_pred, normalized=True)
    plt.savefig('out/plots/'+ args.Perf_plot +'.png', dpi=75)

    #classification report
    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    print(classifier_metrics)
    
    #write classification to txt file
    with open("out/tables/" + args.Report + ".txt","w") as file:  # Use file to refer to the file object
        file.write(classifier_metrics)

    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")

    #cross-validation, in order to test our model's performance and influence of train-test split
    X_vect = vectorizer.fit_transform(X)

    #cross-validation plot
    title = "Learning Curves (Logistic Regression)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = LogisticRegression(random_state=42)
    clf.plot_learning_curve(estimator, title, X_vect, y, cv=cv, n_jobs=4)
    
##-----------------------------------------------------------------------------##
##-----------------------------------------------------------------------------##
    
if __name__ == '__main__':
   main()