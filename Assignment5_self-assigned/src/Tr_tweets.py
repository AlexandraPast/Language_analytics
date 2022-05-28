# data processing tools
import os
import csv
import argparse
import urllib.request

# maths tools
import numpy as np
from scipy.special import softmax

# Huggingface tools
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from transformers import pipeline, set_seed

from transformers import RobertaTokenizer
from sklearn.metrics import (confusion_matrix,
                             ConfusionMatrixDisplay,
                            classification_report)

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

#--------------------------------------------------------------------------------------#

def main():
    # Create the parser
    my_parser = argparse.ArgumentParser(description = 'Sentiment analysis using Transformers')
    
    # Plot name and report name to be specified
    my_parser.add_argument('File',
                           metavar = 'filename',
                           type = str,
                           help = 'input the filename only')
    
    my_parser.add_argument('-Dir',
                           metavar = 'directory',
                           type = str,
                           default = "in/",
                           help = 'input the path to the folder containing the data file (only if the file is not inside "in" folder')
    
    my_parser.add_argument('-Report',
                           metavar = 'name for classification report',
                           type = str,
                           default = "Tr_report",
                           help = 'input your desired name for classification report')
    
    my_parser.add_argument('-Cfm',
                           metavar = 'confusion matrix name',
                           type = str,
                           default = "Tr_cfm",
                           help = 'input your desired name for the confusion matrix')
    
    my_parser.add_argument('-Cfm_plot',
                           metavar = 'confusion matrix plot name',
                           type = str,
                           default = "Tr_cfm",
                           help = 'input your desired name for the confusion matrix plot')
    
    my_parser.add_argument('-Df',
                           metavar = 'dataframe with results',
                           type = str,
                           default = "Tr_Data_predicted",
                           help = 'input your desired name for the output dataframe with data and model predictions')
    
    # Execute parse_args()
    args = my_parser.parse_args()
    
    print('--------------------------------------------------------------------------')
    # print values of arguments for the user to check
    vals = vars(args)
    for k, v in vals.items():
        print(f'{k:<4}: {v}')
     
    print('--------------------------------------------------------------------------')
    
    #-----------------------------------------------------------------------------#

    # Load data
    filename = os.path.join(args.Dir + args.File)
    df = pd.read_csv(filename).dropna()
    
    # Rename column
    df.rename(columns = {'sentiment':'label'}, inplace = True)
    
    # Select only necessary columns
    keep_col = ['text','label']
    data = df[keep_col]
    
    # Reset indices because we removed NAs
    data = data.reset_index(drop=True)
    
    print('--------------------------------------------------------------------------')
    print('----Data sample:')
    print(data.sample(10))
    print('--------------------------------------------------------------------------')
    
    #-----------------------------------------------------------------------------#
    # Load model, labels, tokenizer
    
    task='sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    # download label mapping
    labels=[]
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    # Initialize model
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
    
    #-----------------------------------------------------------------------------#
    
    # Predict text
    scores = []
    print('--------------------------------------------------------------------------')
    print('Predicting labels')
    for text in tqdm(data['text']):
        #encode using tokenizer
        encoded_input = tokenizer(text, return_tensors='tf')

        # get output
        output = model(encoded_input)

        # get outputs as numpy array
        score = output[0][0].numpy()

        # perform softmax classification
        score = softmax(score)

        # organize scores by the highest score
        ranking = np.argsort(score)
        ranking = ranking[::-1]
        for i in range(score.shape[0]):
            l = labels[ranking[i]]
            s = score[ranking[i]]
            # I take only the highest value and label and save it in a list
            if i == 0:
                wee = [l, np.round(float(s), 4)]
                scores.append(wee)
    
    # I am interested in how succesful is the model at predicting the main sentiment of the tweet
    # extract the label only, we don't need the percentage
    pred = [row[0] for row in scores]
    # extracting the true labels from the dataset
    real = data['label']
    
    #-----------------------------------------------------------------------------#
    # RESULTS
    
    # print and save classification report
    labels = ['negative', 'neutral', 'positive']

    report = classification_report(real, pred)
    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")
    print(report)
    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")

    # write classification to txt file
    with open("out/tables/" + args.Report + ".txt","w") as file:  # Use file to refer to the file object
        file.write(report)
    
    # create and save confusion matrix
    cm = pd.DataFrame(confusion_matrix(real, pred), index=labels, columns=labels)
    
    cm.to_csv('out/tables/' + args.Cfm + '.csv')
    
    # create and save plot of confusion matrix because they are pretty and easily readable!
    ConfusionMatrixDisplay.from_predictions(real, pred)
    plt.savefig('out/plots/' + args.Cfm_plot + '.png', dpi=75)
    
    #save a new csv including original data, predicted labels and percentages for future purposes
    dictionary = {'text': data['text'], 'real_label': real, 'pred_label': pred, 'percentage': [row[1] for row in scores]}  
    dataframe = pd.DataFrame(dictionary) 
    dataframe.to_csv('out/tables/' + args.Df + '.csv', index=False)
    
    print('------All done!------')
    
#--------------------------------------------------------------------------------------#

if __name__ == '__main__':
   main()
