# Data analysis
import os
import argparse
import pandas as pd
from collections import Counter
from tqdm import tqdm

# NLP
import spacy
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 3500000

# sentiment analysis VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# sentiment with spacyTextBlob
from spacytextblob.spacytextblob import SpacyTextBlob
nlp.add_pipe('spacytextblob')

# visualisations
import matplotlib.pyplot as plt
import numpy as np
import operator

import warnings
warnings.filterwarnings('ignore')

#-------------------------------------------------------------------------------------#

#FUNCTIONS

# function to get the first sentence (sorta...I can't figure out how to remove the first words such as name of author etc.
# because spacy .sent clumps it together with the first sentence and there is no obvious division between the two)
def get_first_sent(list):
    #Finding the first sentence
    first_sentence = ""
    for sentence in list:
        if len(sentence) > 5:
            first_sentence = sentence
            break
    return first_sentence   

# func to get last sentence (or in some cases half of it due to Spacy not recognising more complex 
# sentences such as usage of double direct speech - if "'!'" are used in one sentence it confuses it I think
# for example see Jane Eyre I have accidentally picked as my 1st attempt book 
def get_last_sent(list):
    #Finding last sentence (the condition is there to prevent some misshaps as single "" as last sentence)
    if len(list[-1]) > 5:
        last_sentence = list[-1]
    else: 
        last_sentence = list[-2]
    return last_sentence

def get_VADER_scores (list):
    vader_scores = []
    # get sentiment scores with VADER
    for sentence in list:
        score = analyzer.polarity_scores(str(sentence))
        vader_scores.append(score)
    # create a dataframe
    vader_df = pd.DataFrame(vader_scores)
    return vader_df

def get_Blob_scores (list):
    # get sentiment scores with spaCyTextBlob
    polarity = []
    for sentence in list:
        doc = nlp(str(sentence))
        score_p = doc._.blob.polarity
        polarity.append(score_p)
    return polarity

def sentiment_plot (sentence_type, Spacy_scores, Vader_scores):
    # make a plot using VADER and Blob scores from two lists with shared grid for easy comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,4))
    fig.suptitle('Sentiment scores for the' +sentence_type+ 'sentences', fontsize = 15, color = (0.2, 0.4, 0.9, 1.0), fontweight = 'normal')
    ax1.plot(Spacy_scores)
    ax2.plot(Vader_scores)
    ax1.set(ylabel="Polarity score")
    ax2.set(ylabel="Vader sentiment score")
    ax1.legend(["spaCyTextBlob score"])
    ax2.legend(["Vader score"])
    fig.savefig("out/sentiment_plot_" + sentence_type + ".png")
    # display
    fig.show
    
def GPEs_bar_chart(GPE_list):
    # sort by second item on the list of lists - the counts
    s = sorted(GPE_list, key = operator.itemgetter(1))

    # assign x and y variables
    x = [item[0] for item in s]
    y = [item[1] for item in s]

    # Set the figure size
    plt.rcParams["figure.figsize"] = [10.00, 7]

    ind = np.arange(len(y))
    fig, ax = plt.subplots()
    # bar settings
    ax.barh(ind, y, color=(0.2, 0.7, 0.5, 0.6)) 
    ax.set_yticks(ind)
    ax.set_yticklabels(x)
    # auto-label bars
    ax.bar_label(ax.containers[0], label_type = 'edge', color = 'red', fontsize = 10, padding=2)
    # pad the spacing between the number and the edge of the figure
    ax.margins(y = 0.01)

    # name labels
    plt.title('Most common GPEs', fontsize = 15, fontweight = 'bold')
    plt.xlabel('Number of times they appear in the whole corpus', fontsize = 12, fontweight = 'bold')
    plt.ylabel('GPEs', fontsize = 12, fontweight = 'bold')   
    plt.savefig("out/bar_chart_GPEs.png")
    plt.show()
    
#-------------------------------------------------------------------------------------#

def main():
    # Firstly the user specified arguments
    # Create the parser
    my_parser = argparse.ArgumentParser(description = 'Sentiment and NER')
    
    # Add the arguments
    # Hyperparameters to specify

    my_parser.add_argument('-Path',
                           metavar = 'Path to the folder with text',
                           type = str,
                           default = "in/",
                           help = 'input path to the folder with text if it is not inside "in" folder eg. "C:/User/folder/folder/"')
    
        
    # Execute parse_args()
    args = my_parser.parse_args()
    
    print('--------------------------------------------------------------------------')
    print('------Your input values:')
    # print values of arguments for the user to check
    vals = vars(args)
    for k, v in vals.items():
        print(f'{k:<4}: {v}')
     
    print('--------------------------------------------------------------------------')
    
    # directory of the texts to be analysed
    directory = args.Path
    # define counter as global so it doesn't throw an error
    global Counter
#-------------------------------------------------------------------------------------#

    # preparing empty lists
    f_VADER_scores = []
    l_VADER_scores = []
    f_Blob_scores = []
    l_Blob_scores = []
    ents = []

    print('------Analysing the texts------')
    print('(This might take a looong time)')
    # loop which takes a loong time to go through 
    for filename in tqdm(os.listdir(directory)):
        f = os.path.join(directory, filename)
        #print(f) #use only to check if code keeps getting stuck
        # checking if it is a file
        if os.path.isfile(f):
            if f.endswith('.txt'):
                # read the file
                with open(f, "r", encoding="utf-8") as file:
                    text = file.read()
                # create nlp object
                doc = nlp(text)      
                # make a list of sentences in a doc
                lst = []
                for sentence in doc.sents:
                    lst.append(sentence)
                # get first and last sentence
                res = []
                res.append(get_first_sent(list = lst))
                res.append(get_last_sent(list = lst))
                # get VADER sentiment scores for each sentence
                vader_df = get_VADER_scores(list = res)
                f_VADER_scores.append(vader_df["compound"][0])
                l_VADER_scores.append(vader_df["compound"][1])
                # get TextBlob sentiment scores for each sentence
                polarity = get_Blob_scores(list = res)
                f_Blob_scores.append(polarity[0])
                l_Blob_scores.append(polarity[1])
                # get all GPEs
                for entity in doc.ents:
                    if entity.label_ == "GPE":
                        ents.append(entity.text) 
    
    # Plotting sentiment scores for opening sentences
    sentiment_plot(sentence_type = "OPENING", Spacy_scores = f_Blob_scores, Vader_scores = f_VADER_scores) 
    print('------Saved first sentiment plot')

    # Plotting sentiment scores for closing sentences
    sentiment_plot(sentence_type = "CLOSING", Spacy_scores = l_Blob_scores, Vader_scores = l_VADER_scores)
    print('------Saved second sentiment plot')
    
    # Finding 20 most common GPEs
    # Create counter variable to keep counts in
    Counter = Counter(ents)
    # find 20 most common from the list
    most_common_GPE = Counter.most_common(20)

    # Behold, the holy bar chart!
    GPEs_bar_chart(GPE_list = most_common_GPE)
    print('------Saved GPEs bar chart')
    
    print('------All done!------')
#-------------------------------------------------------------------------------------#

if __name__ == '__main__':
   main()