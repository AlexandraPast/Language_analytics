# system tools
import os
import sys
#sys.path.append(os.path.join("..","..", "..", "CDS-LANG"))
import argparse
import numpy as np

# simple text processing tools
import re
import tqdm
import unicodedata
import contractions
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')

# data wranling
import pandas as pd
import numpy as np
import utils.classifier_utils as clf

# tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, 
                                    Flatten,
                                    Conv1D, 
                                    MaxPooling1D, 
                                    Embedding)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.regularizers import L2

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, 
                             classification_report,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

# visualisations 
import matplotlib.pyplot as plt
#%matplotlib inline

# handle warnings
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

##-----------------------------------------------------------------------------##
##-----------------------------------------------------------------------------##
## DEFINITIONS

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def pre_process_corpus(docs):
    norm_docs = []
    for doc in tqdm.tqdm(docs):
        doc = strip_html_tags(doc)
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        doc = doc.lower()
        doc = remove_accented_chars(doc)
        doc = contractions.fix(doc)
        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()  
        norm_docs.append(doc)
  
    return norm_docs

def plot_history(H, epochs, plotname):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label='Training loss (' + str(str(format(H.history["loss"][-1],'.5f'))+')'))
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label='Validation loss (' + str(str(format(H.history["val_loss"][-1],'.5f'))+')'), linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label='Training accuracy (' + str(format(H.history["accuracy"][-1],'.5f'))+')')
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label='Validation accuracy (' + str(format(H.history["val_accuracy"][-1],'.5f'))+')', linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig("out/plots/"+ plotname +".png")
    plt.show()

##-----------------------------------------------------------------------------##
##-----------------------------------------------------------------------------##

def main():
    #Create the parser
    my_parser = argparse.ArgumentParser(description = 'Text classification using CNN')
    
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
    
    my_parser.add_argument('-Report',
                           metavar = 'report name for CNN model Keras',
                           type = str,
                           default = "CNN_report",
                           help = 'input your desired name for the classification report')
    
    my_parser.add_argument('-Cfm',
                           metavar = 'confusion matrix name',
                           type = str,
                           default = "CNN_cfm",
                           help = 'input your desired name for the confusion matrix')
    
    my_parser.add_argument('-Cfm_plot',
                           metavar = 'confusion matrix plot name',
                           type = str,
                           default = "CNN_cfm",
                           help = 'input your desired name for the confusion matrix plot')
    
    my_parser.add_argument('-LA_plot',
                           metavar = 'loss accuracy plot',
                           type = str,
                           default = "CNN_loss_acc",
                           help = 'input your desired name for loss and accuracy plot')
    
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
    print("DATA INFO")
    print(data.info())
    print("-----------------------------------------------------------------------------")
    print("DATA SAMPLE")
    print(data.sample(10))
    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")

    #balance function
    data_balanced = clf.balance(data, 1000)

    #var X and y for training and testing 
    X = data["text"]
    y = data["label"]
    data["label"].value_counts()

    #scikit-learn train-test split function
    X_train, X_test, y_train, y_test = train_test_split(X,           
                                                        y,          
                                                        test_size=0.2,   
                                                        random_state=42) 


    #Clean and normalize data
    X_train_norm = pre_process_corpus(X_train)
    X_test_norm = pre_process_corpus(X_test)

    #Preprocessing
    # define out-of-vocabulary token
    t = Tokenizer(oov_token = '<UNK>')

    # fit the tokenizer on then documents
    t.fit_on_texts(X_train_norm)

    # set padding value
    t.word_index["<PAD>"] = 0 

    #Tokenize sequences
    X_train_seqs = t.texts_to_sequences(X_train_norm)
    X_test_seqs = t.texts_to_sequences(X_test_norm)
    
    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")
    print(f"Vocabulary size={len(t.word_index)}")
    print(f"Number of Documents={t.document_count}")
    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")

    #Sequence normalization
    MAX_SEQUENCE_LENGTH = 1000

    #add padding to sequences
    X_train_pad = sequence.pad_sequences(X_train_seqs, maxlen = MAX_SEQUENCE_LENGTH, padding="post")
    X_test_pad = sequence.pad_sequences(X_test_seqs, maxlen = MAX_SEQUENCE_LENGTH, padding="post")

    X_train_pad.shape, X_test_pad.shape

    #Create and compile model
    #define paramaters for model

    #overall vocublarly size
    VOCAB_SIZE = len(t.word_index)
    #number of dimensions for embeddings
    EMBED_SIZE = 300
    #number of epochs to train for
    EPOCHS = 2
    # batch size for training
    BATCH_SIZE = 128

    #create the model
    model = Sequential()
    #embedding layer
    model.add(Embedding(VOCAB_SIZE, 
                        EMBED_SIZE, 
                        input_length=MAX_SEQUENCE_LENGTH))

    #first convolution layer and pooling
    model.add(Conv1D(filters=128, 
                            kernel_size=4, 
                            padding='same',
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    #second convolution layer and pooling
    model.add(Conv1D(filters=64, 
                            kernel_size=4, 
                            padding='same', 
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=32, 
                            kernel_size=4, 
                            padding='same', 
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    #fully-connected classification layer
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                            optimizer='adam', 
                            metrics=['accuracy'])
    #print model summary
    model.summary()

    #Train
    history = model.fit(X_train_pad, y_train,
              epochs = EPOCHS,
              batch_size = BATCH_SIZE,
              validation_split = 0.1,
              verbose = True)

    # Plot and save training loss and accuracy plot
    plot_history(history, EPOCHS, args.LA_plot)
    
    #Evaluate
    #Final evaluation of the model
    scores = model.evaluate(X_test_pad, y_test, verbose=1)
    print(f"Accuracy: {scores[1]}")

    #0.5 decision boundary
    y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")


    #confusion matrix and classification report
    labels = ['negative', 'positive']
    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")
    report = classification_report(y_test, y_pred)
    print(report)
    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")
    
    #write classification to txt file
    with open("out/tables/"+ args.Report +".txt","w") as file:  # Use file to refer to the file object
        file.write(report)

    #save confusion matrix table
    df = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                      index=labels, columns=labels)
    df.to_csv('out/tables/'+ args.Cfm +'.csv')
    
    #save confusion matrix plot
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig('out/plots/' + args.Cfm_plot + '.png', dpi=75)
    
    print('---Results and plots saved')
    print('---All done!---')

##-----------------------------------------------------------------------------##
##-----------------------------------------------------------------------------##
    
if __name__ == '__main__':
   main()