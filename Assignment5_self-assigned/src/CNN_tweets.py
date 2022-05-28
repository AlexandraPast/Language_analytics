import os
import sys
import argparse

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
                                    Embedding,
                                    Bidirectional,
                                    LSTM,
                                    Dropout)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.regularizers import L2
from keras.callbacks import ModelCheckpoint, EarlyStopping

# scikit-learn
from sklearn.metrics import (confusion_matrix, 
                            classification_report,
                            ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split

# visualisations 
import matplotlib.pyplot as plt
# handle warnings

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

#--------------------------------------------------------------------------------------#
# FUNCTIONS

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
    
#--------------------------------------------------------------------------------------#

def main():
    # Create the parser
    my_parser = argparse.ArgumentParser(description = 'Sentiment analysis using CNN and word embeddings')
    
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
    
    my_parser.add_argument('-Batch',
                           metavar = '-batch size',
                           type = int,
                           default = 200,
                           help = 'input integer number')
    
    my_parser.add_argument('-Epochs',
                           metavar = '-epochs',
                           type = int,
                           default = 10,
                           help = 'input integer number')
    
    my_parser.add_argument('-Report',
                           metavar = 'name for classification report',
                           type = str,
                           default = "CNN_report",
                           help = 'input your desired name for classification report')
    
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
    
    # Execute parse_args()
    args = my_parser.parse_args()
    
    print('--------------------------------------------------------------------------')
    # print values of arguments for the user to check
    vals = vars(args)
    for k, v in vals.items():
        print(f'{k:<4}: {v}')
     
    print('--------------------------------------------------------------------------')
    
    #-----------------------------------------------------------------------------#
    # LOAD DATA
    
    filename = os.path.join(args.Dir + args.File)

    df = pd.read_csv(filename).dropna()
    
    # rename column name
    df.rename(columns = {'sentiment':'label'}, inplace = True)
    
    # select only columns we'll use
    keep_col = ['text','label']
    data = df[keep_col]
    
    # reset indices after removing NAs
    data = data.reset_index(drop=True)
    
    print('--------------------------------------------------------------------------')
    print('----Data sample:')
    print(data.sample(10))
    print('--------------------------------------------------------------------------')
    
    #-----------------------------------------------------------------------------#
    # Prepare data for the model
    
    # balance function, I am using largest samples possible
    data_balanced = clf.balance(data, 7781)
    
    # build train and test datasets
    # var X and y for training and testing 
    X = data_balanced["text"]
    y = data_balanced["label"]

    print('--------------------------------------------------------------------------')
    print('----Label count:')
    print(data_balanced["label"].value_counts())
    print('--------------------------------------------------------------------------')
    
    # split data to training and testing
    X_train, X_test, y_train, y_test = train_test_split(X,           
                                                        y,          
                                                        test_size=0.2,   
                                                        random_state=42)
    
    # normalise data
    X_train_norm = pre_process_corpus(X_train)
    X_test_norm = pre_process_corpus(X_test)
    
    # define out-of-vocabulary token
    t = Tokenizer(oov_token = '<UNK>')

    # fit the tokenizer on then documents
    t.fit_on_texts(X_train_norm)

    # set padding value
    t.word_index["<PAD>"] = 0 
    
    X_train_seqs = t.texts_to_sequences(X_train_norm)
    X_test_seqs = t.texts_to_sequences(X_test_norm)
    print('--------------------------------------------------------------------------')
    print(f"Vocabulary size={len(t.word_index)}")
    print(f"Number of Documents={t.document_count}")
    print('--------------------------------------------------------------------------')
    
    
    MAX_SEQUENCE_LENGTH = 1000
    # add padding to sequences
    X_train_pad = sequence.pad_sequences(X_train_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    X_test_pad = sequence.pad_sequences(X_test_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    
    # Encoding labels
    # create one-hot encodings
    lb = LabelBinarizer()
    y_train_lb = lb.fit_transform(y_train)
    y_test_lb = lb.fit_transform(y_test)
    
    #-----------------------------------------------------------------------------#
    # Create and compile model
    
    # overall vocabulary size
    VOCAB_SIZE = len(t.word_index)
    # number of dimensions for embeddings
    EMBED_SIZE = 300
    # number of epochs to train for
    EPOCHS = args.Epochs
    # batch size for training
    BATCH_SIZE = args.Batch
    
    # create the model
    model = Sequential()
    # embedding layer
    model.add(Embedding(VOCAB_SIZE, 
                        EMBED_SIZE, 
                        input_length=MAX_SEQUENCE_LENGTH))

    # first convolution layer and pooling
    model.add(Conv1D(filters=128, 
                            kernel_size=4, 
                            padding='same',
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    # second convolution layer and pooling
    model.add(Conv1D(filters=64, 
                            kernel_size=4, 
                            padding='same', 
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=32, 
                            kernel_size=4, 
                            padding='same', 
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    # fully-connected classification layer
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                            optimizer='adam', 
                            metrics=['accuracy'])
    # print model summary
    model.summary()
    
    #early stopping and model checkpoints
    checkpoint = ModelCheckpoint("sequential_10",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 save_freq="epoch")
    early = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          mode='auto')
    
    #-----------------------------------------------------------------------------#
    # Train and predict data
    
    history = model.fit(X_train_pad, y_train_lb,
                        epochs = EPOCHS,
                        batch_size = BATCH_SIZE,
                        validation_split = 0.1,
                        verbose = True,
                        callbacks = [checkpoint,early])
    
    # Check if EarlyStopping stopped training, if yes, save the value for history plot
    if early.stopped_epoch == 0:
        plot_epoch = EPOCHS
    else:
        plot_epoch = early.stopped_epoch + 1
    
    # Plot and save training loss and accuracy plot
    plot_history(history, plot_epoch, args.LA_plot)
    
    # Evaluate
    # Final evaluation of the model
    scores = model.evaluate(X_test_pad, y_test_lb, verbose=1)
    print(f"Accuracy: {scores[1]}")
    
    # 0.5 decision boundary
    predictions = (model.predict(X_test_pad) > 0.5).astype("int32")
    
    # Transform predicted binary labels back to multi-class labels.
    y_pred = lb.inverse_transform(predictions, threshold=None)
    
    #-----------------------------------------------------------------------------#
    # RESULTS
    
    # print and save classification report
    labels = ['negative', 'neutral', 'positive']
    report = classification_report(y_test, y_pred)
    
    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")
    print(report)
    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")

    with open("out/tables/" + args.Report + ".txt","w") as file:  # Use file to refer to the file object
        file.write(report)
        
    # create and save confusion matrix
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred), index=labels, columns=labels)
   
    cm.to_csv('out/tables/'+ args.Cfm +'.csv')
    
    # save confusion matrix plot
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig('out/plots/' + args.Cfm_plot + '.png', dpi=75)
    
    print('------All done!------')
    
#--------------------------------------------------------------------------------------#

if __name__ == '__main__':
   main()
