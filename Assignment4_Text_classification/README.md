# Assignment 4 - Text classification

The files here are part of the repository for my course in Language analytics.

## 1. Contribution
I worked on this code alone. 

## 2. Assignment description
The aim of this assignment was to write two scripts using two different methods for text classification, one script using methods from classical machine learning and second script using more contemporary method from deep learning. Model like LogisticRegression classifier offers a solid benchmark which we can build on. Additionally word embeddings can be employed to make dense vector representations of words, encoding linguistic information about the word and it's relationship to other words. These can be fed into classifiers and improve performance and generalizability. Using these methods we should be able to analyse the data and predict whether a comment is a certain kind of toxic speech. 

#### The specific tasks were:
- The first script should perform benchmark classification using standard machine learning approaches
  - This means CountVectorizer() or TfidfVectorizer(), LogisticRegression classifier
  - Save the results from the classification report to a text file
- The second script should perform classification using the kind of deep learning methods we saw in class
  - Keras Embedding layer, Convolutional Neural Network
  - Save the classification report to a text file

## 3. Methods
The problem represents the text classification task. In order to predict the toxicity or lack thereof, I first transformed the texts into vectors using TfidfVectorizer() and then trained LogisticRegression classifier as one of the methods from standard machine learning. After training, I made the model predict labels for a subset of the data it has not been exposed to before and saved the results.
For the second approach, I created a CNN with an embedding layer, 3 convolutional layers and one fully connected classification layer. Similarly, I trained the model on the training subset of the data and using model.evaluate() obtained the accuracy of the model. I also saved predictions of labels of the testing set and saved the results. 

## Scripts
#### This repository consists of:
- **in**: a folder containing the input for the code or data. 
  - Data used: https://www.simula.no/sites/default/files/publications/files/cbmi2019_youtube_threat_corpus.pdf
- **out**: a folder containing the output of the script.
  - **Plots**: 
   - LogReg confusion matrix plot, LogReg Performance plot, cross-validation plot for Logistic regression
   - CNN confusion matrix plot, CNN loss and accuracy plot
  - **Tables**:
   - LogReg confusion matrix, LogReg Classification report
   - CNN confusion matrix, CNN Classification report
- **src**:
  - `Text_class_LogReg.py`: script for the LogisticRegression classifier
  - `Text_class_CNN.py `: script for the Convolutional neural network classifier
  - **utils**: contains necessary modules
- `setup.sh`: installs the necessary libraries to run the scripts

#### The script does the following:
**`Text_class_LogReg.py`**
- takes input for several variables (below)
- loads the data
- balances the data so each label has the same number of items/texts
- using TfidfVectorizer() creates vectors
- trains LogisticRegression classifier
- evaluates the model
- makes predictions
- saves various plots of performance and tables (described above)

**`Text_class_CNN.py`**
- takes input for several variables
- loads the data
- balances the data
- tokenizes text
- trains CNN
- evaluates model performance
- makes predictions
- saves plots and tables with results

The script runs with pre-set variables which can also be changed by the user. 

### **`Text_class_LogReg.py`**
**Required positional arguments:**
- **File** (input for filename (the data) eg. VideoCommentsThreatCorpus.csv)

**Default values of optional arguments:**
- **-Dir:** `in/` (if data is elsewhere, input path to the directory)
- **-Perf_plot:** 'LogReg_performance' (model performance plot)
- **-Cfm:** 'LogReg_cfm' (confusion matrix)
- **-Cfm_plot:** 'LogReg_cfm' (confusion matrix plot)
- **-Report:** 'LogReg_report' (classification report name)

### **`Text_class_CNN.py`**
**Required positional arguments:**
- **File** (input for filename (the data) eg. VideoCommentsThreatCorpus.csv)

**Default values of optional arguments:**
- **-Dir:** `in/` (if data is elsewhere, input path to the directory)
- **-Report:** 'CNN_report' (classification report name)
- **-Cfm:** 'CNN_cfm' (confusion matrix)
- **-Cfm_plot:** 'CNN_cfm' (confusion matrix plot)
- **-LA_plot:** 'CNN_loss_acc' (loss and accuracy plot)

## 4. Usage
#### How to correctly run the script:
1. Set your working directory to be the folder containing all the subfolders (in, out, src) using the `cd "path"` command.
2. Open the console and type: `bash setup.sh` (this should install the necessary packages)
3. Type `python` and follow it with the path to `Text_class_CNN.py` which should be `src/Text_class_CNN.py` if you set your working directory correctly.
4. Following the path to the `.py` script should be your parameters. For positional arguments, only value needs to be input. Make sure to input positional arg values in order. For optional arguments specify argument name, then value. It is not required to specify optional arguments as they have a default setting.  
5. Example: 
   - `cd user/file/file/Assignment4_Text_classification` - set working directory
   - `bash setup.sh` - install libraries
   - `python ../src/Text_class_CNN.py VideoCommentsThreatCorpus.csv` - run script on this file (only works if the file is inside the 'in' folder
   - `python ../src/Text_class_CNN.py VideoCommentsThreatCorpus.csv -Cfm_plot MyPrettyPlot` - specified confusion matrix plot name 
   - `python ../src/Text_class_CNN.py VideoCommentsThreatCorpus.csv -Dir C:/user/file/file/` - if data is in other directory than 'in', input path to the directory
   - `python  ../src/Text_class_CNN.py -h` (or `--help`) - for help and explanations

## 5. Discussion of results
The initial Logistic Regression classifier had a weighted average accuracy of 77%. From the performance plot we can see that 27% of toxic comments were predicted to be the opposite. In comparison, the CNN model with word embeddings achieved 83% weighted average accuracy in 4 epochs. In the loss and accuracy plot we can also observe the loss decreasing further and accuracy increasing, suggesting more data could lead to higher accuracy. Overall, I believe larger dataset would improve both models' performances. 



