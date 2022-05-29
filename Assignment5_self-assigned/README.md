# Assignment 5 - self-assigned
The files here are part of the repository for my course in Language analytics.

## 1. Contribution
I worked on this code alone. 

## 2. Assignment description
This assignment is self-assigned. For my task, I chose the sentiment classification task. I downloaded data from Kaggle (https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset). The data represent tweets with the labelled sentiment. Labels are 'negative', 'neutral' or 'positive'. I was interested in seeing the comparison between analysis using Transformers and analysis using a CNN with word embeddings. Transformers offer a simpler solution to sentiment classification while also offering many models pre-trained on millions of samples and so I wanted to compare if that approach would be more efficient than building a CNN from scratch in terms of results but also difficulty.

## 3. Methods
For this comparison, I created two different .py scripts. The first script includes analysis using Transformers (RoBERTa), albeit very simple. Using a for loop, I analysed the sentiment of each tweet in the csv file. Because the output of the model is a score with all 3 labels and their respective percentage, I sorted the scores and for my task purposes assigned the highest percentage as the predicted label. Afterwards, I compared the real labels with the predictions and saved all results into the 'out' folder.
The second script represents a more complicated solution for a beginner in Python. After preprocessing the data, I created a CNN with one embedding layer, 3 convolutional layers, a fully connected classification layer and one softmax layer for multi-class labels. I used Model Checkpoints and Early Stopping to find when the model performs the best and prevent over-fitting. I trained the model on the training subset of the data and made evaluations and predictions. I saved all the results into the 'out' folder.

## Scripts
#### This repository consists of:
- **in**: a folder containing the input for the code or data. 
  - Data used: (https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset)
- **out**: a folder containing the output of the script.
  - **Plots**: 
   - Transformers confusion matrix plot
   - CNN confusion matrix plot, CNN loss and accuracy plot
  - **Tables**:
   - Transformers confusion matrix, Transformers Classification report, csv - data including the predictions and percentages
   - CNN confusion matrix, CNN Classification report
- **src**:
  - `Tr_tweets.py`: script for the Transformers analysis
  - `CNN_tweets.py `: script for the Convolutional neural network classifier
  - **utils**: contains necessary modules
- `setup.sh`: installs the necessary libraries to run the scripts

#### The script does the following:
**`Tr_tweets.py`**
- takes input for several variables
- loads data
- loops over data from csv and tokenizes text
- feeds the text into a model and saves the scores
- pick the highest percentage and saves it into a list of predicted labels
- makes confusion matrix, confusion matrix plot and classification report
- saves all results into `out/`


**`CNN_tweets.py`**
- takes input for several variables
- loads the data
- balances the data
- tokenizes text
- trains CNN
- evaluates model performance
- makes predictions
- saves plots and tables with results into `out/`

The script runs with pre-set variables which can also be changed by the user. 

#### **`Tr_tweets.py`**
**Required positional arguments:**
- **File** (input for filename (the data) eg. Tweets.csv)

**Default values of optional arguments:**
- **-Dir:** `in/` (if data is elsewhere, input path to the directory)

- **-Report:** 'Tr_report' (classification report name)
- **-Cfm:** 'Tr_cfm' (confusion matrix name)
- **-Cfm_plot:** 'Tr_cfm' (confusion matrix plot name)
- **-Df:** 'Tr_Data_predicted' (csv including predictions name)

#### **`CNN_tweets.py`**
**Required positional arguments:**
- **File** (input for filename (the data) eg. Tweets.csv)

**Default values of optional arguments:**
- **-Dir:** `in/` (if data is elsewhere, input path to the directory)
- **-Batch:** 200 (Batch size)
- **-Epochs:** 10

- **-Report:** 'CNN_report' (classification report name)
- **-Cfm:** 'CNN_cfm' (confusion matrix name)
- **-Cfm_plot:** 'CNN_cfm' (confusion matrix plot name)
- **-LA_plot:** 'CNN_loss_acc' (loss and accuracy plot name)

## 4. Usage
#### How to correctly run the script:
1. Set your working directory to be the folder containing all the subfolders (in, out, src) using the `cd "path"` command.
2. Open the console and type: `bash setup.sh` (this should install the necessary packages)
3. Type `python` and follow it with the path to `CNN_tweets.py` which should be `src/CNN_tweets.py` if you set your working directory correctly.
4. Following the path to the `.py` script should be your parameters. For positional arguments, only value needs to be input. Make sure to input positional arg values in order. For optional arguments specify argument name, then value. It is not required to specify optional arguments as they have a default setting.  
5. Example: 
   - `cd user/file/file/Assignment5_self-assigned` - set working directory
   - `bash setup.sh` - install libraries
   - `python ../src/CNN_tweets.py Tweets.csv` - run script on this file (only works if the file is inside the 'in' folder
   - `python ../src/CNN_tweets.py Tweets.csv -Cfm_plot MyPrettyPlot` - specified confusion matrix plot name 
   - `python ../src/CNN_tweets.py Tweets.csv -Dir C:/user/file/file/` - if data is in other directory than 'in', input path to the directory
   - `python  ../src/CNN_tweets.py -h` (or `--help`) - for help and explanations

## 5. Discussion of results
The transformers classifier had a weighted average accuracy of 72%. From the confusion matrix/plot, we can see that it had an issue with misclassifying the neutral statements as either positive or negative. Because it's an already pre-trained model for sentiment classification task using tweets, I did not train the model. In comparison, the CNN model at first achieved 63% weighted average accuracy. From the loss and accuracy plot, I could see that the training reached around 95% and the validation accuracy plateaued at approximately 60%. I believed that the model was overfitting also due to validation loss increasing. I implemented dropout, L2 regularisation, Model checkpoints and Early stopping to mitigate the overfitting. The model reached weighted average accuracy of 68%, nonetheless, the curves of loss and accuracy looked similar. 
Considering the confusion matrix, it also had more issues with misclassifying tweets as neutral or neutral tweets as positive or negative. I tried different model hyperparameters and layer setups, however, the current model seems to have the best performance. Perhaps more data could result in higher accuracy.

To conclude, I think that using Transformers is easier however, during my research, I found the documentation to be either too complex (much-complicated code) or insufficient in explanations (for example, how to get data into shape the transformers' functions work with). That is why I used my approach, which ignores that the model assigns every tweet a percentage of each sentiment rather than one label. However, even using my simplified analysis method, it achieved higher accuracy than my CNN on this dataset.



