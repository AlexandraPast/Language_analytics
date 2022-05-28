# Assignment 2 - Sentiment and NER

The files here are part of the repository for my course in Language analytics.

## 1. Contribution
I worked on this code alone. 

## 2. Assignment description
The aim of this assignment was to perform NER and sentiment analysis using techniques from the class. Common applications of NLP include finding named entities and different forms of sentiment analysis. In our case we used simple ways of calculating sentiment scores for sentences. Using these methods we can extract valuable information such as what is being spoken about and what sentiment is shown in texts analysed. 

We were allowed to choose from two different tasks. I chose the first option. 

#### The specific tasks were:
**For every novel in the corpus:**
- Get the sentiment of the first sentence.
- Get the sentiment of the final sentence.
- Plot the results over time, with one visualisation showing the sentiment of opening sentences over time and one of closing sentences over time.
- Find the 20 most common geopolitical entities mentioned across the whole corpus - plot the result as a bar chart.

## 3. Methods
For this assignment, I wrote a script `Sent_NER.py`. I created a for loop which loops over each text extracting the first and the last sentence. I calculated sentiment scores using two methods, VADER sentiment and SpaCy TextBlob. In the for loop, I also extracted all GPEs present in the text. Afterwards, I compiled all extracted data from all the texts and created 3 plots. Two plots showing side-by-side VADER and TextBlob sentiments of opening and closing sentences. A bar chart showing the 20 most common GPEs present in the whole corpus. 

## Scripts
#### This repository consists of:
- **in**: a folder containing the input for the code or data. In my case: 100-english-novels.
- **out**: a folder containing the output of the script. The output is 3 plots mentioned above.
- **src**:
  - `Sent_NER.py`: script for the sentiment analysis and GPEs extraction
- `setup.sh`: installs the necessary libraries to run the scripts

#### The script does the following:
- Loads all texts (only accepts .txt files)
- Gets the first and the last sentence of each text
- Calculates VADER sentiment and SpaCy TextBlob sentiment score
- Creates plots of all sentiment scores for opening and closing sentences
- Extracts all GPEs
- Plots 20 most common GPEs on a bar chart

The script runs with pre-set variables which can be changed by the user. 

#### Default values of optional arguments:
- **-Path:** `in/` (the script will analyse everything inside the 'in' folder by default)

## 4. Usage
#### How to correctly run the script:
1. Set your working directory to be the folder containing all the subfolders (in, out, src) using the `cd "path"` command.
2. Open the console and type: `bash setup.sh` (this should install the necessary packages)
3. Type `python` and follow it with the path to `Sent_NER.py` which should be `src/Sent_NER.py` if you set your working directory correctly.
4. Following the path to the `.py` script should be your parameters. For positional arguments, only value needs to be input. For optional arguments specify argument name, then value. It is not required to specify optional arguments as they have a default setting. This script only contains one optional argument.
5. Example: 
   - `cd user/file/file/Assignment2_Sentiment_analysis_NER` - set working directory
   - `bash setup.sh` - install libraries
   - `python ../src/Sent_NER.py` - run script (this will analyse all txt files inside 'in' folder)
   - `python ../src/Sent_NER.py -Path C:/user/file/file/folder/` - if you wish to analyse all files in any other directory than 'in'
   - `python  ../src/Sent_NER.py -h` (or `--help`) - for help and explanations

## 5. Discussion of results
In the resulting plots, we see the sentiment of opening and closing sentences jumping from positive to negative and I would assume there is no trend we could observe from that. An interesting point, however, is a comparison between the two different sentiment scores. We can see the VADER score seems to be much more polarised towards higher values in both negative and positive spectrums, while SpaCy TextBlob is more conservative in comparison. Regarding the bar chart, it seems the most common GPEs in the text were London and England, unsurprisingly, since I analysed English novels. Some of the GPEs such as 'thou' and 'Thou' are more suspicious because I think it is meant to be the old English expression for 'you', rather than a geopolitical entity. Also, the names Leonora or Sylvia might be actual places, but might also be names of the characters from the books.
