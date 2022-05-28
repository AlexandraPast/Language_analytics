# Assignment 1 - Collocation tool

The files here are part of the repository for my course in Visual analytics.

## 1. Contribution
I worked on this code alone. 

## 2. Assignment description
The aim of this assignment was to perform collocational analysis using string processing and NLP tools introduced in the lecture. Collocates are words that frequently co-occur together in a given context. It is possible to calculate the strength of association between these words and assess how they are related. For this purpose, I wrote a script that can extract collocates and calculate their MI score.

#### The specific tasks were:
- Take a user-defined search term and user-defined window size.
- Take one specific text that the user can define.
- Find all the context words which appear ± the window size from the search term in that text.
- Calculate the mutual information score for each context word.
- Save the results as a CSV file with (at least) the following columns: the collocate term; how often it appears as a collocate; how often it appears in the text; the mutual information score.

## 3. Methods
To complete the assigned task I wrote a .py script `col_tool.py`. I first cleaned the text from any symbols which could interfere with the analysis. I proceeded to extract all words around the user-defined chosen word within the window size that was specified by the user as well. For testing, I selected the words purple and green, and a window size of 3 as default. Subsequently, I calculated all required variables for the calculation of the mutual information score (MI) from the English Corpora website and finally, the MI score itself for each collocate. I saved all output into a .csv table. 

**The MI formula:**
A = frequency of node word (e.g. purple): 1246
B = frequency of collocate (e.g. color): 112
AB = frequency of collocate near the node word (e.g. color near purple): 22
sizeCorpus= size of corpus (# words; in this case the BNC): 96,263,399
span = span of words (e.g. 3 to left and 3 to right of node word): 6

MI = 11.30 = log ( (22 * 96,263,399) / (1246 * 112 * 6) ) / .30103

(taken from: https://www.english-corpora.org/mutualInformation.asp)

## Scripts

#### This repository consists of:
- **in**: a folder containing the input for the code or data. In this case, the data was 100-english-novels provided by my professor.
- **out**: a folder containing the output of the script. Output is a .csv table with columns specified in the assignment description.
- **src**:
  - `col_tool.py`: script for the collocational analysis
- `setup.sh`: installs the necessary libraries to run the scripts

#### The script does the following:
- Takes search term, window size and file/directory specified by the user.
- It is possible to specify just one file or analyse a whole directory of files. The script will only process .txt, other formats will be ignored.
- For ease, the script automatically analyses all files inside the 'in' folder, even if it is just one.
- Cleans the text of any unwanted symbols.
- Extracts all words within the specified window from the search term.
- Calculates the MI score for each collocate.
- Saves result .csv file with columns: 'collocate' (the collocate term), 'freq_collocate' (how often it appears as a collocate of the searched term), 'freq_text' (how often it appears in the text), 'MI' (Mutual Information score)

The script runs with pre-set variables which can also be changed by the user. 

#### Required positional arguments:
- **Key:** (searched term)

#### Default values of optional arguments:
- **-Path:** `in/` (the script will analyse everything inside the 'in' folder by default)
- **-Window:** 3


## 4. Usage
#### How to correctly run the script:
1. Set your working directory to be the folder containing all the subfolders (in, out, src) using the `cd "path"` command.
2. Open the console and type: `bash setup.sh` (this should install the necessary packages)
3. Type `python` and follow it with the path to `col_tool.py` which should be `src/col_tool.py` if you set your working directory correctly.
4. Following the path to the `.py` script should be your parameters. For positional arguments, only value needs to be input. For optional arguments specify argument name, then value. It is not required to specify optional arguments as they have a default setting.
5. Example: 
   - `cd user/file/file/Assignment1_Collocation_tool` - set working directory
   - `bash setup.sh` - install libraries
   - `python ../src/col_tool.py green` - run script with the searched term (this will analyse all txt files inside 'in' folder)
   - `python ../src/col_tool.py green -window 2` - specified window argument
   - `python ../src/col_tool.py green -Path C:/user/file/file/Assignment1_Collocation_tool/in/yourfile.txt` - if you specifically want to analyse only this one file and not other files present in the folder (you can also input the path as this `in/yourfile.txt` if you set you working directory before)
   - `python ../src/col_tool.py green -Path C:/user/file/file/folder/` - if you wish to analyse all files in any other directory than 'in'
   - `python  ../src/col_tool.py -h` (or `--help`) - for help and explanations

## 5. Discussion of results
As a result, I obtained two CSV tables for the words green and purple. After analysing multiple texts I had a better understanding of what words often appear near the searched terms. Due to data being rather older books, some of the words with very high MI scores were slightly obscure to me. They often appeared in the whole corpus analysed only once and that is why their MI score was so high. If we look closer, however, we can observe some more intuitive pairs. For instance, green had a lot of collocates which had something to do with nature (as most plants are green) or the natural process of nature taking over, words like leafage or apricots. 
The word 'vowler' had the highest MI score but unfortunately, I can't even find what it means.

