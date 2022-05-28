# Assignment 3 - Network analysis

The files here are part of the repository for my course in Language analytics.

## 1. Contribution
I worked on this code alone. 

## 2. Assignment description
The aim of this assignment was to create a script which would work on any data as long as it has the correct format. Specifically, this script should be used for network analysis and be able to work with any undirected, weighted edgelist with correct column names. 

#### The specific tasks were:
- If the user enters a single filename as an argument on the command line:
  - Load that edgelist
  - Perform network analysis using networkx
  - Save a simple visualisation
  - Save a CSV which shows the following for every node: name; degree; betweenness centrality; eigenvector_centrality
- If the user enters a directory name as an argument on the command line:
  - Do all of the above steps for every edgelist in the directory
  - Save a separate visualisation and CSV for each file

## 3. Methods
For this task, I wrote a script `network_analysis.py` which can process one file or multiple files if the directory is input. I used the function networkx_plot to create a plot of network enabling us to describe relations between different objects. Additionally, I used networkx to compute the required results and save them into a csv table.

## Scripts
#### This repository consists of:
- **in**: a folder containing the input for the code or data. We used `network_data` provided by the professor.
- **out**: a folder containing:
  - **plots**: folder for network plots
  - **tables**: folder for the csv with results
- **src**:
  - `network_analysis.py`: script for the network analysis
- `setup.sh`: installs the necessary libraries to run the scripts

#### The script does the following:
- Takes path as an input and decides if it is a file or a folder (no input is needed if data is inside 'in' folder)
- Takes number from 1-6 representing different plotting algorithms: (This step is not necessary)
  - nx.draw_networkx (1)
  - nx.draw_circular (2)
  - nx.draw_kamada_kawai (3)
  - nx.draw_random (4)
  - nx.draw_spectral (5)
  - nx.draw_spring (6)
- Checks if file type is correct
- Creates network plot
- Creates csv with columns: 'name', 'degree', 'betweenness_centrality', 'eigenvector_centrality'

The script runs with pre-set variables which can be changed by the user. 

#### Default values of optional arguments:
- **-Path:** `in/` (the script will analyse everything inside the 'in' folder by default)
- **-Plot:** 1

## 4. Usage
#### How to correctly run the script:
1. Set your working directory to be the folder containing all the subfolders (in, out, src) using the `cd "path"` command.
2. Open the console and type: `bash setup.sh` (this should install the necessary packages)
3. Type `python` and follow it with the path to `network_analysis.py` which should be `src/network_analysis.py` if you set your working directory correctly.
4. Following the path to the `.py` script should be your parameters. For positional arguments, only value needs to be input. For optional arguments specify argument name, then value. It is not required to specify optional arguments as they have a default setting. This script only contains optional arguments.
5. Example: 
   - `cd C:/user/file/file/Assignment3_Network_Analysis` - set working directory
   - `bash setup.sh` - install libraries
   - `python ../src/network_analysis.py` - run script (this will analyse all txt files inside the 'in' folder)
   - `python ../src/network_analysis.py -Path C:/user/file/file/folder/` - if you wish to analyse all files in any other directory than 'in'
   - `python ../src/network_analysis.py -Path C:/user/file/file/Assignment3_Network_Analysis/in/yourfile.csv` - if you wish to analyse one file (if you set the working directory above you can instead write `in/yourfile.csv`)
   - `python ../src/network_analysis.py -Path C:/user/file/file/Assignment3_Network_Analysis/in/yourfile.csv` -Plot 3` - to set different plot option as well
   - `python  ../src/network_analysis.py -h` (or `--help`) - for help and explanations

## 5. Discussion of results
I managed to run the script on all files in the directory and saved the output in the appropriate directories. I don't have much to say about the specific results as I understand, this task was more about the coding ability to write this script and all output looks correct.
