# System tools
import os
import sys
import argparse

# Data analysis
import pandas as pd
from collections import Counter
from itertools import combinations 
from tqdm import tqdm

# NLP
import spacy
nlp = spacy.load("en_core_web_sm")

# Network analysis tools
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)

#-------------------------------------------------------------------------------------#
#FUNCTIONS

def networkx_plot (data, filename, pl_choice):
    G = nx.from_pandas_edgelist(data, "Source", "Target", ["Weight"])
    
    if pl_choice == 1:
        nx.draw_networkx(G, with_labels=True, node_size=20, font_size=10)
    elif pl_choice == 2:
        nx.draw_circular(G, with_labels=True, node_size=20, font_size=10)
    elif pl_choice == 3:
        nx.draw_kamada_kawai(G, with_labels=True, node_size=20, font_size=10)
    elif pl_choice == 4:
        nx.draw_random(G, with_labels=True, node_size=20, font_size=10)
    elif pl_choice == 5:
        nx.draw_spectral(G, with_labels=True, node_size=20, font_size=10)
    elif pl_choice == 6:
        nx.draw_spring(G, with_labels=True, node_size=20, font_size=10)    
        
    name = os.path.splitext(filename)[0]
    outpath_viz = os.path.join('out', 'plots' ,'network_'+name+'.png')
    plt.savefig(outpath_viz, dpi=300, bbox_inches="tight") # save plot
    return G

def networkx_stat (G, filename):
    # Compute node degree
    dg = nx.degree(G)
    ev = nx.eigenvector_centrality(G)
    eigenvector_df = pd.DataFrame(ev.items())
    bc = nx.betweenness_centrality(G)
    betweenness_df = pd.DataFrame(bc.items())
    
    #save all output as a csv
    dictionary = {'name': [item[0] for item in dg], 'degree': [item[1] for item in dg], 'betweenness_centrality': betweenness_df.iloc[:, 1], 'eigenvector_centrality': eigenvector_df.iloc[:, 1] }  
    dataframe = pd.DataFrame(dictionary) 
    name = os.path.splitext(filename)[0]
    dataframe.to_csv('out/tables/nodes_'+name+'.csv', index=False)

#-------------------------------------------------------------------------------------#
# ARG, VAR PREPARATION

def main():
    # Firstly the user specified arguments
    # Create the parser
    my_parser = argparse.ArgumentParser(description = 'Network analysis')
    
    # Add the arguments
    # Hyperparameters to specify
    my_parser.add_argument('-Path',
                           metavar = 'Path to the folder or file',
                           type = str,
                           default = "in/",
                           help = 'input path to the folder or file if it is not inside "in" folder eg. "C:/User/folder/folder/"')
    
    my_parser.add_argument('-Plot',
                           metavar = 'Different plotting algorithms',
                           type = int,
                           default = 1,
                           help = 'input your prefered plotting method (1, 2, 3, 4, 5, 6)')
            
    # Execute parse_args()
    args = my_parser.parse_args()
    
    print('--------------------------------------------------------------------------')
    print('------Your input values:')
    # print values of arguments for the user to check
    vals = vars(args)
    for k, v in vals.items():
        print(f'{k:<4}: {v}')
     
    print('--------------------------------------------------------------------------')
    
    # pass the value into a variable
    user_input = args.Path
    
#-------------------------------------------------------------------------------------#
# EXECUTABLE
    
    # set up a condition to check for a file
    if os.path.isfile(user_input):
        f = user_input
        # extract just the filename
        filename = os.path.basename(user_input)
        # set another condition to check if the file is correct file type
        if f.endswith('.csv'):
            print('------Analysing the file------')
            # read in data
            data = pd.read_csv(f,sep='\t')
            # perform network analysis
            G = networkx_plot(data = data, filename = filename, pl_choice = args.Plot)
            networkx_stat(G = G, filename = filename)
            print('------Saved plot and csv')
        else:
            print(filename + " is not supported .csv file")

   
    # set up a condition to check if the input is a directory
    elif os.path.isdir(user_input):
        print('------Analysing the files------')
        # list through all the files
        for filename in tqdm(os.listdir(user_input)):
            f = os.path.join(user_input, filename)
            #print(f) #use only to check if code keeps getting stuck
            # check if current item is a file
            if os.path.isfile(f):
                # check if the file is correct file type
                if f.endswith('.csv'):
                    # read in data
                    data = pd.read_csv(f,sep='\t')
                    # perform network analysis
                    G = networkx_plot(data = data, filename = filename, pl_choice = args.Plot)
                    networkx_stat(G = G, filename = filename)                    
                else:
                    print(filename + " is not supported .csv file")
                    
        print('------Saved all plots and csvs')

    else:
        print("!! WARNING: No files were analysed, check if the path is correct !!")
 
    print('------All done!------')
    

#-------------------------------------------------------------------------------------#


if __name__ == '__main__':
   main()