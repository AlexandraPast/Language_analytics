import os
import sys
import argparse
# import libraries
import spacy
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 20000000
# for log
import math
import numpy
import re
import pandas as pd 
from tqdm import tqdm

#-------------------------------------------------------------------------------------#

# FUNCTIONS

# cleaning the text
def cleaning_text (text):
    text_cleaned = text.replace("(", " ")
    text_cleaned = text_cleaned.replace(")", " ")
    text_cleaned = text_cleaned.replace('"', " ")
    text_cleaned = text_cleaned.replace("'", " ")
    text_cleaned = text_cleaned.replace(":", " ")
    text_cleaned = text_cleaned.lower()
    # create nlp object
    doc1 = nlp(text_cleaned)
    # removing words shorter than 3
    doc = [ word for word in doc1 if len(word) > 3 ]
    
    return doc


def extract_surround_words(text2, keyword, n, lst):
   
    #extracting all the words from text
    words = words = re.findall(r'\w+', text2)
    
    #iterate through all the words
    for index, word in enumerate(words):

        #check if search keyword matches
        if word == keyword:
            #fetch left side words
            left_side_words = words[index-n : index]
            lst.append(left_side_words)
            
            #fetch right side words
            right_side_words = words[index+1 : index + n + 1]
            lst.append(right_side_words)
            #print(left_side_words, right_side_words)

            
# cleaning the list after changing it to string so I could separate the words
def cleaning_list (list):
    col_string = str(list)
    col_string = col_string.replace("[", "")
    col_string = col_string.replace("]", "")
    col_string = col_string.replace("'", "")
    return col_string


# frequency of word function
def count_word_frequency(word, doc):
    freq = 0    
    for tok in doc:
        if tok.text == word:
            freq = freq + 1
  
    return freq


# frequency of list of words function
def count_word_list_freq(list, doc):
    freq = 0
    res_lst = []
    
    for word in list:
        freq = count_word_frequency(word = word, doc = doc)
        res_lst.append(freq)
        freq = 0
    
    return res_lst


# calculate MI for each collocate (this worked fine and then before handing it in it stopped, I don´t undrstand what is wrong since it
# worked just fine before
def MI_calculate(doc, keyword, collocate_list, collocate_string, n, B_lst, AB_lst, MI_lst):
    A = count_word_frequency(word = keyword, doc = doc)
    B = count_word_list_freq(list = collocate_list, doc = doc)
    AB = count_word_list_freq(list = collocate_list, doc = nlp(collocate_string))
    corpus_size = len(doc)
    span = n*2+1
    MI = 0

    for i in range(len(B)):
        MI = math.log((AB[i] * corpus_size) / (A * B[i] * span)) / math.log(2)
        B_lst.append(B[i])
        AB_lst.append(AB[i])
        MI_lst.append(MI)
    
    #return B_lst, AB_lst, MI_lst

def deduplicate(a):
    source_ips = []
    new_list = []
    for i in range(len(a)):
        if a[i][0] != None:
            if a[i][0] not in source_ips:
                source_ips.append(a[i][0])
                new_list.append(a[i])
            else:
                print('"'+ a[i][0] + '" is already on the list')
    return new_list

#-------------------------------------------------------------------------------------#
# ARG, VAR PREPARATION

def main():
    #Firstly the user specified arguments
    #Create the parser
    my_parser = argparse.ArgumentParser(description = 'Collocation tool')
    
    #Add the arguments
    #Hyperparameters to specify
    
    
    my_parser.add_argument('Key',
                       metavar = 'Keyword',
                       type = str,
                       help = 'input keyword to be searched')

    my_parser.add_argument('-Path',
                           metavar = 'Path to the text',
                           type = str,
                           default = "in/",
                           help = 'input path to the folder or file if data is not inside "in" folder')
    
    my_parser.add_argument('-Window',
                           metavar = 'Window size',
                           type = int,
                           default = 3,
                           help = 'input window size of collocation search')
        
     #Execute parse_args()
    args = my_parser.parse_args()
    
    print('--------------------------------------------------------------------------')
    print('------Your input values:')
    #print values of arguments for the user to check
    vals = vars(args)
    for k, v in vals.items():
        print(f'{k:<4}: {v}')
     
    print('--------------------------------------------------------------------------')
        
    # directory
    text_path = args.Path

    # input word to be searched
    keyword = args.Key

    # input window size 
    n = args.Window
    
#-------------------------------------------------------------------------------------#
# EXECUTABLE

    # preparing empty lists
    lst = []
    B_lst = []
    AB_lst = []
    MI_lst = []
    multi_list = []
    
    # set up a condition to check for a file
    if os.path.isfile(text_path):
        f = text_path
        # extract just the filename
        filename = os.path.basename(text_path)
        # set another condition to check if the file is correct file type
        if f.endswith('.txt'):
            with open(text_path, "r", encoding="utf-8") as file:
                text = file.read()
                
        else:
            print(" ")
            print("!! WARNING: " + filename + " is not supported .txt file")
            print(" ")
            # exit the program because the one file is not supported format
            sys.exit()

            
    elif os.path.isdir(text_path):
        print('------Reading the files------')
        # set counter to 0
        counter = 0
        # list through files
        for filename in tqdm(os.listdir(text_path)):
            f = os.path.join(text_path, filename)
            #print(f) #use only to check if code keeps getting stuck
            # checking if it is a file
            if os.path.isfile(f):
                #check format
                if f.endswith('.txt'):
                    
                    # load data
                    with open(f, "r", encoding="utf-8") as file:
                        w_text = file.read()
                    # first text is put into text var
                    if counter == 0:
                        text = w_text
                    # append the following text with space in between
                    else:
                        text = text + " " + w_text

                    counter = counter + 1
                    
                else:
                    print(" ")
                    print("!! WARNING: " + filename + " is not supported .txt file")
                    print(" ")
                    
    else:
        print('--------------------------------------------------------------------------')
        print("!! WARNING: Something went wrong, check if the path is correct !!")
        print('--------------------------------------------------------------------------')
        # exit the program because path was input incorrectly
        sys.exit()

    
    print('------Cleaning text, Extracting collocates')
    # apply cleaning function
    doc = cleaning_text(text = text)

    # convert `an_object` to a string.
    doc_string = str(doc) 

    #print('--------------------------------------------------------------------------')
    #print('------Extracted collocates within window size:')
    
    # extracting collocate words
    extract_surround_words(text2 = doc_string, keyword = keyword, n = n, lst = lst)

    # string version of collocates
    collocate_string = cleaning_list(list = lst)

    # separate the words, creates a list of collocates
    collocate_list = collocate_string.split(", ")

    # apply MI function, create A, AB, MI lists
    MI_calculate(doc, keyword, collocate_list, collocate_string, n, B_lst, AB_lst, MI_lst)

    # create multi-d list so we can sort the list and remove duplicate words
    for i in range(len(collocate_list)):
        multi_list.append([collocate_list[i],
                           AB_lst[i],
                           B_lst[i],
                           MI_lst[i]])
    
    #print('--------------------------------------------------------------------------')
    print('------Removing duplicates')
    # apply deduplicating function
    final_list = deduplicate(a = multi_list)
    print('--------------------------------------------------------------------------')
    print("Total items in original list :", len(multi_list))
    print("Total items after deduplication :", len(final_list))
    print('--------------------------------------------------------------------------')
    
    # sorting by best MI score
    final_list = sorted(final_list, key=lambda x: x[3], reverse=True)
    
    print('------Saved results in csv')
    # save all output as a csv
    dataframe = pd.DataFrame(final_list, columns =['collocate',
                                               'freq_collocate',
                                               'freq_text',
                                                'MI']) 
    dataframe.to_csv('out/collocate_of_'+keyword+'.csv', index=False)
    
        
    print('------All done!------')
    
#-------------------------------------------------------------------------------------#
    
if __name__ == '__main__':
   main()