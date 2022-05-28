#!/usr/bin/env bash

pip install --upgrade pip
pip install spacy tqdm spacytextblob vaderSentiment networkx scikit-learn tensorflow pandas
python -m spacy download en_core_web_sm