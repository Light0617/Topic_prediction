# Topic_prediction
This is a topic prediction task

## Library install
- nltk
- gensim

## Model Using
- LDA model
## Model description
- I use LDA model to generate topic and predict new topic_conversation.
- First, I preprocessing the sentence in conversation and put all sentence together to generate a list of token per conversation.
- Filter all tokens which frequency is only one and build a dictionary with that and save them.
- Using LDA model to get these topics and get the top topics which sort by its popularity.
- For each topic, we have tokens, first filter the tokens in Brown and others would be keywords and I found the top keywords are in all topics.
Hence, I try to apply TF_IDF to solve the issue to find which keywords are important in each topic.
- For each topic, we use the keyword which has the highest TF_IDF score to be the topic_name and create topic_map to map ID to name.

## First Task
- Once having topic_map and topic_rank, we just output top 10 topics.

## Second task
- For a new conversation, we also do preprocessing to get cleaned conversation.
- With cleaned conversation, we feed into LDA model to find the topic ID and use topic_map to find its name.




## How to run and example
### Version
- Python3 (must python3 instead of python2)
## Run
- train mode with 1000 topic
`python train_model.py 1000`
- run output top 10 topic
`python output_top_topic.py 10`
- predict conversation with file_path
`python predict_new_conversation.py test.tsv`
## If rerun with new model
- clear all data previous
`sh clear.sh`
