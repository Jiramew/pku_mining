## Assignment2
Course webpage: http://www.icst.pku.edu.cn/lcwm/course/WebDataMining/

## Before Running
### Data Preparation 
Download the data from http://www.icst.pku.edu.cn/lcwm/course/WebDataMining/data/data_wdm_assignment_2.rar to get the assignment dataset,      
http://nlp.stanford.edu/data/glove.6B.zip for global word2vec embedding vector,      
https://github.com/stopwords-iso/stopwords-en/blob/master/stopwords-en.txt to get the English stopwords.      
Save all the data in a `data` folder parallel to `src`.
### Environment
Python3 with Keras and Tensorflow. (Better with GPU)

## Running
Two models are implemented, `Simple NN` and `GRU`.      
Just run `python train.py` in directory `src`.