from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import string
import codecs
import collections
import random
import glob
import json
import pickle

allWords = []
xInput = []
yInput = []
table = str.maketrans('', '', string.punctuation)
vocabulary_size = 25000


def readFile(fileName, allWords):


    file = codecs.open(fileName, encoding='utf-8')
    
    for line in file:
        #line = line.lower().encode('utf-8')	
        words = line.split()		
        for word in words:
            word = word.translate(table)
            if word != '':
                allWords.append(word)

    file.close()


def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  word2vec = dict()
  for word, _ in count:
    word2vec[word] = len(word2vec)
  data = list()
  unk_count = 0
  for word in words:
    if word in word2vec:
      index = word2vec[word]
    else:
      index = 0  # word2vec['UNK']
      unk_count = unk_count + 1
    data.append(index)
  count[0][1] = unk_count
  return word2vec



fileList = glob.glob("aclImdb/train/neg/*.txt")
for file in fileList:
    readFile(file, allWords)

fileList = glob.glob("aclImdb/train/pos/*.txt")
for file in fileList:
    readFile(file, allWords)

readFile('custom_dataset.csv', allWords)

print("No of Words: ", len(allWords))

word2vec = build_dataset(allWords)
del allWords  # Hint to reduce memory.

print("Words in word2vec: ",len(word2vec))

with open('word2vec.json', 'w') as fp:
	json.dump(word2vec,fp)


def readFileToConvertWordsToIntegers(word2vec, fileName, xInput, yInput, label):

    file = codecs.open(fileName, encoding='utf-8')
    comment = []
    for line in file:
        #line = line.lower().encode('utf-8')
        words = line.split()
        for word in words:
            word = word.translate(table)
            if word in word2vec:
                index = word2vec[word]
            else:
                index = 0  # word2vec['UNK'] 
            comment.append(index)
        xInput.append(comment)
        yInput.append(label)

    file.close()



	
fileList = glob.glob("aclImdb/train/neg/*.txt")
for file in fileList:
    readFileToConvertWordsToIntegers(word2vec, file, xInput, yInput, 0)

fileList = glob.glob("aclImdb/train/pos/*.txt")
for file in fileList:
    readFileToConvertWordsToIntegers(word2vec, file, xInput, yInput, 1)

x_list =[]
y_list = []
files = []
	
files.append(codecs.open('custom_dataset.csv'))
#files.append(codecs.open('positive.csv'))
for file in files:
	for line in file:
		sentence = line.strip().split('^')
		x = sentence[0]
		y = sentence[1]
		words = x.split()
		temp = []
		flag = False
		for w in words:		
			if w in word2vec:				
				temp.append(word2vec[w])
				if word2vec[w]<vocabulary_size:
					flag= True
		if flag:
			x_list.append(temp)
			y_list.append(y)
	file.close()

for x in x_list:
	xInput.append(x)
for y in y_list:
	yInput.append(y)
	
print("X: ",len(xInput))
print("Y: ",len(yInput))	
	
c = list(zip(xInput, yInput)) 

pickle.dump( c, open( "sentiment_dataset.pkl", "wb" ) )



