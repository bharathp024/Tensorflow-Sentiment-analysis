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
import tensorflow as tf

allWords = []
xInput = []
yInput = []
table = str.maketrans('', '', string.punctuation)

vocabulary_size = 25000
model_path = 'models/'
model_name = 'model_demo'
checkpoint_path ='checkpoints/'
model_save_path = model_path + model_name + '.tfl'
checkpoint_save_path = checkpoint_path + model_name + '/'

def training_module():
    with open('word2vec.json', 'r') as fp:
        word2vec = json.load(fp)
        fp.close()
    
    print('Loading the dataset from pickle file')
    c = pickle.load( open( "sentiment_dataset.pkl", "rb" ) )
    
    #random.shuffle(c)
    
    xValues, yValues = zip(*c)
    del c           #To save memory
    
    x_list =[]
    y_list = []
    files = []
    
    print('Reading new comments from csv')
    files.append(codecs.open('input.csv'))
    
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
    
    xInput.extend(x_list)
    xInput.extend(xValues)
    yInput.extend(y_list)
    yInput.extend(yValues)
    
    
    trainX = xInput
    testX = x_list		#to evaluate the model based of new input
    
    trainY = yInput
    testY = y_list		#to evaluate the model based of new input
    
    
    
    trainX = pad_sequences(trainX, maxlen=100, value=0.)
    testX = pad_sequences(testX, maxlen=100, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)
    
    tf.reset_default_graph() 
    tflearn.config.init_training_mode()
    
    # Network building
    print('Building Neural Network')
    net = tflearn.input_data([None, 100])
    net = tflearn.embedding(net, input_dim=vocabulary_size, output_dim=100)
    net = tflearn.lstm(net, 100, dropout=0.5)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.01,
                             loss='categorical_crossentropy')
     
    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0, best_checkpoint_path=checkpoint_save_path )
    
    #model.fit(trainX, trainY, validation_set=(testX, testY), n_epoch=10,snapshot_epoch=True, show_metric=True,batch_size=200)
    print('Start training...')
    model.fit(trainX, trainY, validation_set=0.1, n_epoch=1,snapshot_epoch=True,shuffle= True, show_metric=True,batch_size=200)
    print('Training completed ')
    model.save(model_save_path)
    print('Saving model into %s' %model_save_path)
#    c = list(zip(trainX, trainY)) 
#    pickle.dump( c, open( "sentiment_dataset.pkl", "wb" ) )
    
    return 'Training Complete'