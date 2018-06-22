# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:58:44 2018

@author: bharath.parameshwara
"""

import tflearn 
import json
import codecs
from tflearn.data_utils import to_categorical, pad_sequences 
from klein import Klein
import predict
import train

app = Klein()

@app.route('/api/ai/getSentimentPrediction/<comment>', methods=['GET'])
def prediction(request,comment):
#    comment = request.args.get('comment', )[0]
    prediction = predict.prediction_module(comment)
    return prediction

@app.route('/api/ai/trainAIEngine', methods=['GET'])
def training(request):
    training = train.training_module()
    return training

@app.route('/api/ai/getSentimentSummary', methods=['GET'])
def summary(request):
    summary = predict.prediction_summary()
    return summary

app.run("localhost", 9080)