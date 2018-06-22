import tflearn 
import json
import codecs
from tflearn.data_utils import to_categorical, pad_sequences 
from klein import Klein
import pymysql.cursors
import pymysql


app = Klein()

model_path = 'models/'
model_name = 'model_new_2'
model_load_path = model_path + model_name + '.tfl'

vocabulary_size = 25000

print('Building Neural Network')
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=vocabulary_size, output_dim=150)
net = tflearn.lstm(net, 150, dropout=0.5)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)

print('Loading model : %s' %model_name)
model.load('model_new_2.tfl')

with open('data_25000.json', 'r') as fp:
    word2vec = json.load(fp)
    fp.close()
	
#word_index = imdb.get_word_index("imdb_word_index.json")
#@app.route('/api/ai/getSentimentPrediction/<comment>')
def prediction_module(comment):    

    x_list =[]
    
    print('Converting input to its vector form.')
    words = comment.split()
    temp = []
    flag = False
    for w in words:		
    	if w in word2vec:				
    		temp.append(word2vec[w])
    		if word2vec[w]<vocabulary_size:
    			flag= True
    if flag:
    	x_list.append(temp)
    
    
    x_list = pad_sequences(x_list, maxlen=100, value=0.)
    
    #trainY = to_categorical(trainY, nb_classes=2)
    '''
    model.fit(trainX, trainY, validation_set=(trainX, trainY), n_epoch=10, show_metric=True, batch_size=150)
    model.save("model_test_1_2_test.tfl")
    '''
    print('Predicting the sentiment for the input...')
    pred= model.predict(x_list)
    print('Predected sentiment is : %.4f' %pred[0][1])
    #print("Prediction: %.4f" %pred[0][1],':::::', comment)
    sentiment = 'undefined'

        

    if(pred[0][1]>=0.7):
        sentiment = 'positive'
    elif(pred[0][1]>=0.3) and (pred[0][1]<0.7):
        sentiment = 'neutral'
    else:
        sentiment = 'negative'
    return "Prediction for the comment '%s' is : %s \n" %(comment, sentiment,)

def prediction_summary():
    
    comments_list = []
    # Connect to the database
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='1234',
                                 db='playlabs_dev_db',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
    
    try:
        with connection.cursor() as cursor:
            # Read a single record
            sql = "SELECT feedback,predictedSentiment,manualSentiment FROM playlabs_dev_db.module_feedbacks where feedback is NOT NULL"
            cursor.execute(sql)
#            result = cursor.fetchone()
            for row in cursor:  
#                print(row)               
                comments_list.append(int(row['predictedSentiment']))
            
    finally:
        connection.close()
    
    total=0
    pos=0
    neg=0
    for comment in comments_list:
        if(comment==1):
            pos = pos +1
        elif(comment==0):
            neg=neg+1
        total = total + comment
    
    length = len(comments_list)
    average = (total/length )
    
    response ={}
    response['Overall Sentiment'] = '' + str(round((average * 100 ),2))+ '%'
    response['Total Positive'] = pos
    response['Total Negative'] = neg

#    response= json.dumps(response)
    return json.dumps(response)  
        
#app.run("localhost", 9080)