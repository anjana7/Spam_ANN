import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import datetime
import simplejson as json
import time


lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

features = []
output = []
words = []

#creating a bag of words
def make_dict(emails):
    size = len(emails)
    for m in range(size):
        clas = (emails["classes"][m])
        if (clas == "spam"):
            mail = emails["data"][m]
            tokens = nltk.word_tokenize(mail)
           
            for t in tokens:
                if t.isalpha() == False:
                    del t
                elif len(t) == 1:
                    del t
                else:
                    words.append(t)

    #remove stopwords
    filtered = [w for w in words if not w in stop_words]
       
    #lemmatize words
    dictn = [lemmatizer.lemmatize(f) for f in filtered]
       
    #find distinct words
    sets = set(dictn)
       
    #for w in sets:
    #    diction[w] = words.count(w)
   
    return(sets)
   
    
#defining feature wectors 
def trainset(emails, diction):
    size = len(emails)
   
    for m in range(size):
        clas = emails["classes"][m]
        mail = emails["data"][m]
        featurev = []
        classv = []
       
        tokens = nltk.word_tokenize(mail)
       
        #feature vector for each mail
        for t in diction:
            if t in tokens:
                featurev.append(1)
            else:
                featurev.append(0)
       
        if (clas == "spam"):
            classv = [1,0]
        else:
            classv = [0,1]
       
        features.append(featurev)
        output.append(classv)
   
    return(features, output)
           
 

#compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output


#convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)


#compute feature vector for test data
def inputv(email1, diction):
    featurev = []
    tokens = nltk.word_tokenize(email1)
    filtered = [w for w in tokens if not w in stop_words]
    words = [lemmatizer.lemmatize(f) for f in filtered]
    wordl=set(words)
    for t in diction:
        if t in wordl:
            featurev.append(1)
        else:
            featurev.append(0)
   
    # input layer is our bag of words
    l0 = featurev
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return(l2)
   
   
   
def training(features, output, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):
   
    np.random.seed(1)
   
    classes = list("spam","ham")
   
    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(features[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, 2)) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
   
   
   
   
    for j in iter(range(epochs+1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = features
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
               
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(features),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = output - layer_2

        if (j% 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
               
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
       
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
       
        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))       
       
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
       
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()
   
   
    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)
   
   

email = pd.read_csv('/home/anjana/MLprograms/testspam.csv', encoding='latin-1')
emails = email.loc[:, ~email.columns.str.contains('^Unnamed')]
   
diction = make_dict(emails)
features,output = trainset(emails, diction)
                                   
X = np.array(features)
y = np.array(output)

start_time = time.time()

training(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")


# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses.json'
with open(synapse_file) as data_file:
    synapse = json.load(data_file)
    synapse_0 = np.asarray(synapse['synapse0'])
    synapse_1 = np.asarray(synapse['synapse1'])

def classify(sentence, show_details=False):
    results = intputv(emailin, diction)

    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s \n classification: %s" % (sentence, return_results))
return return_results


