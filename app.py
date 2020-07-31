  
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import joblib
import pickle
import tensorflow as tf 
from keras.layers import Dense,LSTM,Embedding,Bidirectional,Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd 
import numpy as np
from config import *
from keras import Sequential
from keras.models import model_from_json
import pickle
import joblib


filename = r'transform.pkl'
cv = pickle.load(open(filename, 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
	return render_template(r'home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = (cv.texts_to_sequences(data))
        vect = np.array(pad_sequences(vect,maxlen=30, dtype='int32', value=0))
        clf=pickle.load(open(r'nlp_model.pkl','rb'))
        my_prediction = clf.predict(vect,batch_size=1,verbose = 2)[0]
        if my_prediction<0.5:
            my_prediction = 1
        else:
            my_prediction = 0
    return render_template(r'result.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
