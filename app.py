from flask import Flask,render_template,url_for,request
import pandas as pd 
from tensorflow.keras.layers import Dense,LSTM,Embedding,Bidirectional,Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd 
import numpy as np
from config import *
from tensorflow.keras import Sequential
import pickle
import joblib
from tensorflow.keras.models import model_from_json

# load the model from disk
filename = r'transform.pkl'
cv = pickle.load(open(filename, 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():


	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = (cv.texts_to_sequences(data))
		vect = np.array(pad_sequences(vect,maxlen=30, dtype='int32', value=0))

		json_file = open('model.json', 'r')
		clf = json_file.read()
		json_file.close()
		clf = model_from_json(clf)
		clf.load_weights("model.h5")
		print("Loaded model from disk")
		clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


		my_prediction = clf.predict(vect)[0]
		if my_prediction<0.5:
			my_prediction = 1
		else:
			my_prediction = 0
		
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)