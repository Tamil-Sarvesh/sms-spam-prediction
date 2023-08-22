import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__) #Initialize the flask App

#forest = pickle.load(open('boosting.pkl','rb'))
filename = 'spam-sms-mnb-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))

@app.route('/')

@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/analysis')
def analysis():
	return render_template('analysis.html')

@app.route('/chart')
def chart():
	return render_template('chart.html')

#@app.route('/future')
#def future():
#	return render_template('future.html')    

@app.route('/login')
def login():
	return render_template('login.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        #df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)	

#@app.route('/home')
#def home():
 #   return render_template('home.html')

@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    return render_template('prediction.html', prediction_text="Please fill the details and click on predict")

#@app.route('/upload')
#def upload_file():
#   return render_template('BatchPredict.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv.transform(data).toarray()
    	my_prediction = classifier.predict(vect)
    	return render_template('analysis.html', prediction=my_prediction)
#@app.route('/performance')
#def performance():
	# return render_template('performance.html')   
    
if __name__ == "__main__":
    app.run(debug=True)
