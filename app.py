from time import time
from flask import Flask,redirect, url_for, render_template,request,jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from model_build import model_predict
from labels import change_labels
import warnings
warnings.filterwarnings("ignore")
import json
from flask_cors import CORS

#load the MODEL 
tokenizer = AutoTokenizer.from_pretrained(".\model")
multi_model = TFAutoModelForSequenceClassification.from_pretrained(".\model",num_labels=5)
#WSGI Application
app= Flask(__name__) #Flask App Object
CORS(app)

#Decorator
@app.route('/') #Home Page
def welcome():
    return "Welcome"

@app.route('/model',methods=['POST', 'GET']) #Model Page
def pop_results():
    # data = request.json()
    # data= req.content
    if request.method == "POST":
        data = request.data
        data= json.loads(data)
        print(data)
   # data = {"inputs":[{"Question":"",
   #             "Employer":"Manager",
   #             "Response":"Good"},
   #             {"Question":" ",
   #             "Employer": "Peers",
   #             "Response":"Behaviour is very furious"},
   #             {"Question":"",
   #             "Employer": "Self",
   #             "Response":"Very Rude to colleagues"}
   #             ]}
    l=[]
    for input in data["inputs"]:
        l.append(input["Response"])
    pred = model_predict(multi_model,tokenizer,lis=l)
    pred = change_labels(pred)
    for i in range(0,len(l)):
        data["inputs"][i]["Polarity"] = pred[i]
    # print(data)

    return data

@app.route('/postData', methods=['POST', 'GET'])
def postData():
    req = request.get("https://b0a7-49-249-8-210.in.ngrok.io/")
    content = req.content
    if request.method == "POST":
        l=[]                
        for input in content["inputs"]:
            l.append(input["Response"])
        pred = model_predict(multi_model,tokenizer,lis=l)
        pred = change_labels(pred)
        for i in range(0,len(l)):
            content["inputs"][i]["Polarity"] = pred[i]

    return content

@app.route('/test',methods=['POST', 'GET'])
def testdata():
    data = {"inputs":[{"Question":"",
               "Employer":"Manager",
               "Response":"Good"},
               {"Question":" ",
               "Employer": "Peers",
               "Response":"Behaviour is very furious"},
               {"Question":"",
               "Employer": "Self",
               "Response":"Very Rude to colleagues"}
               ]}
    l=[]
    for input in data["inputs"]:
        l.append(input["Response"])
    pred = model_predict(multi_model,tokenizer,lis=l)
    pred = change_labels(pred)
    for i in range(0,len(l)):
        data["inputs"][i]["Polarity"] = pred[i]
    # print(data)

    return data


if __name__ == '__main__':
    app.run(debug = True)
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)





# @app.route('/login', methods = ["POST"]) #login Page
# def login():
#     username = request.form("fname")
#     passoword = request.form("password")
#     return "Welcome %s" %username

