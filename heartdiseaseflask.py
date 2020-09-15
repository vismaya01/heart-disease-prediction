from flask import Flask,render_template,jsonify,request
import numpy as np
import pandas as pd
import sklearn
import json
import pickle as p
import requests
app=Flask(__name__)
@app.route('/')
def index():
    return render_template("heartdiseasemainpage.html")

@app.route("/heartdiseaseprediction", methods=['POST'])
def predictheartdisease():
    print(model)
    data=request.get_json()
    prediction=np.array2string(model.predict(data))
    return jsonify(prediction)
@app.route('/heartdiseasecondition',methods=['POST'])
def heartdiseasecondition():
    url="http://localhost:5000/heartdiseaseprediction"
    Age = request.form['Age']
    Sex = request.form['Sex']
    CP = request.form['CP']
    RBP = request.form['RBP']
    SC = request.form['SC']
    FBP = request.form['FBP']
    RER= request.form['RER']
    DPF = request.form['DPF']
    exang = request.form['exang']
    OP = request.form['OP']
    slope = request.form['slope']
    CA = request.form['CA']
    thal = request.form['thal']
    data=[[Age,Sex,CP,RBP,SC,FBP,RER,DPF,exang,OP,slope,CA,thal]]
    j_data=json.dumps(data)
    headers={'content-type':'application/json','Accept-Charset':'UTF-8'}
    r=requests.post(url,data=j_data,headers=headers)
    r1=list(r.text)
   
    stat=""
    if r1[2]=='0':
        stat="patient is not affected with Heart disease" 
    else:
        stat="patient affected with Heart disease"
    return render_template("result.html",result=stat)

if __name__=='__main__':
    model_file='final_heartdisease_model.pickle'
    model=p.load(open(model_file,'rb'))
    app.run(debug=True,host='0.0.0.0')

