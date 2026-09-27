import flask
import pandas as pd
import numpy as np
from flask import Flask,render_template,request
import pickle

with open ("scaled_values.pkl","rb") as s:
    scl_obj=pickle.load(s)

with open ("trained_model.pkl","rb") as t:
    mod_obj=pickle.load(t)

app = Flask(__name__)

@app.route("/")
def main_page():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict():
    a = np.array([float(i) for i in request.form.values()])
    a=a.reshape(1,-1)
    vals=scl_obj.transform(a)


    res=mod_obj.predict(vals)



    return render_template('index.html', result=res)


if __name__ == "__main__":
    app.run(debug=True)