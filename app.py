from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os



app = Flask(__name__)


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():

    if request.method == 'POST':
        scalar = StandardScaler()
        scalerfile = 'scalar.sav'
        scalar = pickle.load(open(scalerfile, 'rb'))

        model = pickle.load(open('finalized_model.sav', 'rb'))


        alpha = float(request.form['alpha'])
        delta = float(request.form['delta'])
        u = float(request.form['u'])
        g = float(request.form['g'])
        r = float(request.form['r'])
        i = float(request.form['i'])
        z = float(request.form['z'])
        red = float(request.form['red'])


        data = [[alpha, delta, u, g, r, i, z, red]]
        data = scalar.transform(data)
        if np.round(model.predict(data)) == 0:
            return render_template('index.html',prediction_texts = "It is a Galaxy")
        elif np.round(model.predict(data)) == 1:
            return render_template('index.html',prediction_texts = "It is a QSO")
        elif np.round(model.predict(data)) == 2:
            return render_template('index.html',prediction_texts = "It is a Star")
    else:
        return render_template('index.html')



if __name__=="__main__":
    app.run(debug=True)

