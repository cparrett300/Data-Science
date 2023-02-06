


import numpy as np
from flask import Flask, request, jsonify
import pickle


app = Flask(__name__)
model = pickle.load(open('forest_model.sav', 'rb'))


@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    test = np.array([list(data.values())])
    prediction = model.predict(test)

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
