from flask import Flask, request
from modelo import iris
import numpy as np
import json


app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Ola, mundo!</p>"

@app.route("/bye")
def bye():
    return "<p>Tchau</p>"

@app.route("/predict", methods=['POST'])
def predcit():
  event = json.loads(request.data)
  values = event['values']
  values = list(map(np.float,values))
  print(values)
  pred = iris(values[0], values[1], values[2], values[3])
  return f"A sua flor é {pred}!\n"

if __name__ == '__main__':
  app.run(debug=True)