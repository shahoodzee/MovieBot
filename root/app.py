from flask import Flask, request, jsonify
import json
import pickle
import json
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"












def chat(message):
    with open("data.json") as file:
        data = json.load(file)
        
    # load trained model
    model = keras.models.load_model('cahtmodel.h5')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    while True:
        print("User: ")
        inp = message
        if inp.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                response = np.random.choice(i['responses'])
        return response
        
@app.route("/chatbot/talk", methods=['GET'])
def talk_with_param():
    message = request.args.get('msg')
    message_response = chat(message)
    if message:
        response = {
            "message": message_response,
        }
    else:
        response = {
            "error": "No message provided."
        }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)