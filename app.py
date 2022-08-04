from flask import Flask, request
import tensorflow as tf
print("imported tensorflow")
#load the model into RAM, so that it can be run quickly
model = tf.saved_model.load('models/Query-1')
with open("models/vocab.txt","r") as f:
    vocab = f.read()
vocab = vocab.split("\n")
print("loaded the model and vocabulary into RAM")


app = Flask(__name__)

@app.route("/")
def home():
    return "I'm going to make a pretty webpage, I just have not gotten around to it yet :)"

@app.route("/query", methods=["POST"])
def query():
    query = request.form['query']
    tokens = model(tf.constant(query))
    output_string = ""
    for token in tokens:
        word = vocab[token.numpy()-1]
        output_string += word + " "
    return output_string

@app.errorhandler(404)
def not_found(e):
    return "404, not found"

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
