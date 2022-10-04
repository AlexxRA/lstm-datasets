# Load libraries
from distutils.log import log
import flask
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

# instantiate flask 
app = flask.Flask(__name__)

# declare tokenizer
tokenizer = None

def data_cleaning(raw_data):
    raw_data = raw_data.translate(str.maketrans('', '', string.punctuation + string.digits))
    words = raw_data.lower().split()
    stops = set(stopwords.words("english"))
    useful_words = [w for w in words if not w in stops]
    return( " ".join(useful_words))

# we need to redefine our metric function in order 
# to use it when loading the model 
def initTokenizer():
    print("::::STARTING TOKENIZER INIT:::::")
    max_features = 6000
    df_train = pd.read_csv("labeledTrainData.tsv.zip", header=0, delimiter="\t", quoting=3)
    df_train['review'] = df_train['review'].apply(data_cleaning)
    train_reviews = df_train["review"]
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_reviews))
    print("::::TOKENIZER INIT FINISHED:::::")
    return tokenizer


# load the model, and pass in the custom metric function
# global graph
# graph = tf.get_default_graph()
model = load_model('model.h5')

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        x=[params["msg"]]

        sequence = tokenizer.texts_to_sequences(x)
        sequence = pad_sequences(sequence, maxlen=370)

        data["prediction"] = str("positive" if model.predict(sequence)[0][0] > 0.55 else "negative")
        data["success"] = True

    # return a response in json format 
    return flask.jsonify(data)

# initialize tokenizer at startup
tokenizer = initTokenizer()

# start the flask app, allow remote connections
app.run(host='0.0.0.0')