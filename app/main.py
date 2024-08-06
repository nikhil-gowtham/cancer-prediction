import pickle

import flask
import numpy as np

app = flask.Flask(__name__)

# load model, scaler and label encoder
with open("dependencies/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("dependencies/label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

with open("dependencies/standard_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)


@app.route("/")
def main_page():
    return flask.render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_single():
    """
    This function preprocesses the input and predicts the output
    """
    # read input
    input_data = flask.request.form.to_dict()["input_data"]
    input_data = input_data.replace(" ", "").split(",")

    try:
        input_data = np.array(input_data).reshape(1, 30)
        input_data = input_data.astype(np.float_)
    except Exception as ex:
        print("Exception occurred while pre-processing data: ", ex)
        return flask.render_template("invalid_data.html")

    # apply standard scaler
    scaled_features = scaler.transform(input_data)

    # make prediction
    prediction = model.predict(scaled_features)

    # decode prediction label
    prediction_label = label_encoder.inverse_transform(prediction)

    # respose as per prediction
    if prediction_label[0] == "B":
        return flask.render_template("diagnosis_b.html")
    else:
        return flask.render_template("diagnosis_m.html")
