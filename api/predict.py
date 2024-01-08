import pickle
import numpy as np
from flask import Flask, request, jsonify

input_file = "hospitalization-logistic_model_test.bin"

with open(input_file, "rb") as f_in:
    transformers, model = pickle.load(f_in)


def transform_data(dict_data, transformers):
    dv = transformers["dv"]
    imputer = transformers["imputer"]
    scaler = transformers["scaler"]

    X = dv.transform(dict_data)
    X = imputer.transform(X)
    X = scaler.transform(X)

    return X


def predict_single_patient(patient, transformers):
    
    X_patient = transform_data([patient], transformers)
    y_pred = model.predict_proba(X_patient)[0, 1]
    hospitalization = y_pred >= 0.5

    return y_pred, hospitalization


def revive_nan(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = revive_nan(value)
    elif isinstance(data, list):
        for i, value in enumerate(data):
            data[i] = revive_nan(value)
    elif data == "":
        return np.nan
    else:
        return data


app = Flask("hospitalization")


@app.route("/predict", methods=["POST"])
def predict():
    patient_json = request.get_json()

    print(patient_json)

    patient = dict(patient_json)

    revive_nan(patient)
    y_pred, hospitalization = predict_single_patient(patient, transformers)

    # X = dv.transform([patient])
    # y_pred = model.predict_proba(X)[0, 1]
    # hospitalization = y_pred >= 0.5

    result = {
        "hospitalization_probability": float(y_pred),
        "hospitalization": bool(hospitalization),
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
