import pickle
from flask import Flask, request, jsonify

input_file = "hospitalization-logistic_model_test.bin"

with open(input_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask("hospitalization")

@app.route("/predict", methods=["POST"])
def predict():
    patient = request.get_json()

    print(patient)

    # X = dv.transform([patient])
    # y_pred = model.predict_proba(X)[0, 1]
    # hospitalization = y_pred >= 0.5

    result = {
        "hospitalization_probability" : float(y_pred),
        "hospitalization" : bool(hospitalization)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
