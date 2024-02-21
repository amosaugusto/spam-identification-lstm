from flask import Flask, render_template, request
from helper import Helper

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    return render_template('index.html', result='')

@app.route("/preprocess", methods=["POST", "GET"])
def preprocess():
    preprocessing = "-"
    if request.method == "POST":
        file = request.files['input-file']
        helper = Helper()
        preprocessing = helper.dataset(file)
        return render_template("preprocess.html", preprocessing=preprocessing)
    else:
        return render_template("preprocess.html", preprocessing=preprocessing)

@app.route('/predict', methods=["POST", "GET"])
def predict():
    prediction = "-"
    if request.method == "POST":
        comment = request.form['comment']
        helper = Helper()
        preprocess = helper.preprocessing(comment)
        ubahInput = helper.change_input(preprocess)
        hasil = helper.sequencepred(ubahInput)
        prediction = helper.model_predict(hasil)
        return render_template("prediksi.html", prediction = prediction)
    else:
        return render_template("prediksi.html", prediction = prediction)

@app.route('/train', methods=["POST", "GET"])
def kinerja():
    accuracy = "-"
    precision = "-"
    recall = "-"
    f1 = "-"
    if request.method == "POST":
        file = request.files['input-file']
        helper = Helper()
        X_preprocessed, y = helper.dataset(file)
        X_train, X_test, y_train, y_test = helper.split_data(X_preprocessed, y)
        X_train, X_test, max_length  = helper.sequence(X_preprocessed, X_train, X_test)
        model = helper.train_lstm_model(X_train, y_train, max_length)
        accuracy, precision, recall, f1 = helper.print_kinerja(model, X_test, y_test)
        return render_template("train.html", accuracy = accuracy, precision= precision, recall = recall, f1 = f1, url='static/img/confussion_matrix.png')
    else:
        return render_template("train.html", accuracy = accuracy, precision= precision, recall = recall, f1 = f1 , url='static/img/confussion_matrix.png')



if(__name__) == '__main__':
    app.run(debug=True, host="localhost", port=8000)
