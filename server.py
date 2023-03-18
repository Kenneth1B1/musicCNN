#imports----------------------
import os
from flask import Flask, render_template, send_file, send_from_directory, request, url_for, flash, redirect
from flask_cors import CORS
import json
import ML as ml
import predict

#initialize flask app---------
app = Flask(__name__)
CORS(app)

#page routing ----------------
@app.route("/")
def main():
    return render_template("Main.html")



@app.route("/home")
def home():
    return render_template('Main.html')

@app.route("/statistics")
def statistics():
    return render_template('Statistics.html')

@app.route("/cnn")
def cnn():
    return render_template('CNN.html')

@app.route('/result', methods=["GET"])
def result():

    # call the prediction function in ml.py
    #result = ml.prediction()
    #print(result)
    # make a dictionary from the result
    #resultDict = { "model": "kNN", "accuracy": result[0], "precision":result[2], "recall":result[1]}
    
    # convert dictionary to JSON string
    #resultString = json.dumps(resultDict)
    resultDict = { "Answer": ml.test()}
    print("results server.py")
    return resultDict

@app.route('/result_cnn', methods=["GET"])
def CNNresult():
    #file = open("F:/MusicWebsite/static/upload.wav")
    val = predict.predict("static/upload.wav")
    resultDict = { "Answer": val}
    #print("results server.py")
    return resultDict

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
        file = request.files['file']
        #print(file.filename)
        #dirName = "'static'" + str(file.filename);
        #dirName = '\\'.join(os.path.dirname(file).split("/"))
        #print(dirName)
        #file.save("C:Users/Kenne/Desktop/MusicWebsite/static/upload.wav")
        file.save("./static/upload.wav")
        #predict.plot_mfcc("static/jars.wav")
        #file.save(os.path.join(app.config['UPLOAD_DIR'], file.filename))
        return render_template('CNN.html')
   return 'no upload'


#run main ---------------------
if __name__ == "__main__":
    #app.run(debug=True, host="192.168.0.24", port=5000)
    app.run(port = 8000)
    
