from flask import Flask,render_template,url_for,request
from prediction_pipeline import NLP_Pipeline as Model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		file = request.files['myfile']
		filename = secure_filename(file.filename)
		summary = Model().run(filename=os.path.join(os.getcwd(),"parsed_docs_testing",filename))
		#"test-data-master\\constituency_parsing\\0"
	# else:
	# 	summary = Model().run(context=context)
	return render_template('result_summary.html', prediction=summary)

if __name__ == '__main__':
	app.run(debug=True)