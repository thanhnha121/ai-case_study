from flask import Flask
from image_processing import kmeans
from flask import json
from flask import send_file
from flask import request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/image_processing/resize")
def resize():
    data = kmeans.run()
    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/images')
def get_image():
    name = request.args.get('name')
    if name is None:
        name = 'http://localhost:5000/images?name=home.jpg'

    img_ext = name.split('.')[1]

    return send_file('image_processing/images/{}'.format(name), mimetype='image/' + img_ext)


if __name__ == "__main__":
    app.run()
