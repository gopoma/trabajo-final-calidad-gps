import os

from flask import Flask, request, jsonify, send_from_directory
import uuid

import cv2
import matplotlib
import matplotlib.pyplot as plt

from models.chan_vese_segmentation import ChanVeseSegmentation



app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route('/', methods=['POST'])
def upload_image():
    # Check if the image is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # If no file is selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file with a unique name (to avoid conflicts)
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join("uploads", filename)

    # Make sure the uploads directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Make sure the static directory exists
    os.makedirs(os.path.join("static"), exist_ok=True)

    # Save the file to the server
    file.save(filepath)

    multiple = request.args.get("multiple")
    multiple = (multiple == "true")

    info = ChanVeseSegmentation(os.path.join("uploads", filename), multiple)
    image = cv2.imread(os.path.join("uploads", filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    matplotlib.use('Agg')  # Use a non-interactive backend

    results = []
    for i in range(len(info)):
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.contour(info[i], [0], colors="r", linewidth=2)
        plt.draw()
        plt.show(block=False)

        current_filename = f"{filename}-{i + 1}.jpg"
        plt.savefig(os.path.join("static", current_filename))
        results.append(current_filename)

        plt.cla()

    return jsonify({"images": results})

@app.route('/<path:path>')
def send_report(path):
    # Using request args for path will expose you to directory traversal attacks
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
