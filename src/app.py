import os

from flask import Flask, request, jsonify
import uuid

import cv2
import matplotlib
import matplotlib.pyplot as plt

from models.chan_vese_segmentation import ChanVeseSegmentation



app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

# Endpoint to receive the image and return the image name
@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if the image is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # If no file is selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file with a unique name (to avoid conflicts)
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join("uploads", filename)

    # Make sure the uploads directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the file to the server
    file.save(file_path)

#! work
    print(f"filename:{filename}")
    info = ChanVeseSegmentation(os.path.join("uploads", filename))
    image = cv2.imread(os.path.join("uploads", filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    matplotlib.use('Agg')  # Use a non-interactive backend
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.contour(info[-1], [0], colors="r", linewidth=2)
    plt.draw()
    plt.show(block=False)
    #!plt.pause(0.5)
    plt.savefig(os.path.join("archive", f"{filename}.jpg"))

#! work

    # Return the file name as JSON
    return jsonify({"image_name": filename})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
