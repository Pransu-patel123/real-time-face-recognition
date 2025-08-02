import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, Response
from insightface.app import FaceAnalysis

# Initialize app
app = Flask(__name__)

# Load face recognition model
face_analyzer = FaceAnalysis(name="buffalo_l")
face_analyzer.prepare(ctx_id=0)

# Load embeddings and labels
db_embeddings = np.load("embeddings/face_db.npy")
db_labels = pd.read_csv("embeddings/labels.csv", header=None)[0].tolist()

# Distance threshold for matching
MATCH_THRESHOLD = 0.55

# Logging
LOG_FILE = "recognition_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Name,Timestamp\n")

# Compare embedding with database
def recognize_face(embedding):
    similarities = np.dot(db_embeddings, embedding) / (
        np.linalg.norm(db_embeddings, axis=1) * np.linalg.norm(embedding)
    )
    best_match_index = np.argmax(similarities)
    if similarities[best_match_index] > MATCH_THRESHOLD:
        return db_labels[best_match_index]
    return "Unknown"

# Video generator
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        faces = face_analyzer.get(frame)

        for face in faces:
            bbox = face.bbox.astype(int)
            name = recognize_face(face.embedding)

            # Draw bounding box and label
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Log recognized faces
            if name != "Unknown":
                with open(LOG_FILE, "a") as f:
                    f.write(f"{name},{datetime.now()}\n")

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield to Flask
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run server
if __name__ == "__main__":
    app.run(debug=True)
