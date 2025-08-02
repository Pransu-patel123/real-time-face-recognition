import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import pandas as pd

# Initialize face analysis (CPU only)
app = FaceAnalysis(name="buffalo_l")  # arcface model
app.prepare(ctx_id=0)

# Paths
IMAGE_DIR = "known_faces"
DB_PATH = "embeddings/face_db.npy"
LABELS_PATH = "embeddings/labels.csv"

# Prepare DB lists
face_embeddings = []
labels = []

# Loop through each person's folder
for person_name in os.listdir(IMAGE_DIR):
    person_folder = os.path.join(IMAGE_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue

        faces = app.get(img)
        if len(faces) == 0:
            print(f"No face found in {image_path}")
            continue

        # Take first face only
        face_embedding = faces[0].embedding
        face_embeddings.append(face_embedding)
        labels.append(person_name)
        print(f"Registered: {person_name} ({image_name})")

# Save embeddings and labels
os.makedirs("embeddings", exist_ok=True)
np.save(DB_PATH, np.array(face_embeddings))
pd.DataFrame(labels).to_csv(LABELS_PATH, index=False, header=False)

print("✅ Registration complete.")
