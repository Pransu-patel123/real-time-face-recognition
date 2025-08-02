
# Real-Time Facial Recognition Web App (CPU Optimized)

A real-time facial recognition system built using **Python**, **InsightFace**, and **Flask**, designed to run efficiently on a **CPU**. It detects and recognizes faces from a live webcam stream, shows names, and logs recognition events — all through a user-friendly web interface.

---

## 🚀 Features

- 🎥 Real-time webcam feed in browser
- 😎 Face detection using **RetinaFace**
- 🧠 Face recognition using **ArcFace embeddings**
- 📝 Logs recognized faces with timestamps
- 🗃️ Register new faces using image folders
- 🧰 Runs on **CPU only** (no GPU required)

---

## 📁 Project Structure

```
face_recognition_webapp/
├── app.py                     # Flask app
├── register_faces.py          # Script to build face DB
├── known_faces/               # Labeled images for known users
│   ├── person1/
│   │   └── image.jpg
│   └── person2/
├── embeddings/
│   ├── face_db.npy            # Face embeddings
│   └── labels.csv             # Corresponding names
├── recognition_log.csv        # Log of recognized faces
├── templates/
│   └── index.html             # Web interface
├── static/                    # (Optional CSS/JS)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/face_recognition_webapp.git
cd face_recognition_webapp
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Register Faces

Add folders inside `known_faces/` where each folder name is the **person's name** and contains 1+ clear images of their face.

Example:
```
known_faces/
├── alice/
│   └── alice1.jpg
├── bob/
│   └── bob1.jpg
```

Then run:

```bash
python register_faces.py
```

This creates:
- `face_db.npy` → embeddings
- `labels.csv` → names

---

## 💻 Run the App

```bash
python app.py
```

Visit: [http://localhost:5000](http://localhost:5000)

- Live webcam will open in your browser.
- Recognized names are drawn on-screen.
- All recognized events are logged in `recognition_log.csv`.

---

## 📝 Log Example

```
Name,Timestamp
Alice,2025-08-01 17:45:12
Bob,2025-08-01 17:46:08
```

---

## 📦 Dependencies

- Python 3.8+
- Flask
- OpenCV
- InsightFace
- NumPy
- Pandas
- Pillow

> Install with:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Future Improvements

- Add face registration via webcam
- Add access control for logs
- Deploy to cloud (Render, Railway, etc.)
- Anti-spoofing detection
- GUI for image upload

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Pransu Patel**  
Built as part of a major academic project using CPU-only AI-powered face recognition.

---

## ⭐ GitHub Push Instructions

1. **Initialize git (if not already):**
```bash
git init
```

2. **Add remote & push:**
```bash
git remote add origin https://github.com/your-username/face_recognition_webapp.git
git add .
git commit -m "Initial commit: Real-Time Facial Recognition Web App"
git push -u origin master
```

> Replace `your-username` with your actual GitHub username.

---

### ✅ Done!
You're now ready to show this off as a **professional project** on GitHub.
