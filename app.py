from flask import Flask, Response
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
import os

app = Flask(__name__)

# Initialize FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Image transformations
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# Load dataset images and compute embeddings
def load_dataset_embeddings(dataset_path):
    known_faces = {}
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                image = Image.open(image_path).convert('RGB')
                face_tensor = transform(image).unsqueeze(0).to(device)
                embedding = model(face_tensor).detach().cpu().numpy()
                known_faces[person_name] = embedding
    return known_faces

# Set dataset path
DATASET_PATH = "C:\dataset"
known_faces = load_dataset_embeddings(DATASET_PATH)

# Function to get face embedding
def get_face_embedding(image):
    face_tensor = transform(image).unsqueeze(0).to(device)
    embedding = model(face_tensor).detach().cpu().numpy()
    return embedding

# Recognize faces
def recognize_face(embedding, known_faces, threshold=0.5):
    identity = "Unknown"
    min_distance = float('inf')
    match_confidence = 0.0

    for name, known_embedding in known_faces.items():
        distance = cosine(embedding.squeeze(), known_embedding.squeeze())
        if distance < threshold and distance < min_distance:
            min_distance = distance
            identity = name
            match_confidence = (1 - distance) * 100

    return identity, match_confidence

# Process video frames
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb_frame)
    
    if boxes is not None:
        for box in boxes:
            face = Image.fromarray(rgb_frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])])
            embedding = get_face_embedding(face)
            identity, confidence = recognize_face(embedding, known_faces)
            text = f"{identity}: {confidence:.2f}%"
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, text, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

# Video stream generator
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

# API endpoint for live feed
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.add_url_rule('/video_feed', 'video_feed', video_feed)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
