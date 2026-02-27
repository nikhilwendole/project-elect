import cv2
from deepface import DeepFace
import numpy as np

def get_face_encoding(image_path):
    try:
        # Get face embedding (the 128 numbers that represent the face)
        embedding = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
        
        print(f"Face encoding generated successfully!")
        print(f"Encoding length: {len(embedding[0]['embedding'])} numbers")
        
        return embedding[0]['embedding']
    
    except Exception as e:
        print(f"Error generating encoding: {e}")
        return None

def test_with_image(image_path):
    frame = cv2.imread(image_path)
    
    if frame is None:
        print("Error: Image not found!")
        return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8, minSize=(80,80))
    
    print(f"Faces found: {len(faces)}")
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Voting Booth - Face Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()