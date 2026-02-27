import cv2

def test_with_image(image_path):
    frame = cv2.imread(image_path)
    
    if frame is None:
        print("Error: Image not found!")
        return
    
    # Detect on FULL size image first
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8, minSize=(80,80))
    
    print(f"Faces found: {len(faces)}")
    
    # Draw boxes on full image
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Resize AFTER drawing for display only
    frame = cv2.resize(frame, (640, 480))
    
    cv2.imshow("Voting Booth - Face Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()