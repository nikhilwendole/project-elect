from face_detector import test_with_image, get_face_encoding

if __name__ == "__main__":
    test_with_image("test_face.jpg")
    
    print("\nNow generating face encoding...")
    encoding = get_face_encoding("test_face.jpg")
    
    if encoding:
        print(f"First 5 numbers of your face encoding: {encoding[:5]}")