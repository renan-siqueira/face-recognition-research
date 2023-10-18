import face_recognition
import dlib

def detect_face_from_image(image_path, model="cnn"):
    # Verifica se a dlib está usando CUDA
    if dlib.DLIB_USE_CUDA:
        print("DLIB is using CUDA.")
    else:
        print("DLIB is NOT using CUDA.")
        
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model=model)
    print(f"Detected {len(face_locations)} faces in the image.")

if __name__ == "__main__":
    image_path = 'test_face_recognition.jpg'
    detect_face_from_image(image_path)
