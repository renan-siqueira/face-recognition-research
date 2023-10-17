import os
import cv2
import face_recognition


def learn_from_images(directory):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(image)[0]

            name = filename.split('.')[0]

            known_face_encodings.append(encoding)
            known_face_names.append(name)

    return known_face_encodings, known_face_names

def process_video(video_path, known_face_encodings, known_face_names):
    video_capture = cv2.VideoCapture(video_path)

    output_directory = "output"
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            person_directory = os.path.join(output_directory, name)
            if not os.path.exists(person_directory):
                os.mkdir(person_directory)

            face_image = frame[top*4:bottom*4, left*4:right*4]
            face_filename = os.path.join(person_directory, "{}.png".format(len(os.listdir(person_directory))+1))
            cv2.imwrite(face_filename, face_image)

    video_capture.release()

def main():
    image_directory = 'faces'
    video_path = 'test_video.webm'

    known_face_encodings, known_face_names = learn_from_images(image_directory)
    process_video(video_path, known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
