import os
import cv2
import face_recognition
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading


# Global lock
file_system_lock = threading.Lock()


def learn_from_images(directory):
    known_face_encodings = []
    known_face_names = []

    for person in os.listdir(directory):
        person_path = os.path.join(directory, person)
        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(person_path, filename)
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if face_encodings:
                        known_face_encodings.append(face_encodings[0])
                        known_face_names.append(person)

    return known_face_encodings, known_face_names


def process_video(video_path, known_face_encodings, known_face_names, output_directory, scale_factor=0.25, margin=0):
    video_capture = cv2.VideoCapture(video_path)

    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Acquiring the lock before accessing the file system
            with file_system_lock:
                person_directory = os.path.join(output_directory, name)
                if not os.path.exists(person_directory):
                    os.mkdir(person_directory)

                # Adjusting the coordinates based on the margin
                top = max(top*4 - margin, 0)
                right = min(right*4 + margin, frame.shape[1])
                bottom = min(bottom*4 + margin, frame.shape[0])
                left = max(left*4 - margin, 0)

                face_image = frame[top:bottom, left:right]
                # Modifying the naming logic to include the person's label
                num_files = len(os.listdir(person_directory))
                face_filename = os.path.join(person_directory, "{}_{:06d}.png".format(name, num_files+1))
                cv2.imwrite(face_filename, face_image)

    video_capture.release()


def process_all_videos(videos_directory, known_face_encodings, known_face_names, output_directory, scale_factor=0.25, margin=0):
    allowed_extensions = [".webm", ".mp4", '.avi', '.mkv']
    video_paths = [os.path.join(videos_directory, filename) for filename in os.listdir(videos_directory) if filename.endswith(tuple(allowed_extensions))]

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_video, video_paths, [known_face_encodings]*len(video_paths), [known_face_names]*len(video_paths), [output_directory]*len(video_paths), [scale_factor]*len(video_paths), [margin]*len(video_paths)), total=len(video_paths)))


def main():
    image_directory = 'faces'
    videos_directory = 'videos'
    output_directory = 'output'
    scale_factor = 0.25
    margin = 50

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    known_face_encodings, known_face_names = learn_from_images(image_directory)
    process_all_videos(videos_directory, known_face_encodings, known_face_names, output_directory, scale_factor, margin)


if __name__ == "__main__":
    main()
