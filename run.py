import os
import shutil
import argparse

import cv2
import face_recognition
from tqdm import tqdm

from sklearn.cluster import KMeans, DBSCAN


def learn_from_images(directory, model):
    known_face_encodings = []
    known_face_names = []

    for person in os.listdir(directory):
        person_path = os.path.join(directory, person)
        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(person_path, filename)
                    image = face_recognition.load_image_file(image_path)
                    face_locations = face_recognition.face_locations(image, model=model)
                    face_encodings = face_recognition.face_encodings(image, face_locations)
                    
                    if face_encodings:
                        known_face_encodings.append(face_encodings[0])
                        known_face_names.append(person)

    return known_face_encodings, known_face_names


def process_video(video_path, known_face_encodings, known_face_names, output_directory, image_format, scale_factor, margin, fps, model):
    video_capture = cv2.VideoCapture(video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_skip = fps

    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    print('Video:', video_base_name, 'FPS:', original_fps, 'Total Frames:', total_frames)

    with tqdm(total=total_frames, desc=video_base_name) as pbar:
        frame_count = 0
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
        
            if frame_count % frame_skip == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                rgb_small_frame = small_frame[:, :, ::-1]

                face_locations = face_recognition.face_locations(rgb_small_frame, model=model)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for index, ((top, right, bottom, left), face_encoding) in enumerate(zip(face_locations, face_encodings)):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                    name = "Unknown"
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]

                    person_directory = os.path.join(output_directory, name)
                    if not os.path.exists(person_directory):
                        os.mkdir(person_directory)

                    top = max(top*4 - margin, 0)
                    right = min(right*4 + margin, frame.shape[1])
                    bottom = min(bottom*4 + margin, frame.shape[0])
                    left = max(left*4 - margin, 0)

                    face_image = frame[top:bottom, left:right]
                    face_filename = os.path.join(person_directory, "{}_video_{}_frame_{:06d}_face_{:03d}{}".format(name, video_base_name, frame_count, index + 1, image_format))
                    cv2.imwrite(face_filename, face_image)
                
            frame_count += 1
            pbar.update(1)

    video_capture.release()


def process_all_videos(videos_directory, known_face_encodings, known_face_names, output_directory, image_format, scale_factor, margin, fps, model):
    allowed_extensions = [".webm", ".mp4", '.avi', '.mkv']
    video_paths = [os.path.join(videos_directory, filename) for filename in os.listdir(videos_directory) if filename.endswith(tuple(allowed_extensions))]

    for video_path in video_paths:
        process_video(video_path, known_face_encodings, known_face_names, output_directory, image_format, scale_factor, margin, fps, model)


def process_unknown_images(known_face_encodings, known_face_names, unknown_directory, model):
    for root, _, files in os.walk(unknown_directory):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(root, filename)
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image, model=model)
                face_encodings = face_recognition.face_encodings(image, face_locations)

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                    name = "Unknown"
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]

                    person_directory = os.path.join("output", name)
                    if not os.path.exists(person_directory):
                        os.mkdir(person_directory)
                    shutil.move(image_path, os.path.join(person_directory, filename))


def cluster_unknown_faces(unknown_folder, model, scale_factor, clustering_algorithm="kmeans"):
    image_paths = [os.path.join(unknown_folder, f) for f in os.listdir(unknown_folder) if f.endswith('.png') or f.endswith('.jpg')]
    
    encodings = []
    print("Extracting encodings from images...")
    for image_path in tqdm(image_paths, desc="Encoding"):
        image = face_recognition.load_image_file(image_path)
        
        image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
        
        face_locations = face_recognition.face_locations(image, model=model)
        if face_locations:
            face_enc = face_recognition.face_encodings(image, face_locations)[0]
            encodings.append(face_enc)

    print("\nPerforming clustering...")
    if clustering_algorithm == "kmeans":
        n_clusters = max(1, len(encodings) // 5)
        clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    elif clustering_algorithm == "dbscan":
        clusterer = DBSCAN(eps=0.5, min_samples=5)
    else:
        raise ValueError(f"Invalid clustering algorithm: {clustering_algorithm}")

    labels = clusterer.fit_predict(encodings)

    print("\nOrganizing clustered images...")
    for label, image_path in tqdm(zip(labels, image_paths), total=len(labels), desc="Organizing"):
        cluster_folder = os.path.join(unknown_folder, f"cluster_{label}")
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        shutil.move(image_path, os.path.join(cluster_folder, os.path.basename(image_path)))


def main():
    parser = argparse.ArgumentParser(description="Processa vídeos ou imagens para reconhecimento facial.")
    parser.add_argument("mode", choices=["video", "image", "cluster"], default="video", nargs="?", help="Especifique 'video' para processar vídeos, 'image' para processar imagens na pasta Unknown ou 'cluster' para agrupar imagens desconhecidas.")

    args = parser.parse_args()

    image_directory = 'faces'
    videos_directory = 'videos'
    output_directory = 'output'
    unknown_folder = "output/Unknown"
    image_format = ".jpg"
    scale_factor = 0.25
    margin = 100
    fps = 5
    model = "cnn"
    clustering_algorithm = "dbscan"

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    if args.mode != "cluster":
        known_face_encodings, known_face_names = learn_from_images(image_directory, model)

    if args.mode == "video":
        process_all_videos(videos_directory, known_face_encodings, known_face_names, output_directory, image_format, scale_factor, margin, fps, model)
    elif args.mode == "image":
        process_unknown_images(known_face_encodings, known_face_names, model)
    elif args.mode == "cluster":
        cluster_unknown_faces(unknown_folder, model, scale_factor, clustering_algorithm)


if __name__ == "__main__":
    main()
