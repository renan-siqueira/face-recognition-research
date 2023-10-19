import os
import cv2
import face_recognition
from tqdm import tqdm
import time


known_face_encodings_cache = []
known_face_folders_cache = []
exclusion_encodings = []


def learn_from_images(directory, model):
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
                        known_face_encodings_cache.append(face_encodings[0])
                        known_face_folders_cache.append(person)


def learn_from_exclusions(directory, model):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image, model=model)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if face_encodings:
                exclusion_encodings.extend(face_encodings)


def process_video(video_path, output_directory, image_format, scale_factor, margin, fps, model, tolerance, restricted_recognition):
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
                    matches_known = face_recognition.compare_faces(known_face_encodings_cache, face_encoding, tolerance=tolerance)
                    matches_excluded = face_recognition.compare_faces(exclusion_encodings, face_encoding, tolerance=tolerance)

                    if True in matches_known:
                        first_match_index = matches_known.index(True)
                        name = known_face_folders_cache[first_match_index]
                    elif True in matches_excluded:
                        name = "unrecognized"
                    else:
                        if restricted_recognition:
                            name = "unrecognized"
                        else:
                            person_count = len(known_face_folders_cache) + 1
                            name = f"person_{person_count}"
                            known_face_encodings_cache.append(face_encoding)
                            known_face_folders_cache.append(name)

                    person_directory = os.path.join(output_directory, name)
                    if not os.path.exists(person_directory):
                        os.mkdir(person_directory)

                    top = int(top / scale_factor)
                    right = int(right / scale_factor)
                    bottom = int(bottom / scale_factor)
                    left = int(left / scale_factor)

                    top = max(top - margin, 0)
                    right = min(right + margin, frame.shape[1])
                    bottom = min(bottom + margin, frame.shape[0])
                    left = max(left - margin, 0)

                    face_image = frame[top:bottom, left:right]
                    face_filename = os.path.join(person_directory, "{}_video_{}_frame_{:06d}_face_{:03d}{}".format(name, video_base_name, frame_count, index + 1, image_format))
                    cv2.imwrite(face_filename, face_image)
                
            frame_count += 1
            pbar.update(1)

    video_capture.release()


def process_all_videos(videos_directory, output_directory, image_format, scale_factor, margin, fps, model, tolerance, restricted_recognition):
    allowed_extensions = [".webm", ".mp4", '.avi', '.mkv']
    video_paths = [os.path.join(videos_directory, filename) for filename in os.listdir(videos_directory) if filename.endswith(tuple(allowed_extensions))]

    for video_path in video_paths:
        process_video(video_path, output_directory, image_format, scale_factor, margin, fps, model, tolerance, restricted_recognition)


def print_execution_time(elapsed_time):
    if elapsed_time < 60:
        print(f"Tempo de execução: {elapsed_time:.2f} segundos")
    elif elapsed_time < 3600:
        minutes = elapsed_time / 60
        print(f"Tempo de execução: {minutes:.2f} minutos")
    else:
        hours = elapsed_time / 3600
        print(f"Tempo de execução: {hours:.2f} horas")


def main():
    image_directory = 'faces'
    review_directory = 'review'
    videos_directory = 'videos'
    output_directory = 'output'
    image_format = ".jpg"
    scale_factor = 0.33
    margin = 100
    fps = 5
    model = "cnn"
    tolerance = 0.60
    restricted_recognition = True

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    if os.path.exists(review_directory):
        learn_from_exclusions(review_directory, model)

    start_time = time.time()

    learn_from_images(image_directory, model)
    process_all_videos(videos_directory, output_directory, image_format, scale_factor, margin, fps, model, tolerance, restricted_recognition)

    elapsed_time = time.time() - start_time
    print_execution_time(elapsed_time)

if __name__ == "__main__":
    main()
