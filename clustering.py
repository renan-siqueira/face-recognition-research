import os
import face_recognition
from sklearn.cluster import KMeans
import shutil

def cluster_unknown_faces(unknown_folder):
    image_paths = [os.path.join(unknown_folder, f) for f in os.listdir(unknown_folder) if f.endswith('.png') or f.endswith('.jpg')]
    encodings = [face_recognition.face_encodings(face_recognition.load_image_file(f))[0] for f in image_paths]

    kmeans = KMeans(n_clusters=len(encodings) // 5, random_state=0)  # Você pode ajustar o número de clusters conforme necessário
    labels = kmeans.fit_predict(encodings)

    # Moving images to subfolders based on clusters
    for label, image_path in zip(labels, image_paths):
        cluster_folder = os.path.join(unknown_folder, f"cluster_{label}")
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        shutil.move(image_path, os.path.join(cluster_folder, os.path.basename(image_path)))

if __name__ == "__main__":
    unknown_folder = "output/Unknown"
    cluster_unknown_faces(unknown_folder)
