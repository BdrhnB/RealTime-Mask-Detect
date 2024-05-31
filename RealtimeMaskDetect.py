import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report
from mtcnn.mtcnn import MTCNN
from threading import Thread


def load_data(dataset_path, label):
    data = []
    labels = []
    for image_file in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 64))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(label)
    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    return data, labels


def detect_and_preprocess_face(image, detector):
    faces = detector.detect_faces(image)
    face_arrays = []
    coordinates = []

    for face in faces:
        x, y, w, h = face['box']
        if w < 30 or h < 30:
            continue
        
        face = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (64, 64))
        face_array = img_to_array(face_resized)
        face_array = preprocess_input(face_array)
        face_arrays.append(face_array)
        coordinates.append((x, y, w, h))

    return face_arrays, coordinates


def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    model.save("mask_detector_model.h5")

    print("\nEgitim ozeti:\n")
    print("Egitim Dogrulugu: {:.2f}%".format(100 * max(history.history['accuracy'])))
    print("Egitim Kaybi: {:.4f}".format(min(history.history['loss'])))
    print("Test Dogrulugu: {:.2f}%".format(100 * max(history.history['val_accuracy'])))
    print("Test Kaybi: {:.4f}".format(min(history.history['val_loss'])))

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nSiniflandirma Raporu:")
    print(classification_report(y_test, y_pred))

    return model


def video_stream_processing(model, ip_webcam_url):
    class VideoStreamWidget:
        def __init__(self, src):
            self.capture = cv2.VideoCapture(src)
            self.status, self.frame = self.capture.read()
            Thread(target=self.update, args=()).start()

        def update(self):
            while True:
                if self.capture.isOpened():
                    self.status, self.frame = self.capture.read()

        def get_frame(self):
            return self.frame

    detector = MTCNN()
    video_stream_widget = VideoStreamWidget(ip_webcam_url)
    frame_count = 0
    process_frame_rate = 3

    while True:
        frame = video_stream_widget.get_frame()
        if frame is None:
            continue

        if frame_count % process_frame_rate == 0:
            face_arrays, coordinates = detect_and_preprocess_face(frame, detector)
        for face_array, (x, y, w, h) in zip(face_arrays, coordinates):
            input_image_array = np.expand_dims(face_array, axis=0)
            input_prediction = model.predict(input_image_array)[0][0]
            label = "Maskeli" if input_prediction > 0.5 else "Maskesiz"
            
            color = (0, 255, 0) if label == "Maskeli" else (0, 0, 255)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        cv2.namedWindow('Maske Kontrolu', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Maske Kontrolu', 600, 400)
        cv2.imshow('Maske Kontrolu', frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_stream_widget.capture.release()
    cv2.destroyAllWindows()

def main():
    with_mask_ds = "data/with_mask"
    without_mask_ds = "data/without_mask"

    print("with_mask dataset length:", len(os.listdir(with_mask_ds)))
    print("without_mask dataset length:", len(os.listdir(without_mask_ds)))
    print(" ")

    data_with_mask, labels_with_mask = load_data(with_mask_ds, 1)
    data_without_mask, labels_without_mask = load_data(without_mask_ds, 0)

    X_train_with_mask, X_test_with_mask, y_train_with_mask, y_test_with_mask = train_test_split(
        data_with_mask, labels_with_mask, test_size=0.5, random_state=42)
    X_train_without_mask, X_test_without_mask, y_train_without_mask, y_test_without_mask = train_test_split(
        data_without_mask, labels_without_mask, test_size=0.5, random_state=42)

    X_train = np.concatenate((X_train_with_mask, X_train_without_mask), axis=0)
    X_test = np.concatenate((X_test_with_mask, X_test_without_mask), axis=0)
    y_train = np.concatenate((y_train_with_mask, y_train_without_mask), axis=0)
    y_test = np.concatenate((y_test_with_mask, y_test_without_mask), axis=0)
    
    model = train_and_evaluate_model(X_train, y_train, X_test, y_test)
    
    """urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    http = urllib3.PoolManager(cert_reqs='CERT_NONE')"""
    
    ip_webcam_url = 'https://172.19.35.181:8080/video'
    video_stream_processing(model, ip_webcam_url)

if __name__ == "__main__":
    main()