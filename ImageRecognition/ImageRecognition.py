import os
import cv2
import numpy as np
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load the ResNet152 model
model = ResNet152(weights='imagenet')

# Load and preprocess the image
def load_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Perform prediction on a single image
def predict_single_image(image_path):
    x = load_image(image_path)
    predictions = model.predict(x)
    label = decode_predictions(predictions, top=1)[0][0][1]
    return label

# Perform prediction on a video stream
def predict_video_stream():
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        frame = cv2.resize(frame, (224, 224))
        image = frame[..., ::-1]
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        predictions = model.predict(image)
        label = decode_predictions(predictions, top=1)[0][0][1]
        cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the image file
    image_path = 'image.jpeg'
    # Perform prediction on a single image
    label = predict_single_image(image_path)
    print(f"Prediction for {image_path}: {label}")
    # Perform prediction on a video stream
    predict_video_stream()
