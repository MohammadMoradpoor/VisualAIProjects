# Image Recognition

This project utilizes image recognition techniques to perform predictions on images and video streams using the ResNet152 model. It can be used to identify objects and scenes in real-time using a webcam or process individual images.

## Dependencies

- Python 3.x
- OpenCV
- NumPy
- TensorFlow
- Keras

## Usage

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies by running the following command:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure that your webcam is connected and accessible by the application.
4. Run the script `ImageRecognition.py` to start the webcam image recognition process.

    ```bash
    python ImageRecognition.py
    ```

5. The application will open a window displaying the webcam feed.
6. Objects and scenes in the webcam feed will be recognized and displayed as text on the window.
7. Press the 'q' key to stop the webcam image recognition process.

## Single Image Prediction

To perform predictions on a single image, you can modify the code in the `ImageRecognition.py` file as follows:

```python
# Path to the image file
image_path = 'image.jpeg'
# Perform prediction on a single image
label = predict_single_image(image_path)
print(f"Prediction for {image_path}: {label}")
```

Replace `image.jpeg` with the path to your desired image file. Running the script will output the prediction result for that image.

## Contribution

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request in the repository.

## License

Feel free to use, modify, and distribute the code for personal or commercial purposes.