# Drowsiness Detection

Drowsiness detection is a safety technology that can prevent accidents caused by drivers falling asleep while driving. This project aims to build a drowsy driver alert system that can be implemented in various ways.

## Application Demo

![Drowsiness Detection](DrowsinessDetection1.gif)

## Objective

The objective of this project is to develop a drowsiness detection system that can identify when a person's eyes are closed for a few seconds and alert the driver when drowsiness is detected.

## Code and Resources Used

- Python Version: 3.7
- Packages: OpenCV (for face and eye detection), TensorFlow (Keras uses TensorFlow as a backend), Keras (to build the classification model), Pygame (to play the alarm sound)

## Dataset

The dataset used for this project consists of approximately 7000 images of people's eyes under different lighting conditions.

## Model Architecture

The model used in this project is built with Keras using Convolutional Neural Networks (CNN). A CNN is a specialized type of deep neural network that performs exceptionally well for image classification tasks. The CNN model architecture consists of the following layers:

- Convolutional layer with 32 nodes and a kernel size of 3
- Convolutional layer with 32 nodes and a kernel size of 3
- Convolutional layer with 64 nodes and a kernel size of 3
- Fully connected layer with 128 nodes

The final layer is a fully connected layer with 2 nodes. The ReLU activation function is used in all layers except the output layer, where the Softmax activation function is applied.

## Face and Eye Detection using OpenCV

The project utilizes Haar cascade files to detect faces, left eyes, and right eyes using OpenCV. The `models` folder contains the trained model file `cnnCat2.h5`. When drowsiness is detected, an audio clip `alarm.wav` is played. The program for building the classification model using CNN can be found in the `model.py` file.

## Methodology

The drowsiness detection process involves the following steps:

1. Clone the repository using the following command:

    ```bash
    git clone https://github.com/MohamadsalehMoradpoor/VisualAIProjects.git
    ```

2. Change into the cloned DrowsinessDetection's directory:

    ```bash
    cd DrowsinessDetection
    ```
       
3. Install the required dependencies by running the following command:

    ```bash
    pip install -r requirements.txt
    ```

4. Launch the Drowsiness Detection application by executing the following command:

    ```bash
    python DrowsinessDetection.py
    ```

5. The application will open, utilizing your computer's camera to detect drowsiness in real-time.
6. Keep your face within the camera's view, and the application will analyze your eye movements.
7. As the application runs, it will continuously monitor your eye status and calculate a drowsiness score.
8. If the drowsiness score exceeds 16, indicating a high likelihood of drowsiness, the alarm will be triggered.
9. The alarm sound serves as an alert to prompt you to take necessary actions to avoid drowsiness-related accidents.

Please note that the threshold for the drowsiness score can be adjusted based on your specific requirements. You can modify the value in the code to set a different threshold that suits your needs.

Feel free to explore the application and ensure that you have a clear view of your face for accurate drowsiness detection.

Note: The application is provided as a BAT or EXE file, depending on your operating system. Simply open the file and use the software as intended.

## Files

- `model.py` contains the code used to build the CNN classifier model.
- `DrowsinessDetection.py` is the main file of this project. Run this file to initiate the drowsiness detection process.
- `DrowsinessDetection.exe` is provided for easy installation and usage on the desktop.

## Contribution

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request in the repository.

## License

Feel free to use, modify, and distribute the code for personal or commercial purposes.

---

```javascript
Stay vigilant and drive safely!
```