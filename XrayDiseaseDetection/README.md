# X-ray Disease Detection

This repository contains a Jupyter Notebook for X-ray disease detection using a Convolutional Neural Network (CNN) model. The goal of this project is to classify X-ray images into two categories, including opacity, and normal (healthy) cases.

## Dataset

The dataset used in this project is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle. It consists of X-ray images of the chest categorized into two classes: normal and opacity (which includes cases of pneumonia). The dataset is split into training, validation, and test sets.

## Model Architecture

The CNN model architecture used for disease detection consists of the following layers:

1. Convolutional Layer: 32 nodes, kernel size 3x3, activation ReLU
2. MaxPooling Layer: Pool size 2x2
3. Convolutional Layer: 64 nodes, kernel size 3x3, activation ReLU
4. MaxPooling Layer: Pool size 2x2
5. Convolutional Layer: 128 nodes, kernel size 3x3, activation ReLU
6. MaxPooling Layer: Pool size 2x2
7. Flatten Layer
8. Fully Connected Layer: 256 nodes, activation ReLU
9. Dropout Layer: Dropout rate 0.5
10. Output Layer: Dense layer with the number of classes

[This architecture](https://github.com/MohamadsalehMoradpoor/VisualAIProjects/blob/master/XrayDiseaseDetection/model.png) is designed to extract relevant features from the input X-ray images and make predictions based on those features.

## Usage

To run the code, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/MohamadsalehMoradpoor/VisualAIProjects.git
   ```

2. Install the required dependencies. It is recommended to use a virtual environment:

   ```bash
   cd XrayDiseaseDetection
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Download the dataset from the Kaggle link provided above and place it in the `data` directory.

4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open the `X-ray Disease Detection.ipynb` notebook and execute the cells sequentially.

6. The notebook will guide you through the steps of training the model, evaluating its performance, and making predictions on new X-ray images.

## Results

After training the model, we achieved an accuracy of 95% on the test set. The model demonstrates strong performance in distinguishing between normal and abnormal X-ray images, providing valuable insights for disease diagnosis.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

```css
If you need any further assistance or have additional questions, feel free to ask!
```