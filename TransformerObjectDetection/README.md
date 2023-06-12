# Transformer Object Detection

This repository contains a Python script for object detection using the Transformer-based model called DETR (DEtection TRansformer). The script utilizes the Hugging Face Transformers library and the Timm library for image processing.

## Installation

Install the required dependencies by running the following commands:

```shell
!pip install transformers
!pip install timm
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/MohamadsalehMoradpoor/VisualAIProjects.git
   ```

2. Install the required dependencies. It is recommended to use a virtual environment:

   ```bash
   cd TransformerObjectDetection
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Place the image file you want to perform object detection on in the project directory.

4. Open the Python script `object_detection.py` and modify the following line to specify the path to your image file:

   ```python
   with Image.open("your_image.jpg") as im:
   ```

5. Run the script:

   ```bash
   python object_detection.py
   ```

6. The script will detect objects in the image using the DETR model and draw bounding boxes around them. The modified image with bounding boxes will be saved as `your_image_bboxes.jpg`.

## Results

The object detection script uses the DETR model to accurately detect objects in the image and draw bounding boxes around them. The performance of the model may vary depending on the input image and the objects present in it.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.