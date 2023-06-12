# SegmentAnythingAI

This project utilizes the "segment-anything" library developed by Facebook Research to perform image segmentation tasks. It provides a simple interface to segment objects in images using various techniques.

## Installation and Usage

1. Clone the repository and navigate to the directory:

    ```shell
    git clone https://github.com/MohamadsalehMoradpoor/VisualAIProjects.git
    cd SegmentAnythingAI
    ```

2. To install the required dependencies, run the following commands:

    ```shell
    !pip install git+https://github.com/facebookresearch/segment-anything.git
    !pip install opencv-python pycocotools matplotlib onnxruntime onnx
    ```

3. Download the pre-trained model checkpoint by executing the following command:

    ```shell
    !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    ```

4. Load and display an image of your choice.

5. Create a predictor object using the pre-trained model checkpoint.

6. Set the image for the predictor object.

7. Perform object segmentation by specifying the coordinates and labels of the points of interest.

8. Retrieve the segmentation masks, scores, and logits.

9. Display the segmentation results, showing the image, masks, and points of interest.

10. Repeat steps 6-8 for different segmentation scenarios, such as using multiple points or a bounding box.

Remember to modify the code and adapt it to your specific use case as needed.

## Colab Link

You can access a Colab notebook with the code and sample images [here](https://colab.research.google.com/drive/1t30K3lIPXAZinukUBNqiOnUfmRQRx59C).

## Additional Information

For more details on the "segment-anything" library and its usage, please refer to the [official repository](https://github.com/facebookresearch/segment-anything).