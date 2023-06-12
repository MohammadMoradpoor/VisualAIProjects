from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
from transformers import DetrFeatureExtractor, DetrForObjectDetection

def draw_bounding_box(im, score, label, xmin, ymin, xmax, ymax, index, num_boxes):
    print(f"Drawing bounding box {index} of {num_boxes}...")

    im_with_rectangle = ImageDraw.Draw(im)  
    im_with_rectangle.rectangle((xmin, ymin, xmax, ymax), outline="red", width=2)

    im_with_rectangle.text((xmin+35, ymin-25), label, fill="white", stroke_fill="red")

    return im

def main():
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101')
    object_detector = pipeline("object-detection", model=model, feature_extractor=feature_extractor)

    with Image.open("street.jpeg") as im:
        bounding_boxes = object_detector(im)
        num_boxes = len(bounding_boxes)
        index = 0

        for bounding_box in bounding_boxes:
            box = bounding_box["box"]

            im = draw_bounding_box(im, bounding_box["score"], bounding_box["label"],
                                   box["xmin"], box["ymin"], box["xmax"], box["ymax"], index, num_boxes)

            index += 1

        im.save("street_bboxes.jpg")
        print("Done!")

if __name__ == "__main__":
    main()
