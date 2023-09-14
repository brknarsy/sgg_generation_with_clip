import torchvision
import requests
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
import torch

from transformers import CLIPProcessor, CLIPModel

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import transforms
from torchvision import datapoints as dp
from torchvision.transforms.v2 import functional as F
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

VG_PREDICATE_CLASSES = {
    '1': 'above',
    '2': 'across',
    '3': 'against',
    '4': 'along',
    '5': 'and',
    '6': 'at',
    '7': 'attached to',
    '8': 'behind',
    '9': 'belonging to',
    '10': 'between',
    '11': 'carrying',
    '12': 'covered in',
    '13': 'covering',
    '14': 'eating',
    '15': 'flying in',
    '16': 'for',
    '17': 'from',
    '18': 'growing on',
    '19': 'hanging from',
    '20': 'has',
    '21': 'holding',
    '22': 'in',
    '23': 'in front of',
    '24': 'laying on',
    '25': 'looking at',
    '26': 'lying on',
    '27': 'made of',
    '28': 'mounted on',
    '29': 'near',
    '30': 'of',
    '31': 'on',
    '32': 'on back of',
    '33': 'over',
    '34': 'painted on',
    '35': 'parked on',
    '36': 'part of',
    '37': 'playing',
    '38': 'riding',
    '39': 'says',
    '40': 'sitting on',
    '41': 'standing on',
    '42': 'to',
    '43': 'under',
    '44': 'using',
    '45': 'walking in',
    '46': 'walking on',
    '47': 'watching',
    '48': 'wearing',
    '49': 'wears',
    '50': 'with'
}

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def area_of_union(rect1, rect2, image=None):
    min_x = min(rect1[0], rect1[2], rect2[0], rect2[2])
    min_y = min(rect1[1], rect1[3], rect2[1], rect2[3])
    max_x = max(rect1[0], rect1[2], rect2[0], rect2[2])
    max_y = max(rect1[1], rect1[3], rect2[1], rect2[3])
    if image is None:
        return [min_x, min_y, max_x, max_y]
    new_image = image.crop((min_x, min_y, max_x, max_y))
    return [min_x, min_y, max_x, max_y], new_image

def generate_clip_text_input(label1, label2):
    pairwise_relationships = []
    for relation in list(VG_PREDICATE_CLASSES.values()):
        text = "the " + label1 + " is " + relation + " the " + label2
        pairwise_relationships.append(text)
    return pairwise_relationships


def get_clip_prediction(text, image):
    prediction = {}
    inputs = clip_processor(text=text, images=image, return_tensors="pt",
                       padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    probs_list = probs.tolist()[0]
    for index in range(len(text)):
        prediction[f"{text[index]}"] = probs_list[index]
    return prediction



image_url = "https://media.istockphoto.com/id/1300142880/photo/indian-man-comparing-water-in-different-bottles.jpg?s=612x612&w=0&k=20&c=Gj32nI54TyI2mvZZ--h6agb5PW9Q8OrG-ce5YuNxf4g="
raw_image = Image.open(requests.get(image_url, stream=True).raw)
np_image = np.array(raw_image)
transformed_img = torchvision.transforms.transforms.ToTensor()(
    torchvision.transforms.ToPILImage()(np_image))

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()
predictions = model([transformed_img])

print(predictions)

fig, ax = plt.subplots()
ax.imshow(raw_image)

objects = {
    "labels": [],
    "boxes": [],
    "scores": []
}

for label, box, score in zip(predictions[0]["labels"].tolist(), predictions[0]["boxes"].tolist(), predictions[0]["scores"].tolist()):
    if score < 0.65:
        continue
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    ax.add_patch(
        Rectangle((box[0], box[1]), box_width, box_height,
                  edgecolor='red',
                  fill=False,
                  lw=2
                  ))
    objects["labels"].append(COCO_INSTANCE_CATEGORY_NAMES[label])
    objects["boxes"].append(box)
    objects["scores"].append(score)
    #print(COCO_INSTANCE_CATEGORY_NAMES[label])

# pairwise calculations
for first in range(len(objects)):
    for second in range(first + 1, len(objects)):
        aou, cropped_image = area_of_union(objects["boxes"][first], objects["boxes"][second], raw_image)
        box_width = aou[2] - aou[0]
        box_height = aou[3] - aou[1]
        ax.add_patch(
            Rectangle((aou[0], aou[1]), box_width, box_height,
                      edgecolor='green',
                      fill=False,
                      lw=2
                      ))
        texts = generate_clip_text_input(objects["labels"][first], objects["labels"][second])
        clip_prediction = get_clip_prediction(texts, cropped_image)
        sorted_clip_predictions = sorted(clip_prediction.items(), key=lambda x: x[1], reverse=True)
        print(sorted_clip_predictions)
plt.show()

