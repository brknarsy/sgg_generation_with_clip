from visual_genome import api
import torch
import ijson
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd

from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from transformers import CLIPProcessor, CLIPModel
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import warnings

from torchvision.io import read_image
from PIL import Image
import h5py
import json
import math
import time
import requests

warnings.filterwarnings("ignore")

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


def load_region_descriptions(filepath, parser):
    data = []
    columns = []
    with open(filepath, 'r') as f:
        objects = ijson.items(f, parser)
        first = True
        for item in objects:
            if first:
                columns = list(item.keys())
                first = False
            else:
                data.append(list(item.values()))
    dataframe = pd.DataFrame(data, columns=columns)
    return dataframe


def load_objects(filepath, parser):
    data = []
    columns = []
    with open(filepath, 'r') as f:
        objects = ijson.items(f, parser)
        first = True
        for item in objects:
            if first:
                columns = list(item["objects"][0].keys())
                columns.append("image_id")
                first = False
            for row in item["objects"]:
                selected_row = list(row.values())
                selected_row.append(item["image_id"])
                data.append(selected_row)
        dataframe = pd.DataFrame(data, columns=columns)
        return dataframe


class VisualGenomeDataset(Dataset):
    def __init__(self, img_dir, root_dir, transform, target_transform=None):
        print("Loading the dataset...")
        start_time = time.time()

        self.img_dir = img_dir
        self.img_data = json.load(open(root_dir + "image_data.json"))

        self.region_descriptions = load_region_descriptions(root_dir + "region_descriptions.json", "item.regions.item")
        self.objects = load_objects(root_dir + "objects.json", "item")

        self.transform = transform
        self.target_transform = target_transform

        end_time = time.time()
        print(f"Loaded the dataset in {end_time - start_time}")

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        idx +=1
        img_path = os.path.join(self.img_dir, str(idx) + ".jpg")
        image = Image.open(img_path)
        image_data = self.img_data[idx - 1]
        start_time = time.time()
        regions = self.region_descriptions[self.region_descriptions['image_id'] == idx]
        objects = self.objects[self.objects['image_id'] == idx]
        end_time = time.time()

        image_h = 300
        image_w = 300

        h, w = image.size
        h_ratio = image_h / h
        w_ratio = image_w / w

        image = scale_img(image, image_h, image_w)
        for index in regions.index:
            new_bbox = scale_bbox([regions["x"][index], regions["y"][index], regions["x"][index] + regions["width"][index], regions["y"][index] + regions["height"][index]],
                                  h_ratio,
                                  w_ratio)
            regions.loc[index, "x"] = new_bbox[0]
            regions.loc[index, "y"] = new_bbox[1]
            regions.loc[index, "width"] = new_bbox[2] - new_bbox[0]
            regions.loc[index, "height"] = new_bbox[3] - new_bbox[1]
        regions = list(regions.to_dict("index").values())
        objects = objects.to_dict("index")

        if self.transform:
            image = self.transform(image)
        print(regions)
        return image, regions#, image_data, objects


class BBoxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(8, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 50)
        )

    def forward(self, boxes):
        boxes = self.flatten(boxes)
        logits = self.layers.forward(boxes)
        return logits


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader, start=0):
        print(data)

        """optimizer.zero_grad()

        outputs = bbox_model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0."""

    return last_loss


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def scale_input_data(img, bboxes):
    h, w = img.size()
    img = img.resize(300, 300)
    h_ratio = 300/h
    w_ratio = 300/w
    new_bboxes = []
    for bbox in bboxes:
        selected_bbox = [bbox[0] * w_ratio, bbox[1] * h_ratio, bbox[2] * w_ratio, bbox[3] * h_ratio]
        new_bboxes.append(selected_bbox)
    return img, new_bboxes


def scale_img(img, h, w):
    img = img.resize((h, w), resample=0)
    return img


def scale_bbox(bbox, h_ratio, w_ratio):
    new_bbox = [bbox[0] * w_ratio, bbox[1] * h_ratio, bbox[2] * w_ratio, bbox[3] * h_ratio]
    return new_bbox



class SGGModel:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.frcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.frcnn_model.eval()

    def __call__(self, image, plotting=True):
        raw_image = image
        np_image = np.array(raw_image)
        transformed_img = torchvision.transforms.transforms.ToTensor()(torchvision.transforms.ToPILImage()(np_image))

        frcnn_predictions = self.frcnn_model([transformed_img])

        objects = {
            "labels": [],
            "boxes": [],
            "scores": []
        }

        fig, ax = plt.subplots()
        ax.imshow(raw_image)

        for label, box, score in zip(frcnn_predictions[0]["labels"].tolist(),
                                     frcnn_predictions[0]["boxes"].tolist(),
                                     frcnn_predictions[0]["scores"].tolist()):
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
            # print(COCO_INSTANCE_CATEGORY_NAMES[label])

        for first in range(len(objects)):
            for second in range(first + 1, len(objects)):
                aou, cropped_image = SGGModel.area_of_union(objects["boxes"][first], objects["boxes"][second], raw_image)
                box_width = aou[2] - aou[0]
                box_height = aou[3] - aou[1]
                ax.add_patch(
                    Rectangle((aou[0], aou[1]), box_width, box_height,
                              edgecolor='green',
                              fill=False,
                              lw=2
                              ))
                relationship_texts = SGGModel.generate_clip_text_input(objects["labels"][first], objects["labels"][second])

                clip_prediction = {}
                inputs = self.clip_processor(text=relationship_texts, images=image, return_tensors="pt",
                                        padding=True)
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                probs_list = probs.tolist()[0]

                for index in range(len(relationship_texts)):
                    clip_prediction[f"{relationship_texts[index]}"] = probs_list[index]
                sorted_clip_predictions = sorted(clip_prediction.items(), key=lambda x: x[1], reverse=True)
                print(sorted_clip_predictions)
        return sorted_clip_predictions

    @staticmethod
    def area_of_union(rect1, rect2, image=None):
        min_x = min(rect1[0], rect1[2], rect2[0], rect2[2])
        min_y = min(rect1[1], rect1[3], rect2[1], rect2[3])
        max_x = max(rect1[0], rect1[2], rect2[0], rect2[2])
        max_y = max(rect1[1], rect1[3], rect2[1], rect2[3])
        if image is None:
            return [min_x, min_y, max_x, max_y]
        new_image = image.crop((min_x, min_y, max_x, max_y))
        return [min_x, min_y, max_x, max_y], new_image

    @staticmethod
    def generate_clip_text_input(label1, label2):
        pairwise_relationships = []
        for relation in list(VG_PREDICATE_CLASSES.values()):
            text = "" + label1 + " is " + relation + " " + label2
            pairwise_relationships.append(text)
        return pairwise_relationships


if __name__ == "__main__":
    sgg_model = SGGModel()
    training_dataset = VisualGenomeDataset(
        "C:/Users/Ali Berkin/sgg_benchmark/datasets/vg/images",
        "C:/Users/Ali Berkin/sgg_benchmark/datasets/vg/",
        ToTensor()
    )
    #image, regions, image_data, objects = training_dataset.__getitem__(3)
    #raw_image = transforms.ToPILImage()(image)
    #sgg_model(raw_image)
    #print(regions, image_data, objects)
    #plt.imshow(transforms.ToPILImage()(image))
    #plt.show()

    bbox_model = BBoxModel()
    training_loader = DataLoader(training_dataset, batch_size=64, shuffle=False)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(bbox_model.parameters(), lr=0.001)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 1

    EPOCHS = 5

    best_vloss = 1_000_000.

    train_one_epoch(1, writer)








