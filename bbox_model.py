from visual_genome import api
import torch
import ijson
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.nn.utils.rnn import pad_sequence
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
    '50': 'with',
    '51': 'other',
}


def predicate_translation(value):
    try:
        return int(list(VG_PREDICATE_CLASSES.keys())[list(VG_PREDICATE_CLASSES.values()).index(value.lower())])
    except ValueError:
        return 51


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


def load_relationships(filepath, parser):
    data = []
    columns = []
    with open(filepath, 'r') as f:
        objects = ijson.items(f, parser)
        first = True
        for item in objects:
            if first:
                columns = list(item["relationships"][0].keys())
                columns.append("image_id")
                first = False
            for row in item["relationships"]:
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

        #self.region_descriptions = load_region_descriptions(root_dir + "region_descriptions.json", "item.regions.item")
        #self.objects = load_objects(root_dir + "objects.json", "item")
        self.relationships = load_relationships(root_dir + "relationships.json", "item")

        self.transform = transform
        self.target_transform = target_transform

        end_time = time.time()
        print(f"Loaded the dataset in {end_time - start_time}")

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        relationship = self.relationships.iloc[idx]
        image_id = relationship["image_id"]
        img_path = os.path.join(self.img_dir, str(image_id) + ".jpg")
        img = Image.open(img_path)

        new_h = 480
        new_w = 600
        w, h = img.size
        h_ratio = new_h / h
        w_ratio = new_w / w

        # area-of-union
        object_bbox = [relationship["object"]["x"], relationship["object"]["y"], relationship["object"]["x"] + relationship["object"]["w"], relationship["object"]["y"] + relationship["object"]["h"]]
        subject_bbox = [relationship["subject"]["x"], relationship["subject"]["y"], relationship["subject"]["x"] + relationship["subject"]["w"], relationship["subject"]["y"] + relationship["subject"]["h"]]
        min_x = min(object_bbox[0], object_bbox[2], subject_bbox[0], subject_bbox[2])
        min_y = min(object_bbox[1], object_bbox[3], subject_bbox[1], subject_bbox[3])
        max_x = max(object_bbox[0], object_bbox[2], subject_bbox[0], subject_bbox[2])
        max_y = max(object_bbox[1], object_bbox[3], subject_bbox[1], subject_bbox[3])

        relationship_img = img.crop((min_x, min_y, max_x, max_y))
        relationship_img = relationship_img.resize((new_h, new_w), resample=0)

        relative_object_bbox = [
            int(round((object_bbox[0] - min_x) * w_ratio)),
            int(round((object_bbox[1] - min_y) * h_ratio)),
            int(round((object_bbox[2] - min_x) * w_ratio)),
            int(round((object_bbox[3] - min_y) * h_ratio))]
        relative_subject_bbox = [
            int(round((subject_bbox[0] - min_x) * w_ratio)),
            int(round((subject_bbox[1] - min_y) * h_ratio)),
            int(round((subject_bbox[2] - min_x) * w_ratio)),
            int(round((subject_bbox[3] - min_y) * h_ratio))]

        predicate = relationship["predicate"]
        relationship_id = relationship["relationship_id"]

        if self.transform:
            relationship_img = self.transform(relationship_img)
        return relationship_img, image_id, predicate, relationship_id, relative_object_bbox, relative_subject_bbox


class BBoxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 51)
        )

    def forward(self, boxes):
        boxes = torch.flatten(boxes)
        logits = self.layers.forward(boxes)
        return logits

def collate_fn(data):
    tensors, image_ids, predicates, relationship_ids, object_bboxes, subject_bboxes = zip(*data)

    features = pad_sequence(tensors, batch_first=True)
    key_predicates = []
    for predicate in predicates:
        key_predicates.append(torch.tensor(predicate_translation(predicate)))
    key_predicates = torch.stack(key_predicates)
    bboxes = []

    assert len(object_bboxes) == len(subject_bboxes), f"Expected bboxes equal but found {len(object_bboxes)} - {len(subject_bboxes)}"

    for j in range(len(object_bboxes)):
        bboxes.append(torch.tensor([object_bboxes[j], subject_bboxes[j]], dtype=torch.float))
    return features, image_ids, key_predicates, relationship_ids, bboxes


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for index, [feature, image_id, predicate, relationship_id, bboxes] in enumerate(training_loader):

        optimizer.zero_grad()
        logits = []
        targets = []
        for x in range(batch_size):
            output = bbox_model(bboxes[x])
            logits.append(output)
            targets.append(predicate)

        logits = torch.stack(logits)
        targets = torch.stack(targets)
        # Compute the loss and its gradients
        predicate = torch.tensor([x - 1 for x in predicate])
        loss = loss_fn(input=logits, target=predicate)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if index % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(index + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + index + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

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


if __name__ == "__main__":
    training_dataset = VisualGenomeDataset(
        "C:/Users/Ali Berkin/sgg_benchmark/datasets/vg/images",
        "C:/Users/Ali Berkin/sgg_benchmark/datasets/vg/",
        ToTensor()
    )

    bbox_model = BBoxModel()
    batch_size = 4
    training_loader = DataLoader(training_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)

    """predicate_classes = {}
    class_numbers = {}
    length = 0
    for index, [feature, image_id, predicate, relationship_id, bboxes] in enumerate(training_loader): # for extracting predicate classes
        for x in range(batch_size):
            if predicate[x].lower() not in list(predicate_classes.values()):
                predicate_classes[f"{length + 1}"] = predicate[x].lower()
                class_numbers[f"{length + 1}"] = 1
                print(index, length, predicate[x].lower())
                length += 1
            else:
                class_numbers[list(predicate_classes.keys())[list(predicate_classes.values()).index(predicate[x].lower())]] += 1
    with open("classes.json", "w") as outfile:
        json.dump(predicate_classes, outfile)
    with open("numbers.json", "w") as outfile:
        json.dump(class_numbers, outfile)"""

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(bbox_model.parameters(), lr=0.001)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        bbox_model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        bbox_model.eval()

        with torch.no_grad():
            for v_index, [v_feature, v_image_id, v_predicate, v_relationship_id, v_bboxes] in enumerate(training_loader):
                optimizer.zero_grad()
                v_logits = []
                v_targets = []
                for i in range(batch_size):
                    v_output = bbox_model(v_bboxes[i])
                    v_logits.append(v_output)
                    v_targets.append(v_predicate)
                v_logits = torch.stack(v_logits)
                v_targets = torch.stack(v_targets)

                # Compute the loss and its gradients
                v_predicate = torch.tensor([i - 1 for i in v_predicate])
                v_loss = loss_fn(input=v_logits, target=v_predicate)
                running_vloss += v_loss

        avg_vloss = running_vloss / (v_index + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(bbox_model.state_dict(), model_path)

        epoch_number += 1








