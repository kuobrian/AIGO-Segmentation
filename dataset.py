import os
import cv2
import numpy as np
import torch
import imgaug
import json
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from sklearn.model_selection import train_test_split
from collections import namedtuple

Label = namedtuple('Label', [
                            'name'        , 
                            'id'          ,
                            'trainId'     , 
                            'hasInstanceignoreInEval'       ,
                            'color'       ,
                            ])

Mapillary_labels = [
            #name                    clsId  trainId hasInstanceignoreInEval  color
    Label("Bird"                   ,    0,     0,   True,    [165, 42, 42]),
    Label("Ground Animal"          ,    1,     1,   True,    [0, 192, 0]),
    Label("Curb"                   ,    2,     2,   False,    [196, 196, 196]),
    Label("Fence"                  ,    3,     3,   False,    [190, 153, 153]),
    Label("Guard Rail"             ,    4,     4,   False,    [180, 165, 180]),
    Label("Barrier"                ,    5,     5,   False,    [90, 120, 150]),
    Label("Wall"                   ,    6,     6,   False,    [102, 102, 156]),
    Label("Bike Lane"              ,    7,     7,   False,    [128, 64, 255]),
    Label("Crosswalk - Plain"      ,    8,     8,   True,    [140, 140, 200]),
    Label("Curb Cut"               ,    9,     9,   False,    [170, 170, 170]),
    Label("Parking"                ,   10,    10,   False,    [250, 170, 160]),
    Label("Pedestrian Area"        ,   11,    11,   False,    [96, 96, 96]),
    Label("Rail Track"             ,   12,    12,   False,    [230, 150, 140]),
    Label("Road"                   ,   13,    13,   False,    [128, 64, 128]),
    Label("Service Lane"           ,   14,    14,   False,    [110, 110, 110]),
    Label("Sidewalk"               ,   15,    15,   False,    [244, 35, 232]),
    Label("Bridge"                 ,   16,    16,   False,    [150, 100, 100]),
    Label("Building"               ,   17,    17,   False,    [70, 70, 70]),
    Label("Tunnel"                 ,   18,    18,   False,    [150, 120, 90]),
    Label("Person"                 ,   19,    19,   True,    [220, 20, 60]),
    Label("Bicyclist"              ,   20,    20,   True,    [255, 0, 0]),
    Label("Motorcyclist"           ,   21,    21,   True,    [255, 0, 100]),
    Label("Other Rider"            ,   22,    22,   True,    [255, 0, 200]),
    Label("Lane Marking - Crosswalk",   23,    23,   True,    [200, 128, 128]),
    Label("Lane Marking - General" ,   24,    24,   False,    [255, 255, 255]),
    Label("Mountain"               ,   25,    25,   False,    [64, 170, 64]),
    Label("Sand"                   ,   26,    26,   False,    [230, 160, 50]),
    Label("Sky"                    ,   27,    27,   False,    [70, 130, 180]),
    Label("Snow"                   ,   28,    28,   False,    [190, 255, 255]),
    Label("Terrain"                ,   29,    29,   False,    [152, 251, 152]),
    Label("Vegetation"             ,   30,    30,   False,    [107, 142, 35]),
    Label("Water"                  ,   31,    31,   False,    [0, 170, 30]),
    Label("Banner"                 ,   32,    32,   True,    [255, 255, 128]),
    Label("Bench"                  ,   33,    33,   True,    [250, 0, 30]),
    Label("Bike Rack"              ,   34,    34,   True,    [100, 140, 180]),
    Label("Billboard"              ,   35,    35,   True,    [220, 220, 220]),
    Label("Catch Basin"            ,   36,    36,   True,    [220, 128, 128]),
    Label("CCTV Camera"            ,   37,    37,   True,    [222, 40, 40]),
    Label("Fire Hydrant"           ,   38,    38,   True,    [100, 170, 30]),
    Label("Junction Box"           ,   39,    39,   True,    [40, 40, 40]),
    Label("Mailbox"                ,   40,    40,   True,    [33, 33, 33]),
    Label("Manhole"                ,   41,    41,   True,    [100, 128, 160]),
    Label("Phone Booth"            ,   42,    42,   True,    [142, 0, 0]),
    Label("Pothole"                ,   43,    43,   False,    [70, 100, 150]),
    Label("Street Light"           ,   44,    44,   True,    [210, 170, 100]),
    Label("Pole"                   ,   45,    45,   True,    [153, 153, 153]),
    Label("Traffic Sign Frame"     ,   46,    46,   True,    [128, 128, 128]),
    Label("Utility Pole"           ,   47,    47,   True,    [0, 0, 80]),
    Label("Traffic Light"          ,   48,    48,   True,    [250, 170, 30]),
    Label("Traffic Sign (Back)"    ,   49,    49,   True,    [192, 192, 192]),
    Label("Traffic Sign (Front)"   ,   50,    50,   True,    [220, 220, 0]),
    Label("Trash Can"              ,   51,    51,   True,    [140, 140, 20]),
    Label("Bicycle"                ,   52,    52,   True,    [119, 11, 32]),
    Label("Boat"                   ,   53,    53,   True,    [150, 0, 255]),
    Label("Bus"                    ,   54,    54,   True,    [0, 60, 100]),
    Label("Car"                    ,   55,    55,   True,    [0, 0, 142]),
    Label("Caravan"                ,   56,    56,   True,    [0, 0, 90]),
    Label("Motorcycle"             ,   57,    57,   True,    [0, 0, 230]),
    Label("On Rails"               ,   58,    58,   False,    [0, 80, 100]),
    Label("Other Vehicle"          ,   59,    59,   True,    [128, 64, 64]),
    Label("Trailer"                ,   60,    60,   True,    [0, 0, 110]),
    Label("Truck"                  ,   61,    61,   True,    [0, 0, 70]),
    Label("Wheeled Slow"           ,   62,    62,   True,    [0, 0, 192]),
    Label("Car Mount"              ,   63,    63,   False,    [32, 32, 32]),
    Label("Ego Vehicle"            ,   64,    64,   False,    [120, 10, 10]),
    Label("Unlabeled"              ,   65,    65,   False,    [0, 0, 0])]

class MapillaryDataset(Dataset):
    def __init__(self, dataset, seq=None):
        self.dataset = dataset
        self.img_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                        std=(0.229, 0.224, 0.225))
                                                ])
        self.seq = seq

    def _load_data(self, index):
        img_path = self.dataset[index]["img_path"]
        label_path = self.dataset[index]["instance_path"]
        img = cv2.imread(img_path, -1)
        label = cv2.imread(label_path, -1)
        label = np.array(label/256., dtype=np.uint8)
        return img, label
    
    def  __getitem__(self ,index):
        img, label = self._load_data(index)
        seg = SegmentationMapsOnImage(label, shape=img.shape)
        if self.seq:
            img, seg = self.seq(image=img, segmentation_maps=seg)
        label = seg.get_arr()

        img = img/255.0
        img = img.astype(np.float32)
        img = self.img_transform(img)
        
        label = torch.from_numpy(label)
        return img, label

    def get_num_cls(self):
        return len(Mapillary_labels)
    
    def __len__(self):
        return len(self.dataset)


def MakeTrainValidSet(data_dir):
    with open(data_dir + '/config.json') as config_file:
        config = json.load(config_file)
    labels = config['labels']

    folder_list = ["training", "validation", "testing"]
    for folder in folder_list:
        Image_folder = os.path.join(data_dir, folder, "images")
        if not os.path.exists("./Mapillary_{}.json".format(folder)):
            dataset = []
            for root, dirs, files in os.walk(Image_folder):         
                for img_file in files:
                    img_path = os.path.join(Image_folder, img_file)
                    label_path = img_path.replace("images", "labels")
                    instance_path = img_path.replace("images", "instances")
                    panoptic_path = img_path.replace("images", "panoptic")

                    label_path = os.path.splitext(label_path)[0] + ".png"
                    instance_path = os.path.splitext(instance_path)[0] + ".png"
                    panoptic_path = os.path.splitext(panoptic_path)[0] + ".png"
                    dataset.append({"img_path": img_path,
                                    "label_path": label_path,
                                    "instance_path": instance_path,
                                    "panoptic_path": panoptic_path,})

            with open("./Mapillary_{}.json".format(folder), "w") as fp:
                json.dump(dataset, fp, indent=4)
        
def ComputeClassWeigths(trainset):
    
    ################################################################################
    # compute the class weigths:
    ################################################################################


    num_classes = train_dataset.get_num_cls()
    print ("computing {} class weights".format(num_classes))

    trainId_to_count = {}
    for trainId in range(num_classes):
        trainId_to_count[trainId] = 0

    # get the total number of pixels in all train label_imgs that are of each object class:
    for step, pair_img in enumerate(trainset):
        if step % 100 == 0:
            print (step)

        label_img = cv2.imread(pair_img["instance_path"], -1)
        label_img = np.array(label_img/256, dtype=np.uint8)
        
        for trainId in range(num_classes):
            # count how many pixels in label_img which are of object class trainId:
            trainId_mask = np.equal(label_img, trainId)
            trainId_count = np.sum(trainId_mask)

            # add to the total count:
            trainId_to_count[trainId] += trainId_count

    # compute the class weights according to the ENet paper:
    class_weights = []
    total_count = sum(trainId_to_count.values())
    for trainId, count in trainId_to_count.items():
        trainId_prob = float(count)/float(total_count)
        trainId_weight = 1/np.log(1.02 + trainId_prob)
        class_weights.append(trainId_weight)

    print (class_weights)

    with open("./{}_ClassWeights_Mapillary.pkl".format(num_classes), "wb") as f:
        pickle.dump(class_weights, f)

def ColorLabels(labels, path):
    num_color = len(labels)
    gap = 40
    color_label = np.ones((num_color*gap+20 , 800, 3), dtype=np.uint8) * 255
    for i, label in enumerate(labels):
        # count = i + 1
        color = label.color
        print(type(color))

        if isinstance(color, int):
            r =  color // (256*256)
            g = (color-256*256*r) // 256
            b = (color-256*256*r-256*g)
            color = (r, g, b)
        name = label.name
        print(i, color, name, (i)*gap , 600)
        print(color_label[i:i+200, :500, :].shape)
        start = i*gap
        end = start + gap - 10
        color_label[ start: end , :500, :] = color

        cv2.putText(color_label, name, (550, start+20),   cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        start = end
    cv2.imshow("vis_label", color_label)
    cv2.imwrite(path, color_label)
    cv2.waitKey(0)

def DEMO_Mapillary(data_dir):
    # a nice example
    image_id = 'M2kh294N9c72sICO990Uew'

    # read in config file
    with open(data_dir + '/config.json') as config_file:
        config = json.load(config_file)
    labels = config['labels']
    # print labels
    print("There are {} labels in the config file".format(len(labels)))
    for label_id, label in enumerate(labels):
        print("{:>30} ({:2d}): {:<40} has instances: {}".format(label["readable"], label_id, label["name"], label["instances"]))

    # set up paths for every image
    image_path = data_dir + "/training/images/{}.jpg".format(image_id)
    label_path = data_dir + "/training/labels/{}.png".format(image_id)
    instance_path = data_dir + "/training/instances/{}.png".format(image_id)
    panoptic_path = data_dir + "/training/panoptic/{}.png".format(image_id)

    # load images
    from PIL import Image
    import matplotlib.pyplot as plt
    base_image = Image.open(image_path)
    label_image = Image.open(label_path)
    instance_image = Image.open(instance_path)
    panoptic_image = Image.open(panoptic_path)



    # convert labeled data to numpy arrays for better handling
    label_array = np.array(label_image)
    instance_array = np.array(instance_image, dtype=np.uint16)

    # now we split the instance_array into labels and instance ids
    instance_label_array = np.array(instance_array / 256, dtype=np.uint8)
    instance_ids_array = np.array(instance_array % 256, dtype=np.uint8)

    # for visualization, we apply the colors stored in the config
    colored_label_array = apply_color_map(label_array, labels)
    colored_instance_label_array = apply_color_map(instance_label_array, labels)


    # plot the result
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20,15))

    ax[0][0].imshow(base_image)
    ax[0][0].get_xaxis().set_visible(False)
    ax[0][0].get_yaxis().set_visible(False)
    ax[0][0].set_title("Base image")
    ax[0][1].imshow(colored_label_array)
    ax[0][1].get_xaxis().set_visible(False)
    ax[0][1].get_yaxis().set_visible(False)
    ax[0][1].set_title("Labels")
    ax[1][0].imshow(instance_ids_array)
    ax[1][0].get_xaxis().set_visible(False)
    ax[1][0].get_yaxis().set_visible(False)
    ax[1][0].set_title("Instance IDs")
    ax[1][1].imshow(colored_instance_label_array)
    ax[1][1].get_xaxis().set_visible(False)
    ax[1][1].get_yaxis().set_visible(False)
    ax[1][1].set_title("Labels from instance file (identical to labels above)")
    ax[2][0].imshow(panoptic_image)
    ax[2][0].get_xaxis().set_visible(False)
    ax[2][0].get_yaxis().set_visible(False)
    ax[2][0].set_title("Labels from panoptic")
    ax[2][1].axis('off')

    fig.tight_layout()

    fig.savefig('MVD_plot.png')


    # PANOPTIC HANDLING

    # read in panoptic file
    with open(data_dir + "/training/panoptic/panoptic_2018.json") as panoptic_file:
        panoptic = json.load(panoptic_file)

    # convert annotation infos to image_id indexed dictionary
    panoptic_per_image_id = {}
    for annotation in panoptic["annotations"]:
        panoptic_per_image_id[annotation["image_id"]] = annotation

    # convert category infos to category_id indexed dictionary
    panoptic_category_per_id = {}
    for category in panoptic["categories"]:
        panoptic_category_per_id[category["id"]] = category

    # convert segment infos to segment id indexed dictionary
    example_panoptic = panoptic_per_image_id[image_id]
    example_segments = {}
    for segment_info in example_panoptic["segments_info"]:
        example_segments[segment_info["id"]] = segment_info

    print('')
    print('Panoptic segments:')
    print('')
    panoptic_array = np.array(panoptic_image).astype(np.uint32)
    panoptic_id_array = panoptic_array[:,:,0] + (2**8)*panoptic_array[:,:,1] + (2**16)*panoptic_array[:,:,2]
    panoptic_ids_from_image = np.unique(panoptic_id_array)
    for panoptic_id in panoptic_ids_from_image:
        if panoptic_id == 0:
            # void image areas don't have segments
            continue
        segment_info = example_segments[panoptic_id]
        category = panoptic_category_per_id[segment_info["category_id"]]
        print("segment {:8d}: label {:<40}, area {:6d}, bbox {}".format(
            panoptic_id,
            category["supercategory"],
            segment_info["area"],
            segment_info["bbox"],
        ))

        # remove to show every segment is associated with an id and vice versa
        example_segments.pop(panoptic_id)

    # every positive id is associated with a segment
    # after removing every id occuring in the image, the segment dictionary is empty
    assert len(example_segments) == 0

def apply_color_map(image_array, labels):
    color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    for label_id, label in enumerate(labels):
        # set all pixels with the current label to the color of the current label
        color_array[image_array == label_id] = label["color"]

    return color_array


if __name__ == "__main__":
    if True:
        data_dir = "D://mapillary-vistas-dataset"
        
        if not os.path.exists("./Mapillary_training.json"):
            MakeTrainValidSet(data_dir)
        # DEMO_Mapillary(data_dir)
        # ColorLabels(Mapillary_labels, "./ColorLabels_Mapillary.png")
        
        with open("./Mapillary_training.json", "r") as fp:
            trainingSet = json.load(fp)
        train_seq = iaa.Sequential([iaa.size.Resize({"height": 416, "width": 512}),
                                    iaa.Fliplr(0.5)
                                    ])
        train_dataset = MapillaryDataset(trainingSet, seq=train_seq)

        if not os.path.exists("./{}_ClassWeights_Mapillary.pkl".format(train_dataset.get_num_cls())):
            ComputeClassWeigths(trainingSet)

        for i in range(len(train_dataset)):
            img, label_image = train_dataset[i]
            print(np.unique(label_image))
            label_image = label_image.numpy()

            color_array = np.zeros((label_image.shape[0], label_image.shape[1], 3), dtype=np.uint8)

            for idx, label in enumerate(Mapillary_labels):
                if  label.id in np.unique(label_image):
                    print(label.trainId, " => ", label.name)
                    color_array[label_image == label.id] = label.color
            cv2.imshow("ins", color_array[:, :, ::-1])
            cv2.waitKey(0)
            assert(0)
        