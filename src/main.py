import os
import numpy as np
import cv2
import supervisely as sly
from dotenv import load_dotenv

# init api for uploading data to server
load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api.from_env()

# check the workspace exists
workspace_id = int(os.environ["context.workspaceId"])
workspace = api.workspace.get_info_by_id(workspace_id)
if workspace is None:
    print("you should put correct workspaceId value to local.env")
    raise ValueError(f"Workspace with id={workspace_id} not found")

################################    Part 1    ######################################
###################    create empty project and dataset    #########################
################################    ------    ######################################

# create empty project and dataset on server
project = api.project.create(workspace.id, name="Demo", change_name_if_conflict=True)
dataset = api.dataset.create(project.id, name="berries")
print(f"Project has been sucessfully created, id={project.id}")

# create classes
strawberry = sly.ObjClass("strawberry", sly.Rectangle, color=[0, 0, 255])
raspberry = sly.ObjClass("raspberry", sly.Polygon, color=[0, 255, 0])
blackberry = sly.ObjClass("blackberry", sly.Bitmap, color=[255, 255, 0])
berry_center = sly.ObjClass("berry_center", sly.Point, color=[0, 255, 255])
separator = sly.ObjClass("separator", sly.Polyline)  # color will be generated randomly

# create project meta with all classes and upload them to server
project_meta = sly.ProjectMeta(
    obj_classes=[strawberry, raspberry, blackberry, berry_center, separator]
)
api.project.update_meta(project.id, project_meta.to_json())

################################    Part 2    ######################################
####################    create rectangle, polygon, mask    #########################
######################  on image "data/berries-01.jpg"   ##########################

# create rectangle label (bbox) of class "strawberry"
bbox = sly.Rectangle(top=127, left=1726, bottom=1087, right=2560)
label1 = sly.Label(geometry=bbox, obj_class=strawberry)

# create polygon label of class "raspberry"
polygon = sly.Polygon(
    exterior=[
        [941, 663],  # row, col
        [976, 874],
        [934, 1096],
        [819, 1196],
        [698, 1228],
        [527, 1081],
        [439, 1090],
        [331, 980],
        [359, 808],
        [452, 698],
        [549, 612],
        [762, 564],
        [879, 605],
    ]
)
label2 = sly.Label(geometry=polygon, obj_class=raspberry)

# create masks(sly.Bitmap) labels of class "blackberry"
labels_masks = []
for mask_path in [
    "data/masks/blackberry_01.png",
    "data/masks/blackberry_02.png",
    "data/masks/blackberry_03.png",
]:
    # read only first channel of image
    image_black_and_white = cv2.imread(mask_path)[:, :, 0]
    # supports masks with values (0, 1) or (0, 255) or (False, True)
    mask = sly.Bitmap(image_black_and_white)
    label = sly.Label(geometry=mask, obj_class=blackberry)
    labels_masks.append(label)

image_path = "data/berries-01.jpg"
height, width = cv2.imread(image_path).shape[0:2]

# result image annotation
all_labels = [label1, label2]  # rectangle and polygon
all_labels.extend(labels_masks)  # add three masks to the list
ann = sly.Annotation(img_size=[height, width], labels=all_labels)

# upload image to the dataset on server
image_name = sly.fs.get_file_name_with_ext(image_path)
image_info = api.image.upload_path(dataset.id, image_name, image_path)
print(f"Image has been sucessfully uploaded, id={image_info.id}")

# upload annotation to the image on server
api.annotation.upload_ann(image_info.id, ann)
print(f"Annotation has been sucessfully uploaded to the image {image_name}")

################################    Part 3    ######################################
#######################      create point, polyline      ###########################
######################  on image "data/berries-02.jpg"   ##########################

# create points
labels_points = []
for [row, col] in [
    [1313, 313],
    [1714, 1061],
    [1318, 1851],
    [554, 1912],
    [190, 808],
    [941, 1094],
]:
    point = sly.Point(row, col)
    label = sly.Label(geometry=point, obj_class=berry_center)
    labels_points.append(label)

# create polyline
polyline = sly.Polyline(
    [[883, 443], [1360, 803], [1395, 1372], [928, 1676], [458, 1372], [552, 554]]
)
label_line = sly.Label(geometry=polyline, obj_class=separator)

image_path = "data/berries-02.jpg"
height, width = cv2.imread(image_path).shape[0:2]

# result image annotation
ann = sly.Annotation(img_size=[height, width], labels=[*labels_points, label_line])

# upload image to the dataset on server
image_name = sly.fs.get_file_name_with_ext(image_path)
image_info = api.image.upload_path(dataset.id, image_name, image_path)
print(f"Image has been sucessfully uploaded, id={image_info.id}")

# upload annotation to the image on server
api.annotation.upload_ann(image_info.id, ann)
print(f"Annotation has been sucessfully uploaded to the image {image_name}")
