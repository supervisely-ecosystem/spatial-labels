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
    # You should put correct value to local.env for workspaceId
    raise ValueError(f"Workspace with id={workspace_id} not found")


################################    Part 1    ######################################
###################    create empty project and dataset    #########################
################################    ------    ######################################

# create empty project and dataset
project = api.project.create(workspace.id, name="Demo", change_name_if_conflict=True)
dataset = api.dataset.create(project.id, name="berries")
print(f"Project has been sucessfully created, id={project.id}")

# result project meta with all classes
project_meta = sly.ProjectMeta(obj_classes=[])

################################    Part 2    ######################################
####################    create rectangle, polygon, mask    #########################
######################  on image "data/berries-01.jpeg"   ##########################

# create rectangle label (bbox) of class "strawberry"
strawberry = sly.ObjClass(
    name="strawberry", geometry_type=sly.Rectangle, color=[0, 0, 255]
)
bbox = sly.Rectangle(top=127, left=1726, bottom=1087, right=2560)
label1 = sly.Label(geometry=bbox, obj_class=strawberry)

# create polygon label of class "raspberry"
raspberry = sly.ObjClass(name="raspberry", geometry_type=sly.Polygon, color=[0, 255, 0])
polygon = sly.Polygon(
    exterior=[
        [941, 663],
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
blackberry = sly.ObjClass(
    name="blackberry", geometry_type=sly.Bitmap, color=[255, 255, 0]
)
labels_masks = []
for mask_path in [
    "data/masks/blackberry_01.png",
    "data/masks/blackberry_02.png",
    "data/masks/blackberry_03.png",
]:
    # read only first channel of image
    image_bw = cv2.imread(mask_path)
    # image_bw has only values 0 (black) and 255 (white)
    image_bool = np.array(image_bw / 255, dtype=bool)
    # image_bool has only values False (black) and True (white)
    mask = sly.Bitmap(image_bool)
    label = sly.Label(geometry=mask, obj_class=blackberry)
    labels_masks.append(label)


image_path = "data/berries-01.jpeg"
# get dimensions of image
height, width = cv2.imread(image_path).shape[0:2]

# result image annotation
all_labels = [label1, label2]  # rectangle and polygon
all_labels.extend(labels_masks)  # add three masks to the list
ann = sly.Annotation(img_size=[height, width], labels=all_labels)

# add classes to project before upload annotation
project_meta = project_meta.add_obj_classes([strawberry, raspberry, blackberry])
api.project.update_meta(project.id, project_meta.to_json())

# upload image to the dataset on server
image_name = sly.fs.get_file_name_with_ext(image_path)
image_info = api.image.upload_path(dataset.id, image_name, image_path)
print(f"Image has been sucessfully uploaded, id={image_info.id}")

# upload annotation to the image on server
api.annotation.upload_ann(image_info.id, ann)
print(f"Annotation has been sucessfully uploaded")

################################    Part 3    ######################################
#######################      create point, polyline      ###########################
######################  on image "data/berries-02.jpeg"   ##########################

# create point
center = sly.ObjClass(name="center", geometry_type=sly.Point)
point = sly.Point(row=320, col=1302)
label_point = sly.Label(geometry=point, obj_class=center)

# create polyline
separator = sly.ObjClass(name="separator", geometry_type=sly.Point)
polyline = sly.Polyline(
    [[443, 883], [803, 1360], [1372, 1395], [1676, 928], [1372, 458], [554, 552]]
)
label_line = sly.Label(geometry=polyline, obj_class=separator)

image_path = "data/berries-02.jpeg"
# get dimensions of image
height, width = cv2.imread(image_path).shape[0:2]

# result image annotation
ann = sly.Annotation(img_size=[height, width], labels=[label_point, label_line])

# add classes to project before upload annotation
project_meta = project_meta.add_obj_classes([center, separator])
api.project.update_meta(project.id, project_meta.to_json())

# upload image to the dataset on server
image_name = sly.fs.get_file_name_with_ext(image_path)
image_info = api.image.upload_path(dataset.id, image_name, image_path)
print(f"Image has been sucessfully uploaded, id={image_info.id}")

# upload annotation to the image on server
api.annotation.upload_ann(image_info.id, ann)
print(f"Annotation has been sucessfully uploaded")
