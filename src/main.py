import os
import numpy as np
import supervisely as sly
from dotenv import load_dotenv

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

workspace_id = int(os.environ["context.workspaceId"])
workspace = api.workspace.get_info_by_id(workspace_id)
if workspace is None:
    # You should put correct value to local.env for workspaceId
    raise ValueError(f"Workspace (id={workspace_id}) not found.")

# create empty project and dataset
project = api.project.create(workspace.id, name="Demo", change_name_if_conflict=True)
dataset = api.dataset.create(project.id, name="berries")

# upload image to dataset
image_path = "data/berries.jpeg"
image_name = sly.fs.get_file_name_with_ext(image_path)
image_info = api.image.upload_path(dataset.id, image_name, image_path)
print(f"Image has been sucessfully uploaded (id={image_info.id})")

# create label (rectangle) of class "strawberry"
strawberry = sly.ObjClass(name="strawberry", geometry_type=sly.Rectangle)
bbox = sly.Rectangle(top=127, left=1726, bottom=1087, right=2560)
label1 = sly.Label(geometry=bbox, obj_class=strawberry)

# create polygonal label of class "raspberry"
raspberry = sly.ObjClass(name="raspberry", geometry_type=sly.Polygon)


# classes
classes = {
    "strawberry": sly.Rectangle,
    "raspeberry": sly.Polygon,
    "blackberry": sly.Bitmap,
}

# classes coordinates
strawberry_rectangle_coords = [127, 1726, 1087, 2560]
raspberry_polygon_coords = (
    [933, 338],
    [910, 359],
    [877, 354],
    [829, 357],
    [801, 372],
    [780, 390],
    [769, 408],
    [727, 427],
    [701, 450],
    [679, 480],
    [660, 504],
    [636, 526],
    [622, 549],
    [612, 575],
    [613, 603],
    [607, 618],
    [594, 633],
    [587, 656],
    [587, 678],
    [590, 697],
    [585, 719],
    [575, 732],
    [569, 752],
    [571, 777],
    [578, 795],
    [587, 809],
    [600, 818],
    [605, 832],
    [597, 850],
    [606, 858],
    [604, 875],
    [612, 895],
    [631, 917],
    [648, 928],
    [656, 932],
    [676, 933],
    [681, 940],
    [688, 946],
    [702, 948],
    [709, 955],
    [727, 961],
    [756, 967],
    [784, 969],
    [825, 972],
    [858, 970],
    [898, 968],
    [936, 962],
    [982, 955],
    [1004, 956],
    [1045, 940],
    [1071, 935],
    [1099, 917],
    [1125, 896],
    [1145, 869],
    [1151, 855],
    [1159, 844],
    [1174, 835],
    [1187, 823],
    [1199, 798],
    [1205, 769],
    [1214, 741],
    [1221, 707],
    [1198, 664],
    [1176, 641],
    [1166, 617],
    [1143, 589],
    [1116, 576],
    [1100, 567],
    [1083, 541],
    [1067, 533],
    [1070, 523],
    [1082, 512],
    [1090, 487],
    [1092, 454],
    [1076, 421],
    [1051, 402],
    [1036, 390],
    [1024, 365],
    [991, 343],
    [960, 337],
)
blackberry_masks_paths = [
    os.path.join("../data/masks", mask) for mask in os.listdir("../data/masks")
]

# create project and dataset
project_name = "my_test_project"
project = api.project.create(
    workspace_id=WORKSPACE_ID, name=project_name, change_name_if_conflict=True
)
dataset_name = "ds0"
dataset = api.dataset.create(project_id=project.id, name=dataset_name)

# create project_meta with classes
## create obj classes
strawberrry_objclass = sly.ObjClass(name="strawberry", geometry_type=sly.Rectangle)
raspberry_objclass = sly.ObjClass(name="raspebrry", geometry_type=sly.Polygon)
blackberry_objclass = sly.ObjClass(name="blackberry", geometry_type=sly.Bitmap)

## create project_meta and update project on server
project_meta = sly.ProjectMeta(
    obj_classes=[strawberrry_objclass, raspberry_objclass, blackberry_objclass]
)
api.project.update_meta(id=project.id, meta=project_meta.to_json())

# create annotation with labels
## create labels
strawberry_label = sly.Label(
    geometry=sly.Rectangle(*strawberry_rectangle_coords), obj_class=strawberrry_objclass
)

### convert polygon coords to sly.PointLocation
converted_raspberry_polygon_coords = [
    sly.PointLocation(row=coord[1], col=coord[0]) for coord in raspberry_polygon_coords
]
raspberry_label = sly.Label(
    geometry=sly.Polygon(exterior=converted_raspberry_polygon_coords, interior=[]),
    obj_class=raspberry_objclass,
)

### convert bitmap mask data to boolean 2d numpy array
blackberry_bitmaps = []
for mask_path in blackberry_masks_paths:
    mask = sly.image.read(mask_path)
    mask = mask.reshape(mask.shape[0], (mask.shape[1] * mask.shape[2]))
    bitmap = sly.Bitmap(np.array(mask[:, :, 0] / 255, dtype=bool))
    blackberry_bitmaps.append(bitmap)

blackberry_label_1 = sly.Label(
    geometry=blackberry_bitmaps[0], obj_class=blackberry_objclass
)
blackberry_label_2 = sly.Label(
    geometry=blackberry_bitmaps[1], obj_class=blackberry_objclass
)
blackberry_label_3 = sly.Label(
    geometry=blackberry_bitmaps[2], obj_class=blackberry_objclass
)

labels = [
    strawberry_label,
    raspberry_label,
    blackberry_label_1,
    blackberry_label_2,
    blackberry_label_3,
]

## create annotation
image = sly.image.read(img_path)
image_name = get_file_name_with_ext(img_path)
ann = sly.Annotation(img_size=image.shape, labels=labels)

image_info = api.image.upload_np(dataset_id=dataset.id, name=image_name, img=image)
api.annotation.upload_ann(img_id=image_info.id, ann=ann)
workspace_id = int(os.environ["context.workspaceId"])
workspace = api.workspace.get_info_by_id(workspace_id)
if workspace is None:
    raise ValueError(
        f"Workspace (id={workspace_id}) not found. Put correct value to local.env"
    )
