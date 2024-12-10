import os
import json
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# Paths
base_path = r"c:\Users\Sean\Desktop\TrafficLights\train_dataset"
json_path = os.path.join(base_path, "train.json")
images_folder = os.path.join(base_path, "train_images")
dataset_dir = r"c:\Users\Sean\Desktop\TrafficLights\yolo_dataset"

# Create directories
os.makedirs(os.path.join(dataset_dir, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'labels', 'val'), exist_ok=True)

# Load annotations
with open(json_path, 'r') as f:
    data = json.load(f)

annotations = data["annotations"]
filenames = [ann["filename"].split("\\")[-1] for ann in annotations]

# Split filenames
train_files, val_files = train_test_split(filenames, test_size=0.2, random_state=42)

# Mapping of colors to class IDs
class_map = {"red": 0, "green": 1, "yellow": 2}

def convert_annotation(annotation, img_width, img_height):
    yolo_annotations = []
    for inbox in annotation["inbox"]:
        color = inbox["color"]
        class_id = class_map[color]

        bndbox = annotation["bndbox"]
        xmin = bndbox["xmin"]
        ymin = bndbox["ymin"]
        xmax = bndbox["xmax"]
        ymax = bndbox["ymax"]

        # Calculate normalized coordinates
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    return yolo_annotations

for annotation in annotations:
    filename = annotation["filename"].split("\\")[-1]
    file_path = os.path.join(images_folder, filename)

    # Open image to get dimensions
    with Image.open(file_path) as img:
        img_width, img_height = img.size

    # Convert annotations
    yolo_anns = convert_annotation(annotation, img_width, img_height)

    # Determine if the image is in training or validation set
    if filename in train_files:
        image_dest = os.path.join(dataset_dir, 'images', 'train', filename)
        label_dest = os.path.join(dataset_dir, 'labels', 'train', os.path.splitext(filename)[0] + '.txt')
    else:
        image_dest = os.path.join(dataset_dir, 'images', 'val', filename)
        label_dest = os.path.join(dataset_dir, 'labels', 'val', os.path.splitext(filename)[0] + '.txt')

    # Copy image
    shutil.copyfile(file_path, image_dest)

    # Write annotations to label file
    with open(label_dest, 'w') as f:
        f.write('\n'.join(yolo_anns))

# Create data.yaml
data_yaml = """
path: {}
train: images/train
val: images/val

names:
  0: red
  1: green
  2: yellow
""".format(dataset_dir.replace('\\', '/'))  # Safely replace backslashes


with open(os.path.join(dataset_dir, 'data.yaml'), 'w') as f:
    f.write(data_yaml)

# Train the model
model = YOLO('yolo11n.pt')  

results = model.train(
    data=os.path.join(dataset_dir, 'data.yaml'),
    epochs=4, #Return to 50 (20 took 2.5hours)
    batch=8,
    imgsz=416,
    name='traffic_light_detection',
    project=r'c:\Users\Sean\Desktop\TrafficLights\training',
    device='cpu'  # Set to 'cpu' if you don't have a GPU
)

# Save the model path
model_path = os.path.join(
    r'c:\Users\Sean\Desktop\TrafficLights\training',
    'traffic_light_detection',
    'weights',
    'best.pt'
)
print(f"Model training complete and saved to {model_path}.")

# Load the trained model
model = YOLO(model_path)

# Perform inference on a new image
results = model.predict(
    source=r'c:\Users\Sean\Desktop\TrafficLights\test_dataset\test_images',  # Replace with your image path
    save=True,                       # Save the inference results
    imgsz=416,
    conf=0.25
)

# Display the results
import cv2
from matplotlib import pyplot as plt

for result in results:
    annotated_frame = result.plot()

    # Convert BGR to RGB
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    plt.imshow(annotated_frame)
    plt.axis('off')
    plt.show()
