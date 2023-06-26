import json
import os


folder_path = "./rotated_dir_another"
images = []


image_filenames = os.listdir(folder_path)
image_filenames_sorted_list = sorted(image_filenames, key=lambda x: int(x.split("_")[1]), reverse=True)

for image_file in image_filenames_sorted_list:
    image_path = os.path.join(folder_path, image_file)
    file_name = os.path.basename(image_path)
    angle = file_name.split("_")[1]
    image = {"filename": "/content/rotated_dir_another/" + file_name, "angle": angle}
    images.append(image)
    

# Create a dictionary with the list of images
data = {"images": images}

# Write the data dictionary to a JSON file
with open("image_rotation_data_another.json", "w") as json_file:
    json.dump(data, json_file, indent=4)