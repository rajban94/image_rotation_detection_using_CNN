from PIL import Image
import os

image_dir = "./Dataset"

output_dir = "./rotated_dir_another"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for rotation_angle in range(0,360,90):
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            
            # # Open the image
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path)
            rgb_im = image.convert('RGB')

            # # Rotate the image
            rotated_image = rgb_im.rotate(-rotation_angle, expand=True)

            # # Save the rotated image
            filename = filename.replace(".jpeg",".jpg").replace(".png",'.jpg')
            output_filename = f"rotated_{str(rotation_angle)}_{filename}"
            output_path = os.path.join(output_dir, output_filename)
            rotated_image.save(output_path)