from PIL import Image, ImageSequence
Image.MAX_IMAGE_PIXELS=None
import os
import fitz
import glob
import shutil

############# convert tif to pdf ##################

def tiff_to_pdf(tiff_path: str) -> str:
 
    pdf_path = tiff_path.replace('.tiff', '.pdf').replace('.tif', '.pdf')
    if not os.path.exists(tiff_path): raise Exception(f'{tiff_path} does not find.')
    image = Image.open(tiff_path)

    images = []
    for i, page in enumerate(ImageSequence.Iterator(image)):
        page = page.convert("RGB")
        images.append(page)
    if len(images) == 1:
        images[0].save(pdf_path)
    else:
        images[0].save(pdf_path, save_all=True,append_images=images[1:])
    os.remove(tiff_path)
    return pdf_path

############# convert pdf to image ################

def pdf_to_image(pdf_file: str) -> str:

    folder = os.path.basename(pdf_file).replace(".pdf",'').replace(".PDF","")
    pdf_path = pdf_file

    try:
        dpi=500
        zoom=dpi / 72
        magnify = fitz.Matrix(zoom,zoom)
        doc = fitz.open(pdf_path)
        
        for page in doc:
            pix = page.get_pixmap(matrix=magnify)
            if not os.path.exists("./images/"+folder):
                os.makedirs("./images/"+folder)
            image_name = os.path.basename(pdf_file).replace('.pdf','').replace('.PDF','')+f"_page{page.number+1}"+".jpg"
            pix.save("./images/"+folder+"/"+os.path.basename(pdf_file).replace('.pdf','').replace('.PDF','')+f"_page{page.number+1}"+".jpg")
    except:
        pass

pdf_folder = glob.glob("./pdfs/*")

# pdf_folder = sorted(pdf_folder, key=lambda x: int(os.path.basename(x).split("_")[-1].replace(".pdf","").replace(".PDF","")\
#                                                   .replace('.jpg','').replace('.png','').replace('.tif','').replace('.tiff','')\
#                                                     .replace('.png','').replace('.jpeg','')))

if not os.path.exists("./images"):
    os.makedirs('./images')

for pdf_file in pdf_folder:
    if pdf_file.endswith('.pdf') or pdf_file.endswith('.PDF'):
        # print("\npdf_file::: ",pdf_file)
        pdf_to_image(pdf_file)
        
    
    elif pdf_file.endswith('.tif') or pdf_file.endswith('.tiff'):
        # print("\npdf_file::: ",pdf_file)
        converted_pdf = tiff_to_pdf(pdf_file)
        pdf_to_image(converted_pdf)

    elif pdf_file.endswith('jpg') or pdf_file.endswith('.png') or pdf_file.endswith('jpeg'):
        # print("\npdf_file::: ",pdf_file)
        folder_name = os.path.basename(pdf_file).replace('.jpg','').replace('.png','').replace('jpeg','')
        if not os.path.exists('./images/'+folder_name):
            os.makedirs('./images/'+folder_name)
        shutil.copy(pdf_file, './images/'+folder_name+'/'+os.path.basename(pdf_file))
        

#################### move to Dataset Folder ########################

images = glob.glob("./images/*/*")
dest_folder = "./Dataset"
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

for imgs in images:
    try:
        shutil.move(imgs,dest_folder)
    except:
        pass

##################### rename files #####################
dest = './Dataset'
all_files = os.listdir(dest)
for idx, file in enumerate(all_files):
    dst = f"image_{str(idx+1)}.jpg"
    src =f"{dest}/{file}"
    dst =f"{dest}/{dst}"
    os.rename(src, dst)

##################### rotate images ####################

image_dir = "./Dataset"

output_dir = "./rotated_dir_another"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for rotation_angle in range(0,360,90):
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")\
            or filename.endswith(".tif") or filename.endswith(".tiff"):
            
            # # Open the image
            image_path = os.path.join(image_dir, filename)
            try:
                image = Image.open(image_path)
                rgb_im = image.convert('RGB')

                # # Rotate the image
                rotated_image = rgb_im.rotate(-rotation_angle, expand=True)

                # # Save the rotated image
                filename = filename.replace(".jpeg",".jpg").replace(".png",'.jpg').replace('tif','jpg').replace('.tiff','jpg')
                output_filename = f"rotated_{str(rotation_angle)}_{filename}"
                output_path = os.path.join(output_dir, output_filename)
                rotated_image.save(output_path)
            except:
                pass

################# remove images folder ##################
try:
    shutil.rmtree('./images')
except:
    pass