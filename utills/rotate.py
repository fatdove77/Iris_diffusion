from PIL import Image
import os

# Define the function to rotate and save images
def rotate_and_save_images(image_dir, output_dir):
    # Check if the output directory exists, create if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # List all the image files in the specified directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    # Process each file
    for filename in image_files:
        # Load the image
        img_path = os.path.join(image_dir, filename)
        img = Image.open(img_path)
        original_size = img.size
        
        # Save the original image
        img.save(os.path.join(output_dir, filename))
        
        # Generate rotated images every 30 degrees and save them
        for i in range(1, 12):  # from 30 to 330 degrees
            angle = i * 30
            rotated_img = img.rotate(angle, expand=True)
            
            # Calculate the center crop box
            rotated_size = rotated_img.size
            left = (rotated_size[0] - original_size[0]) / 2
            top = (rotated_size[1] - original_size[1]) / 2
            right = (rotated_size[0] + original_size[0]) / 2
            bottom = (rotated_size[1] + original_size[1]) / 2
            cropped_img = rotated_img.crop((left, top, right, bottom))
            
            rotated_filename = f"{filename.split('.')[0]}_{angle}.jpg"
            cropped_img.save(os.path.join(output_dir, rotated_filename))

            print("succussfully rotated images")

# Assuming the script is in '/scripts' and images are in '/eyes'
rotate_and_save_images('/home/saturn/eyes/realeyes/left', '/home/saturn/eyes/realeyes/rotate_left')
rotate_and_save_images('/home/saturn/eyes/realeyes/right', '/home/saturn/eyes/realeyes/rotate_right')