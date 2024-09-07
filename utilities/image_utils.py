from PIL import Image, ImageOps

def zoom_out_and_pad(image_path, zoom_factor=0.5, padding=100):
    """Zoom out an image and add padding around it."""
    with Image.open(image_path) as img:
        # Resize image to zoom out
        new_size = (int(img.width * zoom_factor), int(img.height * zoom_factor))
        img = img.resize(new_size, Image.LANCZOS)

        # Add padding around the image
        img_with_padding = ImageOps.expand(img, border=padding, fill='white')

        # Save the new image to a temporary file
        new_image_path = image_path.replace('.', '_zoomed_out.')
        img_with_padding.save(new_image_path)
        return new_image_path