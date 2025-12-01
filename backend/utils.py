import cv2
import numpy as np

import pypdfium2 as pdfium

def resize_image(image, width=None, height=None):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def pdf_to_images(pdf_path):
    pdf = pdfium.PdfDocument(pdf_path)
    images = []
    for i in range(len(pdf)):
        page = pdf[i]
        image = page.render(scale=4).to_pil() # Render to PIL image
        # Convert to OpenCV format
        open_cv_image = np.array(image) 
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        images.append(open_cv_image)
    pdf.close()
    return images
