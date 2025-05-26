import cv2

IMAGES = [
    "./data/bom_1.jpg",
    "./data/bom_2.jpg",
    "./data/bom_3.jpg",
    "./data/bom_4.jpg",
    "./data/ruim_1.jpg",
    "./data/ruim_2.jpg",
    "./data/ruim_3.jpg",
    "./data/ruim_4.jpg"
]

def apply_sobel(img):
    img_gray = img
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    img_sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return img_sobel