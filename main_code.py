''' Vehicle Number Plate Recognition using OCR with Authorized Entry'''

# Libraries Importing
import cv2                      # Computer Vision Library
import time                     # Library for delays
import imutils
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

cam = cv2.VideoCapture(0)       # Camera Access

while True:
    # Camera Reading
    img = cam.read()

    # Preprocessing Stage
    img = cv2.resize(img,(620,480))                  # Image Resize to width = 640 and height = 480
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # BGR to Gray Conversion
    blur = cv2.bilateralFilter(gray, 13, 15, 15)     # Blur out the background(Unnecessary) details where pixel
                                                     # diameter = 13, sigma space = 15 and sigma color = 15
    cv2.imshow('Blur',blur)                                            
    # Edge Detection Algorithm - Canny
    edged = cv2.Canny(blur, 30, 200)                 # Threshold 1 = 30 & Threshold 2 = 200
    cv2.imshow('Edged',edged)
    # Contour Detection & Presenting
    contours = cv2.findContours(edged.copy(),cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea,reverse = True)[:10]
    screencnt = None
    for c in contours:
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.018 * peri, True)
    ##    print(len(approx))
        if len(approx) == 4:
            screencnt = approx
            break

    # Masking Number Plate
    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screencnt],0,255,-1)
    new_image = cv2.bitwise_and(img,img,mask=mask)
    cv2.imshow('contour',new_image)
    
    # Segmentation
    (x,y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx,bottomy) = (np.max(x),np.max(y))
    cropped = gray[topx:bottomx+1, topy:bottomy+1]
    cv2.imshow('characters',cropped)
    
    # Character Recognition - Pytesseract
    text = pytesseract.image_to_string(cropped, config = '--psm 11')
    print("Vehicle Number: ",text)
    print(len(text))
    
    if text == " ‘KL 21S 8086)" or text == "  “21 BH 2345 AA":
        print("Authorized entry")
    else:
        print("Unauthorized Entry")
    

