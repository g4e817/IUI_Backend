
import cv2 as cv
import os

def get_most_important_feature():
    
    size =  len(os.listdir("data/cropped_images"))
    counter = 0

    for filename in os.listdir("data/images"):

        if (counter > size):
            image = cv.imread("data/images/" + filename)
            r = cv.selectROI(image)

            croppedImage = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

            cv.imshow(filename, croppedImage)

            cv.imwrite("data/cropped_images/" + filename, croppedImage)

            cv.destroyWindow(filename)
        counter = counter + 1

get_most_important_feature()