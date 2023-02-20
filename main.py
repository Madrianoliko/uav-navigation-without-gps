import cv2
from compareImages import compareImagesMethod

path_main = "sectorMap21.png"
largeMapImage = cv2.imread(path_main, 0)
cap = cv2.VideoCapture('videoFromUav2.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # wyświetlanie okna obrazu z drona
        resized_image = cv2.resize(frame, (900, 500))
        cv2.imshow('Frame', resized_image)

        # porównanie obrazów
        smallMapImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        topleft, bottomright = compareImagesMethod(largeMapImage, smallMapImage)

        # wyświetlanie okna z główną mapą i rysowanie pozycji drona
        imgcv2 = cv2.imread(path_main)
        cv2.rectangle(imgcv2, topleft, bottomright, (255, 0, 0), 5)
        resized_image2 = cv2.resize(imgcv2, (900, 500))
        cv2.imshow('image', resized_image2)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
