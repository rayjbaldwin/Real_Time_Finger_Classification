import cv2
import time

capture = cv2.VideoCapture(0)

while True:
    status, photo = capture.read()
    cv2.imshow("This is the image", photo)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('a'):
        cv2.waitKey(2000)
        cv2.imwrite("images/image_3_{}.jpg".format(time.time()), photo)
        print("Saving the image")

capture.release()
cv2.destroyAllWindows()
