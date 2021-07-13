import cv2
import time
import numpy as np

# to save the output file in output.avi
fourcc = cv2.VideoWriter_fourcc(*'XIVD')
output_file =  cv2.VideoWriter("Output.avi",fourcc,20.0,(640,480))
# allowing the webcam to start by making the code sleep for 2 seconds
time.sleep(2)
bg = 0
cap = cv2.VideoCapture(0)
# capturing the background for 60 frames
for i in range(60):
    ret,bg = cap.read()

bg = np.flip(bg,axis = 1)

# reading the captured frame untill the  camera is open
while(cap.isOpened()):
    ret,img = cap.read()

    if not ret:
        break
    # fliping the image
    img = np.flip(img,axis = 1)
    # converting the color frm bgr to hsv
    # Hue Saturation Value
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # generating the mask to detect red color
    lower_red = np.array([0,120,50])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv,lower_red,upper_red)
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)
    mask1 = mask1+mask2
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    # seleting only the part that does not have mask1 and saving it in mask2
    mask2 = cv2.bitwise_not(mask1)
    resolution1 = cv2.bitwise_and(img,img,mask = mask2)
    resolution2 = cv2.bitwise_and(bg,bg,mask = mask1)

    final_output = cv2.addWeighted(resolution1,1,resolution2,1,0)
    output_file.write(final_output)
    cv2.imshow("Magic",final_output)
    cv2.waitKey(1)

cap.release()
# out.release()
cv2.destroyAllWindows()
