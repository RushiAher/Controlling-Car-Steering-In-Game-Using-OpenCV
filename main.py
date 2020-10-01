import cv2
import numpy as np
import pyautogui


# Capturing video
cap = cv2.VideoCapture(0)

# creating trackbars
cv2.namedWindow("tracking")
cv2.createTrackbar("L_H", "tracking",0,255,nothing)
cv2.createTrackbar("L_S", "tracking",0,255,nothing)
cv2.createTrackbar("L_V", "tracking",0,255,nothing)
cv2.createTrackbar("U_H", "tracking",255,255,nothing)
cv2.createTrackbar("U_S", "tracking",255,255,nothing)
cv2.createTrackbar("U_V", "tracking",255,255,nothing)

# Checking if camera is open or not
while cap.isOpened():
    # Capturing frame
    ret, frame = cap.read()

    # filping the frame
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, "Direction:", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, [0, 255, 0], 2)

    # Drawing rectangle on frame
    cv2.rectangle(frame, (150, 150), (450, 450), (45, 255, 120), 0)

    # Croping rectangle on frame
    crop_image = frame[150:450, 150:450]

    # BGR to HSV conversion
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # getting trackbar values
    l_H = cv2.getTrackbarPos("L_H", "tracking")
    l_S = cv2.getTrackbarPos("L_S", "tracking")
    l_V = cv2.getTrackbarPos("L_V", "tracking")

    u_H = cv2.getTrackbarPos("U_H", "tracking")
    u_S = cv2.getTrackbarPos("U_S", "tracking")
    u_V = cv2.getTrackbarPos("U_V", "tracking")

    l_b = np.array([l_H,l_S,l_V])
   
    u_b = np.array([u_H,u_S,u_V])
   
    # creating mask 
    mask = cv2.inRange(hsv, l_b, u_b)

    # Kernel for morphological transformation
    kernel = np.ones((5, 5))

    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask, kernel, iterations=3)
  
    erosion = cv2.erode(dilation, kernel, iterations=1)
   
    # Apply Gaussian Blur and Threshold
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)
  
    # finding contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    


    try:

        contour = max(contours, key=lambda x: cv2.contourArea(x))
        for contour in contours:
            pyautogui.PAUSE = 0.08
            
            # calculating centroid of rectangle
            M = cv2.moments(contour)
            Cx = int(M['m10'] / M['m00'])
            Cy = int(M['m01'] / M['m00'])

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Contour area less than 800 should ignore
            if cv2.contourArea(contour) < 800:

                continue
            # drawing contours on crop_image
            cv2.drawContours(crop_image, [box], 0, (0, 0, 255), 2)
            cv2.circle(crop_image, (Cx, Cy), 3, [0, 0, 255], -1)

            # condition to check whether to turn left or right
            if (box[0,0]-box[2,0])<0 and abs(box[0,0]-box[2,0])<90 and abs(box[0,0]-box[2,0])>40:
                cv2.putText(frame, "<--", (300, 40), cv2.FONT_HERSHEY_COMPLEX, 1, [0, 255, 0], 2)
                pyautogui.keyDown('left')


            elif (box[2,0]-box[0,0])<0 and abs(box[2,0]-box[0,0])<110 and abs(box[2,0]-box[0,0])>60:
                cv2.putText(frame,"-->",(300,40),cv2.FONT_HERSHEY_COMPLEX,1,[0,255,0],2)
                pyautogui.keyDown('right')
               

            else:
                cv2.putText(frame, "--", (300, 40), cv2.FONT_HERSHEY_COMPLEX, 1, [0, 255, 0], 2)
                pyautogui.keyUp('left')
                pyautogui.keyUp('right')
      
    except:
        pass

    
    # showing frame
    cv2.imshow("Frame",frame)
    

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()