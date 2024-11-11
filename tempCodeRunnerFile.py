import cv2
cv2.putText  (frame,str(gaze_ratio),(50,100)font,2(0,0,255),3)#left_side_white
    #     #cv2.putText(frame,str(right_side_white),(50,150)font,2(0,0,255),3)


    #     threshold_eye=cv2.resize(threshold_eye,None,fx=5,fy=5)
    #     eye=cv2.resize(gray_eye,None,fx=5,fy=5)
    #     #cv2.imshow("Eye",eye)
    #     cv2.imshow("Threshold",threshold_eye)
    #     cv2.imshow("left",left_side_treshold)
    #     cv2.imshow("right",right_side_threshold)
    #    # cv2.imshow("Left Eye",left_eye)