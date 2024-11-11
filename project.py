import cv2
import dlib
import numpy as np
import pyglet
from math import hypot
import time

# load sounds
sound=pyglet.media.load("sound.wav",streaming=False)
left_sound=pyglet.media.load("left_sound.wav",streaming=False)
right_sound=pyglet.media.load("right.mp3",streaming=False)


cam=0
cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
# detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
pradictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#keybord settings

keybord =np.zeros((600,1000,3),np.uint8)
keys_set_1 = {
    0: "Q", 1: "W", 2: "E", 3: "R", 4: "T",
    5: "A", 6: "S", 7: "D", 8: "F", 9: "G",
    10: "Z", 11: "X", 12: "C", 13: "V", 14: "B",
    # 15: "Y", 16: "U", 17: "I", 18: "O", 19: "P",
    # 20: "H", 21: "J", 22: "K", 23: "L",
    # 24: "N", 25: "M"
}
def letter(letter_index,text,litter_light):
    
    #keys
    if letter_index==0:
        x=0
        y=0
    elif letter_index==1:
        x=200
        y=0
    elif letter_index==2:
        x=400
        y=0
    elif letter_index==3:
        x=600
        y=0
    elif letter_index==4:
        x=800
        y=0
    elif letter_index==5:
        x=0
        y=200
    elif letter_index==6:
        x=200
        y=200
    elif letter_index==7:
        x=400
        y=400
    elif letter_index==8:
        x=600
        y=200
    elif letter_index==9:
        x=800
        y=200
    elif letter_index==10:
        x=0
        y=400
    elif letter_index==11:
        x=200
        y=400
    elif letter_index==12:
        x=400
        y=400
    elif letter_index==13:
        x=600
        y=400
    elif letter_index==14:
        x=800
        y=400



    width=200
    height=200
    th=3#thickness
    if litter_light is True:
        cv2.rectangle(keybord,(x + th , y + th) , ( x + width-th , y+height-th),(255,255,255),-1)
    else:
        cv2.rectangle(keybord,(x+th,y+th),(x+width-th,y+height-th),(255,0,0),th)


    #test settings
    font_letter=cv2.FONT_HERSHEY_PLAIN
     #text="A"
    font_scale=10
    font_th=4
    text_size=cv2.getTextSize(text,font_letter,font_scale,font_th)[0]

    width_text,height_text=text_size[0],text_size[1]
    text_x=int((width-width_text)/2)+x
    text_y=int((height+height_text)/2)+y

    cv2.putText(keybord,text,(text_x,text_y),font_letter,font_scale,(255,0,0),font_th)



def midpoint(p1,p2):
    return (p1.x+p2.x)//2,(p1.y+p2.y)//2

# def get_retio(eye_point,face_landmark):
    
#     ## Draw horigental and vertical line and find the mid point of eye
    
#     left_point=(face_landmark.part(eye_point[0]).x,face_landmark.part(eye_point[0]).y)
#     right_point=(face_landmark.part(eye_point[3]).x,face_landmark.part(eye_point[3]).y)
#     center_top=midpoint(face_landmark.part(eye_point[1]),face_landmark.part(eye_point[2]))
#     center_bottom=midpoint(face_landmark.part(eye_point[5]),face_landmark.part(eye_point[4]))
    
#     #hor_line=cv2.line(s,left_point,right_point,(0,255,0),2)
#     #ver_line=cv2.line(s,center_top,center_bottom,(0,255,0),2)

#     ### Find the length of the eye
#     hor_line_len=hypot((left_point[0]-right_point[0]),(left_point[1]-right_point[1]))
#     ver_line_len=hypot((center_top[0]-center_bottom[0]),(center_top[1]-center_bottom[1]))

#     ## find the eye is bilink or not
#     ratio=hor_line_len/ver_line_len
    
#     return ratio



font=cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points,facial_landmarks ):
    left_point=(facial_landmarks.part(eye_points[0]).x,facial_landmarks.part(eye_points[0]).y)
    right_point=(facial_landmarks.part(eye_points[3]).x,facial_landmarks.part(eye_points[3]).y)
    center_top=midpoint(facial_landmarks.part(eye_points[1]),facial_landmarks.part(eye_points[2]))
    center_bottom=midpoint(facial_landmarks.part(eye_points[5]),facial_landmarks.part(eye_points[4]))

    # hor_line=cv2.line(frame,left_point,right_point,(0,255,0),2)
    
    # ver_line=cv2.line(frame,center_top,center_bottom,(0,255,0),2)


    hor_line_length=hypot((left_point[0]-right_point[0]),(left_point[1]-right_point[1]))
    ver_line_length=hypot((center_top[0]-center_bottom[0]),(center_top[1]-center_bottom[1]))

    if ver_line_length == 0:  # Prevent division by zero
        return 0
    ratio = hor_line_length / ver_line_length
    return ratio

def get_gaze_ratio(eye_points,facial_landmarks):
    # left_eye_region=np.array([(facial_landmarks.part(eye_points[0]).x,facial_landmarks.part(eye_points[0]).y),#
    left_eye_region=np.array([(landamrks.part(eye_points[0]).x,landamrks.part(eye_points[0]).y),
                                  (facial_landmarks.part(eye_points[1]).x,facial_landmarks.part(eye_points[1]).y),
                                  (facial_landmarks.part(eye_points[2]).x,facial_landmarks.part(eye_points[2]).y),
                                  (facial_landmarks.part(eye_points[3]).x,facial_landmarks.part(eye_points[3]).y),
                                  (facial_landmarks.part(eye_points[4]).x,facial_landmarks.part(eye_points[4]).y),
                                  (facial_landmarks.part(eye_points[5]).x,facial_landmarks.part(eye_points[5]).y), np.int32])
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    height,width,_=frame.shape
    mask=np.zeros((height,width),np.uint8)
    cv2.polylines(mask,[left_eye_region],True,255,2)
    cv2.fillPoly(mask,[left_eye_region],255)
    eye=cv2.bitwise_and(gray,gray,mask=mask)
    


    min_x=np.min(left_eye_region[:,0])
    max_x=np.max(left_eye_region[:,0])
    min_y=np.min(left_eye_region[:,1])
    max_y=np.max(left_eye_region[:,1])

    gray_eye=eye[min_y:max_y,min_x:max_x]
     #gray_eye=cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)#
    _,threshold=cv2.threshold(gray_eye,70,255,cv2.THRESH_BINARY_INV)
    height,width=threshold.shape
    left_side_treshold=threshold[0:height,0:int(width/2)]
    left_side_white=cv2.countNonZero(left_side_treshold)
    cv2.putText(frame,str(left_side_white))

    right_side_threshold=threshold[0:height,int(width/2):width]
    right_side_white=cv2.countNonZero(right_side_threshold)

    if left_side_white ==0 :
         gaze_ratio =1
    elif right_side_white==0 :
            gaze_ratio=5
    else :
        gaze_ratio=left_side_white/right_side_white
    return gaze_ratio







def get_gaze_ratio2(eye_points, landamrks):
    left_eye_points = []
    for index in eye_points:
        point = landamrks.part(index)
        left_eye_points.append((point.x, point.y))

    # Now create the NumPy array from the list of tuples
    left_eye_region = np.array(left_eye_points, dtype=np.int32)

    # Continue with the rest of your logic
    # ...


#counters
frames=0
letter_index=0  
text=""

while True:
    _,frame=cap.read()
    frame=cv2.resize(frame,None,fx=0.5,fy=0.5)
    keybord[:]=(0,0,0)
    frames+=1
    new_frame=np.zeros((500,500,3),np.uint8)
    blinking_frame=0

    s=cv2.resize(frame,(1000,600),interpolation = cv2.INTER_AREA)
    gray=cv2.cvtColor(s,cv2.COLOR_BGR2GRAY)
    
    

    faces=detector(gray)
    for face in faces:
        landamrks=pradictor(gray,face)
        ## Draw a rectengle over the face 
        x,y=face.left(),face.top()
        x1,y1=face.right(),face.bottom()
        cv2.rectangle(s,(x,y),(x1,y1),(0,255,0),2)
         

        

        #Detect blinking
        left_eye_ratio=get_blinking_ratio([37,38,39,40,41,42],landamrks)  ## this (36,37,38,39,40,41) numbers are the position landmarks of left eye
        right_eye_ratio=get_blinking_ratio([43,44,45,46,47,48],landamrks)  ## this (42,43,44,45,46,47) numbers are the position landmarks of left eye
        blink_ratio=(left_eye_ratio+right_eye_ratio)/2

        if blink_ratio >5.7:
            cv2.putText(frame ,"BLINKING",(50,150),font,4,(255,0,0),thickness=3)#frame/s

            blinking_frame+=1
            frame-=1
            if blinking_frame==2:
                # text+=active_letter
                sound.play()
                time.sleep(1)
    
        #Gaze detection


        # Collect the points in a list
        left_eye_points = [
            (landamrks.part(36).x, landamrks.part(36).y),
            (landamrks.part(37).x, landamrks.part(37).y),
            (landamrks.part(38).x, landamrks.part(38).y),
            (landamrks.part(39).x, landamrks.part(39).y),
            (landamrks.part(40).x, landamrks.part(40).y),
            (landamrks.part(41).x, landamrks.part(41).y)
        ]

        # Convert the list to a NumPy array
        left_eye_region = np.array(left_eye_points, dtype=np.int32)
        # cv2.polylines(frame,left_eye_region,True,(0,0,255),2)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        height,width,_=frame.shape
        mask=np.zeros((height,width),np.uint8)
        cv2.polylines(mask,[left_eye_region],True,255,2)
        cv2.fillPoly(mask,[left_eye_region],255)
        left_eye=cv2.bitwise_and(gray,gray,mask=mask)

        # gray_eye=left_eye[min_y:max_y,min_x:max_x]
    
        # Calculate min and max coordinates for cropping
        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])

        gray_eye = left_eye[min_y:max_y, min_x:max_x]

        # _,threshold_eye=cv2.threshold(gray_eye,70,255,cv2.THRESH_BINARY)#
        
        if gray_eye is not None and gray_eye.size > 0:
            _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
            
            if threshold_eye is not None:
                height, width = threshold_eye.shape

                left_side_threshold = threshold_eye[0:height, 0:int(width/2)]
                left_side_white = cv2.countNonZero(left_side_threshold)

                right_side_threshold = threshold_eye[0:height, int(width/2):width]
                right_side_white = cv2.countNonZero(right_side_threshold)
            else:
                print("Thresholding failed, threshold_eye is None.")
        # else:
        #     print("gray_eye is empty or invalid.")#
        #gray_eye=cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
        # _,threshold=cv2.threshold(gray_eye,70,255,cv2.THRESH_BINARY_INV)
        # height,width=threshold_eye.shape
        # left_side_treshold=threshold_eye[0:height,0:int(width/2)]
        # left_side_white=cv2.countNonZero(left_side_treshold)


        # right_side_threshold=threshold_eye[0:height,int(width/2):width]
        # right_side_white=cv2.countNonZero(right_side_threshold)


        eye_points = [36, 37, 38, 39, 40, 41]
        left_eye_points = []
        for index in eye_points:
            point = landamrks.part(index)
            # print(f"Point {index}: ({point.x}, {point.y})")  # Debugging line#
            left_eye_points.append((point.x, point.y))
            
        # gaze_ratio=left_side_white/right_side_white#
        gaze_ratio_left_eye = get_gaze_ratio2(eye_points, landamrks)
        gaze_ratio_right_eye = get_gaze_ratio2([42, 43, 44, 45, 46, 47], landamrks)

        # Check if either gaze ratio is None
        if gaze_ratio_left_eye is None:
            gaze_ratio_left_eye = 0  # or some default value
        if gaze_ratio_right_eye is None:
            gaze_ratio_right_eye = 0  # or some default value

        # Now safely calculate the average
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2




        if gaze_ratio<1:
            # cv2.putText(frame," RIGHT",(50.100),font,2,(0,0,255),3)#
            cv2.putText(frame, " RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:]=(0,0,255)
        elif 1<gaze_ratio<2:
             cv2.putText(frame,"CENTER",(50,100),font,2,(0,0,255),3)
        else :
            new_frame[:]=(255,0,0)
            cv2.putText(frame,"LEFT",(50,100),font,2,(0,0,255),3)  
    

    #Letters
    if frames==10:
        letter_index+=1
        frames=0
    if letter_index==15:
        letter_index=0 


    for i in range(15):
        if i ==letter_index: 
         light=True
        else:
         light=False

    letter(i,keys_set_1[i],light)



    cv2.imshow("Frame",frame)
    cv2.imshow("new frame",new_frame)
    cv2.imshow("virtual keybord",keybord)

    key=cv2.waitKey(1)
    if key==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()