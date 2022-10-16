import cv2
import dlib
from math import hypot

video=cv2.VideoCapture(1)
nose=cv2.imread('FaceAlgorithms/imageforfilter.jpg')
facedetector=dlib.get_frontal_face_detector()
markpredictor=dlib.shape_predictor('FaceAlgorithms/shape_predictor_68_face_landmarks.dat')


while video.isOpened():
    ret,frame=video.read()
    if ret:
        grayframe=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=facedetector(grayframe)
        for face in faces:
            landmarks=markpredictor(grayframe,face)
            top_nose_point=(landmarks.part(28).x,landmarks.part(28).y)
            left_nose_point=(landmarks.part(31).x,landmarks.part(31).y)
            right_nose_point=(landmarks.part(35).x,landmarks.part(35).y)
            
            nose_width=int((hypot(right_nose_point[0]-left_nose_point[0],
                            right_nose_point[1]-left_nose_point[1]))*3)
            middle_nose=((right_nose_point[0]+left_nose_point[0])/2,
                        (right_nose_point[1]+left_nose_point[1])/2)
            nose_height=int((hypot(middle_nose[0]-top_nose_point[0],
                                middle_nose[1]-top_nose_point[1]))*2.5)
            resized_nose=cv2.resize(nose,(nose_width,nose_height),
                                    interpolation=cv2.INTER_AREA)   

            gray_resized_nose=cv2.cvtColor(resized_nose,cv2.COLOR_BGR2GRAY)
            ret,threshold_nose=cv2.threshold(gray_resized_nose,25,255,cv2.THRESH_BINARY_INV)
            
            center_nose=((middle_nose[0]+top_nose_point[0])/2,
                        (middle_nose[1]+top_nose_point[1])/2)
           
            top_left=(int(center_nose[0]-nose_width/2),int(center_nose[1]-nose_height/2))
            bottom_right=(int(center_nose[0]+nose_width/2),int(center_nose[1]+nose_height/2))
            nose_area=frame[top_left[1]:top_left[1]+nose_height,top_left[0]:top_left[0]+nose_width]
           
            try:
                bitwise=cv2.bitwise_and(nose_area,nose_area,mask=threshold_nose)
                added=cv2.add(bitwise,resized_nose)
                frame[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]=added
            except:
                pass
            
    cv2.imshow('frame',frame)
    cv2.imshow('area',nose_area)
    key=cv2.waitKey(7)
    if key==ord('q'):
        break
video.release()
cv2.destroyAllWindows()


