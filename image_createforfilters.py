import cv2

def mouse(event,x,y,flags,params):
    global bitwise
    if event==cv2.EVENT_LBUTTONDOWN:
        cv2.circle(threshold,(x,y),10,(0,0,0),-1)
        bitwise=cv2.bitwise_and(image,image,mask=threshold)
        cv2.imshow('threshold',threshold)
        cv2.imshow('nose',bitwise)

image=cv2.imread('images/nose.png ')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret,threshold=cv2.threshold(gray,245,255,cv2.THRESH_BINARY_INV)
bitwise=cv2.bitwise_and(image,image,mask=threshold)
cv2.imshow('threshold',threshold)
cv2.imshow('nose',bitwise)
cv2.setMouseCallback('threshold',mouse)
if cv2.waitKey(0)==ord('s'):
    cv2.imwrite('FaceAlgorithms/imageforfilter.jpg',bitwise)


