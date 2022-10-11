import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

w = round(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = 15

fourcc = cv.VideoWriter_fourcc(*'MJPG')
output_video = cv.VideoWriter('output.avi', fourcc, fps, (w, h), 0)

if not output_video.isOpened():
    raise Exception("File open failed")

last_frame = 0
frame_cnt=0
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if not ret:
        break

    if 0<frame_cnt:
        frame_cnt-=1
        output_video.write(~gray)
        frame = ~gray
    else:
        cur_frame = np.mean(gray)

        if last_frame == 0:
            last_frame = cur_frame

        if abs(cur_frame - last_frame) > 30:
            frame_cnt=fps*3-1
            output_video.write(~gray)
            frame = ~gray
        else:
            output_video.write(gray)
            frame = gray

        last_frame = cur_frame

    if cv.waitKey(25) >=0 :
      break

    cv.imshow('task3',frame)

cv.destroyAllWindows()