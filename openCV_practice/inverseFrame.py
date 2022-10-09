import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

w = round(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

fourcc = cv.VideoWriter_fourcc(*'MJPG')
output_video = cv.VideoWriter('output.avi', fourcc, fps, (w, h))

if not output_video.isOpened():
    raise Exception("File open failed")

last_frame = 0

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cur_frame = np.mean(gray)

    if last_frame == 0:
        last_frame = cur_frame

    if not ret:
        break

    if abs(cur_frame - last_frame) > 30:
        inverse = ~frame
        output_video.write(inverse)
    else:
        output_video.write(frame)

    last_frame = cur_frame

    if cv.waitKey(round(1000/fps)) >=0 :
      break

    cv.imshow('task3',frame)

cv.destroyAllWindows()