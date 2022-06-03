import cv2
import dlib
import numpy as np
import torch
from model import Map
from imutils import face_utils

IMG_SIZE = (34,26)
PATH = './weights/classifier_weights_iter_50.pt'

#모델 생성
model = Map()
model.load_state_dict(torch.load(PATH))
model.eval()

#예측 함수
def predict(pred):
  pred = pred.transpose(1, 3).transpose(2, 3)
  outputs = model(pred)
  pred_tag = torch.round(torch.sigmoid(outputs))

  return pred_tag

#검출한 eye_point가 들어오면 좌표로 리턴
def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)
  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

#실시간 영상 속 데이터 프레임을 획득
capture = cv2.VideoCapture(0)

if capture.isOpened()==False:
  raise Exception("카메라 연결 안됨")

#dlib의 face_landmark 이용
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')

n_count = 0

#실시간 영상 작동 중 처리
while True:
  ret, frame = capture.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = detector(gray)

  for face in faces:
    shapes = predictor(gray, face)
    shapes = face_utils.shape_to_np(shapes)

    #눈 부분에 대한 좌표 값을 리턴 받음
    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)

    eye_input_l = torch.from_numpy(eye_input_l)
    eye_input_r = torch.from_numpy(eye_input_r)

    #눈을 감은지 여부에 대한 예측값
    pred_l = predict(eye_input_l)
    pred_r = predict(eye_input_r)
    print(pred_l, pred_r)

    state_l = 'O %.1f' if pred_l > 0.0 else '- %.1f'
    state_r = 'O %.1f' if pred_r > 0.0 else '- %.1f'

    state_l = state_l % pred_l
    state_r = state_r % pred_r

    #양쪽 다 눈을 감았으면 count 해줌
    if pred_l.item() == 0.0 and pred_r.item() == 0.0:
      n_count+=1

    else:
      n_count = 0

    #50 프레임 이상이면 wake up 문구를 출력해줌
    if n_count > 50:
      cv2.putText(frame,"Wake up", (120,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

    #검출해낸 눈에 사각형과 눈을 떴는지, 감았는지 0과 1을 출력
    cv2.putText(frame, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.rectangle(frame, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
    cv2.rectangle(frame, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)


  if not ret : break
  if cv2.waitKey(30)>=0: break
  title = "Wake up Program"
  cv2.imshow(title, frame)
capture.release()