# OpenCV_projects

### videoCapture.py
> 노트북 상의 내장 카메라(0번)에 접근하여 화면의 노출값을 가져옴

<br/>

### videoWriter.py
> 내장 카메라(0번)에서 프레임을 가져와 동영상으로 저장  
> macOS는 프레임 크기를 1280x720으로 설정해주어야 프레임을 가져올 수 있음

<br/>

### readVideo.py
> 저장된 동영상을 가져와 파란색 - 초록색 - 빨간색으로 영상 색 변화

<br />

### coco_test_training
> epoch 50개로만 수행  
> yolov5/train.py 파일로 학습 진행  
> yolov5/data/ 에 데이터셋의 위치를 yaml 파일로 생성  
> yolov5/models/ 에서 데이터 학습 모델 파일 지정

<br />

### eyeDetect
> 데이터셋은 npy 확장자  
> 눈을 감고 있는지 뜨고 있는지 딥러닝 학습  
> 눈을 감고 있는 프레임이 50개가 지속된다면, 사용자 화면에 Wake up 문구 출력
