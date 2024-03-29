# computer_vision / openCV_practice

### video_capture.py
> 노트북 상의 내장 카메라(0번)에 접근하여 화면의 노출값을 가져옴

<br/>

###  video_writer.py
> 내장 카메라(0번)에서 프레임을 가져와 동영상으로 저장  
> macOS는 프레임 크기를 1280x720으로 설정해주어야 프레임을 가져올 수 있음

<br/>

### read_video.py
> 저장된 동영상을 가져와 파란색 - 초록색 - 빨간색으로 영상 색 변화

<br />

### numpy_basic.ipynb
> openCV 사용 시 많이 사용되는 numpy 함수 정리

<br />

### convert_zero_about_dark.py
> sample.jpg 파일을 회색조(grayscale)로 열어서 이미지의 평균 밝기보다 어두운 픽셀들을 0으로 바꿔서 output.jpg로 저장

<br />

### control_contrast_ratio.py
> sample.jpg 이미지의 평균 밝기를 기준으로 명암비를 조절해 contrast.jpg로 저장  
> 명암비 조절 시 원본의 pixel 값에 일정한 값을 곱하는 형태로 하되, 계수는 2.0을 사용  
> 결과 저장 시 saturation 연산을 적용

<br />

### inverse_frame.py
> 웹캠을 사용하여 회색조(grayscale)로 동영상 프레임을 가져옴  
> 현재 프레임이 직전 프레임보다 이미지 전체의 평균 밝기가 30 넘게 바뀔 경우, 그 시점부터 다음 3초간 반전시켜서 output.avi로 저장
