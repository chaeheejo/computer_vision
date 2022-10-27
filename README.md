# computer_vision

## summary
1. [컴퓨터 비전과 영상의 이해](https://blog.naver.com/60cogml/222911689905)
2. [OpenCV와 Matrix](https://blog.naver.com/60cogml/222912097249)
3. [OpenCV의 주요 기능](https://blog.naver.com/60cogml/222912358355)

<br />

## openCV_practice 
> openCV 실습 &nbsp; [README](https://github.com/chaeheejo/computer_vision/blob/main/openCV_practice/README.md)

<br />

## coco_test_training
> yolov5 학습  

<br />

epoch 50
yolov5/train.py 파일로 학습 진행  
yolov5/data/yaml에 데이터셋의 위치 작성  
yolov5/models/ 에서 데이터 학습 모델 파일 지정

<br />

## eye_detect
> 웹캠을 통한 eye detection

<br />

데이터셋 npy 확장자  
눈을 감은 사진, 뜬 사진 yolo-mark를 통해 라벨링  
눈을 감고 있는 프레임이 50개가 지속된다면, 사용자 화면에 Wake up 문구 출력
