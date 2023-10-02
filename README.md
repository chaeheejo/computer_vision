# computer_vision

## summary
1. [컴퓨터 비전과 영상의 이해](https://blog.naver.com/60cogml/222911689905)
2. [OpenCV와 Matrix](https://blog.naver.com/60cogml/222912097249)
3. [OpenCV의 주요 기능](https://blog.naver.com/60cogml/222912358355)
4. [영상의 밝기와 명암비 조절](https://blog.naver.com/60cogml/222912697887)
5. [영상의 산술 및 논리 연산](https://blog.naver.com/60cogml/222912728552)
6. [필터링](https://blog.naver.com/60cogml/222913272892)
7. [영상의 기하학적 변환](https://blog.naver.com/60cogml/222913438097)
8. [에지 검출과 응용](https://blog.naver.com/60cogml/222941767204)
9. [컬러 영상 처리](https://blog.naver.com/60cogml/222941840534)
10. [이진화와 모폴로지](https://blog.naver.com/60cogml/222943483536)
11. [레이블링과 외곽선 검출](https://blog.naver.com/60cogml)
12. [객체 검출](https://blog.naver.com/60cogml/222948044815)
13. [지역 특징점 검출과 매칭](https://blog.naver.com/60cogml/222948291010)
14. [머신 러닝](https://blog.naver.com/60cogml/222948928848)

---

## coco_test_training
> training results on the coco dataset trained with yolov5

### install yolov5
```
git clone https://github.com/ultralytics/yolov5  
pip install -r requirements.txt
```
### write location of dataset
```
cd yolov5/data/[YOUR_CUSTOM].yaml  
```
at line 10, write your datasets location
at line 11, write your train images location relative to path
at line 12, write your validation images location relative to path

### select dataset and yolo model
```
python train.py --img 640 --epochs [EPOCH_NUM] --data [YOUR_CUSTOM].yaml --weight [YOLO_WEIGHT].pt
```
### result with visualize
```
cd yolov5/runs/train/
```

<br />

## openCV_practice 
> openCV 실습 &nbsp; [README](https://github.com/chaeheejo/computer_vision/blob/main/openCV_practice/README.md)

<br />

## eye_detect
> 사용자가 눈을 감고 있는 프레임이 50개가 지속된다면 화면에 Wake up 문구 출력 &nbsp; [REPORT](https://github.com/chaeheejo/computer_vision/tree/main/eye_detect/report.pdf)
