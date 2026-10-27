# 사용 모델

## 동물 탐지: Ultralytics YOLO11n

- 파일: models/yolo11n.pt
- 역할: COCO의 cat(15), dog(16)만 탐지
- 공식 안내: https://docs.ultralytics.com/models/yolo11/
- 파일 출처: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt
- SHA-256: 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1
- Ultralytics는 AGPL-3.0 및 Enterprise 라이선스를 안내합니다. https://www.ultralytics.com/license

## 품종 분류: Oxford-IIIT Pet으로 학습된 ResNet50

- 배포자/모델: https://huggingface.co/flaviodell/oxford-pets-resnet50
- 커뮤니티 배포 사전 학습 모델입니다. Oxford 대학의 공식 모델 배포를 뜻하지 않습니다.
- 고정 리비전: 3253ad292cf0778f5b64fa5a9156f2bdb0a84a3b
- 원본 파일: best_model.pth (95,263,895 bytes)
- 로컬 파일: models/oxford-pets-resnet50.pth
- SHA-256: 889b2d4919ea0c48cb8a89e2b25ff7c56af1bd3e1627e36b36d63c495b3842d4
- 모델 카드 라이선스 표기: MIT
- 구조: ResNet50 + Dropout(0.3) + Linear(2048,37)
- 전처리: RGB, 224×224 Resize, ToTensor, ImageNet 평균·표준편차 정규화. 배포 예제와 동일합니다.
- 체크포인트의 실제 클래스 목록과 37차원 출력을 검증합니다. 파일 해시 검증 후 torch.load(weights_only=True)를 사용합니다.

원본 Oxford 데이터셋은 사진·정답 자료이고, 위 모델 파일은 그 자료를 학습한 가중치입니다. 프로그램은 추가 학습을 하지 않습니다. 탐지한 영역을 자르므로 학습 시 전체 사진과 입력 구도가 달라질 수 있으며 오분류 가능성이 있습니다.

모델 카드의 제공자 성능 수치를 이 프로젝트의 정확도로 사용하지 않습니다. 학습 분할과 공식 test의 중복 여부는 미확인입니다. 데이터셋 라이선스는 모델 라이선스와 별개입니다: https://www.robots.ox.ac.uk/~vgg/data/pets/

큰 파일은 GitHub에 올리지 않습니다. 모델은 실행 시 다운로드하고, 원본 데이터셋은 datasets에 직접 준비합니다.
