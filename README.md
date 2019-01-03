# SimilarPictures
이미지 유사도 측정하여 군집화 하기

## Requirements
Python 3.5
```
tf-slim
```

trained ckpt link: [Ckpt based on naver_dataset](https://1drv.ms/u/s!AtpHpqkl2-8C1wvs5ne0aRNFaA2O)


## GOAL
STEP 1
```
Convert data (tfrecord)
이미지 데이터 tfrecord 포맷으로 바꾸기
기존에 받은 네이버 데이터셋을 utils안에 subdir_label.py를 실행하여 label로 서브 폴더를 만들어 나누어 줍니다. 다음 slim에 본래 있던 download_and_convert_data.py와, datasets디렉토리 안에 naver.py, download_and_convert_naver.py를 추가해주고 dataset_factory.py 를 수정하였습니다. 
다음 [1]:python download_and_convert_data.py --dataset_name=[데이터셋이름] --dataset_dir=[데이터셋경로] 
를 실행하여 자동으로 validation 과 train을 나누어서 tfrecord로 변환하였습니다. 저희가 올린 프로젝트에서는 
[1]을 실행해주시면 됩니다. 예시는 다음과 같습니다. 
python download_and_convert_data.py --dataset_name=naver --dataset_dir=C:\Users\iceba\develop\data\naver
```

STEP 2
```
Train
https://github.com/tensorflow/models 을 clone 합니다.
data를 directory의 path에 맞춰 넣어 놓으십시오.해당 명령어는 저희의 directory path에 맞춰넣은 것이므로 path는 사용자의 path에 맞게 수정하여 주십시오.

python train_image_classifier.py 
    --train_dir=/tfpath/naver/log_inception_resnet_v2_naver 
    --dataset_dir=/tfpath/naver/naver/ 
    --dataset_name=naver 
    --dataset_split_name=train 
    --max_number_of_steps=20000  
    --batch_size=35 
    --model_name=inception_resnet_v2 
    --checkpoint_path=/tfpath/naver/inception_resnet_pretrained/inception_resnet_v2_2016_08_30.ckpt 
    --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits 
    --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits
```

STEP 3
```
Evaluate
slim에 eval_image_classifier.py를 통해 train된 이미지 결과 확인.
python eval_image_classifier.py alsologtostderr 
    --checkpoint_path=/tfpath/naver/log_inception_resnet_v2_naver/ 
    --dataset_dir=/tfpath/naver/naver/ 
    --dataset_name=naver 
    --dataset_split_name=validation 
    --model_name=inception_resnet_v2
```

STEP4
```
Test
저희가 짜놓은 make_labels_pred.py에 안에 path를 적절하게 바꾸어서 .ckpt를 적용과 test데이터셋 경로를 설정합니다.
다음 실행을 해주시면 다음 label_pred.txt에 군집화된 label과 image_paths.txt를 얻으실수 있습니다.
```