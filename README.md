# SimilarPictures
Inception Resnet V2를 사용하여 이미지 유사도 측정하여 군집화 하기

## Requirements
Python 3.5
```
tf-slim
```

trained ckpt link: [inception_resnet_v2.ckpt based on naver_dataset](https://1drv.ms/u/s!AtpHpqkl2-8C1wvs5ne0aRNFaA2O)

## GOAL
### STEP 1: Convert data (tfrecord)

이미지 데이터 tfrecord 포맷으로 바꾸기
데이터셋 구성은 [data_foler_view.txt](https://github.com/jaemin93/SimilarPictures/blob/master/data_folder_view.txt) 에서 볼 수 있습니다.
기존에 받은 네이버 데이터셋을 utils안에 subdir_label.py를 실행하여 label로 서브 폴더를 만들어 나누어 줍니다. 
다음 slim에 본래 있던 download_and_convert_data.py와, datasets디렉토리 안에 naver.py, download_and_convert_naver.py를 추가해주고 dataset_factory.py 를 수정하였습니다. 다음 코드를 실행

```
python download_and_convert_data.py --dataset_name=[데이터셋이름] --dataset_dir=[데이터셋경로]
```

자동으로 validation 과 train을 나누어서 tfrecord로 변환하였습니다. 저희는 Data Augmentation을 하여서
'train': 50844, 'validation': 12711으로 나누었습니다 


### STEP 2: Train

https://github.com/tensorflow/models 을 clone 합니다.
해당 명령어는 저희의 directory path에 맞춰넣은 것이므로 path는 사용자의 path에 맞게 수정가능 합니다.
다른 옵션의 의문점은 https://github.com/tensorflow/models/tree/master/research/slim 에서 확인 가능합니다.


```
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


### STEP 3: Evaluate

train이 완료되었다면 평가할 필요가 있습니다 아래의 실행 명령어로 얼마나 훈련이 잘되었는지 봅시다.

```
python eval_image_classifier.py alsologtostderr 
    --checkpoint_path=/tfpath/naver/log_inception_resnet_v2_naver/ 
    --dataset_dir=/tfpath/naver/naver/ 
    --dataset_name=naver 
    --dataset_split_name=validation 
    --model_name=inception_resnet_v2
```

### STEP 4: Test

make_labels_pred.py에 path를 적절하게 바꾸어서 *.ckpt를 적용하고 test를 진행할 데이터 경로를 설정합니다.

```
python make_labels_pred.py
```

실행을 하면 군집화된 label_pred.txt와 image_paths.txt를 얻을수 있습니다.