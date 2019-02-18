# SimilarPictures
이미지 클러스터링에 대한 모든 것!

## Requirements
Python Version: 3.6
pip list (based on 아나콘다에서 pip install ** in 2019.02.17 )
```
tensorflow      1.12.0
tensorflow hub  0.2.0
scikit image    0.14.2
sklearn         0.20.2
matplotlib      3.0.2
numpy           1.15.4
pillow          5.4.1
six             1.12.0
scipy           1.2.0
keras           2.2.4
```

## How can I use it?

<a href="#Install">설치하기</a><br>
<a href='#Clustering'>제공되는 모델로 Clustering</a><br>
<a href='#Finetuning'>제공된 모델로 tuning</a><br>
<a href='#Export'>pb 파일 내보내기</a><br>
<a href='#Eval'>평가하기</a><br>
<a href='#Update'>업데이트</a><br>

## Install
<a id='Install'></a>
```
cd $HOME/workspace
git clone https://github.com/jaemin93/SimilarPictures.git
```

## Model(Feature Vector) Compare (NaverDataSet에서 20% 샘플링하여 K-means 사용) 
Model | Hub Module | Output size | Score 
:------:|:---------------:|:---------------------:|:-----------:
inception_v3 | [inception_v3](https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1)| 2048 | 39.4 
inception_resnet_v2 |[inception_resnet_v2](https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1)| 1536 | 29.1
mobilenet_v2_140_224| [mobilenet_v2_140_224](https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2)| 1792 | 45.9
resnet_v2_152|[resnet_v2_152](https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1)| 2048 | 42.1


## **원하는 모델로 Clustering**
<a id='Clustering'></a>

어떠한 이미지 데이터셋이든 자신이 원하는 모델로 옵션을 줘서 Clustering을 진행할 수 있습니다.

i) cluster 개수를 알고 있을 경우
```
python run_all.py --model_name=[원하는 모델이름] --number_of_cluster=[cluster 개수]
```

ii) cluster 개수를 모를 경우
```
python run_all.py --model_name=[원하는 모델이름] --number_of_cluster=-1
```
이경우 Tensorflow Hub의 [DELF](https://tfhub.dev/google/delf/1) 모듈을 사용하여 특정 랜덤이미지 하나에 대하여 테스트 하고자 하는디렉토리 내에 **View Point** 와 무관하게 유사한 이미지 개수를 찾고 각 군집화안의 이미지 개수는 비슷하다는 전제로 클러스터 개수를 예측하게 됩니다.

### **자신만의 이미지 데이터로 재학습하여 Clustering**
<a id='Finetuning'></a>

자신만의 데이터로 구글에서 제공하는 모델의 weight, bias 등을 재학습하여 이미지 Feature Vector를 뽑아내고 싶으면 어떻게 할까요?
https://github.com/tensorflow/models/tree/master/research/slim <- 링크에서 Image-Net으로 pre-trained 된 .ckpt를 구할수 있습니다. 

저희는 [Inception_resent_v2](http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz)로 진행하였고 이것에 대한 Tutorial은 아래와 같습니다.

#### Augmentation

데이터 개수가 충분하지 않다면 적절한 Augmentatino을 통해 데이터개수를 불려주시길 바랍니다.
Augmentatino 방법은 매우 다양합니다. 저희가 기본적으로 제공하는 것을 이용안하셔도 됩니다! 적절하게 불려주세요.
Augmentation(SimilarPictures/src/utils/Augmentation.py):

    1. Affine            :이미지를 비트는 효과를 줍니다.
    2. Blur              :이미지를 뿌옇게 효과를 줍니다.
    3. Flip              :이미지의 반전 효과를 줍니다.
    4. Translate left    :이미지를 일정 거리만큼 왼쪽으로 움직입니다.
    5. Translate right   :이미지를 일정 거리만큼 오른쪽으로 움직입니다.

#### Convert data (tfrecord)

이제 데이터 개수가 충분해졌습니다. 이미지 데이터는 기본적으로 용량이 큽니다. 따라서 tfrecord형태로 바꿔주겠습니다. 네이버 데이터셋을 기준으로 말하지만 이것을 자신만의 데이터셋으로 생각해도 좋습니다.
네이버 데이터셋을 utils안에 subdir_label.py를 실행하여 label로 서브 폴더를 만들어 나누어 줍니다. 
다음 slim에 본래 있던 download_and_convert_data.py와, datasets디렉토리 안에 naver.py, download_and_convert_naver.py를 추가해주고 dataset_factory.py 를 수정하였습니다. 다음 코드를 실행

```
python download_and_convert_data.py --dataset_name=[데이터셋이름] --dataset_dir=[데이터셋경로]
```

#### Retrain (fine tuning)

이제 변환된 tfrecord들을 Retrain 합시다!
https://github.com/tensorflow/models 을 clone 합니다.
해당 명령어는 저희의 directory path에 맞춰넣은 것이므로 path는 사용자의 path에 맞게 수정가능 합니다.
다른 옵션의 의문점은 https://github.com/tensorflow/models/tree/master/research/slim 에서 확인 가능합니다.

--trainble_scopes 옵션을 주의하세요. Feature Vector가 나온이후의 layer를 재학습한다면 clustering에 Input에 해당하는 Feature Vector는 의미가 사라집니다. Feature Vecotr 이전 layer를 적절히 학습하세요. Tensorboard를 적극 활용하면 매우 쉽습니다.

```
python train_image_classifier.py 
    --train_dir=/save/model/path/
    --dataset_dir=/your/dataset/
    --dataset_name=[your dataset name]
    --dataset_split_name=train 
    --max_number_of_steps=[how many do you want to train]
    --batch_size=[number of batch size]
    --model_name=[model name]
    --checkpoint_path=[pretrained/model/.ckpt]
    --checkpoint_exclude_scopes=[재학습할 Layer] 
    --trainable_scopes=[재학습할 Layer]
```

## Make pb file
<a id='Export'></a>
거의 다왔습니다! 나만의 데이터로 retrain된 .ckpt를 pb로 바꿔주기만 하면 됩니다!!
src/utils.freeze_graph.py 를 활용하여 ckpt->pb 로 바꿔줍시다.


```
python freeze_graph.py \
    --input_graph=/path/to/your/graph.pbtxt \
    --input_checkpoint=/path/to/your/model.ckpt-19248 \
    --output_graph=/path/to/your/Inception_resnet_V2.pb \
    --output_node_names=[재학습된 Layer]
```

#### Run

마지막으로 클러스터링을 해봅시다. 아래와 같이 당장 실행 가능합니다!

```
python run_all.py \
    --fine_tuning=[pb파일 위치 파일이름까지] \
    --fine_tuned_layer=[재학습된 layer] \
    --number_of_cluster=[개수를 모른다면 -1]
```

## Evaluate
<a id='Eval'></a>
Clustering이 얼마나 잘되었는지 확인해 봅시다. run_all.py를 실행하면 데이터들이 얼마나 군집화가 잘되었는지 t-sne를 통하여 확인하실수 있습니다. 이제 score값을 확인해볼까요?

```
python evaluation
```

좋은 점수가 나오셨길 바랍니다! 


## Update
<a id='Update'></a>
클러스터 개수를 모를때:

1. DELF 모델 효율적으로 사용하여 클러스터 개수 예측하기
2. 계층적 군집화 알고리즘 사용하기