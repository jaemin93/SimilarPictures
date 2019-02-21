# Triplet loss 
https://github.com/akarshzingade/image-similarity-deep-ranking 깃 repo에서 triplet loss를 train할수 있다. 이 디렉토리 안의 코드들은 여기서 가져오거나 수정한 것들이며 실행이 잘 안될시 위의 repo에 가서 확인하길 바란다.

## triplet loss?
triplet loss는 이미지를 훈련할 이미지와 postive이미지 negative이미지 를 동시에 넣어서 clustering 하기 좋은 결과값이 나온다. 자세한 설명은 [링크](https://wwiiiii.tistory.com/entry/Pairwise-Triplet-Loss)에서 볼수 있다.

# Run

## step 1: preprocess
먼저 dummy형태의 데이터셋이 구성되어있다면 모델 넘버 기준으로 sub 디렉토리를 만들고 안에 할당 해줘야 한다.
```
---DATASET---
        ├───────1: images01, imagse02, ....
        ├───────2: images01, images02, ....
```
이러한 데이터셋 형테는 utils 폴더안에 original_subdir_label.py 를 사용하면 된다.

## step 2: tirplets dataset list
데이터셋이 준비 되었다면 다음을 실행한다.
```
$ python tripletSampler.py --input_directory <<path to the directory>> \
     --output_directory <<path to the directory>> \
     --num_pos_images <<Number of positive images you want>> \
     --num_neg_images <<Number of negative images you want>>
```
이제 자신과 맞는 이미지와 안맞는 이미지의 목록을 만들게되고 이 목록을 기반으로 triplet loss를 train할 수 있다.

## step 3: train
step 1에서 분할한 데이터 셋경로와 step 2에서 만든 triplets.txt 를 로드하면 train 할수 있다.
deepRanking.py에서 경로를 수정할 수 있고 epoch과 step per epoch을 설정할 수 있다.
```
$ python deepRanking.py
```

## step 4: clustering
```
$ python test.py
```

# Trouble

하드웨어 스펙이 좋지 않을시 로드하는 데이터의 양에 따라 프로세스가 죽는 문제가 있다. 이부분을 해결중에 있고 해결을 못하면 하드웨어 스펙을 올려서 해결할 생각이다.

출처: https://github.com/akarshzingade/image-similarity-deep-ranking, https://wwiiiii.tistory.com/entry/Pairwise-Triplet-Loss