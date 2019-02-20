# Tiplet loss 

출처의 repository를 클론하여 실행하면 deepranking.h5와 deepranking.json을 얻을 수 있다.
이것으로 triplet loss를 이용한 fine tuning된 vgg16(다른모델도 업데이트 할 예정)를 로드 하여 feature vector를 뽑아내 클러스터링을 할수 있다. 현재 이 것은 지난주 밋업에서 힌트를 얻어 추후 업데이트 될 예정이다.


## Trouble

하드웨어 스펙이 좋지 않을시 로드하는 데이터의 양에 따라 프로세스가 죽는 문제가 있다. 이부분을 해결중에 있고 해결을 못하면 하드웨어 스펙을 올려서 해결할 생각이다.

출처: https://github.com/akarshzingade/image-similarity-deep-ranking