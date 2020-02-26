# YBIGTA Project: DACON 천체유형 분류
<br>
ML 2팀: 노시영, 양정열, 안세현, 최정윤, 박솔희
<br>
![](https://en.wikipedia.org/wiki/Galaxy#/media/File:NGC_4414_(NASA-med).jpg)
<br>
<br>

## 프로젝트의 목적 및 의의
DACON에서 제공하는 천체 트레이닝 데이터를 활용하여 테스트 데이터의 천체 유형을 예측해보는 것이 궁극적인 목표. 여기서 채점되는 기준은 log loss이다. 이 수치를 작게하면 할 수록 더욱 높은 등수를 얻을 수 있다. 각 예측의 확률을 .csv형태로 저장한 후 DACON에 제출하면 자동으로 log loss값이 계산되어 출력된다. 천문학 전문가가 아님에도 불구하고 본 프로젝트가 중요한 의미를 가지는 이유는 데이터를 시각화하고 중요 변수를 파악하며, 적절한 모델을 선택하여 학습시키는 과정을 처음부터 끝까지 경험 할 수 있기 때문이다. 
<br>

## 목차(진행과정)

1. 변수의 의미 파악

2. Training Data 시각화

3. Training Data 전처리

4. Training Data 샘플링

5. 적합한 모델 찾기(XGBoost, CatBoost, RandomForest, LightGBM)

6. 그리드 서치 
<br>
<br>
<br>
## 1. 변수의 의미 파악

-type: 천체 유형으로 예측해야 하는 변수(종속변수) 

  train=pd.read_csv('/content/gdrive/My Drive/train.csv')
test=pd.read_csv('/content/gdrive/My Drive/test.csv')
train['type'].unique()






