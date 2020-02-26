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
```
  train=pd.read_csv('/content/gdrive/My Drive/train.csv')
  test=pd.read_csv('/content/gdrive/My Drive/test.csv')
  train['type'].unique()

array(['QSO', 'STAR_RED_DWARF', 'SERENDIPITY_BLUE', 'STAR_BHB',
       'STAR_CATY_VAR', 'SERENDIPITY_DISTANT', 'GALAXY',
       'SPECTROPHOTO_STD', 'REDDEN_STD', 'ROSAT_D', 'STAR_WHITE_DWARF',
       'SERENDIPITY_RED', 'STAR_CARBON', 'SERENDIPITY_FIRST',
       'STAR_BROWN_DWARF', 'STAR_SUB_DWARF', 'SKY', 'SERENDIPITY_MANUAL',
       'STAR_PN'], dtype=object
```
총 19종류의 천체 유형으로 분류된다

-psfMag(Point spread function magnitudes) : 먼 천체를 한 점으로 가정하여 측정한 빛의 밝기

-fiberMag(Fiber magnitudes) : 3인치 지름의 광섬유를 사용하여 광스펙트럼을 측정합니다. 광섬유를 통과하는 빛의 밝기

-petroMag(Petrosian Magnitudes) : 은하처럼 뚜렷한 표면이 없는 천체에서는 빛의 밝기를 측정하기 어렵다. 천체의 위치와 거리에 상관없이 빛의 밝기를 비교하기 위한 수치

-modelMag(Model magnitudes) : 천체 중심으로부터 특정 거리의 밝기

-FiberID:관측에 사용된 광섬유의 구분자

-참고: u(ultraviolet), g(green), r(red), i,z(very-near-infrared)

## 2. Training Data 시각화

```
  plt.figure(figsize=(12,8))
  ax = sns.countplot(y="type", data=df)
  plt.title('Distribution of orb types\n')
  plt.ylabel('Number of type\n')

  # Make twin axis
  ax2=ax.twiny()

  # Switch so count axis is on right, frequency on left
  ax2.xaxis.tick_top()
  ax.xaxis.tick_bottom()

  # Also switch the labels over
  ax.xaxis.set_label_position('bottom')
  ax2.xaxis.set_label_position('top')

  ax2.set_xlabel('Frequency [%]')
```

![](https://github.com/sehyeona/ybigta-project/blob/master/visualization1.png)


