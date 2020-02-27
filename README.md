# YBIGTA Project: DACON 천체유형 분류
<br>
ML 2팀: 노시영, 양정열, 안세현, 최정윤, 박솔희

![](https://github.com/sehyeona/ybigta-project/blob/master/Title.jpg)


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

-type의 출현 빈도 파악

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

-변수(feature)들의 분포

```

print(len(features))
for col in features :
    plt.figure(figsize=(12,4))
    sns.distplot(df[col])
    plt.title('Distribution of %s\n'%col)
    
```

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC1.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC2.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC3.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC4.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC5.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC6.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC7.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC8.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC9.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC10.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC11.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC21.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC12.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC13.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC14.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC15.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC16.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC17.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC18.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC19.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EB%B3%80%EC%88%98%EB%B6%84%ED%8F%AC20.png)

분포 확인 결과

1. fiberMag_u 와 psfMag_u 가 두개의 분포형태가 거의 동일하다고 볼 수 있음

2. 대부분의 분포들이 평균에서 매우 떨어져있는 아웃라이어를 가지고 있음을 알 수 있음

3. 주로 모든 아웃라이어들은 양방향으로 분포해있기 보다 한방향으로 치우쳐서 분포해 있음을 알 수 있음

-천체타입에 의한 변수간의 상관관계

```
for x in types:    
    plt.figure(figsize=(12,8))
    ax =  sns.heatmap(df[df['type'] == x].corr(method='pearson'), annot = True,   
                fmt = '.2f',linewidths = 1, cmap="summer")
    buttom, top = ax.get_ylim()
    ax.set_ylim(buttom + 0.5, top - 0.5)
    plt.title("Correlations when type is %s"%x)
```


