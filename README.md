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

<br>

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

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%841.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%842.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%843.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%844.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%845.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%846.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%847.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%848.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%849.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8410.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8411.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8412.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8413.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8414.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8415.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8416.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8417.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8418.png)

![](https://github.com/sehyeona/ybigta-project/blob/master/%EC%83%81%EA%B4%80%EA%B4%80%EA%B3%8419.png)

데이터의 outlier값을 가지고 있는 천체 타입에 대한 정보 확인 

위에서 데이터 타이별 상관관계를 확인해보면 가로 세로 직선으로 진한 부분들을 관찰 할 수 있는데 이는 하나의 특징이 다른 모든 특징들과 매우 강한 상관관계를 가지고 있다는 것을 의미한다. 특징이 일대일로 강한 상관관계를 가지고 있는것은 가능한 일이지만 하나의 특징이 다른 모든 특징들과 매우 강한 상관관계를 가지는 것은 거의 불가능한 상황이다. 이에 대한 원인으로 두가지를 생각하였다

1.특징이 매우 큰 outliers 값을 가지기 때문에 다른 특징들과의 수리적 계산에서 강한 상관관계를 초래한다.

2.타입 샘플의 개수가 적어서

## 3. Training Data 전처리

-스케일링

변수들간의 스케일이 대부분 맞기는 하지만 어느정도 아웃라이어가 존재하기도 하고 해보기 전까지 모르기 때문에 아래의 4가지 방법을 이용하여 스케일링 하였다

종류

-standardscaler : 정규분포 이용

-minmaxscaler : 최대/최소값이 각각 0, 1

-maxabsscaler : 최대절대값과 0이 각각 1, 0 이 되도록하는 scaling

-robustscaler : median과 IQR 사용 outlier의 영향을 최소화 한다

모든 스케일러는 sklearn.preprocessing 안에 각자 이름으로 들어있음

```
def scaling_func(df, scaler) :
    '''
    param : dataframe / scaler object
    return : scaled dataframe / fitting scaler
    '''
    scaler = scaler()
    # type과 id를 제외하고 학습
    data_for_scaling = df.drop(['id', 'type', 'fiberID'], axis = 1)
    scaler.fit(data_for_scaling)
    # 학습후 변환
    train_scaled = scaler.transform(data_for_scaling)
    # 학습후 변환한 데이터를 다시 원래 데이터로 만들기
    result = pd.DataFrame(train_scaled, columns = data_for_scaling.columns)
    result = pd.concat([df[['id','type', "fiberID"]], result], axis=1)
    return result, scaler
    
```

## 4. Training Data 샘플링

-샘플링

종속변수인 type의 데이터 클래스가 불균형한 것으로 나타난다. 이처럼 데이터 클래스의 비율 차가 심하다면 단순히 우세한 클래스를 택하는 모형의 정확도가 높아져 모형의 성능 판별이 어려워지는 문제가 생길 수 있다. 이에 샘플링을 통해 비대칭 데이터를 다루고자 하였다.

- oversampling : 소수 클래스 데이터를 증가시키는 방법이다.
- undersampling : 다수 클래스 데이터의 일부만 사용하는 방법이다.

앞서 시각화 파트에서 천체 type 별 데이터 개수를 비교해보았을 때 majority class와 minority class의 비율 차이가 극심한 것을 확인하였다. 때문에 oversampling을 통해 다수 클래스를 기준으로 소수 클래스 데이터를 증가시켜 균형을 맞춤으로써 데이터 손실을 줄이고자 하였다. 

