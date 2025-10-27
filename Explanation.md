# VQVAE 기반 이상 탐지 모델 설명

## 1. 전체 모델 구조

```
Input Image (512x512)
       │
       ▼
┌─────────────┐
│   VQVAE     │
│  Encoder    │
└─────────────┘
       │
       ▼
┌─────────────┐
│  Codebook   │
│  (32 codes) │
└─────────────┘
       │
       ▼
┌─────────────┐
│   VQVAE     │
│  Decoder    │
└─────────────┘
       │
       ▼
┌──────────────────────────┐
│        VIT-MLP           │
│  ┌────────────────────┐  │
│  │ Vision Transformer │  │
│  │  (Layers: 12)      │  │
│  └────────────────────┘  │
│  ┌────────────────────┐  │
│  │ MLP Head           │  │
│  │  (Layers: 3)       │  │
│  └────────────────────┘  │
└──────────────────────────┘
       │
       ▼
┌─────────────┐
│  MLP Prior  │
└─────────────┘
```

## 2. 주요 Loss 함수

### 2.1 VQVAE Loss
\[
\mathcal{L}_{VQVAE} = \mathcal{L}_{recon} + \mathcal{L}_{commit} + \mathcal{L}_{codebook}
\]

여기서:
- \(\mathcal{L}_{recon} = \|x - \hat{x}\|_2^2\) (재구성 손실)
- \(\mathcal{L}_{commit} = \beta\|z_e(x) - sg[e]\|_2^2\) (커밋먼트 손실)
- \(\mathcal{L}_{codebook} = \|sg[z_e(x)] - e\|_2^2\) (코드북 손실)

### 2.2 ALM (Anomaly Likelihood Map)
\[
ALM(x) = \|z_e(x) - e\|_2^2
\]

여기서:
- \(z_e(x)\): 인코더 출력
- \(e\): 가장 가까운 코드북 벡터

### 2.3 NLL (Negative Log-Likelihood) Map
\[
NLL(x) = -\log p(z|x)
\]

여기서:
- \(p(z|x)\): VIT-MLP 모델이 예측한 코드북 인덱스의 확률

## 3. 모델 특징

1. **VQVAE 구조**
   - 입력 이미지 크기: 512x512
   - 잠재 공간 크기: 64x64
   - 코드북 크기: 32개의 코드

2. **VIT-MLP 구조**
   - 입력 크기: 256x256
   - Vision Transformer 기반 특징 추출
   - MLP Prior를 통한 코드북 인덱스 예측
   - Vision Transformer 레이어 수: 12
   - MLP Head 레이어 수: 3

3. **이상 탐지 방식**
   - ALM: 재구성 오차 기반 이상 점수
   - NLL: 확률 기반 이상 점수
   - 두 지도의 결합을 통한 종합적인 이상 탐지

## 4. 추론 과정

1. 입력 이미지를 VQVAE로 인코딩
2. 코드북을 통한 양자화
3. VQVAE 디코더를 통한 재구성
4. VIT-MLP를 통한 코드북 인덱스 예측
5. ALM과 NLL 맵 생성
6. 최종 이상 탐지 결과 도출
