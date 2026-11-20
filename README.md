# 욕설, 악플 분류 모델

(사)한국인공지능연구소 AI 오픈랩 7기에서 진행한 프로젝트입니다.

게임 채팅이나 댓글 필터는 단어 목록 기반이라 변형된 욕설이나 문맥상 악플을 잘 잡지 못합니다. 이를 보완하기 위해 **욕설**과 **악플** 두 가지 이진 분류 문제를 정의하고, 1D CNN, BiLSTM, BERT 세 가지 모델로 비교 실험했습니다.

## 결과

테스트셋 정확도(%)입니다.

| 모델 | 욕설 탐지 | 악플 탐지 |
|---|---|---|
| BERT | 88.9 | 78.5 |
| 1D CNN | **89.3** | 81.0 |
| BiLSTM | 88.0 | **82.0** |

욕설 탐지는 1D CNN, 악플 탐지는 BiLSTM이 가장 높았습니다.

## 데이터

| 파일 | 문장 수 | 레이블 |
|---|---|---|
| `악플_train.csv` / `악플_test.csv` | 약 70,500 / 17,600 | 0: 일반, 1: 악플 |
| `욕설_train.csv` / `욕설_test.csv` | 약 63,800 / 16,000 | 0: 일반, 1: 욕설 |

- 학습 데이터는 다시 8:2로 나눠 검증에 사용했습니다.
- 띄어쓰기와 맞춤법이 엉망인 댓글이 많아서, 학습 전에 `맞춤법 교정.py`로 hanspell 맞춤법 검사기를 돌려 `악플맞춤법_train.csv` 같은 교정본을 만들어 썼습니다.

## 모델

| 모델 | 파일 | 입력 처리 | 주요 설정 |
|---|---|---|---|
| 1D CNN | `1DCNN 악플.py`, `1DCNN 욕설.py` | Komoran 형태소, 불용어 제거, 길이 60 | 임베딩 300, 필터 크기 2~6 각 50개, SELU, 드롭아웃 0.5, 7 epoch |
| BiLSTM | `BiLSTM 악플.py`, `BiLSTM 욕설.py` | Komoran 형태소, 불용어 제거 | 양방향 LSTM(hidden 256), 배치 100, 학습률 9e-4 |
| BERT | `BERT 악플.py`, `BERT 욕설.py` | `bert-base-multilingual-cased` 토크나이저 | multilingual BERT 파인튜닝, [CLS] 출력 → Linear, 배치 40, 학습률 3e-5 |

- 단어 사전은 5번 이상 나온 단어로 최대 2만 개까지 만듭니다.
- 손실은 `BCEWithLogitsLoss`, 평가는 이진 정확도로 합니다.

## 실행

```bash
# 1. 맞춤법 교정본 만들기 (파일 경로는 코드 안에서 지정)
python "맞춤법 교정.py"

# 2. 모델 학습과 평가
python "1DCNN 악플.py"
```

- 실행 폴더에 `stopwords.csv`(불용어 목록)와 맞춤법 교정본 CSV가 있어야 합니다. 두 파일은 저장소에 포함되어 있지 않습니다.
- `torchtext`의 예전 API(`data.Field`, `BucketIterator`)를 쓰므로 torchtext 0.8 이하가 필요합니다.

## 환경

Python 3.7.8, PyTorch 1.7.0, torchtext, KoNLPy(Komoran), transformers, py-hanspell
