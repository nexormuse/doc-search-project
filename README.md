# LLM Infra Document Search Project

LLM Infra 마일스톤 과제로 진행하는 기술 문서 검색 프로젝트입니다.

## Week 1: 문서 데이터 준비 및 기초 탐색

### 구현 기능

- CSV 데이터 불러오기
- 데이터 행·열 수 및 컬럼 구조 확인
- 카테고리별 문서 수와 비율 계산
- 반복문과 딕셔너리를 이용한 카테고리별 평균 단어 수 계산
- 컬럼별 결측치 수, 비율, 심각도 확인
- NumPy를 이용한 문서 길이 통계 계산
- 50단어 미만 문서 필터링
- pandas와 NumPy 통계 결과 비교

## 프로젝트 구조

```text
doc-search-project/
├── data/
│   └── tech_docs.csv
├── week1/
│   └── main.py
├── learning-log/
├── requirements.txt
└── README.md