import re  # 정규식을 이용한 텍스트 전처리를 위해 re 모듈을 불러온다.
from pathlib import Path  # 운영체제와 관계없이 파일 경로를 관리하기 위해 Path를 불러온다.

import numpy as np  # 벡터 계산, 정렬, 수치 비교에 사용할 NumPy를 불러온다.
import pandas as pd  # CSV 데이터 로드와 DataFrame 처리를 위해 pandas를 불러온다.
from sklearn.feature_extraction.text import TfidfVectorizer  # 문서를 TF-IDF 벡터로 변환하기 위해 불러온다.

DATA_PATH = Path("data/tech_docs.csv")  # 프로젝트 루트를 기준으로 기술 문서 CSV 파일 경로를 지정한다.
TOP_K = 3  # 검색 평가에서 확인할 상위 문서 개수를 3개로 지정한다.

EVAL_SET = [  # 기능 1에서 사용할 검색 평가셋 목록을 정의한다.
    {  # 첫 번째 평가 질문 정보를 담는 딕셔너리를 시작한다.
        "query": "how can I create lists with a concise for expression",  # Python comprehension 관련 질문을 작성한다.
        "relevant_doc_ids": ["D001", "D059"],  # 질문에 답이 되는 실제 문서 ID를 정답으로 지정한다.
    },  # 첫 번째 평가 항목을 끝낸다.
    {  # 두 번째 평가 질문 정보를 담는 딕셔너리를 시작한다.
        "query": "how do I handle errors with try except and finally",  # Python 예외 처리 관련 질문을 작성한다.
        "relevant_doc_ids": ["D005"],  # Python Exception Handling 문서를 정답으로 지정한다.
    },  # 두 번째 평가 항목을 끝낸다.
    {  # 세 번째 평가 질문 정보를 담는 딕셔너리를 시작한다.
        "query": "how do I open a file and make sure it closes automatically",  # Python 파일과 자원 정리 관련 질문을 작성한다.
        "relevant_doc_ids": ["D008", "D051"],  # File I/O와 Context Manager 문서를 정답으로 지정한다.
    },  # 세 번째 평가 항목을 끝낸다.
    {  # 네 번째 평가 질문 정보를 담는 딕셔너리를 시작한다.
        "query": "how do I combine changes from two git branches",  # Git 브랜치 병합 관련 질문을 작성한다.
        "relevant_doc_ids": ["D014"],  # Git Merging Strategies 문서를 정답으로 지정한다.
    },  # 네 번째 평가 항목을 끝낸다.
    {  # 다섯 번째 평가 질문 정보를 담는 딕셔너리를 시작한다.
        "query": "how can I temporarily save uncommitted changes in git",  # Git 임시 변경사항 저장 관련 질문을 작성한다.
        "relevant_doc_ids": ["D019"],  # Git Stashing Changes 문서를 정답으로 지정한다.
    },  # 다섯 번째 평가 항목을 끝낸다.
    {  # 여섯 번째 평가 질문 정보를 담는 딕셔너리를 시작한다.
        "query": "how does gradient descent update model parameters",  # 경사하강법 관련 질문을 작성한다.
        "relevant_doc_ids": ["D023"],  # What is Gradient Descent 문서를 정답으로 지정한다.
    },  # 여섯 번째 평가 항목을 끝낸다.
    {  # 일곱 번째 평가 질문 정보를 담는 딕셔너리를 시작한다.
        "query": "what is the difference between overfitting and underfitting",  # 과적합과 과소적합 관련 질문을 작성한다.
        "relevant_doc_ids": ["D026"],  # Overfitting and Underfitting 문서를 정답으로 지정한다.
    },  # 일곱 번째 평가 항목을 끝낸다.
    {  # 여덟 번째 평가 질문 정보를 담는 딕셔너리를 시작한다.
        "query": "how are gradients computed through neural network layers",  # 역전파 관련 질문을 작성한다.
        "relevant_doc_ids": ["D030"],  # Backpropagation Algorithm 문서를 정답으로 지정한다.
    },  # 여덟 번째 평가 항목을 끝낸다.
    {  # 아홉 번째 평가 질문 정보를 담는 딕셔너리를 시작한다.
        "query": "how do numpy arrays operate when their shapes are different",  # NumPy 브로드캐스팅 관련 질문을 작성한다.
        "relevant_doc_ids": ["D034"],  # NumPy Broadcasting Rules 문서를 정답으로 지정한다.
    },  # 아홉 번째 평가 항목을 끝낸다.
    {  # 열 번째 평가 질문 정보를 담는 딕셔너리를 시작한다.
        "query": "how can I filter numpy array elements with boolean conditions",  # NumPy 불리언 필터링 관련 질문을 작성한다.
        "relevant_doc_ids": ["D033", "D039"],  # Array Indexing과 Boolean Masking 문서를 정답으로 지정한다.
    },  # 열 번째 평가 항목을 끝낸다.
    {  # 열한 번째 평가 질문 정보를 담는 딕셔너리를 시작한다.
        "query": "how do I remove or fill missing values in pandas",  # pandas 결측치 처리 관련 질문을 작성한다.
        "relevant_doc_ids": ["D043"],  # Handling Missing Data in pandas 문서를 정답으로 지정한다.
    },  # 열한 번째 평가 항목을 끝낸다.
    {  # 열두 번째 평가 질문 정보를 담는 딕셔너리를 시작한다.
        "query": "what is the difference between loc and iloc in pandas",  # pandas 인덱싱 관련 질문을 작성한다.
        "relevant_doc_ids": ["D046"],  # pandas Indexing with loc and iloc 문서를 정답으로 지정한다.
    },  # 열두 번째 평가 항목을 끝낸다.
]  # 기능 1 평가셋 정의를 끝낸다.


def load_data(file_path: Path) -> pd.DataFrame:  # CSV 파일을 불러오는 2주차 함수를 정의한다.
    """CSV 파일을 불러와 DataFrame으로 반환한다."""  # 함수의 역할을 설명하는 문서 문자열이다.
    if not file_path.exists():  # 전달받은 경로에 실제 파일이 존재하는지 확인한다.
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")  # 파일이 없으면 명확한 오류를 발생시킨다.

    df = pd.read_csv(file_path)  # CSV 파일을 읽어 pandas DataFrame으로 저장한다.
    required_columns = {"doc_id", "title", "category", "content"}  # 실행에 필요한 필수 컬럼 집합을 정의한다.
    missing_columns = required_columns - set(df.columns)  # 필수 컬럼 중 실제 데이터에 없는 컬럼을 계산한다.

    if missing_columns:  # 누락된 필수 컬럼이 하나라도 있는지 확인한다.
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing_columns)}")  # 누락 컬럼 목록을 포함한 오류를 발생시킨다.

    return df  # 검증이 완료된 DataFrame을 호출한 위치로 반환한다.


def preprocess(text: str) -> str:  # 문자열 하나를 검색용 형태로 전처리하는 2주차 함수를 정의한다.
    """영문 텍스트를 검색에 적합한 형태로 정제한다."""  # 함수의 역할을 설명하는 문서 문자열이다.
    text = text.lower()  # 대문자와 소문자를 동일한 단어로 처리하기 위해 모두 소문자로 변환한다.
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # 영문 소문자, 숫자, 공백을 제외한 특수문자를 공백으로 바꾼다.
    text = re.sub(r"\s+", " ", text).strip()  # 여러 공백을 하나로 줄이고 문자열 앞뒤 공백을 제거한다.
    return text  # 전처리가 끝난 문자열을 반환한다.


def cosine_similarity_numpy(  # NumPy만 이용해 코사인 유사도를 계산하는 2주차 함수를 정의한다.
    a: np.ndarray,  # 첫 번째 벡터를 NumPy 배열로 전달받는다.
    b: np.ndarray,  # 두 번째 벡터를 NumPy 배열로 전달받는다.
) -> float:  # 계산된 코사인 유사도를 실수로 반환한다.
    """NumPy를 이용해 두 벡터의 코사인 유사도를 계산한다."""  # 함수의 역할을 설명하는 문서 문자열이다.
    a = np.asarray(a, dtype=float)  # 첫 번째 입력을 실수형 NumPy 배열로 변환한다.
    b = np.asarray(b, dtype=float)  # 두 번째 입력을 실수형 NumPy 배열로 변환한다.

    if a.shape != b.shape:  # 두 벡터의 차원과 크기가 동일한지 검사한다.
        raise ValueError("두 벡터의 크기가 같아야 합니다.")  # 크기가 다르면 내적할 수 없으므로 오류를 발생시킨다.

    norm_a = np.linalg.norm(a)  # 첫 번째 벡터의 유클리드 노름을 계산한다.
    norm_b = np.linalg.norm(b)  # 두 번째 벡터의 유클리드 노름을 계산한다.

    if norm_a == 0 or norm_b == 0:  # 두 벡터 중 하나라도 영벡터인지 확인한다.
        return 0.0  # 분모가 0이 되는 문제를 막기 위해 유사도 0.0을 반환한다.

    dot_product = np.dot(a, b)  # 두 벡터의 내적을 계산한다.
    similarity = dot_product / (norm_a * norm_b)  # 내적을 두 벡터 노름의 곱으로 나눈다.
    return float(similarity)  # NumPy 실수 값을 일반 Python float로 변환해 반환한다.


def keyword_search(  # 질문과 문서의 단어 교집합을 이용하는 2주차 검색 함수를 정의한다.
    query: str,  # 사용자가 입력한 검색 질문을 전달받는다.
    df: pd.DataFrame,  # 전처리된 문서가 들어 있는 DataFrame을 전달받는다.
    top_k: int = 3,  # 반환할 상위 검색 결과 수를 기본 3개로 지정한다.
) -> pd.DataFrame:  # 검색 결과를 DataFrame으로 반환한다.
    """질문과 문서에서 겹치는 단어 수를 기준으로 검색한다."""  # 함수의 역할을 설명하는 문서 문자열이다.
    query_clean = preprocess(query)  # 질문에도 문서와 동일한 전처리 함수를 적용한다.
    query_words = set(query_clean.split())  # 질문을 단어 집합으로 만들어 중복 단어를 제거한다.
    result_df = df.copy()  # 원본 DataFrame을 변경하지 않도록 복사본을 만든다.
    result_df["score"] = result_df["content_clean"].apply(  # 각 문서의 키워드 검색 점수를 계산한다.
        lambda content: len(query_words & set(content.split()))  # 질문 단어와 문서 단어의 교집합 크기를 점수로 사용한다.
    )  # 모든 문서에 대한 점수 계산을 끝낸다.
    result_df = result_df.sort_values(  # 계산된 점수를 기준으로 검색 결과를 정렬한다.
        by="score",  # score 컬럼을 정렬 기준으로 사용한다.
        ascending=False,  # 높은 점수가 먼저 나오도록 내림차순으로 정렬한다.
        kind="stable",  # 점수가 같은 문서는 기존 순서를 유지하도록 안정 정렬을 사용한다.
    )  # 검색 결과 정렬을 끝낸다.
    return result_df[["doc_id", "title", "category", "score"]].head(top_k)  # 상위 top_k개 결과만 반환한다.


def build_tfidf(df: pd.DataFrame) -> tuple:  # 전체 문서를 TF-IDF 벡터 행렬로 변환하는 2주차 함수를 정의한다.
    """전체 문서를 TF-IDF 벡터 행렬로 변환한다."""  # 함수의 역할을 설명하는 문서 문자열이다.
    vectorizer = TfidfVectorizer(  # TF-IDF 벡터화를 담당할 객체를 생성한다.
        max_features=5000,  # 전체 단어 중 최대 5,000개 특성까지만 사용한다.
        min_df=2,  # 최소 2개 이상의 문서에 등장한 단어만 단어 목록에 포함한다.
        stop_words="english",  # the, is, a 같은 영어 불용어를 검색 단어에서 제외한다.
    )  # TfidfVectorizer 설정을 끝낸다.
    tfidf_matrix = vectorizer.fit_transform(df["content_clean"])  # 전체 전처리 문서를 TF-IDF 행렬로 변환한다.
    feature_count = len(vectorizer.get_feature_names_out())  # TF-IDF에 실제 사용된 단어 수를 계산한다.
    print(f"TF-IDF 행렬 크기: {tfidf_matrix.shape} | 사용된 단어 수: {feature_count}")  # 벡터화 결과를 출력한다.
    return tfidf_matrix, vectorizer  # 검색 단계에서 재사용할 문서 행렬과 vectorizer를 함께 반환한다.


def tfidf_search(  # TF-IDF와 코사인 유사도를 이용하는 2주차 검색 함수를 정의한다.
    query: str,  # 사용자가 입력한 검색 질문을 전달받는다.
    df: pd.DataFrame,  # 검색 결과에 표시할 문서 정보가 담긴 DataFrame을 전달받는다.
    tfidf_matrix,  # build_tfidf에서 만든 전체 문서 TF-IDF 행렬을 전달받는다.
    vectorizer: TfidfVectorizer,  # 문서에 학습된 동일한 TfidfVectorizer를 전달받는다.
    top_k: int = 3,  # 반환할 상위 검색 결과 수를 기본 3개로 지정한다.
) -> pd.DataFrame:  # TF-IDF 검색 결과를 DataFrame으로 반환한다.
    """질문과 각 문서의 TF-IDF 코사인 유사도를 계산한다."""  # 함수의 역할을 설명하는 문서 문자열이다.
    query_clean = preprocess(query)  # 질문에 문서와 동일한 텍스트 전처리를 적용한다.
    query_vector = vectorizer.transform([query_clean]).toarray()[0]  # 질문을 문서와 같은 TF-IDF 벡터 공간에 변환한다.
    similarities = []  # 질문과 각 문서의 코사인 유사도를 저장할 빈 리스트를 만든다.

    for index in range(tfidf_matrix.shape[0]):  # 전체 문서 수만큼 반복하면서 각 문서와 질문을 비교한다.
        document_vector = tfidf_matrix[index].toarray()[0]  # 현재 문서 벡터를 NumPy 1차원 배열로 변환한다.
        similarity = cosine_similarity_numpy(query_vector, document_vector)  # 질문과 현재 문서의 유사도를 계산한다.
        similarities.append(similarity)  # 계산된 유사도를 리스트에 추가한다.

    similarities_array = np.array(similarities)  # 모든 문서 유사도를 NumPy 배열로 변환한다.
    top_indices = similarities_array.argsort()[::-1][:top_k]  # 유사도가 높은 상위 top_k 문서 위치를 구한다.
    result_df = df.iloc[top_indices][["doc_id", "title", "category"]].copy()  # 해당 문서 정보를 복사한다.
    result_df["similarity"] = similarities_array[top_indices]  # 각 문서의 유사도를 결과 컬럼에 저장한다.
    return result_df  # 유사도 순으로 정렬된 상위 검색 결과를 반환한다.


def precision_at_k(  # 기능 2인 Precision@k 계산 함수를 정의한다.
    retrieved_doc_ids: list[str],  # 검색 결과 문서 ID 목록을 순위 순서대로 전달받는다.
    relevant_doc_ids: list[str],  # 질문에 대한 정답 문서 ID 목록을 전달받는다.
    k: int,  # 평가할 상위 검색 결과 개수를 전달받는다.
) -> float:  # 상위 k개 중 정답 문서가 차지하는 비율을 실수로 반환한다.
    """검색 결과 상위 k개 중 정답 문서가 차지하는 비율을 계산한다."""  # 함수의 역할을 설명하는 문서 문자열이다.
    if k <= 0:  # k가 0 이하인지 확인한다.
        raise ValueError("k는 1 이상의 정수여야 합니다.")  # 0으로 나누는 오류를 막기 위해 명확한 예외를 발생시킨다.

    top_k_doc_ids = retrieved_doc_ids[:k]  # 검색 결과 목록에서 상위 k개 문서 ID만 자른다.
    retrieved_set = set(top_k_doc_ids)  # 상위 검색 결과를 중복 없는 집합으로 변환한다.
    relevant_set = set(relevant_doc_ids)  # 정답 문서 ID 목록을 빠르게 비교할 수 있는 집합으로 변환한다.
    correct_count = len(retrieved_set & relevant_set)  # 검색 결과와 정답 목록의 교집합 크기를 계산한다.
    return correct_count / k  # 정답 개수를 k로 나누어 Precision@k를 반환한다.


def reciprocal_rank(  # 기능 3인 첫 정답 순위의 역수 계산 함수를 정의한다.
    retrieved_doc_ids: list[str],  # 검색 결과 문서 ID 목록을 순위 순서대로 전달받는다.
    relevant_doc_ids: list[str],  # 질문에 대한 정답 문서 ID 목록을 전달받는다.
) -> float:  # 첫 번째 정답 순위의 역수를 실수로 반환한다.
    """검색 결과에서 처음 등장한 정답 문서 순위의 역수를 계산한다."""  # 함수의 역할을 설명하는 문서 문자열이다.
    relevant_set = set(relevant_doc_ids)  # 정답 포함 여부를 빠르게 확인하도록 정답 목록을 집합으로 변환한다.

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):  # 검색 결과를 1위부터 순서대로 확인한다.
        if doc_id in relevant_set:  # 현재 문서 ID가 정답 목록에 포함되는지 확인한다.
            return 1.0 / rank  # 처음 발견한 정답 순위의 역수를 즉시 반환한다.

    return 0.0  # 검색 결과 안에 정답이 하나도 없으면 0.0을 반환한다.


def run_day1_checks() -> None:  # 오늘 구현한 기능 1~3만 독립적으로 확인하는 임시 실행 함수를 정의한다.
    """3주차 첫날 기능 1~3의 동작을 확인한다."""  # 함수의 역할을 설명하는 문서 문자열이다.
    df = load_data(DATA_PATH)  # 평가셋 문서 ID가 들어 있는 기술 문서 데이터를 불러온다.
    existing_doc_ids = set(df["doc_id"])  # 실제 데이터에 존재하는 모든 문서 ID를 집합으로 만든다.
    eval_doc_ids = {  # 평가셋에 작성한 모든 정답 문서 ID를 하나의 집합으로 모은다.
        doc_id  # 각 정답 문서 ID를 집합의 원소로 추가한다.
        for item in EVAL_SET  # 평가셋의 모든 질문 항목을 순회한다.
        for doc_id in item["relevant_doc_ids"]  # 각 질문의 정답 문서 ID 목록을 순회한다.
    }  # 평가셋 정답 문서 ID 집합 생성을 끝낸다.
    invalid_doc_ids = eval_doc_ids - existing_doc_ids  # 실제 데이터에 없는 잘못된 정답 ID를 계산한다.

    if invalid_doc_ids:  # 잘못된 정답 문서 ID가 하나라도 있는지 확인한다.
        raise ValueError(f"평가셋에 존재하지 않는 문서 ID가 있습니다: {sorted(invalid_doc_ids)}")  # 잘못된 ID 목록을 출력한다.

    print(f"데이터 로드 완료: {df.shape[0]}행 × {df.shape[1]}열")  # 데이터 로드 결과를 출력한다.
    print(f"\n=== 기능 1: 평가셋 구성 ===")  # 기능 1 출력 영역의 제목을 표시한다.
    print(f"평가셋 크기: {len(EVAL_SET)}개 질문")  # 평가셋 질문 개수를 출력한다.

    for index, item in enumerate(EVAL_SET, start=1):  # 평가셋의 각 질문을 1번부터 순서대로 확인한다.
        print(f"{index:02d}. {item['query']} -> {item['relevant_doc_ids']}")  # 질문과 정답 문서 ID를 함께 출력한다.

    test_retrieved = ["D001", "D012", "D059"]  # Precision@k 확인용 검색 결과 문서 ID를 준비한다.
    test_relevant = ["D001", "D059"]  # Precision@k 확인용 정답 문서 ID를 준비한다.
    precision_result = precision_at_k(test_retrieved, test_relevant, k=TOP_K)  # Precision@3을 계산한다.

    print("\n=== 기능 2: Precision@k 테스트 ===")  # 기능 2 출력 영역의 제목을 표시한다.
    print(f"검색 결과: {test_retrieved}")  # 테스트 검색 결과를 출력한다.
    print(f"정답 문서: {test_relevant}")  # 테스트 정답 문서 목록을 출력한다.
    print(f"Precision@{TOP_K}: {precision_result:.4f}")  # 계산 결과를 소수점 넷째 자리까지 출력한다.

    first_rank_result = reciprocal_rank(["D001", "D012", "D059"], ["D001"])  # 정답이 1위인 경우를 계산한다.
    second_rank_result = reciprocal_rank(["D012", "D001", "D059"], ["D001"])  # 정답이 2위인 경우를 계산한다.
    missing_result = reciprocal_rank(["D012", "D013", "D014"], ["D001"])  # 정답이 없는 경우를 계산한다.

    print("\n=== 기능 3: Reciprocal Rank 테스트 ===")  # 기능 3 출력 영역의 제목을 표시한다.
    print(f"정답이 1위인 경우: {first_rank_result:.4f}")  # 1.0000이 나오는지 출력한다.
    print(f"정답이 2위인 경우: {second_rank_result:.4f}")  # 0.5000이 나오는지 출력한다.
    print(f"정답이 없는 경우: {missing_result:.4f}")  # 0.0000이 나오는지 출력한다.

    assert np.isclose(precision_result, 2 / 3)  # Precision@3 결과가 2/3과 같은지 검증한다.
    assert np.isclose(first_rank_result, 1.0)  # 첫 번째 정답의 역수가 1.0인지 검증한다.
    assert np.isclose(second_rank_result, 0.5)  # 두 번째 정답의 역수가 0.5인지 검증한다.
    assert np.isclose(missing_result, 0.0)  # 정답이 없는 경우 결과가 0.0인지 검증한다.
    print("\n기능 1~3 테스트 통과")  # 모든 검증이 성공했음을 출력한다.


if __name__ == "__main__":  # 이 파일이 직접 실행됐는지 확인한다.
    run_day1_checks()  # 직접 실행된 경우 오늘 구현한 기능 1~3의 테스트만 실행한다.