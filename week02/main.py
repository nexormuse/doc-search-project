import re  # 정규식을 이용한 텍스트 전처리를 위해 re 모듈을 불러온다.
from pathlib import Path  # 운영체제와 관계없이 파일 경로를 관리하기 위해 Path를 불러온다.

import numpy as np  # 벡터 내적, 노름, 배열 정렬 등에 사용할 NumPy를 불러온다.
import pandas as pd  # CSV 데이터 로드와 DataFrame 처리를 위해 pandas를 불러온다.
from sklearn.feature_extraction.text import TfidfVectorizer  # 문서를 TF-IDF 벡터로 변환하기 위해 불러온다.


DATA_PATH = Path("data/tech_docs.csv")  # 프로젝트 루트를 기준으로 기술 문서 CSV 파일 경로를 지정한다.
TOP_K = 3  # 검색 결과에서 반환할 상위 문서 개수를 3개로 지정한다.
QUERY = "how does gradient descent work in machine learning"  # 두 검색 방식에 공통으로 사용할 질문을 지정한다.


def load_data(file_path: Path) -> pd.DataFrame:  # CSV 파일을 불러오는 함수를 정의한다.
    """CSV 파일을 불러와 DataFrame으로 반환한다."""  # 함수의 역할을 설명하는 문서 문자열이다.

    if not file_path.exists():  # 전달받은 경로에 실제 파일이 존재하는지 확인한다.
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")  # 파일이 없으면 명확한 오류를 발생시킨다.

    df = pd.read_csv(file_path)  # CSV 파일을 읽어 pandas DataFrame으로 저장한다.

    required_columns = {  # 과제 실행에 필요한 필수 컬럼 집합을 정의한다.
        "doc_id",  # 각 문서를 구분하는 문서 ID 컬럼이다.
        "title",  # 문서 제목 컬럼이다.
        "category",  # 문서 카테고리 컬럼이다.
        "content",  # 실제 검색 대상이 되는 문서 본문 컬럼이다.
    }  # 필수 컬럼 집합 정의를 끝낸다.

    missing_columns = required_columns - set(df.columns)  # 필수 컬럼 중 실제 데이터에 없는 컬럼을 계산한다.

    if missing_columns:  # 누락된 필수 컬럼이 하나라도 있는지 확인한다.
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing_columns)}")  # 누락 컬럼 목록을 포함한 오류를 발생시킨다.

    return df  # 검증이 완료된 DataFrame을 호출한 위치로 반환한다.


def preprocess(text: str) -> str:  # 문자열 하나를 검색용 형태로 전처리하는 함수를 정의한다.
    """영문 텍스트를 검색에 적합한 형태로 정제한다."""  # 함수의 역할을 설명하는 문서 문자열이다.

    text = text.lower()  # 대문자와 소문자를 동일한 단어로 처리하기 위해 모두 소문자로 변환한다.
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # 영문 소문자, 숫자, 공백을 제외한 특수문자를 공백으로 바꾼다.
    text = re.sub(r"\s+", " ", text).strip()  # 여러 공백을 하나로 줄이고 문자열 앞뒤 공백을 제거한다.

    return text  # 전처리가 끝난 문자열을 반환한다.


def cosine_similarity_numpy(  # NumPy만 이용해 코사인 유사도를 계산하는 함수를 정의한다.
    a: np.ndarray,  # 첫 번째 벡터를 NumPy 배열로 전달받는다.
    b: np.ndarray,  # 두 번째 벡터를 NumPy 배열로 전달받는다.
) -> float:  # 계산된 코사인 유사도를 실수로 반환한다.
    """NumPy를 이용해 두 벡터의 코사인 유사도를 계산한다."""  # 함수의 역할을 설명하는 문서 문자열이다.

    a = np.asarray(a, dtype=float)  # 첫 번째 입력을 실수형 NumPy 배열로 변환한다.
    b = np.asarray(b, dtype=float)  # 두 번째 입력을 실수형 NumPy 배열로 변환한다.

    if a.shape != b.shape:  # 두 벡터의 차원과 크기가 동일한지 검사한다.
        raise ValueError("두 벡터의 크기가 같아야 합니다.")  # 크기가 다르면 내적할 수 없으므로 오류를 발생시킨다.

    norm_a = np.linalg.norm(a)  # 첫 번째 벡터의 유클리드 노름, 즉 벡터 크기를 계산한다.
    norm_b = np.linalg.norm(b)  # 두 번째 벡터의 유클리드 노름, 즉 벡터 크기를 계산한다.

    if norm_a == 0 or norm_b == 0:  # 두 벡터 중 하나라도 모든 값이 0인 영벡터인지 확인한다.
        return 0.0  # 분모가 0이 되는 문제를 막기 위해 유사도 0.0을 반환한다.

    dot_product = np.dot(a, b)  # 두 벡터의 내적 A·B를 계산한다.
    similarity = dot_product / (norm_a * norm_b)  # 내적을 두 벡터 노름의 곱으로 나누어 코사인 유사도를 계산한다.

    return float(similarity)  # NumPy 실수 값을 일반 Python float로 변환해 반환한다.


def keyword_search(  # 질문과 문서의 단어 교집합을 이용하는 Baseline 검색 함수를 정의한다.
    query: str,  # 사용자가 입력한 검색 질문을 전달받는다.
    df: pd.DataFrame,  # 전처리된 문서가 들어 있는 DataFrame을 전달받는다.
    top_k: int = 3,  # 반환할 상위 검색 결과 수를 기본 3개로 지정한다.
) -> pd.DataFrame:  # 검색 결과를 DataFrame으로 반환한다.
    """질문과 문서에서 겹치는 단어 수를 기준으로 검색한다."""  # 함수의 역할을 설명하는 문서 문자열이다.

    query_clean = preprocess(query)  # 질문에도 문서와 동일한 전처리 함수를 적용한다.
    query_words = set(query_clean.split())  # 전처리된 질문을 단어 단위로 나누고 중복을 제거한 집합으로 만든다.

    result_df = df.copy()  # 원본 DataFrame을 변경하지 않도록 복사본을 만든다.

    result_df["score"] = result_df["content_clean"].apply(  # 각 문서에 키워드 검색 점수를 계산해 score 컬럼에 저장한다.
        lambda content: len(query_words & set(content.split()))  # 질문 단어와 문서 단어의 교집합 크기를 점수로 사용한다.
    )  # 모든 문서에 대한 점수 계산을 끝낸다.

    result_df = result_df.sort_values(  # 계산된 점수를 기준으로 검색 결과를 정렬한다.
        by="score",  # score 컬럼을 정렬 기준으로 사용한다.
        ascending=False,  # 높은 점수가 먼저 나오도록 내림차순으로 정렬한다.
        kind="stable",  # 점수가 같은 문서는 기존 순서를 유지하도록 안정 정렬을 사용한다.
    )  # 검색 결과 정렬을 끝낸다.

    return result_df[  # 제출 요구사항에 필요한 컬럼만 선택해 반환한다.
        ["doc_id", "title", "category", "score"]  # 문서 ID, 제목, 카테고리, 키워드 점수만 포함한다.
    ].head(top_k)  # 정렬된 결과 중 상위 top_k개 문서만 반환한다.


def build_tfidf(  # 전체 문서를 TF-IDF 벡터 행렬로 변환하는 기능 4 함수를 정의한다.
    df: pd.DataFrame,  # content_clean 컬럼이 포함된 DataFrame을 전달받는다.
) -> tuple:  # TF-IDF 행렬과 학습된 vectorizer를 튜플로 반환한다.
    """전체 문서를 TF-IDF 벡터 행렬로 변환한다."""  # 함수의 역할을 설명하는 문서 문자열이다.

    vectorizer = TfidfVectorizer(  # TF-IDF 벡터화를 담당할 객체를 생성한다.
        max_features=5000,  # 전체 단어 중 최대 5,000개 특성까지만 사용한다.
        min_df=2,  # 최소 2개 이상의 문서에 등장한 단어만 TF-IDF 단어 목록에 포함한다.
        stop_words="english",  # the, is, a 같은 영어 불용어를 검색 단어에서 제외한다.
    )  # TfidfVectorizer 설정을 끝낸다.

    tfidf_matrix = vectorizer.fit_transform(  # 문서에서 단어 목록과 IDF를 학습하고 전체 문서를 벡터로 변환한다.
        df["content_clean"]  # 전처리가 완료된 content_clean 컬럼을 벡터화 대상으로 사용한다.
    )  # 문서 TF-IDF 행렬 생성을 끝낸다.

    feature_names = vectorizer.get_feature_names_out()  # TF-IDF에 실제 사용된 전체 단어 목록을 가져온다.
    feature_count = len(feature_names)  # 사용된 전체 단어 수를 계산한다.

    print(  # TF-IDF 벡터화 결과를 터미널에 출력한다.
        f"TF-IDF 행렬 크기: {tfidf_matrix.shape} | 사용된 단어 수: {feature_count}"  # 문서 수와 단어 수 및 전체 특성 수를 출력한다.
    )  # TF-IDF 정보 출력을 끝낸다.

    return tfidf_matrix, vectorizer  # 검색 단계에서 재사용할 문서 행렬과 vectorizer를 함께 반환한다.


def tfidf_search(  # TF-IDF와 코사인 유사도를 이용하는 기능 5 검색 함수를 정의한다.
    query: str,  # 사용자가 입력한 검색 질문을 전달받는다.
    df: pd.DataFrame,  # 검색 결과에 표시할 문서 정보가 담긴 DataFrame을 전달받는다.
    tfidf_matrix,  # build_tfidf에서 만든 전체 문서 TF-IDF 행렬을 전달받는다.
    vectorizer: TfidfVectorizer,  # 문서에 학습된 동일한 TfidfVectorizer를 전달받는다.
    top_k: int = 3,  # 반환할 상위 검색 결과 수를 기본 3개로 지정한다.
) -> pd.DataFrame:  # TF-IDF 검색 결과를 DataFrame으로 반환한다.
    """질문과 각 문서의 TF-IDF 코사인 유사도를 계산한다."""  # 함수의 역할을 설명하는 문서 문자열이다.

    query_clean = preprocess(query)  # 질문에 문서와 동일한 텍스트 전처리를 적용한다.

    query_vector = vectorizer.transform(  # 기존 문서에 학습된 vectorizer로 질문을 같은 벡터 공간에 변환한다.
        [query_clean]  # transform은 문서 목록 형태를 받으므로 질문 문자열을 리스트로 감싼다.
    ).toarray()[0]  # 희소 행렬을 NumPy 배열로 변환하고 첫 번째 질문 벡터를 선택한다.

    similarities = []  # 질문과 각 문서의 코사인 유사도를 저장할 빈 리스트를 만든다.

    for index in range(tfidf_matrix.shape[0]):  # 전체 문서 수만큼 반복하면서 각 문서와 질문을 비교한다.
        document_vector = tfidf_matrix[index].toarray()[0]  # 현재 문서 한 행만 NumPy 1차원 배열로 변환한다.

        similarity = cosine_similarity_numpy(  # 기능 2에서 직접 구현한 코사인 유사도 함수를 호출한다.
            query_vector,  # 첫 번째 벡터로 TF-IDF 질문 벡터를 전달한다.
            document_vector,  # 두 번째 벡터로 현재 문서의 TF-IDF 벡터를 전달한다.
        )  # 현재 문서와 질문의 유사도 계산을 끝낸다.

        similarities.append(similarity)  # 계산된 현재 문서의 유사도를 리스트에 추가한다.

    similarities_array = np.array(similarities)  # 모든 문서 유사도를 정렬하기 쉽도록 NumPy 배열로 변환한다.

    top_indices = similarities_array.argsort()[::-1][:top_k]  # 유사도가 높은 순으로 정렬한 상위 top_k 문서 위치를 구한다.

    result_df = df.iloc[top_indices][  # 상위 문서의 위치를 이용해 원본 DataFrame에서 해당 행을 선택한다.
        ["doc_id", "title", "category"]  # 결과에 문서 ID, 제목, 카테고리 컬럼을 포함한다.
    ].copy()  # 원본 DataFrame에 영향을 주지 않도록 검색 결과를 복사한다.

    result_df["similarity"] = similarities_array[top_indices]  # 선택된 각 문서의 코사인 유사도를 similarity 컬럼에 저장한다.

    return result_df  # 유사도 순으로 정렬된 상위 TF-IDF 검색 결과를 반환한다.


def test_cosine_similarity() -> None:  # 기능 2의 코사인 유사도 함수를 검증하는 보조 함수를 정의한다.
    """같은 벡터, 직교 벡터, 영벡터의 결과를 확인한다."""  # 함수의 역할을 설명하는 문서 문자열이다.

    same_result = cosine_similarity_numpy(  # 같은 두 벡터의 유사도를 계산한다.
        np.array([1, 2, 3]),  # 첫 번째 테스트 벡터를 생성한다.
        np.array([1, 2, 3]),  # 첫 번째 벡터와 동일한 두 번째 테스트 벡터를 생성한다.
    )  # 같은 벡터 유사도 계산을 끝낸다.

    orthogonal_result = cosine_similarity_numpy(  # 서로 직교하는 두 벡터의 유사도를 계산한다.
        np.array([1, 0]),  # x축 방향 벡터를 생성한다.
        np.array([0, 1]),  # y축 방향 벡터를 생성한다.
    )  # 직교 벡터 유사도 계산을 끝낸다.

    zero_result = cosine_similarity_numpy(  # 영벡터가 포함된 경우의 결과를 계산한다.
        np.array([0, 0]),  # 크기가 0인 영벡터를 생성한다.
        np.array([1, 1]),  # 비교할 일반 벡터를 생성한다.
    )  # 영벡터 유사도 계산을 끝낸다.

    print(f"같은 벡터: {same_result:.4f}")  # 같은 벡터 결과가 1.0000인지 출력한다.
    print(f"직교 벡터: {orthogonal_result:.4f}")  # 직교 벡터 결과가 0.0000인지 출력한다.
    print(f"영벡터 포함: {zero_result:.4f}")  # 영벡터 입력 결과가 0.0000인지 출력한다.


def main() -> None:  # 기능 1부터 기능 6까지 순서대로 실행하는 최종 main 함수를 정의한다.
    """2주차 벡터화 및 코사인 유사도 과제 전체 기능을 실행한다."""  # 함수의 역할을 설명하는 문서 문자열이다.

    df = load_data(DATA_PATH)  # 기능 1~6에서 사용할 기술 문서 CSV 데이터를 불러온다.

    print(f"데이터 로드 완료: {df.shape[0]}행 × {df.shape[1]}열")  # 불러온 데이터의 행과 열 개수를 출력한다.

    df = df.dropna(subset=["content"]).copy()  # content가 비어 있는 행을 제거하고 새로운 DataFrame으로 복사한다.
    df["content_clean"] = df["content"].apply(preprocess)  # 모든 문서 본문에 기능 1 전처리를 적용해 새 컬럼을 만든다.

    print("\n=== 기능 1: 텍스트 전처리 ===")  # 기능 1 출력 영역의 제목을 표시한다.
    print(  # 전처리 전후 내용을 비교할 수 있도록 출력한다.
        df[["content", "content_clean"]].head(3).to_string(index=False)  # 상위 3개 문서의 원문과 전처리 결과를 표시한다.
    )  # 전처리 결과 출력을 끝낸다.

    print("\n=== 기능 2: 코사인 유사도 테스트 ===")  # 기능 2 출력 영역의 제목을 표시한다.
    test_cosine_similarity()  # 같은 벡터, 직교 벡터, 영벡터 테스트를 실행한다.

    print("\n=== 기능 4: TF-IDF 벡터화 ===")  # 기능 4 출력 영역의 제목을 표시한다.
    tfidf_matrix, vectorizer = build_tfidf(df)  # 전체 문서를 TF-IDF 벡터로 바꾸고 행렬과 vectorizer를 전달받는다.

    print(f"\n질문: {QUERY}")  # Keyword Baseline과 TF-IDF 검색에 사용할 동일한 질문을 출력한다.

    print("\n=== 기능 3: Keyword Baseline ===")  # 기능 3 검색 결과 영역의 제목을 표시한다.
    keyword_result = keyword_search(  # 키워드 교집합 기반 검색을 실행한다.
        query=QUERY,  # 미리 정의한 공통 질문을 전달한다.
        df=df,  # 전처리가 완료된 문서 DataFrame을 전달한다.
        top_k=TOP_K,  # 상위 3개 문서를 요청한다.
    )  # Keyword Baseline 검색을 끝낸다.
    print(keyword_result.to_string(index=False))  # Keyword Baseline Top-3 결과를 인덱스 없이 출력한다.

    print("\n=== 기능 5: TF-IDF Search ===")  # 기능 5 검색 결과 영역의 제목을 표시한다.
    tfidf_result = tfidf_search(  # TF-IDF와 코사인 유사도 기반 검색을 실행한다.
        query=QUERY,  # Keyword Baseline과 동일한 질문을 전달한다.
        df=df,  # 문서 정보가 담긴 DataFrame을 전달한다.
        tfidf_matrix=tfidf_matrix,  # 기능 4에서 생성한 전체 문서 TF-IDF 행렬을 전달한다.
        vectorizer=vectorizer,  # 기능 4에서 문서에 학습한 동일한 vectorizer를 전달한다.
        top_k=TOP_K,  # 상위 3개 문서를 요청한다.
    )  # TF-IDF 검색을 끝낸다.

    print(  # TF-IDF 검색 결과를 지정된 형식으로 출력한다.
        tfidf_result.to_string(  # DataFrame을 터미널 출력용 문자열로 변환한다.
            index=False,  # DataFrame의 기본 숫자 인덱스를 출력하지 않는다.
            formatters={"similarity": lambda value: f"{value:.4f}"},  # similarity 값을 소수점 넷째 자리까지 표시한다.
        )  # 검색 결과 문자열 변환을 끝낸다.
    )  # TF-IDF 검색 결과 출력을 끝낸다.

    # 기능 6 비교 결과: Keyword Baseline은 겹치는 단어 수만 계산하므로 흔한 단어가 포함된 무관한 문서도 상위에 나올 수 있다.
    # 기능 6 비교 결과: TF-IDF는 흔한 단어의 가중치를 낮추고 구별력 높은 단어의 가중치를 높여 관련 문서를 더 잘 구분한다.


if __name__ == "__main__":  # 이 파일이 다른 파일에서 import된 것이 아니라 직접 실행됐는지 확인한다.
    main()  # 직접 실행된 경우에만 기능 1부터 기능 6까지 전체 프로그램을 실행한다.