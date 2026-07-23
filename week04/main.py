import re  # 정규식을 이용한 텍스트 전처리를 위해 re 모듈을 불러온다.
from pathlib import Path  # 운영체제와 관계없이 파일 경로를 관리하기 위해 Path를 불러온다.

import numpy as np  # 벡터 계산, 정렬, 평균 계산에 사용할 NumPy를 불러온다.
import pandas as pd  # CSV 데이터 로드와 DataFrame 처리를 위해 pandas를 불러온다.
from sklearn.feature_extraction.text import TfidfVectorizer  # 문서를 TF-IDF 벡터로 변환하기 위해 불러온다.


DATA_PATH = Path("data/tech_docs.csv")  # 프로젝트 루트를 기준으로 기술 문서 CSV 파일 경로를 지정한다.
TOP_K = 3  # 검색과 평가에서 확인할 상위 문서 개수를 3개로 지정한다.
EXAMPLE_QUERY = "git merge conflicts"  # 전체 파이프라인의 검색 동작을 확인할 예시 질문을 지정한다.


EVAL_SET = [  # 검색 성능 평가에 사용할 질문과 정답 문서 ID 목록을 정의한다.
    {
        "query": "how can I create lists with a concise for expression",
        "relevant_doc_ids": ["D001", "D059"],
    },
    {
        "query": "how do I handle errors with try except and finally",
        "relevant_doc_ids": ["D005"],
    },
    {
        "query": "how do I open a file and make sure it closes automatically",
        "relevant_doc_ids": ["D008", "D051"],
    },
    {
        "query": "how do I combine changes from two git branches",
        "relevant_doc_ids": ["D014"],
    },
    {
        "query": "how can I temporarily save uncommitted changes in git",
        "relevant_doc_ids": ["D019"],
    },
    {
        "query": "how does gradient descent update model parameters",
        "relevant_doc_ids": ["D023"],
    },
    {
        "query": "what is the difference between overfitting and underfitting",
        "relevant_doc_ids": ["D026"],
    },
    {
        "query": "how are gradients computed through neural network layers",
        "relevant_doc_ids": ["D030"],
    },
    {
        "query": "how do numpy arrays operate when their shapes are different",
        "relevant_doc_ids": ["D034"],
    },
    {
        "query": "how can I filter numpy array elements with boolean conditions",
        "relevant_doc_ids": ["D033", "D039"],
    },
    {
        "query": "how do I remove or fill missing values in pandas",
        "relevant_doc_ids": ["D043"],
    },
    {
        "query": "what is the difference between loc and iloc in pandas",
        "relevant_doc_ids": ["D046"],
    },
]


def load_data(file_path: Path) -> pd.DataFrame:
    """CSV 파일을 불러와 필수 컬럼을 확인한 뒤 DataFrame으로 반환한다."""
    if not file_path.exists():  # 전달받은 경로에 실제 파일이 존재하는지 확인한다.
        raise FileNotFoundError(  # 파일이 없을 때 실행 위치와 경로를 확인할 수 있도록 오류를 발생시킨다.
            f"데이터 파일을 찾을 수 없습니다: {file_path}\n"
            "프로젝트 루트에서 `python week04/main.py`로 실행했는지 확인하세요."
        )

    df = pd.read_csv(file_path)  # CSV 파일을 읽어 pandas DataFrame으로 저장한다.
    required_columns = {"doc_id", "title", "category", "content"}  # 파이프라인 실행에 필요한 필수 컬럼을 정의한다.
    missing_columns = required_columns - set(df.columns)  # 실제 데이터에서 누락된 필수 컬럼을 계산한다.

    if missing_columns:  # 누락된 필수 컬럼이 하나라도 있는지 확인한다.
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing_columns)}")  # 누락된 컬럼 목록을 포함해 오류를 발생시킨다.

    return df  # 검증이 끝난 DataFrame을 반환한다.


def preprocess(text: str) -> str:
    """영문 텍스트를 검색에 적합한 형태로 정제한다."""
    text = str(text).lower()  # 입력값을 문자열로 바꾸고 대소문자 차이를 없애기 위해 소문자로 변환한다.
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # 영문, 숫자, 공백을 제외한 특수문자를 공백으로 바꾼다.
    text = re.sub(r"\s+", " ", text).strip()  # 연속된 공백을 하나로 줄이고 앞뒤 공백을 제거한다.
    return text  # 전처리가 끝난 문자열을 반환한다.


def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """NumPy를 이용해 두 벡터의 코사인 유사도를 계산한다."""
    a = np.asarray(a, dtype=float)  # 첫 번째 입력을 실수형 NumPy 배열로 변환한다.
    b = np.asarray(b, dtype=float)  # 두 번째 입력을 실수형 NumPy 배열로 변환한다.

    if a.shape != b.shape:  # 두 벡터의 차원과 크기가 같은지 확인한다.
        raise ValueError("두 벡터의 크기가 같아야 합니다.")  # 크기가 다르면 내적할 수 없으므로 오류를 발생시킨다.

    norm_a = np.linalg.norm(a)  # 첫 번째 벡터의 크기를 계산한다.
    norm_b = np.linalg.norm(b)  # 두 번째 벡터의 크기를 계산한다.

    if norm_a == 0 or norm_b == 0:  # 두 벡터 중 하나라도 영벡터인지 확인한다.
        return 0.0  # 0으로 나누는 문제를 막기 위해 유사도 0.0을 반환한다.

    dot_product = np.dot(a, b)  # 두 벡터의 내적을 계산한다.
    similarity = dot_product / (norm_a * norm_b)  # 내적을 두 벡터 크기의 곱으로 나눈다.
    return float(similarity)  # NumPy 실수 값을 Python float로 변환해 반환한다.


def keyword_search(
    query: str,
    df: pd.DataFrame,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """질문과 문서에서 겹치는 단어 수를 기준으로 검색한다."""
    query_clean = preprocess(query)  # 질문에도 문서와 같은 전처리를 적용한다.
    query_words = set(query_clean.split())  # 질문 단어를 집합으로 바꾸어 중복을 제거한다.
    result_df = df.copy()  # 원본 DataFrame을 변경하지 않도록 복사본을 만든다.

    result_df["score"] = result_df["content_clean"].apply(  # 각 문서의 키워드 점수를 계산한다.
        lambda content: len(query_words & set(content.split()))  # 질문과 문서의 공통 단어 개수를 점수로 사용한다.
    )

    result_df = result_df.sort_values(  # 계산된 키워드 점수를 기준으로 정렬한다.
        by="score",  # score 컬럼을 정렬 기준으로 사용한다.
        ascending=False,  # 높은 점수가 먼저 나오도록 내림차순으로 정렬한다.
        kind="stable",  # 점수가 같으면 원래 문서 순서를 유지한다.
    )

    return result_df[["doc_id", "title", "category", "score"]].head(top_k)  # 상위 top_k개 결과만 반환한다.


def build_tfidf(df: pd.DataFrame) -> tuple:
    """전처리된 전체 문서를 TF-IDF 벡터 행렬로 변환한다."""
    vectorizer = TfidfVectorizer(  # TF-IDF 벡터화를 담당할 객체를 생성한다.
        max_features=5000,  # 전체 단어 중 최대 5,000개 특성까지만 사용한다.
        min_df=2,  # 최소 2개 이상의 문서에 등장한 단어만 특성에 포함한다.
        stop_words="english",  # 영어 불용어를 제외한다.
    )

    tfidf_matrix = vectorizer.fit_transform(df["content_clean"])  # 전체 문서를 TF-IDF 행렬로 변환한다.
    feature_count = len(vectorizer.get_feature_names_out())  # 실제 사용된 단어 특성 수를 계산한다.

    print(  # 벡터화 결과를 실행 화면에 출력한다.
        f"TF-IDF 행렬 크기: {tfidf_matrix.shape} | "
        f"사용된 단어 수: {feature_count}"
    )

    return tfidf_matrix, vectorizer  # 검색에 재사용할 문서 행렬과 vectorizer를 반환한다.


def tfidf_search(
    query: str,
    df: pd.DataFrame,
    tfidf_matrix,
    vectorizer: TfidfVectorizer,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """질문과 각 문서의 TF-IDF 코사인 유사도를 계산해 상위 문서를 반환한다."""
    query_clean = preprocess(query)  # 질문에 문서와 동일한 전처리를 적용한다.
    query_vector = vectorizer.transform([query_clean]).toarray()[0]  # 질문을 같은 TF-IDF 공간의 벡터로 변환한다.
    similarities = []  # 질문과 각 문서의 유사도를 저장할 리스트를 만든다.

    for index in range(tfidf_matrix.shape[0]):  # 전체 문서를 하나씩 순회한다.
        document_vector = tfidf_matrix[index].toarray()[0]  # 현재 문서 벡터를 NumPy 1차원 배열로 변환한다.
        similarity = cosine_similarity_numpy(query_vector, document_vector)  # 질문과 현재 문서의 유사도를 계산한다.
        similarities.append(similarity)  # 계산된 유사도를 리스트에 저장한다.

    similarities_array = np.array(similarities)  # 전체 유사도 목록을 NumPy 배열로 변환한다.
    top_indices = similarities_array.argsort()[::-1][:top_k]  # 유사도가 높은 상위 top_k 문서의 위치를 구한다.

    result_df = df.iloc[top_indices][["doc_id", "title", "category"]].copy()  # 상위 문서 정보를 복사한다.
    result_df["similarity"] = similarities_array[top_indices]  # 각 문서의 유사도를 결과에 추가한다.

    return result_df  # 유사도가 높은 순서의 검색 결과를 반환한다.


def precision_at_k(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
    k: int,
) -> float:
    """검색 결과 상위 k개 중 정답 문서가 차지하는 비율을 계산한다."""
    if k <= 0:  # k가 올바른 범위인지 확인한다.
        raise ValueError("k는 1 이상의 정수여야 합니다.")  # 0으로 나누는 문제를 막기 위해 오류를 발생시킨다.

    top_k_doc_ids = retrieved_doc_ids[:k]  # 검색 결과 중 상위 k개 문서 ID만 가져온다.
    retrieved_set = set(top_k_doc_ids)  # 상위 검색 결과를 집합으로 변환한다.
    relevant_set = set(relevant_doc_ids)  # 정답 문서 ID를 집합으로 변환한다.
    correct_count = len(retrieved_set & relevant_set)  # 검색 결과와 정답의 교집합 크기를 계산한다.

    return correct_count / k  # 정답 문서 수를 k로 나누어 Precision@k를 반환한다.


def reciprocal_rank(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
) -> float:
    """검색 결과에서 처음 등장한 정답 문서 순위의 역수를 계산한다."""
    relevant_set = set(relevant_doc_ids)  # 정답 포함 여부를 빠르게 확인하도록 집합으로 변환한다.

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):  # 검색 결과를 1위부터 순서대로 확인한다.
        if doc_id in relevant_set:  # 현재 문서가 정답 목록에 포함되는지 확인한다.
            return 1.0 / rank  # 첫 정답 순위의 역수를 반환한다.

    return 0.0  # 검색 결과 안에 정답이 없으면 0.0을 반환한다.


def validate_eval_set(
    eval_set: list[dict],
    df: pd.DataFrame,
) -> None:
    """평가셋의 개수, 필수 키, 정답 수, 실제 문서 ID 존재 여부를 검사한다."""
    if len(eval_set) < 10:  # 평가 질문이 최소 요구 수보다 적은지 확인한다.
        raise ValueError("평가셋은 최소 10개 이상의 질문으로 구성해야 합니다.")

    existing_doc_ids = set(df["doc_id"].astype(str))  # 실제 CSV에 존재하는 문서 ID를 집합으로 만든다.

    for index, item in enumerate(eval_set, start=1):  # 평가 항목을 하나씩 검사한다.
        if "query" not in item or "relevant_doc_ids" not in item:  # 필수 키가 있는지 확인한다.
            raise ValueError(f"평가셋 {index}번 항목에 필수 키가 없습니다.")

        query = str(item["query"]).strip()  # 질문을 문자열로 바꾸고 앞뒤 공백을 제거한다.
        relevant_doc_ids = item["relevant_doc_ids"]  # 정답 문서 ID 목록을 가져온다.

        if not query:  # 질문이 비어 있는지 확인한다.
            raise ValueError(f"평가셋 {index}번 질문이 비어 있습니다.")

        if not isinstance(relevant_doc_ids, list):  # 정답 문서 ID가 리스트인지 확인한다.
            raise TypeError(f"평가셋 {index}번 relevant_doc_ids는 리스트여야 합니다.")

        if not 1 <= len(relevant_doc_ids) <= 3:  # 질문마다 정답이 1~3개인지 확인한다.
            raise ValueError(f"평가셋 {index}번 정답 문서는 1~3개여야 합니다.")

        invalid_doc_ids = set(relevant_doc_ids) - existing_doc_ids  # 실제 데이터에 없는 정답 ID를 계산한다.

        if invalid_doc_ids:  # 잘못된 문서 ID가 하나라도 있는지 확인한다.
            raise ValueError(
                f"평가셋 {index}번에 존재하지 않는 문서 ID가 있습니다: "
                f"{sorted(invalid_doc_ids)}"
            )


def run_evaluation(
    eval_set: list[dict],
    search_fn,
    k: int = TOP_K,
) -> dict[str, float]:
    """평가셋 전체의 평균 Precision@k와 MRR을 계산한다."""
    if not eval_set:  # 평가셋이 비어 있는지 확인한다.
        raise ValueError("평가셋이 비어 있습니다.")

    if k <= 0:  # k가 올바른 범위인지 확인한다.
        raise ValueError("k는 1 이상의 정수여야 합니다.")

    precision_scores = []  # 각 질문의 Precision@k를 저장할 리스트를 만든다.
    reciprocal_rank_scores = []  # 각 질문의 Reciprocal Rank를 저장할 리스트를 만든다.

    for item in eval_set:  # 모든 평가 질문을 차례대로 순회한다.
        query = item["query"]  # 현재 질문을 가져온다.
        relevant_doc_ids = item["relevant_doc_ids"]  # 현재 질문의 정답 문서 ID를 가져온다.
        result = search_fn(query)  # 전달받은 검색 함수로 질문을 검색한다.

        if "doc_id" not in result.columns:  # 검색 결과에 평가용 doc_id 컬럼이 있는지 확인한다.
            raise ValueError("검색 결과 DataFrame에 doc_id 컬럼이 필요합니다.")

        retrieved_doc_ids = result["doc_id"].astype(str).tolist()  # 검색 결과 문서 ID를 순서가 유지된 리스트로 만든다.

        precision_scores.append(  # 현재 질문의 Precision@k를 계산해 저장한다.
            precision_at_k(retrieved_doc_ids, relevant_doc_ids, k)
        )
        reciprocal_rank_scores.append(  # 현재 질문의 Reciprocal Rank를 계산해 저장한다.
            reciprocal_rank(retrieved_doc_ids, relevant_doc_ids)
        )

    return {  # 전체 질문의 평균 지표를 딕셔너리로 반환한다.
        f"Precision@{k}": float(np.mean(precision_scores)),
        "MRR": float(np.mean(reciprocal_rank_scores)),
    }


def analyze_failures(
    eval_set: list[dict],
    search_fn,
    k: int = TOP_K,
) -> None:
    """정답 문서를 Top-k 안에서 찾지 못한 질문과 검색 결과를 출력한다."""
    if k <= 0:  # k가 올바른 범위인지 확인한다.
        raise ValueError("k는 1 이상의 정수여야 합니다.")

    failures = []  # 실패한 질문 정보를 저장할 리스트를 만든다.

    for item in eval_set:  # 평가셋의 모든 질문을 순회한다.
        query = item["query"]  # 현재 질문을 가져온다.
        relevant_doc_ids = item["relevant_doc_ids"]  # 현재 질문의 정답 문서 ID를 가져온다.
        result = search_fn(query)  # 검색 결과를 가져온다.
        retrieved_doc_ids = result["doc_id"].astype(str).tolist()[:k]  # 상위 k개 문서 ID만 가져온다.

        if reciprocal_rank(retrieved_doc_ids, relevant_doc_ids) == 0.0:  # Top-k 안에 정답이 없는지 확인한다.
            failures.append(
                {
                    "query": query,
                    "relevant_doc_ids": relevant_doc_ids,
                    "retrieved_doc_ids": retrieved_doc_ids,
                }
            )

    print(f"\n=== 실패 케이스: Top-{k} 안에 정답 없음 ===")  # 실패 분석 영역의 제목을 출력한다.

    if not failures:  # 실패 사례가 하나도 없는지 확인한다.
        print("실패 케이스가 없습니다.")  # 실패가 없음을 명확히 출력한다.
        return  # 더 출력할 내용이 없으므로 함수를 종료한다.

    for failure in failures:  # 실패 사례를 하나씩 출력한다.
        print(f"질문: {failure['query']}")
        print(f"정답 doc_id: {failure['relevant_doc_ids']}")
        print(f"검색 결과: {failure['retrieved_doc_ids']}")
        print("-" * 60)


def main() -> None:
    """로드 → 전처리 → 벡터화 → 검색 → 평가 → 오류 분석 전체 파이프라인을 실행한다."""
    print("=== [LLM] 과제 4 기능 1: 전체 파이프라인 통합 ===")  # 프로그램 시작과 오늘 구현 범위를 출력한다.

    df = load_data(DATA_PATH)  # 기술 문서 CSV를 불러온다.
    print(f"데이터 로드 완료: {df.shape[0]}행 × {df.shape[1]}열")  # 데이터 크기를 출력한다.

    df["content"] = df["content"].fillna("").astype(str)  # 본문 결측치를 빈 문자열로 바꾸고 문자열 형식으로 통일한다.
    df["content_clean"] = df["content"].apply(preprocess)  # 모든 문서 본문을 검색용 텍스트로 전처리한다.

    validate_eval_set(EVAL_SET, df)  # 평가셋의 형식과 문서 ID를 검증한다.
    print(f"평가셋 크기: {len(EVAL_SET)}개 질문")  # 평가 질문 수를 출력한다.

    tfidf_matrix, vectorizer = build_tfidf(df)  # 전체 문서의 TF-IDF 행렬을 만든다.

    keyword_search_fn = lambda query: keyword_search(  # Keyword Baseline을 질문 하나만 받는 형태로 감싼다.
        query,
        df,
        top_k=TOP_K,
    )

    tfidf_search_fn = lambda query: tfidf_search(  # TF-IDF 검색기를 질문 하나만 받는 형태로 감싼다.
        query,
        df,
        tfidf_matrix,
        vectorizer,
        top_k=TOP_K,
    )

    example_result = tfidf_search_fn(EXAMPLE_QUERY)  # 예시 질문으로 TF-IDF 검색을 실행한다.
    print(f"\n=== 예시 검색: {EXAMPLE_QUERY} ===")  # 예시 검색 영역의 제목을 출력한다.
    print(  # doc_id, 제목, 카테고리, 유사도를 표 형태로 출력한다.
        example_result.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    keyword_metrics = run_evaluation(  # 전체 평가셋으로 Keyword Baseline 성능을 계산한다.
        EVAL_SET,
        keyword_search_fn,
        k=TOP_K,
    )
    tfidf_metrics = run_evaluation(  # 같은 평가셋으로 TF-IDF 검색 성능을 계산한다.
        EVAL_SET,
        tfidf_search_fn,
        k=TOP_K,
    )

    comparison_df = pd.DataFrame(  # 두 검색 방식의 지표를 비교할 표를 만든다.
        [
            {"검색 방식": "Keyword Baseline", **keyword_metrics},
            {"검색 방식": "TF-IDF", **tfidf_metrics},
        ]
    )

    print("\n=== 성능 비교 ===")  # 성능 비교 영역의 제목을 출력한다.
    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    analyze_failures(  # TF-IDF 검색이 정답을 Top-3 안에서 찾지 못한 질문을 출력한다.
        EVAL_SET,
        tfidf_search_fn,
        k=TOP_K,
    )


if __name__ == "__main__":  # 이 파일이 직접 실행되었는지 확인한다.
    main()  # 전체 검색 파이프라인을 실행한다.