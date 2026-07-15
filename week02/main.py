import re
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/tech_docs.csv")
TOP_K = 3
QUERY = "how does gradient descent work in machine learning"


def load_data(file_path: Path) -> pd.DataFrame:
    """CSV 파일을 불러와 DataFrame으로 반환한다."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = {
        "doc_id",
        "title",
        "category",
        "content",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"필수 컬럼이 없습니다: {sorted(missing_columns)}"
        )

    return df


def preprocess(text: str) -> str:
    """영문 텍스트를 검색에 적합한 형태로 정제한다."""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def cosine_similarity_numpy(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """NumPy를 이용해 두 벡터의 코사인 유사도를 계산한다."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("두 벡터의 크기가 같아야 합니다.")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    dot_product = np.dot(a, b)
    similarity = dot_product / (norm_a * norm_b)

    return float(similarity)


def keyword_search(
    query: str,
    df: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """질문과 문서에서 겹치는 단어 수를 기준으로 검색한다."""

    query_clean = preprocess(query)
    query_words = set(query_clean.split())

    result_df = df.copy()

    result_df["score"] = result_df["content_clean"].apply(
        lambda content: len(
            query_words & set(content.split())
        )
    )

    result_df = result_df.sort_values(
        by="score",
        ascending=False,
        kind="stable",
    )

    return result_df[
        ["doc_id", "title", "category", "score"]
    ].head(top_k)


def run_features_1_to_3() -> None:
    """오늘 구현한 기능 1~3을 실행하고 확인한다."""

    df = load_data(DATA_PATH)

    print(
        f"데이터 로드 완료: "
        f"{df.shape[0]}행 × {df.shape[1]}열"
    )

    # 기능 1: 텍스트 전처리
    df = df.dropna(subset=["content"]).copy()
    df["content_clean"] = df["content"].apply(preprocess)

    print("\n=== 기능 1: 텍스트 전처리 ===")
    print(
        df[["content", "content_clean"]]
        .head(3)
        .to_string(index=False)
    )

    # 기능 2: 코사인 유사도
    print("\n=== 기능 2: 코사인 유사도 테스트 ===")

    same_result = cosine_similarity_numpy(
        np.array([1, 2, 3]),
        np.array([1, 2, 3]),
    )

    orthogonal_result = cosine_similarity_numpy(
        np.array([1, 0]),
        np.array([0, 1]),
    )

    zero_result = cosine_similarity_numpy(
        np.array([0, 0]),
        np.array([1, 1]),
    )

    print(f"같은 벡터: {same_result:.4f}")
    print(f"직교 벡터: {orthogonal_result:.4f}")
    print(f"영벡터 포함: {zero_result:.4f}")

    # 기능 3: 키워드 Baseline 검색
    print(f"\n질문: {QUERY}")
    print("\n=== 기능 3: Keyword Baseline ===")

    keyword_result = keyword_search(
        query=QUERY,
        df=df,
        top_k=TOP_K,
    )

    print(keyword_result.to_string(index=False))


if __name__ == "__main__":
    run_features_1_to_3()