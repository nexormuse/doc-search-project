"""LLM Infra 마일스톤 1주차 과제.

기술 문서 CSV 데이터를 불러온 뒤 다음 내용을 탐색합니다.

1. 데이터의 기본 구조
2. 카테고리별 문서 분포
3. 컬럼별 결측치 현황
4. 문서 길이 통계
5. 50단어 미만의 짧은 문서
6. NumPy와 pandas의 통계 결과 비교
"""

# 운영체제에 상관없이 파일 경로를 안전하게 처리하기 위해 가져옵니다.
from pathlib import Path

# 데이터 파일이 없거나 읽을 수 없을 때 프로그램을 종료하기 위해 가져옵니다.
import sys

# 문서 길이 배열과 통계 계산에 사용합니다.
import numpy as np

# CSV 로딩과 DataFrame 기반 데이터 탐색에 사용합니다.
import pandas as pd


# 현재 파일(main.py)의 절대 경로를 기준으로 프로젝트 루트를 찾습니다.
#
# main.py의 위치:
# 프로젝트 루트/week01/main.py
#
# parent는 week01 폴더이고,
# parent.parent는 프로젝트 루트입니다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 프로젝트 루트 아래의 data/tech_docs.csv 경로를 생성합니다.
DATA_PATH = PROJECT_ROOT / "data" / "tech_docs.csv"


def print_section(title: str) -> None:
    """출력 결과의 각 영역을 구분하기 위한 제목을 출력합니다."""

    # 이전 출력과 구분하기 위해 빈 줄을 하나 추가합니다.
    print()

    # 제목 위쪽에 구분선을 출력합니다.
    print("=" * 70)

    # 전달받은 제목을 출력합니다.
    print(title)

    # 제목 아래쪽에도 구분선을 출력합니다.
    print("=" * 70)


def load_data(file_path: str | Path) -> pd.DataFrame:
    """CSV 파일을 읽어 pandas DataFrame으로 반환합니다."""

    # 문자열로 전달된 경로도 Path 방식으로 사용할 수 있도록 변환합니다.
    path = Path(file_path)

    # CSV 파일이 실제로 존재하는지 확인합니다.
    if not path.exists():
        # 파일을 찾지 못한 경우 현재 확인한 경로를 출력합니다.
        print(f"[오류] 데이터 파일을 찾을 수 없습니다: {path}")

        # 사용자가 확인해야 할 파일 위치를 안내합니다.
        print("data 폴더에 tech_docs.csv 파일이 있는지 확인해 주세요.")

        # 오류 상태 코드 1로 프로그램을 종료합니다.
        sys.exit(1)

    try:
        try:
            # UTF-8 BOM이 포함된 CSV도 읽을 수 있도록 utf-8-sig를 사용합니다.
            df = pd.read_csv(path, encoding="utf-8-sig")

        except UnicodeDecodeError:
            # UTF-8로 읽을 수 없는 경우 Windows 한글 인코딩인 cp949로 재시도합니다.
            print("[안내] UTF-8로 읽을 수 없어 cp949 인코딩으로 다시 시도합니다.")

            # 동일한 파일을 cp949 인코딩으로 다시 읽습니다.
            df = pd.read_csv(path, encoding="cp949")

    except pd.errors.EmptyDataError:
        # 파일은 존재하지만 내부 데이터가 비어 있는 경우 처리합니다.
        print("[오류] CSV 파일이 비어 있습니다.")

        # 정상적으로 분석할 수 없으므로 프로그램을 종료합니다.
        sys.exit(1)

    except pd.errors.ParserError as error:
        # 쉼표 구분이나 따옴표 형식 등이 잘못되어 CSV를 해석하지 못한 경우 처리합니다.
        print(f"[오류] CSV 파일 형식을 해석하지 못했습니다: {error}")

        # 정상적으로 분석할 수 없으므로 프로그램을 종료합니다.
        sys.exit(1)

    # DataFrame의 행 수와 열 수를 각각 가져옵니다.
    rows, columns = df.shape

    # 정상적으로 읽은 데이터의 크기를 출력합니다.
    print(f"데이터 로드 완료: {rows}행 × {columns}열")

    # 불러온 DataFrame을 호출한 위치로 반환합니다.
    return df


def explore_structure(df: pd.DataFrame) -> None:
    """데이터의 행·열 수, 컬럼명, 자료형, 상위 5행을 출력합니다."""

    # 데이터 구조 탐색 영역의 제목을 출력합니다.
    print_section("[1] 데이터 구조 확인")

    # shape에서 전체 행 수와 열 수를 가져옵니다.
    rows, columns = df.shape

    # 전체 행 수를 출력합니다.
    print(f"전체 행 수: {rows}")

    # 전체 열 수를 출력합니다.
    print(f"전체 열 수: {columns}")

    # 컬럼명 출력 영역을 구분합니다.
    print("\n컬럼명:")

    # DataFrame의 모든 컬럼명을 하나씩 반복합니다.
    for column in df.columns:
        # 현재 컬럼명을 출력합니다.
        print(f"- {column}")

    # 컬럼별 자료형 출력 영역을 구분합니다.
    print("\n컬럼별 자료형:")

    # 컬럼명과 자료형을 한 쌍씩 반복합니다.
    for column, dtype in df.dtypes.items():
        # 현재 컬럼명과 해당 자료형을 출력합니다.
        #
        # pandas 버전에 따라 문자열 컬럼은
        # object 또는 str로 표시될 수 있습니다.
        print(f"- {column}: {dtype}")

    # 상위 데이터 출력 영역을 구분합니다.
    print("\n상위 5행:")

    # DataFrame의 첫 5개 행을 인덱스 없이 출력합니다.
    print(df.head(5).to_string(index=False))


def show_category_distribution(
    df: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    """카테고리별 문서 수, 비율, 평균 단어 수를 계산합니다."""

    # 카테고리 분석 영역의 제목을 출력합니다.
    print_section("[2] 카테고리 분포 확인")

    # DataFrame의 전체 문서 수를 저장합니다.
    total_documents = len(df)

    # 데이터가 비어 있으면 계산하지 않고 빈 딕셔너리를 반환합니다.
    if total_documents == 0:
        print("데이터가 없어 카테고리 분포를 계산할 수 없습니다.")
        return {}

    # category 컬럼에서 각 카테고리가 몇 번 등장했는지 계산합니다.
    category_counts = df["category"].value_counts()

    # 카테고리별 분석 결과를 저장할 빈 딕셔너리를 생성합니다.
    category_result: dict[str, dict[str, float | int]] = {}

    # 첫 번째 분석 결과의 제목을 출력합니다.
    print("카테고리별 문서 수와 비율:")

    # 각 카테고리명과 문서 수를 하나씩 반복합니다.
    for category, count in category_counts.items():
        # 카테고리명을 문자열로 변환합니다.
        category_name = str(category)

        # NumPy 정수 등이 들어올 수 있으므로 일반 int로 변환합니다.
        document_count = int(count)

        # 현재 카테고리의 전체 데이터 대비 비율을 계산합니다.
        ratio = document_count / total_documents * 100

        # 현재 카테고리에 속하는 문서만 필터링합니다.
        category_documents = df[df["category"] == category]

        # 현재 카테고리의 문서별 단어 수를 저장할 리스트입니다.
        word_counts: list[int] = []

        # content 컬럼에서 결측치를 제거한 뒤 각 문서를 반복합니다.
        for content in category_documents["content"].dropna():
            # 문서를 공백 기준으로 나누고 단어 개수를 계산합니다.
            word_count = len(str(content).split())

            # 계산한 단어 수를 리스트에 추가합니다.
            word_counts.append(word_count)

        # 단어 수 리스트가 비어 있지 않은 경우 평균을 계산합니다.
        if word_counts:
            average_word_count = sum(word_counts) / len(word_counts)

        # 문서 본문이 하나도 없는 경우 평균을 0으로 처리합니다.
        else:
            average_word_count = 0.0

        # 현재 카테고리의 계산 결과를 딕셔너리에 저장합니다.
        category_result[category_name] = {
            "count": document_count,
            "ratio": round(ratio, 2),
            "average_word_count": round(average_word_count, 2),
        }

        # 현재 카테고리의 문서 수와 비율을 출력합니다.
        print(
            f"- {category_name}: "
            f"{document_count}개 "
            f"({ratio:.2f}%)"
        )

    # 평균 단어 수 출력 영역을 구분합니다.
    print("\n카테고리별 평균 단어 수:")

    # 앞에서 만든 카테고리별 결과 딕셔너리를 반복합니다.
    for category, statistics in category_result.items():
        # 현재 카테고리의 평균 단어 수를 가져옵니다.
        average = statistics["average_word_count"]

        # 평균 단어 수를 소수점 둘째 자리까지 출력합니다.
        print(f"- {category}: 평균 {average:.2f}단어")

    # 이후에도 사용할 수 있도록 전체 결과 딕셔너리를 반환합니다.
    return category_result


def check_missing(
    df: pd.DataFrame,
) -> dict[str, dict[str, float | int | str]]:
    """컬럼별 결측치 수, 비율, 심각도를 계산합니다."""

    # 결측치 분석 영역의 제목을 출력합니다.
    print_section("[3] 결측치 현황 확인")

    # DataFrame의 전체 행 수를 저장합니다.
    total_rows = len(df)

    # 데이터가 비어 있으면 계산하지 않고 빈 딕셔너리를 반환합니다.
    if total_rows == 0:
        print("데이터가 없어 결측치를 계산할 수 없습니다.")
        return {}

    # 컬럼별 결측치 분석 결과를 저장할 딕셔너리입니다.
    missing_result: dict[str, dict[str, float | int | str]] = {}

    # 결측치가 없는 컬럼명을 저장할 리스트입니다.
    no_missing_columns: list[str] = []

    # 결측치가 있는 컬럼명을 저장할 리스트입니다.
    columns_with_missing: list[str] = []

    # DataFrame의 모든 컬럼을 하나씩 반복합니다.
    for column in df.columns:
        # 현재 컬럼의 결측치 수를 계산합니다.
        missing_count = int(df[column].isnull().sum())

        # 전체 행 수를 기준으로 결측치 비율을 계산합니다.
        missing_ratio = missing_count / total_rows * 100

        # 결측치가 하나도 없는 경우입니다.
        if missing_count == 0:
            # 심각도를 없음으로 지정합니다.
            severity = "없음"

            # 결측치가 없는 컬럼 목록에 추가합니다.
            no_missing_columns.append(column)

        # 결측치 비율이 5% 미만인 경우입니다.
        elif missing_ratio < 5:
            # 심각도를 낮음으로 지정합니다.
            severity = "낮음"

            # 결측치가 있는 컬럼 목록에 추가합니다.
            columns_with_missing.append(column)

        # 결측치 비율이 5% 이상 20% 미만인 경우입니다.
        elif missing_ratio < 20:
            # 심각도를 주의로 지정합니다.
            severity = "주의"

            # 결측치가 있는 컬럼 목록에 추가합니다.
            columns_with_missing.append(column)

        # 결측치 비율이 20% 이상인 경우입니다.
        else:
            # 심각도를 높음으로 지정합니다.
            severity = "높음"

            # 결측치가 있는 컬럼 목록에 추가합니다.
            columns_with_missing.append(column)

        # 현재 컬럼의 계산 결과를 딕셔너리에 저장합니다.
        missing_result[column] = {
            "count": missing_count,
            "ratio": round(missing_ratio, 2),
            "severity": severity,
        }

    # 결측치가 있는 컬럼이 하나 이상인 경우입니다.
    if columns_with_missing:
        print("결측치가 있는 컬럼:")

        # 결측치가 있는 컬럼을 하나씩 반복합니다.
        for column in columns_with_missing:
            # 현재 컬럼의 결측치 분석 결과를 가져옵니다.
            result = missing_result[column]

            # 결측치 수, 비율, 심각도를 출력합니다.
            print(
                f"- {column}: "
                f"{result['count']}개, "
                f"{result['ratio']:.2f}%, "
                f"심각도 {result['severity']}"
            )

    # 결측치가 있는 컬럼이 하나도 없는 경우입니다.
    else:
        print("결측치가 있는 컬럼: 없음")

    # 결측치가 없는 컬럼이 존재하면 목록을 출력합니다.
    if no_missing_columns:
        print(
            "\n결측치가 없는 컬럼: "
            + ", ".join(no_missing_columns)
        )

    # 이후에도 사용할 수 있도록 전체 결과 딕셔너리를 반환합니다.
    return missing_result


def numpy_doc_stats(df: pd.DataFrame) -> None:
    """NumPy로 문서 길이 통계와 50단어 미만 문서를 확인합니다."""

    # 문서 길이 통계 영역의 제목을 출력합니다.
    print_section("[4] NumPy 문서 길이 통계")

    # content 컬럼에서 결측치가 아닌 데이터만 가져옵니다.
    valid_contents = df["content"].dropna()

    # 각 문서의 단어 수를 계산하고 NumPy 정수 배열로 변환합니다.
    document_lengths = np.array(
        [
            # 문서를 문자열로 변환한 뒤 공백 기준으로 나눠 단어 수를 계산합니다.
            len(str(content).split())
            for content in valid_contents
        ],
        dtype=np.int64,
    )

    # 유효한 문서가 하나도 없으면 통계를 계산하지 않습니다.
    if document_lengths.size == 0:
        print("통계를 계산할 수 있는 문서가 없습니다.")
        return

    # 전체 문서 길이의 평균을 계산합니다.
    mean_length = float(np.mean(document_lengths))

    # pandas와 같은 표본표준편차를 얻기 위해 ddof=1을 지정합니다.
    standard_deviation = float(np.std(document_lengths, ddof=1))

    # 문서 길이의 중앙값을 계산합니다.
    median_length = float(np.median(document_lengths))

    # 가장 짧은 문서의 단어 수를 계산합니다.
    minimum_length = int(np.min(document_lengths))

    # 가장 긴 문서의 단어 수를 계산합니다.
    maximum_length = int(np.max(document_lengths))

    # 통계 계산에 사용한 문서 수를 출력합니다.
    print(f"문서 수: {document_lengths.size}개")

    # 평균 단어 수를 출력합니다.
    print(f"평균 단어 수: {mean_length:.2f}")

    # 표본표준편차를 출력합니다.
    print(f"표준편차: {standard_deviation:.2f}")

    # 중앙값을 출력합니다.
    print(f"중앙값: {median_length:.2f}")

    # 최솟값을 출력합니다.
    print(f"최솟값: {minimum_length}")

    # 최댓값을 출력합니다.
    print(f"최댓값: {maximum_length}")

    # 각 문서 길이가 50보다 작은지 판단하는 불리언 배열을 만듭니다.
    short_document_mask = document_lengths < 50

    # True의 개수를 합해 50단어 미만 문서 수를 계산합니다.
    short_document_count = int(np.sum(short_document_mask))

    # 50단어 미만 문서 수를 출력합니다.
    print(f"\n50단어 미만 문서 수: {short_document_count}개")

    # 50단어 미만 문서가 하나 이상 존재하는 경우입니다.
    if short_document_count > 0:
        # 결측치를 제거한 content와 동일한 인덱스를 사용해
        # 원본 데이터에서 doc_id와 title을 가져옵니다.
        valid_documents = df.loc[
            valid_contents.index,
            ["doc_id", "title"],
        ].copy()

        # NumPy로 계산한 단어 수를 word_count 컬럼으로 추가합니다.
        valid_documents["word_count"] = document_lengths

        # NumPy 불리언 배열로 50단어 미만 문서만 선택합니다.
        short_documents = valid_documents.loc[short_document_mask]

        # 짧은 문서 목록의 제목을 출력합니다.
        print("\n50단어 미만 문서 목록:")

        # 문서 ID, 제목, 단어 수를 인덱스 없이 출력합니다.
        print(short_documents.to_string(index=False))

    # 50단어 미만 문서가 하나도 없는 경우입니다.
    else:
        print("50단어 미만 문서가 없습니다.")

    # pandas 문자열 기능으로 각 문서의 단어 수를 다시 계산합니다.
    pandas_lengths = valid_contents.str.split().str.len()

    # pandas의 describe()를 사용해 주요 통계값을 계산합니다.
    pandas_statistics = pandas_lengths.describe()

    # NumPy와 pandas의 동일한 통계값을 비교할 수 있도록 묶습니다.
    comparisons = {
        "평균": (
            mean_length,
            float(pandas_statistics["mean"]),
        ),
        "표준편차": (
            standard_deviation,
            float(pandas_statistics["std"]),
        ),
        "중앙값": (
            median_length,
            float(pandas_statistics["50%"]),
        ),
        "최솟값": (
            float(minimum_length),
            float(pandas_statistics["min"]),
        ),
        "최댓값": (
            float(maximum_length),
            float(pandas_statistics["max"]),
        ),
    }

    # 통계 비교 영역의 제목을 출력합니다.
    print("\nNumPy와 pandas 통계 비교:")

    # 각 통계 이름과 NumPy·pandas 값을 반복합니다.
    for name, (numpy_value, pandas_value) in comparisons.items():
        # 부동소수점 오차를 고려해 두 값이 가까운지 확인합니다.
        is_same = np.isclose(numpy_value, pandas_value)

        # 비교 결과에 따라 표시할 한글 문구를 결정합니다.
        result = "일치" if is_same else "불일치"

        # NumPy 값, pandas 값, 비교 결과를 출력합니다.
        print(
            f"- {name}: "
            f"NumPy={numpy_value:.2f}, "
            f"pandas={pandas_value:.2f} "
            f"→ {result}"
        )


def main() -> None:
    """1주차 데이터 탐색 프로그램을 순서대로 실행합니다."""

    # 프로그램 시작 제목을 출력합니다.
    print_section("LLM Infra 마일스톤 1주차 데이터 탐색 시작")

    # CSV 파일을 불러와 DataFrame으로 반환받습니다.
    df = load_data(DATA_PATH)

    # 행·열 수, 컬럼명, 자료형, 상위 5행을 확인합니다.
    explore_structure(df)

    # 카테고리별 문서 수, 비율, 평균 단어 수를 계산합니다.
    show_category_distribution(df)

    # 컬럼별 결측치 수, 비율, 심각도를 확인합니다.
    check_missing(df)

    # NumPy 문서 길이 통계와 짧은 문서를 확인합니다.
    numpy_doc_stats(df)

    # 전체 탐색이 정상적으로 끝났음을 표시합니다.
    print_section("데이터 탐색 완료")


# 이 파일을 직접 실행했을 때만 main()을 호출합니다.
#
# 다른 Python 파일에서 main.py를 import할 경우에는
# main()이 자동 실행되지 않습니다.
if __name__ == "__main__":
    main()