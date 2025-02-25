import os
import pandas as pd
from glob import glob
from jiwer import wer, cer

def calculate_accuracy(reference_path, hypothesis_path):
    """
    reference_path: 실제 대사 파일 (ground truth)
    hypothesis_path: 자동 변환된 텍스트 파일
    """
    # ✅ 실제 대사 파일 로드 확인
    try:
        with open(reference_path, 'r', encoding='utf-8') as f:
            reference = " ".join(f.read().splitlines()).strip()
        print(f"✅ 실제 대사 파일 정상 로드: {reference_path}")
    except Exception as e:
        print(f"❌ 오류 발생 (실제 대사 파일 읽기 실패): {e}")
        return {'WER': None, 'CER': None}  # 오류 발생 시 기본값 반환

    # ✅ 변환된 텍스트 파일 로드 확인
    try:
        with open(hypothesis_path, 'r', encoding='utf-8') as f:
            hypothesis = []
            for line in f:
                try:
                    hypothesis.append(line.split(']  ')[1].strip())  # 시간 정보 제거
                except IndexError:
                    continue  # 잘못된 형식의 줄은 무시
            hypothesis = " ".join(hypothesis)
        print(f"✅ 변환된 대사 파일 정상 로드: {hypothesis_path}")
    except Exception as e:
        print(f"❌ 오류 발생 (대사 파일 읽기 실패): {e}")
        return {'WER': None, 'CER': None}  # 오류 발생 시 기본값 반환

    return {
        'WER': wer(reference, hypothesis),
        'CER': cer(reference, hypothesis)
    }

def evaluate_all_combinations(result_root, reference_path):
    """
    모든 파라미터 조합별로 변환된 텍스트를 평가하여 WER 및 CER을 계산
    """
    results = []

    print(f"🔍 결과 폴더 탐색 중: {result_root}")

    # ✅ 폴더가 존재하는지 먼저 확인
    if not os.path.exists(result_root):
        print(f"❌ 오류: 결과 폴더가 존재하지 않습니다. ({result_root})")
        return pd.DataFrame()

    for param_dir in glob(os.path.join(result_root, "*")):
        print(f"📂 검색 중: {param_dir}")

        # ✅ 폴더 내부 파일 목록 확인
        try:
            actual_files = os.listdir(param_dir)
            print(f"📑 내부 파일 목록: {actual_files}")
        except Exception as e:
            print(f"❌ 오류 발생 (폴더 읽기 실패): {e}")
            continue  # 오류 발생 시 해당 폴더 무시

        # ✅ 올바른 텍스트 파일 찾기
        base_name = "dusdoQkwlsfhaostm"  # 영상 기본 파일명 (수동 설정)
        expected_filename = f"{base_name}_text.txt"

        # ✅ 파일이 존재하는지 체크
        text_file = os.path.join(param_dir, expected_filename)
        try:
            file_exists = os.path.exists(text_file)
            print(f"📄 찾으려는 파일: {text_file}, 존재 여부: {file_exists}")
        except Exception as e:
            print(f"❌ 오류 발생 (파일 존재 여부 확인 중): {e}")
            continue

        if not file_exists:
            continue  # 파일이 없는 경우 무시

        # ✅ WER & CER 계산
        metrics = calculate_accuracy(reference_path, text_file)

        # ✅ 파라미터 값 파싱 (파일명에서 추출)
        param_parts = os.path.basename(param_dir).split('_')
        try:
            param_dict = {
                'fd': int(param_parts[1][2:]),
                'pd': int(param_parts[2][2:]),
                'vs': int(param_parts[3][2:]),
                'sd': int(param_parts[4][2:])
            }
        except Exception as e:
            print(f"❌ 오류 발생 (파라미터 파싱 중): {e}")
            continue

        results.append({
            **param_dict,
            **metrics,
            'file_path': text_file
        })

    # ✅ DataFrame 변환 및 정렬
    df = pd.DataFrame(results)

    if df.empty:
        print("❌ 오류: 평가할 데이터가 없습니다. 결과 폴더를 확인하세요.")
        return df  # 빈 데이터프레임 반환

    df.sort_values(by=['WER', 'CER'], inplace=True)

    # ✅ 결과 저장
    report_path = os.path.join(result_root, "accuracy_report.csv")
    try:
        df.to_csv(report_path, index=False)
        print(f"✅ 평가 결과 저장 완료: {report_path}")
    except Exception as e:
        print(f"❌ CSV 저장 중 오류 발생: {e}")

    return df

# 🎯 실행 코드
if __name__ == "__main__":
    result_root = "result/dusdoQkwlsfhaostm/auto_combination"  # 결과 폴더 경로
    reference_path = "dusdoQkwlsfhaostm_lines_truth.txt"  # 실제 대사 파일
    
    # ✅ 실행 중 오류 확인
    try:
        print("🚀 실행 시작")
        result_df = evaluate_all_combinations(result_root, reference_path)

        if result_df.empty:
            print("❌ 최적 파라미터를 찾을 데이터가 없습니다.")
        else:
            print("✅ 평가 완료")
    except Exception as e:
        print(f"❌ 프로그램 실행 중 오류 발생: {e}")