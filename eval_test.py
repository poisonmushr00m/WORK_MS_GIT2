import os
import jiwer
import pandas as pd
import glob

# 실제 대본 파일 경로 (1단계에서 준비한 파일)
actual_dialogue_path = "duseoQkwlsfhaostm_lines_truth.txt"  # 실제 대본 파일 경로를 여기에 입력하세요.

# Whisper API 결과 텍스트 파일들이 저장된 기본 폴더 경로
results_base_folder = "result"

# 결과 분석을 저장할 CSV 파일 경로
output_csv_path = "parameter_accuracy_results.csv"

def calculate_wer(reference_path, hypothesis_path):
    with open(reference_path, "r", encoding="utf-8") as ref_file, open(hypothesis_path, "r", encoding="utf-8") as hyp_file:
        reference_text = ref_file.read()
        hypothesis_text = hyp_file.read()

    wer = jiwer.wer(reference_text, hypothesis_text)
    return wer

if __name__ == "__main__":
    wer_results = []

    # 'result' 폴더 아래의 모든 조합 폴더를 순회
    for combination_folder in glob.glob(os.path.join(results_base_folder, "*_fd*_pd*_vs*_sd*")):
        text_file_path = glob.glob(os.path.join(combination_folder, "*_text.txt"))[0]  # 해당 조합의 텍스트 파일 경로
        if text_file_path:
            wer_value = calculate_wer(actual_dialogue_path, text_file_path)

            # 폴더 이름에서 파라미터 값 추출
            folder_name = os.path.basename(combination_folder)
            params = folder_name.split('_')
            fd = int(params[1][2:])
            pd = int(params[2][2:])
            vs = int(params[3][2:])
            sd = int(params[4][2:])

            wer_results.append({
                'folder_name': folder_name,
                'frame_duration_ms': fd,
                'padding_duration_ms': pd,
                'vad_sensitivity': vs,
                'silence_duration_ms': sd,
                'wer': wer_value
            })
        else:
            print(f"텍스트 파일을 찾을 수 없습니다: {combination_folder}")

    # WER 결과를 DataFrame으로 변환 및 CSV 파일로 저장
    df_results = pd.DataFrame(wer_results)
    df_results_sorted = df_results.sort_values(by='wer')  # WER 낮은 순으로 정렬
    df_results_sorted.to_csv(output_csv_path, index=False, encoding="utf-8")

    print(f"WER 결과가 '{output_csv_path}' 파일에 저장되었습니다.")

    # 가장 낮은 WER 값을 가진 파라미터 조합 출력
    best_combination = df_results_sorted.iloc[0]
    print("\n가장 낮은 WER 값을 가진 파라미터 조합:")
    print(f"폴더 이름: {best_combination['folder_name']}")
    print(f"WER 값: {best_combination['wer']:.4f}")
    print(f"Frame Duration (ms): {best_combination['frame_duration_ms']}")
    print(f"Padding Duration (ms): {best_combination['padding_duration_ms']}")
    print(f"VAD Sensitivity: {best_combination['vad_sensitivity']}")
    print(f"Silence Duration (ms): {best_combination['silence_duration_ms']}")