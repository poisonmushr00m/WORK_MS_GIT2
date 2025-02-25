import re
import glob
import json
import csv
from jiwer import wer
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import Levenshtein

# nltk의 punkt 토크나이저 데이터 다운로드 (최초 실행 시 한 번만 필요)
nltk.download('punkt')

def remove_timestamps(text):
    """
    [00:00:00 - 00:00:05]와 같이 시작하는 타임스탬프를 제거합니다.
    """
    return re.sub(r'\[\d{2}:\d{2}:\d{2} - \d{2}:\d{2}:\d{2}\]\s*', '', text)

def save_results_to_csv(results, output_file):
    """
    평가 결과를 CSV 파일로 저장합니다.
    """
    fieldnames = ["file_path", "WER", "BLEU", "ROUGE-1", "ROUGE-L", "Normalized Levenshtein"]
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for file_path, metrics in results.items():
            row = {"file_path": file_path}
            row.update(metrics)
            writer.writerow(row)
    print(f"Results saved to {output_file}")

def save_results_to_json(results, output_file):
    """
    평가 결과를 JSON 파일로 저장합니다.
    """
    with open(output_file, mode='w', encoding='utf-8') as jsonfile:
        json.dump(results, jsonfile, indent=4, ensure_ascii=False)
    print(f"Results saved to {output_file}")

def main():
    # 1. 실제 대사(ground truth) 파일 읽기
    ground_truth_file = "duseoQkwlsfhaostm_lines_truth.txt"
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truth = f.read().strip()

    # BLEU 계산을 위해 ground truth 토크나이즈
    ground_truth_tokens = word_tokenize(ground_truth)

    # ROUGE scorer 설정 (ROUGE-1, ROUGE-L)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    # 음성 인식 결과 파일들을 glob로 찾습니다.
    # 실제 파일 저장 경로에 맞게 패턴을 수정하세요.
    result_files = glob.glob("result/**/*.txt", recursive=True)

    results = {}

    for file_path in result_files:
        with open(file_path, "r", encoding="utf-8") as f:
            transcript = f.read().strip()

        # 타임스탬프 제거
        transcript_clean = remove_timestamps(transcript)

        # 평가 지표 계산
        error_rate = wer(ground_truth, transcript_clean)
        candidate_tokens = word_tokenize(transcript_clean)
        bleu_score = sentence_bleu([ground_truth_tokens], candidate_tokens)
        rouge_scores = scorer.score(ground_truth, transcript_clean)
        rouge1 = rouge_scores["rouge1"].fmeasure
        rougeL = rouge_scores["rougeL"].fmeasure
        lev_distance = Levenshtein.distance(ground_truth, transcript_clean)
        norm_lev = lev_distance / max(len(ground_truth), 1)

        results[file_path] = {
            "WER": error_rate,
            "BLEU": bleu_score,
            "ROUGE-1": rouge1,
            "ROUGE-L": rougeL,
            "Normalized Levenshtein": norm_lev
        }

        print(f"File: {file_path}")
        print(f"  WER: {error_rate:.3f}")
        print(f"  BLEU: {bleu_score:.3f}")
        print(f"  ROUGE-1: {rouge1:.3f}")
        print(f"  ROUGE-L: {rougeL:.3f}")
        print(f"  Normalized Levenshtein: {norm_lev:.3f}")
        print("-" * 50)

    # WER 기준으로 가장 낮은 결과(최적)를 찾습니다.
    best_file = min(results, key=lambda k: results[k]["WER"])
    print(f"\nBest result (lowest WER): {best_file}")
    print(results[best_file])

    # 평가 결과를 CSV와 JSON 파일로 저장합니다.
    save_results_to_csv(results, "evaluation_results.csv")
    save_results_to_json(results, "evaluation_results.json")

if __name__ == "__main__":
    main()
