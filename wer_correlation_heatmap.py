import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 📌 1️⃣ 특정 파일 제외하고 CSV 데이터 로드
excluded_files = ["accuracy_report_boyhood.csv", "accuracy_report_동화지만 청불입니다.csv"]
file_paths = [file for file in glob.glob("result/*/auto_combination/accuracy_report_*.csv") if not any(ex in file for ex in excluded_files)]

if not file_paths:
    print("❌ CSV 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")
    exit()
else:
    print(f"🔍 총 {len(file_paths)}개의 CSV 파일을 찾았습니다. (Boyhood, 동화지만 청불입니다 제외)")

df_list = [pd.read_csv(file) for file in file_paths]
df = pd.concat(df_list, ignore_index=True)

print(f"✅ 총 {len(df)}개의 행이 로드되었습니다.")

# 📌 2️⃣ WER과 각 변수의 상관관계 분석
correlation_matrix = df[['fd', 'pd', 'vs', 'sd', 'WER']].corr()

# 변수명 영어로 변경
correlation_matrix = correlation_matrix.rename(index={'fd': 'Frame Length (fd)', 'pd': 'Signal Sensitivity (pd)',
                                                      'vs': 'Voice Separation Strength (vs)', 'sd': 'Sliding Window (sd)',
                                                      'WER': 'Word Error Rate (WER)'},
                                               columns={'fd': 'Frame Length (fd)', 'pd': 'Signal Sensitivity (pd)',
                                                        'vs': 'Voice Separation Strength (vs)', 'sd': 'Sliding Window (sd)',
                                                        'WER': 'Word Error Rate (WER)'})

# 📌 3️⃣ 상관관계 히트맵 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title("Correlation Between WER and Parameters (Fd, Pd, Vs, Sd)")
plt.show()


'''
📊 WER과 각 변수(fd, pd, vs, sd) 간의 상관관계 분석 결과
fd (프레임 길이)와 WER: 약한 양의 상관관계 → fd 값이 커질수록 WER이 다소 증가하는 경향
pd (신호 감도)와 WER: 상관관계 거의 없음 → pd 값이 WER에 큰 영향을 미치지 않음
vs (음성 구분 강도)와 WER: 음의 상관관계 → vs 값이 높을수록 WER이 감소하는 경향
sd (슬라이딩 윈도우)와 WER: 약한 음의 상관관계 → sd 값이 커질수록 WER이 감소하는 경향
📌 해석

WER을 줄이려면 vs(음성 구분 강도)와 sd(슬라이딩 윈도우)를 조절하는 것이 중요
fd(프레임 길이)는 너무 크면 WER이 높아질 수 있음
pd(신호 감도)는 WER과 직접적인 관련이 크지 않음 → 다른 변수와 조합해서 분석 필요
'''