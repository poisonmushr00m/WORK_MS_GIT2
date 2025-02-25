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

# 📌 2️⃣ 변수 리스트 및 영어 라벨 매핑
variable_labels = {
    'fd': 'Frame Length (fd)',
    'pd': 'Signal Sensitivity (pd)',
    'vs': 'Voice Separation Strength (vs)',
    'sd': 'Sliding Window (sd)'
}

# 📌 3️⃣ 각 변수별 WER 변화 추세 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

variables = ['fd', 'pd', 'vs', 'sd']
titles = [variable_labels[var] + ' vs WER' for var in variables]

for i, var in enumerate(variables):
    ax = axes[i // 2, i % 2]
    sns.lineplot(data=df, x=var, y="WER", marker='o', ax=ax)
    ax.set_title(titles[i])
    ax.set_xlabel(variable_labels[var])
    ax.set_ylabel("WER")

plt.tight_layout()
plt.show()



'''
📊 변수 값 변화에 따른 WER 추세 분석 결과
✅ 프레임 길이(fd) vs WER

fd(프레임 길이)가 증가할수록 WER이 다소 증가하는 경향
fd=20~30에서 가장 낮은 WER을 기록하는 경우가 많음
fd가 너무 크면 오류가 증가할 가능성이 있음
✅ 신호 감도(pd) vs WER

pd(신호 감도)는 WER과 큰 연관성이 없음
특정 값(200~250)에서 안정적인 성능을 보임
✅ 음성 구분 강도(vs) vs WER

vs(음성 구분 강도)가 높을수록 WER이 감소하는 경향
vs=2~3일 때 가장 좋은 성능을 보임
✅ 슬라이딩 윈도우(sd) vs WER

sd(슬라이딩 윈도우)가 클수록 WER이 감소하는 경향
sd=250~300에서 가장 안정적인 성능을 보임
'''