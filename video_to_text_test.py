import os
import subprocess
from datetime import timedelta
from openai import OpenAI
import json
from dotenv import load_dotenv
import spleeter
from spleeter.separator import Separator
import librosa 
import noisereduce as nr 
import soundfile as sf 


# .env 파일 로드
def open_ai_load():
    load_dotenv()

    ### API Key 불러오기
    openai_api_key = os.getenv('OPENAI_API_KEY')
    # print(openai_api_key)

    ### OpenAi 함수 호출
    client = OpenAI()
    whisper_client= client 
    return whisper_client


# 디렉토리 생성 함수
def create_dirs(base_path, relative_path):
    base_name = os.path.splitext(relative_path)[0]  # 확장자 제외한 파일 이름만 가져옴
    os.makedirs(f"./result", exist_ok=True)
    result_folder_path = f"./result/{base_name}"
    os.makedirs(result_folder_path, exist_ok=True)
    output_audio_path = result_folder_path + f"/{base_name}_audio_output" + f"/{base_name}_audio.mp3"
    output_images_path = result_folder_path + f"/{base_name}_images_output" + f"/frame_%03d.png"
    output_text_path = result_folder_path + f"/{base_name}_text_output" + f"/{base_name}_text.txt"

    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_images_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_text_path), exist_ok=True)
    os.makedirs(result_folder_path+"/result_json", exist_ok=True)
    return output_audio_path, output_images_path, output_text_path

# 오디오 추출 (ffmpeg)
def extract_audio(input_video_path, output_audio_path, start_time=None, duration=None):
    base_name = os.path.splitext(os.path.basename(input_video_path))[0] # 파일 이름 (확장자 제외)
    output_vocals_path = os.path.join(os.path.dirname(output_audio_path), f"{base_name}_vocals.wav") # vocals.wav 경로

    # 1. FFmpeg를 사용하여 원본 오디오 추출 (기존 코드 유지)
    command = ["ffmpeg", "-i", input_video_path, "-vn", "-acodec", "mp3"]
    if start_time:
        command.extend(["-ss", start_time])
    if duration:
        command.extend(["-t", duration])
    command.append(output_audio_path)
    ffmpeg_result = subprocess.run(command) # FFmpeg 실행

    if ffmpeg_result.returncode != 0: # FFmpeg 오류 체크
        print("FFmpeg 오디오 추출 중 오류 발생")
        return None # 오류 발생 시 None 반환

    # 2. Spleeter를 사용하여 음원 분리 (vocals 분리)
    separator = Separator('spleeter:2stems')
    try:
        separator.separate_to_file(output_audio_path, os.path.dirname(output_audio_path))
    except Exception as e:
        print(f"Spleeter 음원 분리 중 오류 발생: {e}")
        return None

    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    vocals_filename = f"{base_name}_vocals.wav"  # 파일 이름 구성

    # ***핵심 변경 부분***: os.path.join()을 사용하여 정확한 경로 생성
    output_vocals_path = os.path.join(os.path.dirname(output_audio_path), vocals_filename)

    return output_vocals_path  # 수정된 경로 반환

# 이미지 추출
def extract_images(input_video_path, output_images_path, start_time=None, duration=None):
    command = ["ffmpeg", "-i", input_video_path, "-vf", "fps=1"]
    
    if start_time:
        command.extend(["-ss", start_time])
    if duration:
        command.extend(["-t", duration])
    
    command.append(output_images_path)
    return subprocess.run(command)

# Whisper 텍스트 변환
def transcribe_audio(client, output_audio_path,language):
    with open(output_audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    return transcription

# 시간 포맷 변환 함수
def format_time(seconds):
    return str(timedelta(seconds=int(seconds))).zfill(8)

# 텍스트 파일에 기록
def write_text(output_text_path, result):
    with open(output_text_path, 'w', encoding='utf-8') as f:
        for segment in result.segments:
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])
            text = segment["text"]
            f.write(f"[{start_time} - {end_time}]  {text}\n")
            print(f"[{start_time} - {end_time}]  {text}")

# 메인 프로세스 실행
def process_video(input_video_path,start_time=None, duration=None,language='ko'):
    client=open_ai_load()
    base_path, relative_path = input_video_path.split("video_data/")

    # 디렉토리 생성 및 경로 반환
    output_audio_path, output_images_path, output_text_path = create_dirs(base_path, relative_path)

    # 오디오 추출
    audio_vocals_path = extract_audio(input_video_path, output_audio_path, start_time, duration)
    if audio_vocals_path is not None:
        print("오디오 추출 완료")
    else:
        print("오디오 추출 중 오류 발생")

    # 이미지 추출
    if extract_images(input_video_path, output_images_path,start_time, duration).returncode == 0:
        print("이미지 추출 완료")
    else:
        print("이미지 추출 중 오류 발생")

    # Whisper로 텍스트 변환
    result = transcribe_audio(client, output_audio_path,language=language)
    write_text(output_text_path, result)

    print("프로세스 완료")
    
    
    
# import한 후 호출 예시
# from this_module import process_video
# process_video("video_data/dusdoQkwlsfhaostm.mp4",)
# video_data 폴더 만들고 영상넣으시면됩니다

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # 윈도우에서 필요할 수 있음
    process_video("video_data/수리남_예고편.mp4")