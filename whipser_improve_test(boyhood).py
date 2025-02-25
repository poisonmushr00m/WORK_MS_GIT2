import os
import subprocess
from datetime import timedelta
import openai
from spleeter.separator import Separator
import webrtcvad
import collections
import contextlib
import wave
import librosa
import noisereduce as nr   # 노이즈 제거 (옵션, 여기서는 사용하지 않음)
import soundfile as sf
from dotenv import load_dotenv
import itertools
import time
import gc
import multiprocessing as mp


# .env 파일 로드 및 API 키 설정
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 로컬 비디오 파일 경로 (수정 필요)
video_path = "video_data/boyhood.mp4"  # 예시 경로

########################################
# 1. 음원 분리 및 Whisper API 파이프라인 #
########################################

def create_dirs(file_name, param_suffix=""):
    """
    결과 폴더 및 출력 파일 경로를 생성합니다.
    최종 폴더 구조:
      result/
         영상제목/
             auto_combination/
                 영상제목_파라미터접미사/
                     영상제목_audio.mp3
                     영상제목_text.txt
    param_suffix: 파라미터 조합을 나타내는 문자열 (예: "_fd30_pd200_vs1_sd200")
    """
    # 영상제목 추출 (확장자 제거)
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    # 결과 폴더 경로를 "result/영상제목/auto_combination/영상제목_파라미터접미사"로 생성
    result_folder = os.path.join("result", base_name, "auto_combination", base_name + param_suffix)
    os.makedirs(result_folder, exist_ok=True)
    
    # 출력 파일 경로 생성
    output_audio_path = os.path.join(result_folder, f"{base_name}_audio.mp3")
    output_text_path = os.path.join(result_folder, f"{base_name}_text.txt")
    return output_audio_path, output_text_path


def extract_audio(input_video_path, output_audio_path, start_time=None, duration=None, target_rate=16000):
    """
    FFmpeg를 사용해 비디오에서 오디오를 추출하며, target_rate(예: 16000 Hz)로 리샘플링하고,
    Spleeter를 통해 보컬을 분리한 후 보컬 파일의 경로를 반환합니다.
    """
    command = [
        "ffmpeg", "-y", "-i", input_video_path, "-vn", "-acodec", "mp3",
        "-ar", str(target_rate), output_audio_path
    ]
    process = subprocess.run(command, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        print("FFmpeg 오류 발생:")
        print(process.stderr)
        raise subprocess.CalledProcessError(process.returncode, command)
    
    # Spleeter로 보컬 분리 (2stems 모델 사용)
    separator = Separator('spleeter:2stems')
    try:
        separator.separate_to_file(output_audio_path, os.path.dirname(output_audio_path))
    except Exception as e:
        print(f"Spleeter 음원 분리 중 오류 발생: {e}")
        return None

    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    # Spleeter의 출력 경로는 상황에 따라 다를 수 있으므로 여러 경로를 시도
    vocals_path = os.path.join(os.path.dirname(output_audio_path), f"{base_name}_audio", "vocals.wav")
    if not os.path.exists(vocals_path):
        vocals_path = os.path.join(os.path.dirname(output_audio_path), f"{base_name}", "vocals.wav")
    return vocals_path

def transcribe_audio(audio_path):
    """
    Whisper API를 호출하여 오디오를 텍스트로 변환한 후, 결과를 딕셔너리로 반환합니다.
    """
    with open(audio_path, "rb") as audio_file:
        transcription = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ko",
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    return transcription.to_dict()

def format_time(seconds):
    return str(timedelta(seconds=int(seconds))).zfill(8)

def write_text(output_text_path, result):
    """
    Whisper API 결과에서 세그먼트별로 텍스트를 추출하여 파일로 저장합니다.
    """
    with open(output_text_path, 'w', encoding='utf-8') as f:
        for segment in result['segments']:
            start_time = format_time(segment['start'])
            end_time = format_time(segment['end'])
            text = segment['text']
            f.write(f"[{start_time} - {end_time}]  {text}\n")
            print(f"[{start_time} - {end_time}]  {text}")

###############################################
# 2. VAD 및 샘플레이트 최적화 (노이즈 제거 생략) #
###############################################

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1, "오디오 파일은 Mono여야 합니다."
        sample_width = wf.getsampwidth()
        assert sample_width == 2, "오디오 파일은 16-bit여야 합니다."
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000), "지원되지 않는 샘플레이트입니다."
        frames = wf.readframes(wf.getnframes())
        return frames, sample_rate

def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = frame_duration_ms / 1000.0
    frames = []
    while offset + n <= len(audio):
        frame_bytes = audio[offset:offset+n]
        frames.append(Frame(frame_bytes, timestamp, duration))
        timestamp += duration
        offset += n
    return frames

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                voiced_frames.extend([f for f, s in ring_buffer])
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

def convert_to_mono(input_wav_path, output_wav_path):
    command = ["ffmpeg", "-y", "-i", input_wav_path, "-ac", "1", output_wav_path]
    subprocess.run(command, stderr=subprocess.PIPE, text=True, check=True)

def convert_sample_rate(input_wav, output_wav, target_rate=16000):
    command = ["ffmpeg", "-y", "-i", input_wav, "-ar", str(target_rate), output_wav]
    subprocess.run(command, stderr=subprocess.PIPE, text=True, check=True)

def merge_segments_with_silence(segments, sample_rate, silence_duration_ms=200):
    """
    각 세그먼트 사이에 silence_duration_ms 만큼의 무음을 삽입하여 세그먼트를 병합합니다.
    """
    num_silence_samples = int(sample_rate * (silence_duration_ms / 1000.0))
    silence = b'\x00' * (num_silence_samples * 2)  # 16-bit mono
    merged_audio = silence.join(segments)
    return merged_audio

def apply_vad(vocals_path, frame_duration_ms=30, padding_duration_ms=200, vad_sensitivity=1, silence_duration_ms=200):
    # Mono 변환
    mono_vocals_path = os.path.splitext(vocals_path)[0] + "_mono.wav"
    convert_to_mono(vocals_path, mono_vocals_path)
    
    # 샘플레이트 확인 및 재샘플링
    with contextlib.closing(wave.open(mono_vocals_path, 'rb')) as wf:
        sample_rate = wf.getframerate()
    if sample_rate not in (8000, 16000, 32000, 48000):
        resampled_path = os.path.splitext(mono_vocals_path)[0] + "_resampled.wav"
        convert_sample_rate(mono_vocals_path, resampled_path, target_rate=16000)
        mono_vocals_path = resampled_path
    
    # 노이즈 제거 부분은 생략 (VAD만 적용)
    # Mono 파일 읽기 및 VAD 적용
    audio, sample_rate = read_wave(mono_vocals_path)
    frames = frame_generator(frame_duration_ms, audio, sample_rate)
    vad = webrtcvad.Vad(vad_sensitivity)
    segments = list(vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames))
    
    # 각 세그먼트 사이에 지정한 무음 삽입
    merged_audio = merge_segments_with_silence(segments, sample_rate, silence_duration_ms)
    
    vad_audio_path = os.path.splitext(mono_vocals_path)[0] + "_vad.wav"
    write_wave(vad_audio_path, merged_audio, sample_rate)
    return vad_audio_path

#################################################
# 3. 통합 파이프라인: 음원 분리 -> VAD -> Whisper #
#################################################

def process_video(input_video_path, start_time=None, duration=None, language='ko',
                  frame_duration_ms=30, padding_duration_ms=200, vad_sensitivity=1, silence_duration_ms=200):
    # 결과 폴더 생성: 파라미터 값을 파일명에 포함
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    param_suffix = f"_fd{frame_duration_ms}_pd{padding_duration_ms}_vs{vad_sensitivity}_sd{silence_duration_ms}"
    result_folder = os.path.join("result", base_name + param_suffix)
    os.makedirs(result_folder, exist_ok=True)
    output_audio_path = os.path.join(result_folder, f"{base_name}_audio.mp3")
    output_text_path = os.path.join(result_folder, f"{base_name}_text.txt")
    
    print("1. 오디오 추출 및 음원 분리 시작")
    vocals_path = extract_audio(input_video_path, output_audio_path, start_time, duration)
    print("2. 오디오 추출 및 음원 분리 완료")
    
    print("3. VAD 적용 시작")
    vad_vocals_path = apply_vad(vocals_path, frame_duration_ms, padding_duration_ms, vad_sensitivity, silence_duration_ms)
    print("4. VAD 적용 완료")
    
    print("5. Whisper API 호출 시작")
    transcription_result = transcribe_audio(vad_vocals_path)
    print("6. Whisper API 호출 완료")
    
    print("7. 텍스트 파일 저장 시작")
    write_text(output_text_path, transcription_result)
    print("8. 텍스트 파일 저장 완료")
    print("프로세스 완료")

##########################################
# 4. 그리드 서치: 파라미터 조합 자동 테스트 #
##########################################

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    from multiprocessing import freeze_support
    freeze_support()
    
    # 다른 파라미터 후보 목록은 동일하게 두고, vad_sensitivities는 개별적으로 처리합니다.
    vad_sensitivities = [1]  # 먼저 민감도 1로 실행
    frame_durations = [20]            # ms
    padding_durations = [200]     # ms
    silence_durations = [250,300]     # ms

    print(">>> VAD 민감도 1 조합 실행 시작 <<<")
    for vad_sensitivity, frame_duration, padding_duration, silence_duration in itertools.product(
        vad_sensitivities, frame_durations, padding_durations, silence_durations):
        
        print("\n=======================================")
        print(f"테스트 조합: VAD 민감도={vad_sensitivity}, 프레임={frame_duration}ms, 패딩={padding_duration}ms, 무음삽입={silence_duration}ms")
        try:
            process_video(
                video_path,
                frame_duration_ms=frame_duration,
                padding_duration_ms=padding_duration,
                vad_sensitivity=vad_sensitivity,
                silence_duration_ms=silence_duration,
                language='ko'
            )
        except Exception as e:
            print(f"조합 실행 중 오류 발생: {e}")
        finally:
            gc.collect()
            time.sleep(1)

    # # 다음 단계: 민감도를 2로 실행
    # vad_sensitivities = [2]
    # print("\n>>> VAD 민감도 2 조합 실행 시작 <<<")
    # for vad_sensitivity, frame_duration, padding_duration, silence_duration in itertools.product(
    #     vad_sensitivities, frame_durations, padding_durations, silence_durations):
        
    #     print("\n=======================================")
    #     print(f"테스트 조합: VAD 민감도={vad_sensitivity}, 프레임={frame_duration}ms, 패딩={padding_duration}ms, 무음삽입={silence_duration}ms")
    #     try:
    #         process_video(
    #             video_path,
    #             frame_duration_ms=frame_duration,
    #             padding_duration_ms=padding_duration,
    #             vad_sensitivity=vad_sensitivity,
    #             silence_duration_ms=silence_duration,
    #             language='ko'
    #         )
    #     except Exception as e:
    #         print(f"조합 실행 중 오류 발생: {e}")
    #     finally:
    #         gc.collect()
    #         time.sleep(1)

    # # 마지막 단계: 민감도를 3로 실행
    # vad_sensitivities = [3]
    # print("\n>>> VAD 민감도 3 조합 실행 시작 <<<")
    # for vad_sensitivity, frame_duration, padding_duration, silence_duration in itertools.product(
    #     vad_sensitivities, frame_durations, padding_durations, silence_durations):
        
    #     print("\n=======================================")
    #     print(f"테스트 조합: VAD 민감도={vad_sensitivity}, 프레임={frame_duration}ms, 패딩={padding_duration}ms, 무음삽입={silence_duration}ms")
    #     try:
    #         process_video(
    #             video_path,
    #             frame_duration_ms=frame_duration,
    #             padding_duration_ms=padding_duration,
    #             vad_sensitivity=vad_sensitivity,
    #             silence_duration_ms=silence_duration,
    #             language='ko'
    #         )
    #     except Exception as e:
    #         print(f"조합 실행 중 오류 발생: {e}")
    #     finally:
    #         gc.collect()
    #         time.sleep(1)
