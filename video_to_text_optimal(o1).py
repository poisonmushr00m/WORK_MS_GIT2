import os
import subprocess
from datetime import timedelta
from openai import OpenAI       # 최신 방식
from spleeter.separator import Separator
import webrtcvad
import collections
import contextlib
import wave
import time
import gc
import soundfile as sf
from dotenv import load_dotenv
import multiprocessing as mp

# .env 파일 로드 및 API 키 설정
load_dotenv()

# 최신 OpenAI 객체 생성
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")


#######################################################
# 1. 오디오 추출 (부분 구간) + Spleeter(보컬 분리) 함수  #
#######################################################
def extract_audio(input_video_path, output_audio_path, start_time=None, duration=None, target_rate=16000):
    """
    FFmpeg를 사용해 비디오에서 오디오를 추출하며, start_time과 duration으로 부분 구간을 지정할 수 있음.
    추출 후 target_rate (기본: 16000Hz)로 리샘플링하고, Spleeter를 통해 보컬만 추출.
    최종적으로 보컬 wav 파일의 경로를 반환.
    """
    # FFmpeg 명령어 구성
    command = ["ffmpeg", "-y"]
    if start_time is not None:
        command += ["-ss", str(start_time)]
    if duration is not None:
        command += ["-t", str(duration)]
    command += [
        "-i", input_video_path,
        "-vn", 
        "-acodec", "mp3",
        "-ar", str(target_rate),
        output_audio_path
    ]

    process = subprocess.run(
        command, 
        stderr=subprocess.PIPE, 
        text=True,
        encoding='utf-8',
        errors='replace'  # 디코딩 실패시 � 대체
        )

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

    # 분리된 보컬 wav 경로 찾기
    base_name = os.path.splitext(os.path.basename(output_audio_path))[0]
    vocals_path = os.path.join(os.path.dirname(output_audio_path), f"{base_name}", "vocals.wav")
    if not os.path.exists(vocals_path):
        # 폴더 구조가 다른 경우를 대비
        alt_path = os.path.join(os.path.dirname(output_audio_path), f"{base_name}_audio", "vocals.wav")
        if os.path.exists(alt_path):
            vocals_path = alt_path

    return vocals_path


##################################
# 2. Whisper API 호출 및 결과 저장 #
##################################
def transcribe_audio(audio_path):
    """
    최신 OpenAI 라이브러리( from openai import OpenAI )를 통해
    Whisper API를 호출하여 오디오를 텍스트로 변환하고, 결과(딕셔너리형)를 반환합니다.
    """
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ko",
            response_format="verbose_json"  
        )
    return transcription


def format_time(seconds):
    return str(timedelta(seconds=int(seconds))).zfill(8)

def write_text(output_text_path, transcription):
    """
    Whisper API 결과( TranscriptionVerbose 객체 )에서
    세그먼트별로 텍스트를 추출하여 파일로 저장합니다.
    """
    with open(output_text_path, 'w', encoding='utf-8') as f:
        for segment in transcription.segments:
            # segment.start, segment.end, segment.text 속성 사용
            start_time = format_time(segment.start)
            end_time = format_time(segment.end)
            text = segment.text
            f.write(f"[{start_time} - {end_time}]  {text}\n")
    print(f"텍스트가 저장되었습니다: {output_text_path}")


#########################################
# 3. VAD 관련 함수 (webrtcvad 활용)      #
#########################################
class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

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

def frame_generator(frame_duration_ms, audio, sample_rate):
    """
    오디오를 일정 길이(frame_duration_ms)로 나누어 리스트로 반환
    """
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
    """
    VAD가 '말소리'라고 판단한 프레임들만 합쳐서 세그먼트로 묶어 반환
    """
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
    세그먼트 사이에 일정 길이 무음을 삽입하여 하나의 오디오로 병합
    """
    num_silence_samples = int(sample_rate * (silence_duration_ms / 1000.0))
    silence = b'\x00' * (num_silence_samples * 2)  # 16-bit mono
    merged_audio = silence.join(segments)
    return merged_audio

def apply_vad(vocals_path, 
              frame_duration_ms=30, padding_duration_ms=200, 
              vad_sensitivity=1, silence_duration_ms=200):
    """
    1) Mono 변환
    2) 필요 시 리샘플링
    3) webrtcvad로 음성 구간만 추출 후, 구간 사이에 무음(silence_duration_ms) 삽입
    4) 최종 결과 wav 경로 반환
    """
    # 1) Mono 변환
    mono_vocals_path = os.path.splitext(vocals_path)[0] + "_mono.wav"
    convert_to_mono(vocals_path, mono_vocals_path)
    
    # 2) 샘플레이트 확인 및 재샘플링
    with contextlib.closing(wave.open(mono_vocals_path, 'rb')) as wf:
        sample_rate = wf.getframerate()
    if sample_rate not in (8000, 16000, 32000, 48000):
        resampled_path = os.path.splitext(mono_vocals_path)[0] + "_resampled.wav"
        convert_sample_rate(mono_vocals_path, resampled_path, target_rate=16000)
        mono_vocals_path = resampled_path
    
    # 3) VAD 적용
    audio, sample_rate = read_wave(mono_vocals_path)
    frames = frame_generator(frame_duration_ms, audio, sample_rate)
    vad = webrtcvad.Vad(vad_sensitivity)
    segments = list(vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames))
    
    # 세그먼트 사이에 무음 삽입
    merged_audio = merge_segments_with_silence(segments, sample_rate, silence_duration_ms)
    
    # 결과 wav 파일 기록
    vad_audio_path = os.path.splitext(mono_vocals_path)[0] + "_vad.wav"
    write_wave(vad_audio_path, merged_audio, sample_rate)
    
    return vad_audio_path


################################################
# 4. 최종 파이프라인: (부분 추출) -> VAD -> Whisper
################################################
def process_video(input_video_path, output_text_path,
                  start_time=None, duration=None,
                  frame_duration_ms=30, padding_duration_ms=200, 
                  vad_sensitivity=1, silence_duration_ms=250,
                  language='ko'):

    # 1) 오디오 추출 (부분 구간) + 음원 분리
    print(f"\n[{os.path.basename(input_video_path)}] 오디오 추출 시작...")
    temp_audio_path = "temp_extracted.mp3"  # 임시 mp3 파일 경로
    vocals_path = extract_audio(
        input_video_path=input_video_path,
        output_audio_path=temp_audio_path,
        start_time=start_time,
        duration=duration,
        target_rate=16000
    )
    print("  -> 오디오 추출 및 음원 분리 완료")

    # 2) VAD 적용
    print("  -> VAD 적용 시작...")
    vad_vocals_path = apply_vad(
        vocals_path,
        frame_duration_ms=frame_duration_ms,
        padding_duration_ms=padding_duration_ms,
        vad_sensitivity=vad_sensitivity,
        silence_duration_ms=silence_duration_ms
    )
    print("  -> VAD 적용 완료")

    # 3) Whisper API 호출
    print("  -> Whisper API 호출 시작...")
    transcription_result = transcribe_audio(vad_vocals_path)
    print("  -> Whisper API 완료")

    # 4) 텍스트 저장
    print("  -> 텍스트 파일 저장 중...")
    write_text(output_text_path, transcription_result)
    print(f"최종 완료: {output_text_path}")

    # 중간 파일/메모리 정리
    gc.collect()
    time.sleep(1)


###################################################
# 5. video_data 내 전체 파일을 처리 + 일부 영상만 구간 지정
###################################################
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    from multiprocessing import freeze_support
    freeze_support()

    '''
    # 특정 영상에 대해 구간 추출을 지정 (초 단위)
    partial_extractions = {
        "완득이":   (30, 180),   # 30초 ~ 3분30초
        "아저씨":   (570, 300),  # 9분30초 ~ 14분30초
        "베테랑":   (210, None)  # 3분30초 ~ 끝까지
    }
    

    # video_data 폴더 내의 mp4 파일을 전부 읽어 처리
    video_dir = "video_data"
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    
    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]  # 확장자 제거
        input_path = os.path.join(video_dir, video_file)
        output_txt = os.path.join("result", "optimal", f"{video_name}_text.txt")
        
        # 만약 partial_extractions에 정의된 영상이면 구간 추출, 아니면 전체 추출
        if video_name in partial_extractions:
            start_sec, dur_sec = partial_extractions[video_name]
            print(f"\n=== [처리 시작] {video_name} (구간추출) ===")
        else:
            start_sec, dur_sec = None, None
            print(f"\n=== [처리 시작] {video_name} (전체추출) ===")
        
        try:
            process_video(
                input_video_path=input_path,
                output_text_path=output_txt,
                start_time=start_sec,
                duration=dur_sec,
                # 고정 파라미터
                frame_duration_ms=30,
                padding_duration_ms=200,
                vad_sensitivity=1,
                silence_duration_ms=250,
                language='ko'
            )
        except Exception as e:
            print(f"[오류 발생] {video_name}: {e}")
        finally:
            print(f"=== [처리 종료] {video_name} ===\n")
    '''

    # 결과 저장 폴더 준비
    os.makedirs("result/optimal", exist_ok=True)

    # 처리할 단일 파일 경로 (예시)
    input_path = "video_data/30일.mp4"   # 원하는 mp4 경로
    output_txt = "result/optimal/30일_text.txt"  # 결과 텍스트 저장 경로

    # # 부분 구간 추출 (start_time=30, duration=180) 예시
    # # 30초부터 180초(3분) 길이 추출
    # start_sec = 30
    # dur_sec = 180

    # (D) process_video 직접 호출
    process_video(
        input_video_path=input_path,
        output_text_path=output_txt,
        # start_time=start_sec,
        # duration=dur_sec,
        # VAD 파라미터
        frame_duration_ms=30,
        padding_duration_ms=200,
        vad_sensitivity=1,
        silence_duration_ms=250,
        language='ko'
    )