import os
import subprocess
from datetime import timedelta
import openai
import spleeter
from spleeter.separator import Separator
import webrtcvad
import collections
import contextlib
import wave
import librosa
import noisereduce as nr
import soundfile as sf
from dotenv import load_dotenv

# .env 파일 로드 및 API 키 설정
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 로컬 비디오 파일 경로 (원하는 경로로 수정)
video_path = "video_data/dusdoQkwlsfhaostm.mp4"  # 예시 경로

#########################################
# 1. 음원 분리 및 Whisper API 파이프라인  #
#########################################

def create_dirs(file_name):
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    result_folder = f"./result/{base_name}"
    os.makedirs(result_folder, exist_ok=True)
    output_audio_path = f"{result_folder}/{base_name}_audio.mp3"
    output_vocals_path = f"{result_folder}/{base_name}_vocals.wav"
    output_text_path = f"{result_folder}/{base_name}_text.txt"
    return output_audio_path, output_vocals_path, output_text_path

def extract_audio(input_video_path, output_audio_path, start_time=None, duration=None, target_rate=16000):
    """
    FFmpeg로 비디오에서 오디오를 추출할 때 16kHz로 리샘플링하고,
    Spleeter로 보컬(대사)와 반주를 분리한 후 보컬 파일 경로를 반환합니다.
    """
    # FFmpeg 옵션에 -ar를 추가하여 target_rate로 리샘플링
    command = ["ffmpeg", "-y", "-i", input_video_path, "-vn", "-acodec", "mp3", "-ar", str(target_rate), output_audio_path]
    process = subprocess.run(command, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        print("FFmpeg 오류 발생:")
        print(process.stderr)
        raise subprocess.CalledProcessError(process.returncode, command)
    
    # Spleeter로 보컬 분리
    separator = Separator('spleeter:2stems')
    try:
        separator.separate_to_file(output_audio_path, os.path.dirname(output_audio_path))
    except Exception as e:
        print(f"Spleeter 음원 분리 중 오류 발생: {e}")
        return None
    
    # Spleeter는 보통 "<base_name>/vocals.wav" 경로에 결과를 저장합니다.
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    vocals_path = os.path.join(os.path.dirname(output_audio_path), f"{base_name}_audio", "vocals.wav")
    # 만약 위 경로가 다르다면, 아래와 같이 기본 경로로도 시도해볼 수 있습니다:
    if not os.path.exists(vocals_path):
        vocals_path = os.path.join(os.path.dirname(output_audio_path), f"{base_name}/vocals.wav")
    return vocals_path

def transcribe_audio(audio_path):
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
    with open(output_text_path, 'w', encoding='utf-8') as f:
        for segment in result['segments']:
            start_time = format_time(segment['start'])
            end_time = format_time(segment['end'])
            text = segment['text']
            f.write(f"[{start_time} - {end_time}]  {text}\n")
            print(f"[{start_time} - {end_time}]  {text}")

#########################################
# 2. 노이즈 제거, VAD 및 샘플레이트 최적화 #
#########################################

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

def apply_noise_reduction(vocals_path):
    # 노이즈 제거를 위해 librosa로 오디오 로드, noisereduce 적용, soundfile로 저장
    y, sr = librosa.load(vocals_path, sr=None, mono=True)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    denoised_path = os.path.splitext(vocals_path)[0] + "_denoised.wav"
    sf.write(denoised_path, reduced_noise, sr)
    return denoised_path

def apply_vad(vocals_path):
    # 먼저, vocals 파일을 Mono로 변환 (변환된 파일명은 _mono.wav)
    mono_vocals_path = os.path.splitext(vocals_path)[0] + "_mono.wav"
    convert_to_mono(vocals_path, mono_vocals_path)
    
    # 샘플레이트 확인 및 재샘플링
    with contextlib.closing(wave.open(mono_vocals_path, 'rb')) as wf:
        sample_rate = wf.getframerate()
    if sample_rate not in (8000, 16000, 32000, 48000):
        resampled_path = os.path.splitext(mono_vocals_path)[0] + "_resampled.wav"
        convert_sample_rate(mono_vocals_path, resampled_path, target_rate=16000)
        mono_vocals_path = resampled_path
    
    # # 노이즈 제거 적용 (추가된 부분)
    # denoised_path = apply_noise_reduction(mono_vocals_path)
    
    # # denoised 파일을 읽어서 VAD 적용
    # audio, sample_rate = read_wave(denoised_path)
    # frames = frame_generator(30, audio, sample_rate)
    # vad = webrtcvad.Vad(1)
    # segments = list(vad_collector(sample_rate, 30, 300, vad, frames))
    # speech_audio = b''.join(segments)
    # vad_audio_path = os.path.splitext(denoised_path)[0] + "_vad.wav"
    # write_wave(vad_audio_path, speech_audio, sample_rate)
    # return vad_audio_path

    # Mono 파일 읽기 및 VAD 적용
    audio, sample_rate = read_wave(mono_vocals_path)
    frames = frame_generator(30, audio, sample_rate)
    vad = webrtcvad.Vad(1)  # 민감도 1로 설정 (필요에 따라 1 또는 2 등으로 조정)
    segments = list(vad_collector(sample_rate, 30, 300, vad, frames))
    speech_audio = b''.join(segments)
    vad_audio_path = os.path.splitext(mono_vocals_path)[0] + "_vad.wav"
    write_wave(vad_audio_path, speech_audio, sample_rate)
    return vad_audio_path

##############################################
# 3. 통합 파이프라인: 음원 분리 -> VAD -> Whisper #
##############################################

def process_video(input_video_path, start_time=None, duration=None, language='ko'):
    # 여기서는 간단하게 파일 경로를 기준으로 디렉토리 생성
    output_audio_path, output_vocals_path, output_text_path = create_dirs(input_video_path)
    
    print("1. 오디오 추출 및 음원 분리 시작")
    vocals_path = extract_audio(input_video_path, output_audio_path, start_time, duration)
    print("2. 오디오 추출 및 음원 분리 완료")
    
    print("3. VAD 적용 시작")
    vad_vocals_path = apply_vad(vocals_path)
    print("4. VAD 적용 완료")
    
    print("5. Whisper API 호출 시작")
    transcription_result = transcribe_audio(vad_vocals_path)
    print("6. Whisper API 호출 완료")
    
    print("7. 텍스트 파일 저장 시작")
    write_text(output_text_path, transcription_result)
    print("8. 텍스트 파일 저장 완료")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # 윈도우에서 필요할 수 있음
    process_video("video_data/dusdoQkwlsfhaostm.mp4")
