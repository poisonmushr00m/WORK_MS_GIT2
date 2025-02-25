import os
import subprocess
from datetime import timedelta
from openai import OpenAI
from spleeter.separator import Separator
import webrtcvad
import collections
import contextlib
import wave
import time
import gc
import soundfile as sf
import librosa
import noisereduce as nr  # pip install noisereduce
from dotenv import load_dotenv
import multiprocessing as mp

load_dotenv()

client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

#######################################################
# 1. 오디오 추출 + Spleeter
#######################################################
def extract_audio(input_video_path, output_audio_path, start_time=None, duration=None, target_rate=16000, stems=2):
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
    # ffmpeg 실행
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
    if process.returncode != 0:
        print("FFmpeg 오류 발생:")
        print(process.stderr.decode('utf-8', errors='replace'))
        raise subprocess.CalledProcessError(process.returncode, command)
    
    # Spleeter 실행
    separator = Separator(f"spleeter:2stems")
    try:
        separator.separate_to_file(output_audio_path, os.path.dirname(output_audio_path))
    except Exception as e:
        print(f"Spleeter 분리 오류: {e}")
        return None

    base_name = os.path.splitext(os.path.basename(output_audio_path))[0]
    vocals_dir = os.path.join(os.path.dirname(output_audio_path), base_name)
    vocals_path = os.path.join(vocals_dir, "vocals.wav")
    if not os.path.exists(vocals_path):
        alt_path = os.path.join(os.path.dirname(output_audio_path), f"{base_name}_audio", "vocals.wav")
        if os.path.exists(alt_path):
            vocals_path = alt_path

    return vocals_path

#######################################################
# 2. (옵션) 볼륨 증폭 (FFmpeg volume 필터)
#######################################################
def amplify_volume(input_wav, output_wav, gain_db=10):
    """
    FFmpeg volume 필터로 오디오 볼륨을 `gain_db` 만큼 증폭
    - ex) gain_db=10 → 10dB 증폭
    - 클리핑이나 잡음이 커질 위험이 있으므로 적당한 값으로 실험 필요
    """
    command = [
        "ffmpeg", "-y",
        "-i", input_wav,
        "-filter:a", f"volume={gain_db}dB",
        output_wav
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
    if process.returncode != 0:
        print("볼륨 증폭 중 FFmpeg 오류 발생:")
        print(process.stderr.decode('utf-8', errors='replace'))
        raise subprocess.CalledProcessError(process.returncode, command)
    return output_wav

#######################################################
# 3. (옵션) 노이즈 제거 (noisereduce 라이브러리)
#######################################################
def denoise_audio(input_wav, output_wav):
    """
    noisereduce 라이브러리를 사용해 간단한 노이즈 제거
    - 더 정교한 파라미터 조절 가능
    """
    data, rate = librosa.load(input_wav, sr=None)
    reduced_data = nr.reduce_noise(y=data, sr=rate)
    sf.write(output_wav, reduced_data, rate)
    return output_wav

##################################
# 4. Whisper API + 결과 저장
##################################
def transcribe_audio(audio_path):
    """
    최신 OpenAI 라이브러리로 Whisper API 호출
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
    return str(int(seconds // 3600)).zfill(2) + ":" + \
           str(int((seconds % 3600) // 60)).zfill(2) + ":" + \
           str(int(seconds % 60)).zfill(2)

def write_text(output_text_path, transcription):
    """
    TranscriptionVerbose 객체에서 segments 속성 사용
    """
    with open(output_text_path, 'w', encoding='utf-8') as f:
        for segment in transcription.segments:
            start_time = format_time(segment.start)
            end_time = format_time(segment.end)
            text = segment.text
            f.write(f"[{start_time} - {end_time}] {text}\n")
    print(f"텍스트가 저장되었습니다: {output_text_path}")

#########################################
# 5. VAD (webrtcvad)
#########################################
class Frame:
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1, "Mono만 지원"
        sample_width = wf.getsampwidth()
        assert sample_width == 2, "16-bit PCM만 지원"
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000), "지원되지 않는 샘플레이트"
        frames = wf.readframes(wf.getnframes())
        return frames, sample_rate

def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

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
    import collections
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, s in ring_buffer if s])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                voiced_frames.extend([f for f, s in ring_buffer])
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, s in ring_buffer if not s])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

def convert_to_mono(input_wav_path, output_wav_path):
    command = ["ffmpeg", "-y", "-i", input_wav_path, "-ac", "1", output_wav_path]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False, check=True)

def convert_sample_rate(input_wav, output_wav, target_rate=16000):
    command = ["ffmpeg", "-y", "-i", input_wav, "-ar", str(target_rate), output_wav]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False, check=True)

def merge_segments_with_silence(segments, sample_rate, silence_duration_ms=200):
    num_silence_samples = int(sample_rate * (silence_duration_ms / 1000.0))
    silence = b'\x00' * (num_silence_samples * 2)  # 16-bit mono
    return silence.join(segments)

def apply_vad(vocals_path,
              frame_duration_ms=30, padding_duration_ms=200,
              vad_sensitivity=1, silence_duration_ms=200):
    """
    - Mono 변환
    - 필요 시 16kHz 리샘플
    - webrtcvad로 음성 구간만 추출 + 구간 사이에 무음 삽입
    """
    # 1) Mono 변환
    mono_path = os.path.splitext(vocals_path)[0] + "_mono.wav"
    convert_to_mono(vocals_path, mono_path)

    # 2) 샘플레이트 체크
    with contextlib.closing(wave.open(mono_path, 'rb')) as wf:
        sample_rate = wf.getframerate()
    if sample_rate not in (8000, 16000, 32000, 48000):
        resampled_path = os.path.splitext(mono_path)[0] + "_resampled.wav"
        convert_sample_rate(mono_path, resampled_path, 16000)
        mono_path = resampled_path

    # 3) webrtcvad
    audio, sample_rate = read_wave(mono_path)
    vad = webrtcvad.Vad(vad_sensitivity)
    frames = frame_generator(frame_duration_ms, audio, sample_rate)
    segments = list(vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames))

    # 무음 삽입 후 병합
    merged_audio = merge_segments_with_silence(segments, sample_rate, silence_duration_ms)

    # 결과 파일
    vad_path = os.path.splitext(mono_path)[0] + "_vad.wav"
    write_wave(vad_path, merged_audio, sample_rate)
    return vad_path


################################################
# 6. 최종 파이프라인 (볼륨 증폭 & 노이즈 제거 옵션)
################################################
def process_video(
    input_video_path, 
    output_text_path,
    start_time=None, 
    duration=None,
    # 볼륨 증폭 옵션
    apply_volume_gain=False, gain_db=10,
    # 노이즈 제거 옵션
    apply_noise_reduce=False,
    # VAD 파라미터
    frame_duration_ms=30,
    padding_duration_ms=200,
    vad_sensitivity=1,
    silence_duration_ms=250,
    language='ko'
):
    """
    1) ffmpeg로 오디오 추출 & Spleeter(2stems)
    2) (옵션) 볼륨 증폭
    3) (옵션) 노이즈 제거
    4) VAD
    5) Whisper
    6) 텍스트 파일 저장
    """

    print(f"\n[ 영상 처리 시작 ] {os.path.basename(input_video_path)}")

    # 1) 오디오 추출 & Spleeter
    temp_audio_path = "temp_extracted.mp3"
    vocals_path = extract_audio(
        input_video_path=input_video_path,
        output_audio_path=temp_audio_path,
        start_time=start_time,
        duration=duration,
        target_rate=16000
    )
    print("  -> 오디오 추출 & Spleeter 완료")

    # 2) (옵션) 볼륨 증폭
    #    vocals.wav => vocals_amplified.wav
    if apply_volume_gain and vocals_path and os.path.exists(vocals_path):
        vocals_amp = os.path.splitext(vocals_path)[0] + f"_amp{gain_db}.wav"
        amplified_path = amplify_volume(vocals_path, vocals_amp, gain_db=gain_db)
        vocals_path = amplified_path
        print(f"  -> 볼륨 증폭 ({gain_db}dB) 적용 완료")

    # 3) (옵션) 노이즈 제거
    #    vocals_amp.wav => vocals_amp_denoised.wav
    if apply_noise_reduce and vocals_path and os.path.exists(vocals_path):
        vocals_denoise = os.path.splitext(vocals_path)[0] + "_denoised.wav"
        denoised_path = denoise_audio(vocals_path, vocals_denoise)
        vocals_path = denoised_path
        print("  -> 노이즈 제거 적용 완료")

    # 4) VAD
    vad_vocals_path = apply_vad(
        vocals_path,
        frame_duration_ms=frame_duration_ms,
        padding_duration_ms=padding_duration_ms,
        vad_sensitivity=vad_sensitivity,
        silence_duration_ms=silence_duration_ms
    )
    print("  -> VAD 완료")

    # 5) Whisper
    transcription_result = transcribe_audio(vad_vocals_path)
    print("  -> Whisper 완료")

    # 6) 텍스트 저장
    write_text(output_text_path, transcription_result)
    print(f"[ 영상 처리 완료 ] 결과: {output_text_path}")

    # 자원 정리
    gc.collect()
    time.sleep(1)


###################################################
# 실제 실행 예시
###################################################
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    from multiprocessing import freeze_support
    freeze_support()

    os.makedirs("result/optimal", exist_ok=True)

    # (A) 파일 설정
    input_path = "video_data/30일.mp4"
    output_txt = "result/amplify/30일_text.txt"

    # (B) 파라미터 설정
    #  - 일단 볼륨 증폭만 해보고, 성능이 안 좋으면
    #    apply_noise_reduce=True 로 노이즈 제거도 해본다.
    process_video(
        input_video_path=input_path,
        output_text_path=output_txt,
        # start_time=30,       # 부분 구간 추출 (30초부터)
        # duration=180,        # 180초 (3분) 길이
        apply_volume_gain=True,
        gain_db=10,          # 10dB 증폭
        apply_noise_reduce=True,  # 우선 꺼둠
        frame_duration_ms=30,
        padding_duration_ms=200,
        vad_sensitivity=1,
        silence_duration_ms=250,
        language='ko'
    )
