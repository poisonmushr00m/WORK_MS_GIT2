import os
import ffmpeg
import numpy as np
import webrtcvad
from openai import OpenAI
from spleeter.separator import Separator
from pydub import AudioSegment
from pydub.utils import make_chunks

# 클라이언트 초기화
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# 파라미터 설정
params = {
    'frame_duration_ms': 30,
    'padding_duration_ms': 200,
    'vad_sensitivity': 1,
    'silence_duration_ms': 250
}

video_time_ranges = {
    '완득이': (30, 210),
    '아저씨': (570, 870),
    '베테랑': (210, None)
}

def time_to_sec(t):
    return t if isinstance(t, (int, float)) else None

def trim_video(input_path, output_path, start, end):
    try:
        (
            ffmpeg.input(input_path)
            .output(output_path, ss=start, to=end, c='copy')
            .run(overwrite_output=True)
        )
    except ffmpeg.Error as e:
        print(f"비디오 트리밍 오류: {e}")

def extract_audio(video_path, audio_path):
    try:
        (
            ffmpeg.input(video_path)
            .output(audio_path, ar=16000, ac=1)
            .run(overwrite_output=True)
        )
    except ffmpeg.Error as e:
        print(f"오디오 추출 오류: {e}")

def separate_vocals(audio_path, output_dir):
    separator = Separator('spleeter:2stems', stft_backend="librosa")
    separator.separate_to_file(audio_path, output_dir)

def vad_processing(audio_path, params):
    vad = webrtcvad.Vad(params['vad_sensitivity'])
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    frame_duration = params['frame_duration_ms']
    chunks = make_chunks(audio, frame_duration)
    speech_frames = []
    
    for i, chunk in enumerate(chunks):
        pcm_data = np.array(chunk.get_array_of_samples(), dtype=np.int16)
        if vad.is_speech(pcm_data.tobytes(), 16000):
            speech_frames.append(i)

    intervals = []
    if speech_frames:
        start = speech_frames[0] * frame_duration
        end = (speech_frames[0] + 1) * frame_duration
        
        for frame in speech_frames[1:]:
            current_time = frame * frame_duration
            if current_time <= end + params['silence_duration_ms']:
                end = (frame + 1) * frame_duration
            else:
                intervals.append((start, end))
                start = current_time
                end = (frame + 1) * frame_duration
        intervals.append((start, end))
    
    return intervals

def apply_padding(intervals, padding):
    padded = []
    for start, end in intervals:
        padded_start = max(0, start - padding)
        padded_end = end + padding
        padded.append((padded_start, padded_end))
    return padded

def transcribe_audio(audio_path, intervals):
    full_text = []
    audio = AudioSegment.from_file(audio_path)
    
    for idx, (start, end) in enumerate(intervals):
        chunk = audio[start:end]
        temp_path = f"temp_{idx}.wav"
        chunk.export(temp_path, format="wav")
        
        try:
            with open(temp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ko"
                )
                full_text.append(transcription.text.strip())
        except Exception as e:
            print(f"전사 실패: {str(e)}")
            full_text.append("[전사 실패 구간]")
        finally:
            os.remove(temp_path)
    
    return '\n'.join(full_text)

def process_video(video_path, output_path, time_range):
    # 1. 비디오 트리밍
    trimmed_path = "temp_trimmed.mp4"
    trim_video(video_path, trimmed_path, *map(time_to_sec, time_range))
    
    # 2. 오디오 추출 및 전처리
    audio_path = "temp_audio.wav"
    extract_audio(trimmed_path, audio_path)
    audio = AudioSegment.from_file(audio_path).set_frame_rate(16000).set_channels(1).apply_gain(6)
    audio.export(audio_path, format="wav")
    
    # 3. 보컬 분리
    separate_vocals(audio_path, "temp_output")
    vocals_path = os.path.join("temp_output", os.path.basename(audio_path).replace(".wav", "/vocals.wav"))
    
    # 4. VAD 처리
    intervals = vad_processing(vocals_path, params)
    padded_intervals = apply_padding(intervals, params['padding_duration_ms'])
    
    # 5. Whisper API 전사
    final_text = transcribe_audio(vocals_path, padded_intervals)
    
    # 6. 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_text)
    
    # 7. 임시 파일 정리
    for path in [trimmed_path, audio_path, vocals_path]:
        if os.path.exists(path):
            os.remove(path)

if __name__ == "__main__":
    video_dir = "video_data"
    output_dir = "result/optimal"
    
    for filename in os.listdir(video_dir):
        if not filename.endswith(('.mp4', '.avi', '.mov')):
            continue
            
        name = os.path.splitext(filename)[0]
        if name not in video_time_ranges:
            continue
            
        video_path = os.path.join(video_dir, filename)
        output_path = os.path.join(output_dir, f"{name}_text.txt")
        time_range = video_time_ranges[name]
        
        print(f"처리 시작: {name}")
        process_video(
            video_path=video_path,
            output_path=output_path,
            time_range=(time_range[0], time_range[1])
        )
        print(f"처리 완료: {name}")