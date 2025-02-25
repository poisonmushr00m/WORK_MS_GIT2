import openai
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import re

### .env file 로드
load_dotenv()

### API Key 불러오기
openai_api_key = os.getenv('OPENAI_API_KEY')

### OpenAI 함수 호출
client = OpenAI(api_key=openai_api_key)

# 1. 텍스트 파일 읽기
def load_texts(file_path):
    """텍스트 파일에서 데이터를 읽어 리스트로 반환"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# 2. GPT-4o를 사용하여 폭력성 판단
import re

def detect_violence(texts):
    results = []
    summary = None

    for text in texts:
        try:
            # GPT-4o에 요청
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 주어진 대사에서 폭력성을 분석하는 전문가입니다. "                            
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"""
                        다음 대사에서 폭력성이 있는지 JSON 형식으로 알려주세요. 한글로 대답해 주세요. 
                        
                        {{
                            "text": "대사 내용",
                            "is_violent": "False" 또는 "True",
                            "explanation": "폭력성 여부에 대한 이유"
                        }}

                        대사: "{text}"

                        추가 요청:
                        모든 대사를 분석한 결과를 종합하여 요약 데이터를 생성해 주세요. 
                        요약 데이터에는 분석한 텍스트의 수, 폭력적인 대사의 수, 폭력적이지 않은 대사의 수, 폭력적인 대사의 비율, 폭력적이지 않은 대사의 비율을 넣어주세요.
                        """
                    },
                ],
                max_tokens=800,
                temperature=0.7
            )

            # GPT-4o 응답 파싱
            answer = response.choices[0].message.content.strip()

            # ```json과 ``` 사이의 JSON 블록 추출
            json_blocks = re.findall(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', answer, re.DOTALL)

            for block in json_blocks:
                block = block.strip()
                try:
                    parsed_json = json.loads(block)
                    if "total_scenes" in parsed_json:
                        summary = parsed_json
                    else:
                        results.append(parsed_json)
                except json.JSONDecodeError as e:
                    print(f"JSON 파싱 오류: {e}")
                    print(f"문제의 JSON 블록: {block}")

        except Exception as e:
            print(f"텍스트 처리 중 오류 발생: {text.strip()} - {e}")
            results.append({
                "text": text.strip(),
                "is_violent": None,
                "explanation": f"오류 발생: {str(e)}"
            })

    return results, summary


# 3. 결과를 JSON 파일로 저장
def save_results(results, summary, output_path):
    """결과 리스트와 요약 데이터를 JSON 파일로 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"results": results, "summary": summary}, f, ensure_ascii=False, indent=4)
    print(f"분석 결과가 {output_path}에 저장되었습니다.")

# 4. 메인 함수
def main():
    """메인 실행 함수"""
 
    # 파일 경로 설정
    text_path = "핸섬가이즈_text.txt"  # 텍스트 파일 경로
    output_path = "핸섬가이즈_violence_text_prompt(3).json"  # 분석 결과 저장 경로

    # Whisper로 추출된 텍스트 로드
    texts = load_texts(text_path)
    
    # GPT-4o를 사용한 폭력성 분석
    results, summary = detect_violence(texts)
    
    # 결과 저장
    save_results(results, summary, output_path)

if __name__ == "__main__":
    main()