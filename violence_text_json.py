import openai
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

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
def detect_violence(texts):
    """
    텍스트 리스트를 받아 GPT-4o를 사용하여 폭력성 판단.
    :param texts: 분석할 텍스트 리스트
    :return: 분석 결과 리스트
    """
    results = []
    for text in texts:
        try:
            # GPT-4o에 요청
            response = client.chat.completions.create(  # 최신 API 호출 방식
                model="gpt-4o",  # 사용할 모델 이름
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 주어진 대사에서 폭력성을 분석하는 전문가입니다. "
                        ),
                    },
                    {
                        "role": "user",
                        "content": 
                         f"""
                            다음 대사에서 폭력성이 있는지 JSON 형식으로 알려주세요. 한글로 대답해 주세요. 문장으로 대답해 주세요.
                            {{
                            "text": "대사 내용"
                            "is_violent: "False" 또는 "True"
                            "explanation": "폭력성 여부에 대한 이유"                          
                            }}
                            대사: {text}
                            """}
                    ],
                max_tokens=800,  # 응답의 최대 토큰 수
                temperature=0.7  # 창의성 조정
            )
            # GPT-4o 응답 파싱
            print(text +' 완료')
            answer = response.choices[0].message.content.strip()  # 응답에서 텍스트 추출
            if answer:
                try:
                    json_start = answer.find('{')
                    json_end = answer.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_output = answer[json_start:json_end]
                        parsed_json = json.loads(json_output)
                        results.append(parsed_json) 
                    else:
                        print("API 응답에서 JSON 형식을 찾을 수 없습니다.")
                        print(f"API 텍스트 응답: {answer}")
                        return answer
                except (ValueError, json.JSONDecodeError) as e:
                    print(f"JSON 파싱 오류: {e}")
                    print(f"API 텍스트 응답: {answer}") # 전체 텍스트 응답 출력
                    return None
            else:
                print("API 응답에 텍스트가 없습니다.")
                return None
        except Exception as e:
            print(f"텍스트 처리 중 오류 발생: {text.strip()} - {e}")
            results.append({
                "text": text.strip(),
                "is_violent": None,
                "explanation": f"오류 발생: {str(e)}"
            })
    return results

# 3. 결과를 JSON 파일로 저장
def save_results(results, output_path):
    """결과 리스트를 JSON 파일로 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"분석 결과가 {output_path}에 저장되었습니다.")

# 4. 메인 함수
def main():
    """메인 실행 함수"""
     
    # 파일 경로 설정
    text_path = "C:/Users/kth45/ASIA_Video_rating_classification/result/핸섬가이즈/핸섬가이즈_text_output/핸섬가이즈_text.txt"  # 텍스트 파일 경로
    output_path = "C:/Users/kth45/work/핸섬가이즈_violence_result.json"  # 분석 결과 저장 경로

    # Whisper로 추출된 텍스트 로드
    texts = load_texts(text_path)
    
    # GPT-4o를 사용한 폭력성 분석
    results = detect_violence(texts)
    print(results)

    # 결과 저장
    save_results(results, output_path)

if __name__ == "__main__":
    main()