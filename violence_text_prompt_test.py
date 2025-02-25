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
                            "당신은 한국 영상물 등급 위원회의 기준에 따라 대사의 폭력성을 분석하는 전문가입니다."
                            "대사를 기준에 따라 분석한 후 JSON 형식으로만 응답하세요. "
                            "불필요한 설명이나 추가 텍스트 없이, 오직 JSON 형식의 응답만 제공하세요."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"""
                            대사 : {text}

                            요청사항:
                            1. 대사 내용
                            2. 폭력성 여부 (True/False)
                            3. 각 등급 기준에 대한 O/X 판별
                            4. 최종 등급 (전체관람가, 12세 관람가, 15세 관람가, 청소년 관람불가)
                            5. 분석 결과 설명

                            한국 영상물 등급 분류 기준 (폭력성)
                            1) 전체관람가:
                            - 신체 부위, 도구 등을 이용한 물리적 폭력이 없거나 매우 약하게 표현된 것.
                            - 상해, 유혈, 신체훼손 등의 표현이 없는 것.
                            - 성폭력이 표현되지 않은 것.
                            - 폭력적인 느낌을 주는 음향, 시각 효과 등이 거의 없는 것.

                            2) 12세 관람가:
                            - 신체 부위, 도구 등을 이용한 물리적 폭력이 간결하고 경미하게 표현된 것.
                            - 상해, 유혈, 신체훼손 등이 강조되지 않고 빈도가 낮은 것.
                            - 성희롱, 성추행이 암시적으로 표현된 것.
                            - 폭력적인 느낌을 주는 음향, 시각 효과 등이 경미하게 표현된 것.

                            3) 15세 관람가:
                            - 신체 부위, 도구 등을 이용한 물리적 폭력과 학대가 지속적이지 않은 것.
                            - 상해, 유혈, 신체훼손 등이 지속적, 직접적으로 표현되지 않은 것.
                            - 성희롱, 성추행, 성폭행이 간접적으로 묘사된 것.
                            - 폭력적인 느낌을 주는 음향, 시각 효과가 자극적이지 않은 것.

                            4) 청소년 관람불가:
                            - 신체 부위, 도구 등을 이용한 물리적 폭력과 학대가 구체적이고 지속적이며 노골적으로 표현된 것.
                            - 상해, 유혈, 신체훼손 등이 직접적이고 자극적으로 묘사된 것.
                            - 성희롱, 성추행, 성폭행 등이 구체적이고 직접적으로 표현된 것.
                            - 폭력적인 느낌을 주는 음향, 시각 효과가 지속적이고 자극적으로 표현된 것.

                            결과는 반드시 다음 JSON 형식으로만 제공:
                            {{
                                "text": "대사 내용",
                                "is_violent": true/false,
                                "criteria": {{
                                    "전체관람가": "O/X",
                                    "12세 관람가": "O/X",
                                    "15세 관람가": "O/X",
                                    "청소년 관람불가": "O/X"
                                }},
                                "classification": "최종 등급",
                                "explanation": "폭력성 여부 및 이유"
                            }}
                            """
                        },
                ],
                max_tokens=1200,  # 응답의 최대 토큰 수
                temperature=0.7  # 창의성 조정
            )

            # GPT-4o 응답 파싱
            print(text +' 완료')
            answer = response.choices[0].message.content.strip()  # 응답에서 텍스트 추출
            try:
                # JSON 검증 및 파싱
                parsed_json = json.loads(answer)
                results.append(parsed_json)
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}")
                print(f"응답 내용: {answer}")
                results.append({
                    "text": text.strip(),
                    "is_violent": None,
                    "explanation": f"JSON 파싱 오류 발생: {str(e)}"
                })

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
    text_path = "핸섬가이즈_text.txt"  # 텍스트 파일 경로
    output_path = "./핸섬가이즈_violence_prompt_test(2).json"  # 분석 결과 저장 경로

     # Whisper로 추출된 텍스트 로드
    texts = load_texts(text_path)
    
    # GPT-4o를 사용한 폭력성 분석
    results = detect_violence(texts)
    
    # 결과 저장
    save_results(results, output_path)

if __name__ == "__main__":
    main()