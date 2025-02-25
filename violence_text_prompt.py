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
    두 단계(대사별 분석, 최종 등급결정)로 나누어 영상물 등급 결정하도록 설정.
    (한 번에 프롬프트를 작성하니 대사별 분석과 최종 등급 분석이 뒤섞임)
    """
    results = []
    for text in texts:
        try:
            # 🎯 2-1 GPT-4o 개별 대사 분석석
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 한국 영상물 등급 위원회의 기준에 따라 대사를 분석하는 전문가입니다. "
                            "대사의 맥락과 사회 통념을 충분히 고려해 폭력성을 분석하고, 분석 결과를 JSON 형식으로 제공해 주세요. "
                            "정확하고 공정한 분석을 수행해야 합니다."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "다음 대사들을 한국 영상물 등급 위원회의 폭력성 분류 기준에 따라 분석해 주세요. "
                            "맥락과 사회 통념을 충분히 고려해야 하며, 결과는 JSON 형식으로 반환해 주세요. 다음 세부 사항을 포함해야 합니다:\n\n"
                            "1. 각 대사의 폭력성 여부(True/False)."
                            "2. 폭력성 여부에 대한 이유."
                            "3. 각 대사의 한국 영상물 등급(전체관람가, 12세 관람가, 15세 관람가, 청소년 관람불가)."

                            "한국 영상물 등급 위원회의 폭력성 기준은 다음과 같습니다:"
                            "- 전체관람가:"
                            "  * 신체 부위, 도구 등을 이용한 물리적 폭력이 없거나 매우 약하게 표현된 것."
                            "  * 상해, 유혈, 신체훼손 등의 표현이 없는 것.\n"
                            "  * 성폭력이 표현되지 않은 것."
                            "  * 폭력적인 느낌을 주는 음향, 시각 효과 등이 거의 없는 것."
                            "- 12세 관람가:"
                            "  * 신체 부위, 도구 등을 이용한 물리적 폭력이 간결하고 경미하게 표현된 것."
                            "  * 상해, 유혈, 신체훼손 등이 강조되지 않고 빈도가 낮은 것."
                            "  * 성희롱, 성추행이 암시적으로 표현된 것.\n"
                            "  * 폭력적인 느낌을 주는 음향, 시각 효과 등이 경미하게 표현된 것."
                            "- 15세 관람가:"
                            "  * 신체 부위, 도구 등을 이용한 물리적 폭력과 학대가 지속적이지 않은 것."
                            "  * 상해, 유혈, 신체훼손 등이 지속적, 직접적으로 표현되지 않은 것."
                            "  * 성희롱, 성추행, 성폭행이 간접적으로 묘사된 것."
                            "  * 폭력적인 느낌을 주는 음향, 시각 효과가 자극적이지 않은 것."
                            "- 청소년 관람불가:"
                            "  * 신체 부위, 도구 등을 이용한 물리적 폭력과 학대가 구체적이고 지속적이며 노골적으로 표현된 것."
                            "  * 상해, 유혈, 신체훼손 등이 직접적이고 자극적으로 묘사된 것."
                            "  * 성희롱, 성추행, 성폭행 등이 구체적이고 직접적으로 표현된 것."
                            "  * 폭력적인 느낌을 주는 음향, 시각 효과가 지속적이고 자극적으로 표현된 것."

                            f"분석할 대사: \"{text.strip()}\""                            
                        )

                    }
                ]
            )


            # 2-2 GPT-4o 응답 파싱
            answer = response.choices[0].message.content.strip()
            if answer:
                # 응답에서 JSON 추출
                json_start = answer.find('{')
                json_end = answer.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_output = answer[json_start:json_end]
                    parsed_json = json.loads(json_output)
                    # 대사별 결과 추가
                    results["dialogues"].append(parsed_json)
                else:
                    print("API 응답에서 JSON 형식을 찾을 수 없습니다.")
                    print(f"API 텍스트 응답: {answer}")
        except Exception as e:
            print(f"텍스트 처리 중 오류 발생: {text.strip()} - {e}")
            results["dialogues"].append({
                "text": text.strip(),
                "is_violent": None,
                "reason": f"오류 발생: {str(e)}",
                "rating": None
            })

    # 🎯 2-3 GPT에게 최종 등급 판단 요청
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 한국 영상물 등급 위원회의 기준에 따라 대사를 분석하는 전문가입니다. "
                    "대사의 맥락과 사회 통념을 충분히 고려해 폭력성을 분석하고, 최종 영상물의 등급을 판단해야 합니다. "
                    "모든 대사의 분석 결과를 보고 최종 영상물의 등급을 결정한 후, 결정 이유를 설명해 주세요. "
                    "결과는 JSON 형식으로 제공해야 합니다."
                )
            },
            {
                "role": "user",
                "content": f(
                    "다음은 각각의 대사별 분석 결과입니다. 이를 바탕으로 최종 영상물의 등급을 판정해 주세요."
                    f"{json.dumps(results['dialogues'], ensure_ascii=False, indent=4)}"
                    "1. 최종 영상물의 등급을 결정해 주세요. (전체관람가, 12세 관람가, 15세 관람가, 청소년 관람불가 중 하나)"
                    "2. 왜 해당 등급이 선택되었는지 이유를 설명해 주세요."
                    "결과는 JSON 형식으로 반환해 주세요."
                )
            }
        ]
    )

    # 2-4 GPT 최종 판단 결과 파싱
    final_answer = final_response.choices[0].message.content.strip()
    if final_answer:
        json_start = final_answer.find('{')
        json_end = final_answer.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_output = final_answer[json_start:json_end]
            parsed_json = json.loads(json_output)
            results["final_rating"] = parsed_json.get("final_rating", "전체관람가")
            results["final_reason"] = parsed_json.get("final_reason", "이유 없음")
        else:
            print("API 응답에서 JSON 형식을 찾을 수 없습니다.")
            print(f"API 텍스트 응답: {final_answer}")

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
    output_path = "C:/Users/kth45/work/핸섬가이즈_violence_text_prompt(2).json"  # 분석 결과 저장 경로

     # Whisper로 추출된 텍스트 로드
    texts = load_texts(text_path)
    
    # GPT-4o를 사용한 폭력성 분석
    results = detect_violence(texts)
    
    # 결과 저장
    save_results(results, output_path)

if __name__ == "__main__":
    main()
