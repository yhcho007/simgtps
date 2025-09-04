# 1. 라이브러리 설치: pip install websockets google-cloud-speech
# 2. Google Cloud 설정
#    - Google Cloud 프로젝트를 생성합니다.
#    - "Speech-to-Text API"를 활성화합니다.
#    - 서비스 계정 키 파일을 JSON 형식으로 생성하고 다운로드합니다.
#    - 환경 변수 GOOGLE_APPLICATION_CREDENTIALS에 다운로드한 파일의 경로를 설정합니다. (예: export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-key.json")

import asyncio
import websockets
import json
from google.cloud import speech_v1p1beta1 as speech

# Google Cloud Speech 클라이언트 생성
client = speech.SpeechClient()

# 음성 인식 설정 (한국어 및 화자 분리 활성화)
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code="ko-KR",
    enable_speaker_diarization=True,
    diarization_speaker_count=2,  # 최소 2명의 화자를 가정합니다. 필요에 따라 변경 가능.
)

streaming_config = speech.StreamingRecognitionConfig(
    config=config,
    interim_results=True,
)


async def recognize_audio(websocket):
    """클라이언트로부터 음성 데이터를 받아 Google Cloud API로 전달합니다."""

    # 비동기 제너레이터 (클라이언트로부터 데이터를 받는 역할)
    async def request_stream():
        async for message in websocket:
            yield speech.StreamingRecognizeRequest(audio_content=message)

    try:
        # Google Speech-to-Text API 스트리밍 호출
        responses = client.streaming_recognize(config=streaming_config, requests=request_stream())

        # API 응답을 기다리고 처리합니다.
        async for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            # 화자 정보가 포함된 텍스트를 구성
            transcript_parts = []
            for word_info in result.alternatives[0].words:
                speaker_tag = word_info.speaker_tag
                word = word_info.word

                # 화자가 변경될 때마다 새로운 줄과 화자 태그를 추가
                if len(transcript_parts) == 0 or transcript_parts[-1]["speaker_tag"] != speaker_tag:
                    transcript_parts.append({"speaker_tag": speaker_tag, "text": word})
                else:
                    transcript_parts[-1]["text"] += " " + word

            # 클라이언트로 결과 전송
            # is_final은 결과가 확정되었는지 여부를 나타냅니다.
            await websocket.send(json.dumps({
                "is_final": result.is_final,
                "transcript": transcript_parts
            }))

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")


async def main():
    """WebSocket 서버를 시작합니다."""
    # 8765 포트에서 WebSocket 서버를 시작합니다.
    async with websockets.serve(recognize_audio, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # 서버를 계속 실행 상태로 유지


if __name__ == "__main__":
    asyncio.run(main())
