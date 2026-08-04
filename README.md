ODIA - 음성 기반 실시간 객체 인식
=================================

On Device AI 과정 1차 프로젝트
개발: 김현수, 김우석
GitHub: https://github.com/wooseokkim1992-byte/detect_objects


1. 프로젝트 소개
----------------

ODIA는 사용자가 음성 또는 텍스트로 찾고 싶은 물체를 지정하면, 카메라 영상에서
해당 객체를 실시간으로 탐지해 Bounding Box, 클래스 이름, 신뢰도를 표시하는
온디바이스 AI 애플리케이션이다.

예시:

    사용자: "휴대폰과 사람을 찾아줘"
        -> Whisper: 한국어 음성을 텍스트로 변환
        -> Text Manager: "휴대폰", "사람"을 YOLO 클래스에 매핑
        -> YOLO-World: cell phone, person을 탐지 대상으로 적용
        -> OpenCV/PySide6: 객체 위치와 결과를 실시간으로 표시

음성 인식, 명령 분석, 객체 탐지, 영상 처리와 사용자 인터페이스를 하나의
파이프라인으로 연결하는 것이 프로젝트의 핵심이다. 외부 STT API를 사용하지 않고
Whisper와 YOLO-World를 로컬에서 실행하므로 음성 데이터 보호와 오프라인 동작에도
유리하다.


2. 해결하려는 문제
------------------

- 편의점이나 마트에서 원하는 상품을 찾는 데 걸리는 시간을 줄인다.
- 집, 창고, 작업장에서 분실물이나 필요한 도구의 위치를 빠르게 확인한다.
- 키보드나 화면 조작이 어려운 상황에서 음성만으로 탐지 대상을 지정한다.
- 고정된 클래스만 탐지하는 일반 YOLO의 한계를 넘어 실행 중 대상을 변경한다.


3. 주요 기능
------------

- 마이크의 한국어 음성을 Whisper로 인식
- RMS 기반 VAD로 음성 시작과 발화 종료 감지
- 한국어 객체명과 YOLO 영문 클래스 매핑
- YOLO-World v2 Small 기반 Open-vocabulary 객체 탐지
- 카메라 프레임에 Bounding Box, 클래스 이름, 신뢰도 표시
- 음성 명령에 따른 실시간 탐지 클래스 변경
- 클래스 임베딩 캐시를 이용한 전환 지연 최적화
- Textual 기반 장치 및 모델 설정 마법사
- Classic OpenCV 런타임과 PySide6 Desktop 대시보드 제공
- MPS, CUDA, CPU 순서의 추론 장치 선택 및 MPS 실패 시 CPU fallback


4. 사용 모델
-------------

4.1 YOLO-World v2 Small

YOLO-World는 이미지 특징과 텍스트 임베딩을 비교하여 사용자가 텍스트로 지정한
객체를 탐지하는 Open-vocabulary 객체 탐지 모델이다. 일반 YOLO처럼 탐지 클래스가
완전히 고정되지 않으므로 음성 명령에 따라 탐지 대상을 동적으로 바꾸는 본
프로젝트에 적합하다.

- Weight: yolov8s-worldv2.pt
- Parameters: 약 12.76M
- 기본 vocabulary: COCO 80 classes
- Weight format: 주요 parameter FP16
- Text embedding dimension: 512
- 기본 입력 크기: 640 x 640

4.2 OpenAI Whisper

Whisper는 음성 파형을 Log-Mel Spectrogram으로 변환하고 Transformer로 분석하여
텍스트를 생성하는 다국어 Speech-to-Text 모델이다. 본 프로젝트에서는 16 kHz,
mono, float32 PCM 음성을 입력받아 한국어 명령으로 변환한다.

- 기본 선택 모델: Whisper Base - Korean
- 추가 선택 모델: Whisper Tiny - Korean
- 추론 장치: CPU
- 입력 sample rate: 16 kHz
- 입력 형식: mono float32 NumPy array
- VAD: RMS threshold, pre-roll, 최소 발화 길이, 종료 침묵 시간 적용


5. 시스템 아키텍처
------------------

전체 데이터 흐름:

    Microphone
        -> SoundDevice InputStream
        -> Audio Queue
        -> RMS VAD
        -> Whisper STT
        -> Korean Text
        -> Text Manager
        -> YOLO Class Queue (latest only)
        -> Cached Text Embedding Switch
        -> YOLO-World
        -> Bounding Boxes / Labels / Confidence
        -> OpenCV 또는 PySide6 화면

동시 실행 구조:

    Voice Thread                          Main / Vision Thread
    ------------                         --------------------
    마이크 입력                           카메라 프레임 입력
    VAD 및 Whisper 추론                   YOLO-World 추론
    객체 클래스 추출                      최신 클래스 Queue 확인
             \                           /
              +---- Queue(maxsize=1) ----+

음성과 카메라 처리를 서로 다른 실행 흐름으로 분리하여 Whisper 추론 중에도 카메라
스트림이 멈추지 않게 했다. 클래스 Queue는 최신 요청 하나만 유지하므로 사용자가
연속으로 명령하더라도 오래된 명령이 뒤늦게 적용되는 것을 방지한다.


6. YOLO 클래스 변경 최적화
--------------------------

기존 model.set_classes() 방식은 클래스 이름만 바꾸는 단순 setter가 아니다.
새 텍스트를 토큰화하고 CLIP Text Encoder로 임베딩을 생성한 다음 txt_feats와
Detection Head 상태를 변경한다. 실시간 추론 중 반복 호출하면 다음 문제가 있다.

- 동일 클래스의 텍스트 임베딩을 반복 계산
- CLIP 실행 및 CPU/MPS device 이동 비용 발생
- txt_feats, nc, names가 서로 다른 시점에 변경될 가능성
- 카메라 추론과 다른 Thread의 모델 변경이 충돌할 가능성
- Apple MPS에서 Placeholder storage 오류가 발생할 가능성
- 클래스 변경 시점마다 불규칙한 latency 발생

본 프로젝트에서는 YOLO-World checkpoint에 포함된 COCO 클래스 임베딩을 시작 시
CPU 캐시에 분리해 저장한다. 클래스가 바뀌면 CLIP을 다시 실행하거나 모델 전체를
재로딩하지 않고 필요한 Tensor만 선택하여 결합한다.

    COCO 80-class embeddings
        -> 클래스별 CPU cache 생성
        -> 음성으로 요청한 클래스 선택
        -> torch.cat(..., dim=1)
        -> world_model.txt_feats 교체
        -> Detection Head nc 변경
        -> world_model.names / predictor.names 동기화
        -> 다음 frame에서 새 클래스 탐지

클래스 변경은 predict() 실행 중이 아니라 프레임 사이에서 Vision Thread가 직접
적용한다. 이를 통해 모델 상태 변경과 추론이 동시에 일어나지 않도록 했다.

포트폴리오 측정 결과:

    기존 평균 클래스 변경 시간: 2083.23 ms
    캐시 방식 평균 변경 시간:      47.54 ms


7. Whisper 성능 측정
--------------------

측정 조건:

- 기준 문장: "폰과 사람을 찾아줘"
- 동일한 5초 음성 사용
- 모델별 3회 평균
- 최초 warm-up 추론 제외
- CPU 환경

Model      Load(s)  Avg(s)  Min(s)  RTF    CER    Result
---------  -------  ------  ------  -----  -----  ----------------------
base         0.510   0.283   0.283  0.057  0.375  콩과 사람을 찾아둬요
small        1.329   0.795   0.785  0.159  0.125  콘과 사람을 찾아줘
medium       4.755   2.499   2.469  0.500  0.125  돈과 사람을 찾아줘
large        8.563   4.396   4.284  0.879  0.125  혼과 사람을 찾아줘
turbo        3.959   3.342   3.278  0.668  0.000  폰과 사람을 찾아줘

모든 모델의 RTF가 1보다 작아 음성 길이보다 빠르게 처리했다. Turbo는 기준 문장을
유일하게 완전히 인식했지만 CPU 추론 시간이 길다. 실제 프로젝트에서는 객체명
하나가 틀리면 YOLO 클래스 매핑 전체가 실패할 수 있으므로 CER뿐 아니라 객체
키워드 정확도도 중요하다. 현재 UI의 기본 선택은 Base이며, 정확도 우선 환경에서는
Turbo, 속도와 자원 효율 우선 환경에서는 Small과 도메인 단어 보정을 고려할 수 있다.


8. 기술 스택
-------------

- Python 3.11
- Ultralytics 8.4.107
- YOLO-World v2 Small
- OpenAI Whisper
- PyTorch / Torchvision
- CLIP (Ultralytics fork)
- OpenCV 5.0.0.93
- NumPy 2.4.6+
- sounddevice 0.5.5
- Textual 8.2.8+
- PySide6 6.11.1
- uv / uv.lock


9. 설치 및 실행
--------------

요구 사항:

- Python 3.11 호환 환경
- 카메라와 마이크
- macOS, Linux 또는 Windows
- 카메라 및 마이크 접근 권한

macOS / Linux:

    cd /Users/wooseok_kim/projects/On_Device_AI/Project1
    ./bootstrap/setup.sh

Windows PowerShell:

    .\bootstrap\setup.ps1

Bootstrap은 프로젝트 로컬 .odia 디렉터리에 uv, Python 환경, 패키지와 cache를
준비하고 환경 검증 후 ODIA를 실행한다. Shell profile은 변경하지 않는다.

환경이 이미 준비된 경우:

    uv run odia

또는:

    uv run python -m detect_objects

실행 단계:

1. Textual 화면에서 UI theme를 선택한다.
2. 스피커를 선택하고 sample 재생을 확인한다.
3. 마이크를 선택하고 입력 level과 녹음을 확인한다.
4. 카메라를 선택하고 preview를 확인한다.
5. YOLO와 Whisper 모델을 선택한다.
6. Classic 또는 Desktop runtime을 선택한다.
7. 음성 또는 텍스트로 찾을 객체를 명령한다.
8. Classic OpenCV 창에서는 q를 눌러 종료한다.


10. 주요 디렉터리
-----------------

    Project1/
    |-- bootstrap/                         설치, 검증, 실행 자동화
    |-- config/models.toml                 YOLO weight, confidence, imgsz 설정
    |-- src/detect_objects/
    |   |-- main.py                        공통 애플리케이션 진입점
    |   |-- runtime.py                     Local runtime과 Thread lifecycle
    |   |-- camera_cv/                     Classic 카메라 및 탐지 루프
    |   |-- desktop/                       PySide6 Desktop runtime
    |   |-- device_setup/                  오디오/카메라 장치 검사
    |   |-- models/                        모델 선택, 로딩, 추론, cache 전환
    |   |-- tui/                           Textual 설정 마법사
    |   `-- voice_text_convert/            Whisper, VAD, 텍스트 클래스 매핑
    |-- tests/                             기능별 자동 테스트
    |-- docs/                              아키텍처 및 실행 문서
    |-- pyproject.toml                     패키지와 실행 명령 정의
    `-- uv.lock                            재현 가능한 dependency lock


11. 현재 한계
-------------

- 기본 dictionary와 cache에 등록된 클래스 중심으로 동작한다.
- "백팩을 멘 사람"처럼 속성과 관계가 결합된 복합 표현 이해는 제한적이다.
- 짧은 한국어 객체명은 발음, 억양, 배경 소음에 따라 오인식될 수 있다.
- RMS 기반 VAD threshold는 마이크와 주변 소음에 맞게 조정해야 한다.
- 고해상도 입력이나 큰 Whisper 모델은 latency와 memory 사용량을 증가시킨다.
- MPS에서 일부 YOLO-World Tensor 연산이 불안정하면 CPU fallback이 발생할 수 있다.


12. 확장 방향
-------------

- 색상, 의복, 소지품 등 복합 속성 명령 지원
- Grounding/VLM을 이용한 객체 간 관계 이해
- Tracking을 이용한 대상 이동 경로 추적
- 특정 상품과 산업용 도구를 위한 도메인 데이터 fine-tuning
- 소음 환경별 VAD 및 Whisper keyword 정확도 개선
- 음성, 시각 또는 진동 기반 사용자 알림
- 로봇의 인지, 이동, 안내 및 물체 탐색 기능과 결합


13. 참고 자료
-------------

YOLO-World: Real-Time Open-Vocabulary Object Detection
https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html

YOLO-World 공식 구현
https://github.com/AILab-CVC/YOLO-World

Whisper: Robust Speech Recognition via Large-Scale Weak Supervision
https://arxiv.org/abs/2212.04356

OpenAI Whisper 공식 구현
https://github.com/openai/whisper
