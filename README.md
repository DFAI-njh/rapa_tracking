# rapa_tracking

RAPA-R radar detections에 맞춰 개발한 RoadTrack V3와 ROS2/Foxglove용
시각화 모듈을 함께 관리하는 독립 Python 패키지다.

이 저장소는 detector나 mmdet3d 자체를 포함하지 않는다. 외부 inference
프로그램이 NumPy detection 배열을 RoadTrack V3에 전달하고, 반환된 track을
필요한 ROS 메시지로 변환하는 구조다.

## 구성

```text
src/rapa_tracking/
├── preprocess/       # tracker 입력용 class-agnostic rotated 3D NMS
├── roadtrack_v3/     # association, motion filter, lifecycle, 설정과 테스트
└── visualization/    # mesh, lane, velocity arrow, CRI와 Foxglove asset
```

- `roadtrack_v3`: RAPA-R의 class jitter, 근접 중복 detection, 빠른 상대 운동과
  일시 누락에 대응하는 3D MOT backend
- `visualization`: tracker 원본 상태를 바꾸지 않고 mesh/lane/arrow/CRI 토픽을
  만드는 display-only renderer
- `preprocess`: detector ID나 GT를 사용하지 않는 tracker-side NMS


## 개발 설치

```bash
source /path/to/DFAI_mmdet3d/venv/bin/activate
python -m pip install -e /root/workspace/rapa_tracking
```

editable 설치이므로 이 저장소의 Python 코드를 수정한 뒤 재설치할 필요는
없다. 실행 중인 inference 프로세스만 재시작하면 변경이 반영된다.

## 기본 사용

```python
from rapa_tracking.roadtrack_v3 import RoadTrackV3Adapter

tracker = RoadTrackV3Adapter(
    "/root/workspace/rapa_tracking/src/rapa_tracking/roadtrack_v3/"
    "configs/bosch_roadtrack_v3.yaml"
)
```

시각화 renderer는 ROS2 node와 메시지 타입을 호출자가 전달하므로 이 패키지가
별도의 ROS 노드를 실행하지 않는다.

```python
from rapa_tracking.visualization import TrackingVisualizationRenderer
```

## 테스트

```bash
cd /root/workspace/rapa_tracking
/root/workspace/DFAI_mmdet3d/venv/bin/python -m pytest -q
```

