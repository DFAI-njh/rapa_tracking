# Tracking visualization

이 폴더는 RAPA-R 추론 결과를 추적한 뒤 Foxglove에서 보기 쉬운 형태로 표현하는
**트래킹 전용 시각화 모듈**이다. 차량 mesh, track ID, 속도 벡터, 충돌 위험
지수(CRI), ego 차량 및 가상 차선을 `visualization_msgs/MarkerArray`로 생성한다.

이 모듈은 별도의 ROS 노드나 subscriber가 아니다. Bosch inference 노드가 각
tracker의 `update()`를 수행하고 원래 track 토픽을 발행하는 과정에서
`SimpleTrackVisualizationRenderer.publish()`를 직접 호출한다. 따라서 별도
시각화 노드를 실행할 필요가 없다.

## 가장 중요한 원칙

시각화에서 사용하는 bbox는 tracker 출력의 복사본이다. 다음 항목은 표시용
복사본에만 적용되며 tracker의 Kalman filter, association, track ID, class,
평가 결과에는 영향을 주지 않는다.

- lane 중심 방향의 lateral 보정
- 입력 주기 사이의 30 Hz 위치·크기·yaw 보간
- mesh 종류, 방향 및 색상 결정
- 역방향 차량의 mesh 숨김
- 속도 화살표 및 CRI 계산

원래 `/rapar/bosch/tracks` 계열 토픽은 이 폴더의 처리와 무관하게 그대로
발행된다. 즉 Foxglove용 결과가 차선 중심으로 이동하더라도 tracker 자체의
box 좌표가 변경된 것은 아니다.

## 파일 구성

| 파일/디렉터리 | 역할 |
|---|---|
| `renderer.py` | Marker 생성, mesh HTTP 서버, lane 보정, 보간, 속도 화살표와 CRI 계산 |
| `config.yaml` | 토픽, mesh, 색상, 크기, 차선, 보간 및 CRI 파라미터 |
| `assets/small_centered.obj` | 길이 6 m 미만 객체와 ego에 사용하는 중심 정렬 승용차 mesh |
| `assets/large_centered.obj` | 길이 6 m 이상 객체에 사용하는 중심 정렬 대형 차량 mesh |
| `test_renderer_math.py` | yaw 보간, 설정 및 CRI 수학 검증 |
| `__init__.py` | renderer 공개 import |

## 입력과 출력

### 입력

inference 노드가 tracker별 결과를 아래 배열 형태로 `publish()`에 전달한다.

```text
boxes:       N x 7 = [x, y, bottom_z, length, width, height, yaw]
scores:      N
labels:      N
track_ids:   N
alpha:       N       # missed track fade 표시용
velocities:  N x 3   # [vx, vy, vz], 제공 가능한 tracker에 한함
```

좌표계는 `ego_vehicle`이며 `x`는 차량 전방, `y`는 좌우 방향이다. RoadTrack
V3는 Kalman filter가 추정한 속도를 전달한다. 속도를 제공하지 않는 backend는
동일 ID의 연속 bbox 차분으로 표시용 속도만 추정한다. 이 fallback 속도 역시
tracker 상태로 되돌아가지 않는다.

### 출력 토픽

기본 설정은 다음과 같다.

| 토픽 | 내용 |
|---|---|
| `/rapar/bosch/tracks_mesh` | 차량 mesh, track ID/CRI text, 속도 화살표, ego mesh |
| `/rapar/bosch/lanes` | 일반 차선 경계와 ego 차선 경계 |
| `/rapar/bosch/tracks` | 기존 tracker marker. 이 모듈이 수정하지 않음 |

SimpleTrack과 RoadTrack V3 비교 모드에서는 backend 이름이 suffix로 붙는다.
예를 들어 `/rapar/bosch/tracks_mesh/simpletrack`과
`/rapar/bosch/tracks_mesh/roadtrack_v3`처럼 분리되어 동시에 비교할 수 있다.

## 처리 흐름

```text
RAPA-R detections
        ↓
selected tracker update/association
        ├── 원본 track MarkerArray 발행 ──→ 평가·원본 시각화
        ↓ 복사본
lane 표시 보정 → 30 Hz 보간 → mesh/ID/arrow/CRI MarkerArray 발행
```

### 1. Mesh 선택과 정렬

mesh 선택에는 detector class를 사용하지 않는다. bbox의 `length`가
`large_mesh_min_length` 이상이면 large mesh, 그보다 짧으면 small mesh를
사용한다. 기본 임계값은 6.0 m이며 정확히 6.0 m인 객체는 large로 분류한다.

OBJ 원본 축, 크기 및 원점 차이는 다음 설정으로 보정한다.

- `*_native_dimensions`: OBJ 원본의 실제 크기
- `*_dimension_order`: bbox length/width/height와 OBJ 축의 대응 순서
- `*_mesh_rpy_deg`: OBJ 기본 자세 보정
- `*_mesh_z_offset`: 필요할 때 사용하는 높이 오프셋

현재 `small_centered.obj`, `large_centered.obj`는 원점 편향을 제거해 둔
asset이다. 다른 OBJ로 교체할 때는 파일명만 바꾸지 말고 위 네 설정도 해당
mesh에 맞게 갱신해야 bbox 중심과 mesh 중심이 일치한다.

### 2. 표시 대상과 색상

현재 mesh 토픽에서는 `cos(yaw) < 0`인 역방향 track을 숨긴다. 이는 표시만
생략하는 것이며 해당 track은 원본 track 토픽에 계속 존재한다.

같은 방향 차량의 mesh 색상은 ego와의 평면 거리에 따라 설정된다.

- 10 m 이하: 빨간색
- 10~30 m: 빨간색에서 노란색
- 30~70 m: 노란색에서 흰색
- 70 m 이상: 흰색

구간 내부는 smoothstep 색상 보간을 사용해 경계에서 색이 갑자기 바뀌지
않는다. ego mesh는 별도의 `ego_color`로 초록색을 사용한다.

### 3. Lane 시각화 보정

`use_lane_visual_follow`가 활성화되면 진행 방향 차량의 표시용 `y` 좌표를
가장 가까운 lane 중심 쪽으로 조금씩 이동시킨다. 한 프레임에 lane 중심으로
강제로 붙이지 않고 `lane_visual_gain`과 `lane_damping_gain`만큼 반영한다.

다음 객체는 보정하지 않는다.

- `lane_follow_yaw_disable_th`보다 yaw 절댓값이 큰 객체
- 가장 바깥쪽 두 차선 경계 밖에 있는 객체
- lane follow가 비활성화된 경우

따라서 도로 밖 객체를 바깥 lane으로 끌어오는 현상은 발생하지 않는다.
표시용 상태는 track ID별로 관리되며 사라진 ID의 상태는 즉시 정리된다.

### 4. 30 Hz 시각화 보간

tracker 입력은 대략 7.5 Hz지만 Foxglove marker는 ROS timer를 통해 30 Hz로
발행한다. 새 tracker 결과가 들어오면 직전 표시 상태와 새 목표 상태 사이를
smoothstep으로 보간한다.

- 위치와 크기: 선형 보간
- yaw: `-pi~pi` 최단 회전 방향으로 보간
- alpha와 표시용 속도: 선형 보간
- 입력 간격: 최소/최대 범위로 제한한 뒤 EMA로 갱신

이 방식은 tracker 결과를 다시 추정하는 것이 아니라 두 결과 사이의 표시만
채운다. 별도 시각화 thread, 임계감쇠 필터, adaptive smoothing 및 lane
hysteresis는 사용하지 않는다.

### 5. 속도 화살표

속도가 `velocity_arrow_min_speed_mps` 이상인 track에 이동 방향 화살표를
그린다. 길이는 `speed × velocity_arrow_horizon_sec`이며
`velocity_arrow_max_length_m`에서 제한된다. 굵기와 화살촉 크기, 높이,
불투명도 및 색상은 `velocity_arrow_*` 설정으로 변경할 수 있다.

현재 기본 색상은 어두운 남색이다.

```yaml
velocity_arrow_color: [0.05, 0.12, 0.55]
```

### 6. Track ID와 CRI

text marker는 다음 형식이다.

```text
<track_id> CRI <0~100>%
```

CRI는 확률 모델로 보정된 충돌 확률이 아니라 **표시용 정규화 경고 지수**다.
현재 상대속도가 일정하다고 가정하고 다음 값을 조합한다.

- ego/target 확장 footprint에 진입하는 시간(TTC)
- 최근접점 도달 시간과 여유 거리(CPA)
- 현재 box 간 근접도

따라서 `CRI 80%`를 실제 충돌 확률 80%로 해석하면 안 된다. 운전자 경고용
기준으로 사용하려면 실제 주행 데이터로 별도 calibration과 검증이 필요하다.

## Mesh asset 제공 방식

Foxglove 브라우저는 원격 서버의 `file:///root/...` 경로를 직접 읽을 수 없다.
renderer는 OBJ 파일을 CORS가 허용된 HTTP asset 서버로 제공하며 marker에는
`http://<server>:<port>/<mesh>.obj` URL을 넣는다.

관련 설정은 다음과 같다.

```yaml
asset_bind_address: 0.0.0.0
asset_server_port: 8876
asset_public_base_url: http://192.168.100.201:8876
```

`Address already in use`가 발생하면 기존 inference/asset server 프로세스가
남아 있는지 먼저 확인한다. 실제로 다른 서비스가 해당 포트를 사용한다면
`asset_server_port`와 `asset_public_base_url`의 포트를 함께 변경한다.

## 주요 설정

| 설정 | 의미 |
|---|---|
| `interpolation_publish_hz` | mesh/lane 표시 발행 주기 |
| `interpolation_*duration_sec` | 새 입력 사이 보간 시간 범위 |
| `large_mesh_min_length` | large mesh 선택 길이 임계값 |
| `same_direction_color_stops` | 거리별 차량 mesh 색상 |
| `track_id_font_size` | ID/CRI text 크기 |
| `velocity_arrow_*` | 속도 화살표 길이, 굵기, 색상 및 표시 임계값 |
| `collision_risk_*` | CRI horizon과 정규화 계수 |
| `lanes_num`, `ego_lane_num`, `lane_width` | 가상 차선 배치 |
| `line_x_min`, `line_x_max` | 차선 표시 시작·끝 거리 |
| `lane_visual_gain`, `lane_damping_gain` | lane 중심 표시 보정 강도 |
| `tracking_marker_lifetime` | inference 인자에서 전달되는 marker 수명 |

## 실행

RoadTrack V3 시각화를 실행하는 일반적인 예시는 다음과 같다.

```bash
source /opt/ros/humble/setup.bash
source /root/workspace/poc_ws/install/setup.bash
source /root/workspace/DFAI_mmdet3d/venv/bin/activate
export ROS_DOMAIN_ID=98

cd /root/workspace/DFAI_mmdet3d
./inference_roadtrack_v3.sh
```

Foxglove에서는 bridge에 연결한 뒤 3D panel에 mesh 토픽과 lane 토픽을
추가한다. 원본 bbox와 보정된 mesh를 비교하려면 `/rapar/bosch/tracks`와
`/rapar/bosch/tracks_mesh`를 같은 3D panel에서 함께 켜면 된다.

## 검증

시각화 수학 및 설정 테스트는 다음 명령으로 실행한다.

```bash
cd /root/workspace/rapa_tracking
/root/workspace/DFAI_mmdet3d/venv/bin/python -m pytest -q \
  src/rapa_tracking/visualization/test_renderer_math.py
```

설정을 변경한 뒤에는 inference 프로세스를 재시작해야 `config.yaml`이 다시
로드된다.
