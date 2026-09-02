# RoadTrack V3

RoadTrack V3는 RAPA-R의 radar-only 3D detection 출력에 맞춰 설계한 온라인 다중 객체 추적기(MOT)이다. 카메라 특징, detector가 만든 ID, 미래 프레임은 사용하지 않는다. 각 시점의 회전 3D box를 입력받아 **엄격한 1 detection ↔ 1 track** 관계로 연결하고, 짧은 detection 누락·점수 하락·class 흔들림에도 가능한 한 같은 ID를 유지한다.

현재 production selector에는 `3d_optimized`, `bev_optimized` 두 모드만 노출한다.

> 핵심 요약: RoadTrack V3는 단순히 GIoU에 중심 거리를 더한 tracker가 아니다. **tracker 입력 NMS → timestamp 예측 → 분리형 KF → covariance/motion hard gate → score-aware 1:1 association → 상태 기반 lifecycle → 숨김 re-identification**을 하나의 파이프라인으로 구성한다.

## 1. 입출력 계약

### 입력

```text
boxes:        float[N, 7] = [x, y, z, length, width, height, yaw]
scores:       float[N]
labels:       int[N]
timestamp_ns: int
ego_pose:     optional 3x3 planar pose
radar_points: optional structured PointCloud2 array (experimental)
```

- `boxes`는 detector가 현재 프레임에서 추론한 회전 3D box이다.
- detector가 내부적으로 가지고 있던 ID가 있더라도 tracker는 사용하지 않는다.
- class는 출력용 상태로 유지하지만 association 조건에서는 제외한다.
- `ego_pose=None`이면 입력 좌표계 안에서 그대로 추적한다. 현재 평가도 ego-motion 없이 수행했다.
- `radar_points` 경로는 실험용이며 기본 설정에서는 꺼져 있다.

### 출력

각 출력 track은 적어도 다음 의미를 갖는다.

```text
track_id, box[x,y,z,l,w,h,yaw], score, label, velocity, lifecycle state
```

출력 box는 tracker의 물리 상태이다. 차선 중심으로 mesh를 당기는 처리, mesh 원점/크기 보정, 색상 그라데이션은 **visualization-only**이며 association 및 원본 track 좌표를 바꾸지 않는다.

## 2. RAPA-R에서 해결하려는 문제

Radar-only detection에는 다음 현상이 반복된다.

1. 동일 차량의 class가 `s_vehicle ↔ l_vehicle`처럼 순간적으로 흔들릴 수 있다.
2. 한두 프레임 score가 낮아지거나 box가 누락될 수 있다.
3. 빠른 상대 운동에서는 직전 box와 현재 box의 IoU가 0에 가까울 수 있다.
4. yaw와 length/width가 흔들리면서 하나의 결합 KF에 잘못된 속도 정보가 들어갈 수 있다.
5. detector의 class별 NMS를 통과한 동일 위치의 cross-class 중복 box가 tracker에 함께 들어올 수 있다.
6. 차량이 밀집한 경우 넓은 거리 gate만 사용하면 기존 ID가 인접 차량로 넘어갈 수 있다.

RoadTrack V3는 gate 하나를 무조건 느슨하게 만드는 대신, 각 문제를 서로 다른 단계에서 처리한다. 빠른 차량을 위해 예측 범위를 확보하면서도, lateral motion·속도 변화·shape·overlap을 함께 확인해 인접 객체 간 ID 탈취를 제한한다.

## 3. 전체 처리 흐름

```text
RAPA-R detections
        │
        ▼
class-agnostic rotated 3D NMS (IoU 0.001)
        │
        ▼
timestamp 기반 predict + 분리형 KF
        │
        ├─ high-score detections (>= 0.62)
        └─ low-score detections  (0.27 ~ 0.62)
        │
        ▼
confirmed/lost track miss-age cascade
        │  Mahalanobis + center/z/size/velocity hard gates
        │  3D GIoU 또는 BEV hybrid cost
        ▼
strict Hungarian 1:1 matching
        │
        ├─ tentative confirmation (high only)
        ├─ low-score rescue (기존 track만)
        ├─ dormant re-identification (high only)
        └─ unmatched high-score detection birth
        │
        ▼
tentative → confirmed → lost → dormant → dead
        │
        ▼
physical track output + 별도 visualization output
```

## 4. 단계별 상세 설명

### 4.1 Tracker 입력 NMS

기본 detector NMS는 class별로 동작할 수 있기 때문에 같은 차량이 서로 다른 class로 검출되면 두 box가 함께 남을 수 있다. V3는 association 직전에 class-agnostic rotated 3D NMS를 한 번 더 수행한다.

```yaml
input_nms:
  enabled: true
  iou_threshold: 0.001
```

현재 `0.001`은 작은 겹침도 중복 후보로 보는 공격적인 값이다. 이는 일반적인 detector NMS 권장값이 아니라 현재 RAPA-R/평가 slice에 맞춘 값이다. 서로 가까워도 실제 3D volume이 겹치지 않는 차량은 합치지 않으며, 이미 생성된 인접 track끼리 병합하는 기능도 없다.

### 4.2 Timestamp 기반 예측

프레임 번호가 아니라 실제 timestamp 차이 `dt`로 predict한다.

```yaml
default_dt: 0.10
min_predict_dt: 0.02
max_predict_dt: 1.00
```

rosbag 지연이나 처리 주기 변화가 있어도 동일한 프레임 간격으로 가정하지 않는다. timestamp가 뒤로 되감기면 이전 시퀀스의 상태가 새 재생에 섞이지 않도록 tracker를 reset한다.

### 4.3 분리형 Kalman filter

SimpleTrack 계열의 결합 상태 대신, 서로 성격이 다른 상태를 분리한다.

| 상태 | 모델/업데이트 | 목적 |
|---|---|---|
| `[x, y, vx, vy]` | planar CV KF | 평면 위치와 속도 예측 |
| `[z, vz]` | vertical CV KF | 높이 방향 noise가 평면 속도 covariance에 미치는 영향 차단 |
| `[yaw, yaw_rate]` | angle-aware KF | `±π` wrap을 처리하며 회전 상태 추정 |
| `[length, width, height]` | EMA (`alpha=0.10`) | detector box 크기 jitter 완화 |

따라서 순간적인 yaw/size 이상치가 planar velocity covariance를 직접 오염시키지 않는다. motion model 자체는 기본적으로 CV(Constant Velocity)이며, 관측 위치 차이로 얻은 observation velocity를 약하게 주입한다.

```yaml
observation_velocity:
  gain: 0.20
  gain_after_miss: 0.60
  gain_low: 0.05
  gain_low_after_miss: 0.15
```

누락 뒤에는 새 관측을 더 빠르게 반영하지만, low-score box는 apparent velocity를 크게 흔들 수 있으므로 gain을 낮춘다.

### 4.4 Score 분리와 low-score rescue

```yaml
low_score_threshold: 0.27
high_score_threshold: 0.62
new_track_score_threshold: 0.62
low_score_update: position_only
```

- high-score detection은 기존 track 매칭, tentative 확인, dormant re-ID, 새 birth에 사용할 수 있다.
- low-score detection은 이미 존재하는 track을 살리는 데만 사용한다.
- low-score detection만으로 새 ID를 만들거나 tentative track을 확정하지 않는다.
- low-score match에서는 position만 갱신해 불안정한 yaw/shape가 track 상태를 흔들지 않게 한다.

### 4.5 Association: hard gate와 cost의 역할

`gate`와 `cost`는 역할이 다르다.

- **Hard gate**: 물리적으로 말이 안 되는 track-detection 쌍을 Hungarian 입력 전에 제거한다.
- **Cost**: gate를 통과한 후보 중 어떤 조합이 가장 자연스러운지 순위를 매긴다.
- **Hungarian**: 전체 cost 합이 최소가 되도록 strict 1:1 배정을 수행한다.

기본 gate에는 다음 항목이 들어간다.

1. center distance
2. z difference (`3d_optimized`만)
3. box size ratio
4. KF covariance 기반 Mahalanobis distance
5. observation/prediction velocity residual
6. mode별 longitudinal/lateral distance

#### Mahalanobis gate

Mahalanobis distance는 단순 유클리드 거리와 달리 KF의 예측 covariance를 사용한다.

```text
d² = (z - Hx)ᵀ S⁻¹ (z - Hx)
S  = HPHᵀ + R
```

예측 불확실성이 커진 track에는 같은 미터 오차를 상대적으로 덜 엄격하게 보고, 안정적으로 추적 중인 track에는 더 엄격하게 본다. V3에서는 기본적으로 **중복 cost 항이 아니라 hard gate**로 사용한다.

#### Velocity consistency gate

track의 예측 속도와 새 detection까지 필요한 순간 속도의 residual이 지나치게 크면 후보를 제거한다. 누락 프레임 수에 따라 허용량을 제한적으로 늘려 빠른 차량의 재연결은 허용하지만, 넓은 center gate 때문에 옆 차량로 ID가 넘어가는 현상은 억제한다.

#### Miss-age cascade

모든 track을 한 번에 경쟁시키지 않고 miss가 적은 track부터 매칭한다. 최근까지 정상 관측된 track이 오래 누락된 불확실한 track보다 먼저 올바른 detection을 선택하게 한다.

### 4.6 두 production association 모드

#### `3d_optimized` — 기본/권장

- 실제 회전 3D intersection과 enclosing volume을 사용하는 GIoU3D cost
- `[x, y, z]` covariance의 Mahalanobis gate
- z difference 및 height ratio 포함
- stage별 center/cost threshold

box 높이 정보가 충분히 안정적이면 상하 위치와 height도 구별 정보로 활용할 수 있다.

#### `bev_optimized`

- z/height를 association에서 제외
- 회전 BEV GIoU가 주 cost (`weight=0.95`)
- `[x, y]` Mahalanobis gate
- predicted yaw 좌표계의 longitudinal/lateral distance
- velocity, yaw, size의 약한 보조 cost

```text
C_bev = 0.95 C_giou
      + w_long C_longitudinal
      + w_lat  C_lateral
      + w_vel  C_velocity
      + w_yaw  C_yaw
      + w_size C_size
```

중심 거리를 하나의 원형 반경으로만 보지 않고 차량 진행 방향 기준 종·횡 방향으로 나눈다. 빠른 종방향 이동은 넓게 허용하면서도 불필요한 횡방향 ID 전이는 더 제한할 수 있다.

| 항목 | `3d_optimized` | `bev_optimized` |
|---|---|---|
| 주 geometry cost | rotated GIoU3D | rotated BEV GIoU + 약한 motion/shape cost |
| Mahalanobis 차원 | XYZ | XY |
| z/height gate | 사용 | 미사용 |
| 중심 거리 | XYZ/center gate | center + heading-frame 종·횡 gate |
| 권장 상황 | z/height가 안정적인 현재 기본값 | z/height noise가 큰 데이터 비교 실험 |

### 4.7 Lifecycle과 birth/삭제

```text
tentative ──충분한 관측──> confirmed
    │                        │
    └─timeout────────> dead  └─짧은 누락──> lost
                                      │
                           재관측 ─────┤
                                      └─visible timeout──> dormant
                                                           │
                                                re-ID 성공 ─┤
                                                           └─timeout──> dead
```

- **tentative**: 아직 화면에 내보내지 않는 birth 후보.
- **confirmed**: 정상적으로 publish되는 track.
- **lost**: 짧게 detection이 누락됐지만 예측 상태로 유지되는 track.
- **dormant**: 화면에는 내보내지 않지만 짧은 재등장을 위해 KF와 ID를 보관하는 숨김 상태.
- **dead**: re-ID 허용 시간까지 지난 최종 삭제 상태.

확정 조건은 연속 3프레임 고정이 아니라 **6프레임 window 안의 3회 high-score 관측**이다. 1~2프레임 noise birth는 숨기면서 중간의 짧은 score dip은 허용한다.

```yaml
confirmation:
  hits: 3
  window: 6
  require_consecutive: false
lifecycle:
  visible_max_lost_seconds: 0.30
  reid_max_lost_seconds: 2.00
```

약 5프레임 누락까지만 visible 상태로 유지하고, 이후에는 ghost를 계속 보여주지 않는다. 대신 hidden dormant 상태에서 같은 ID의 re-identification을 시도한다.

### 4.8 Class 처리

Class는 제거하지 않는다. 다만 class mismatch로 association을 막지 않는다.

- `s_vehicle → l_vehicle` 오탐이 한 번 발생해도 기존 ID로 매칭한다.
- track의 대표 class는 즉시 바꾸지 않는다.
- 다른 class가 10회 일관되게 관측될 때만 published class를 변경한다.
- 따라서 ID continuity와 mesh 종류 결정에 필요한 class 안정성을 분리한다.

```yaml
class_update_min_hits: 10
```

Class 변경은 같은 track 내부 상태 변경이며, 그 자체로 새 `track_id`를 만들지 않는다.

## 5. SimpleTrack과의 구조 비교

| 구분 | SimpleTrack 기준 구현 | RoadTrack V3 |
|---|---|---|
| 입력 중복 제거 | detector 출력 의존 | tracker-side class-agnostic rotated 3D NMS |
| 상태 추정 | box/motion을 하나의 KF 상태로 관리 | planar, vertical, yaw를 분리한 KF + size EMA |
| 시간 | frame/timestamp 기반 기본 예측 | 실제 timestamp `dt` clamp 및 rewind reset |
| 주 association | GIoU3D + Hungarian | hard gates + mode별 geometry cost + Hungarian |
| Mahalanobis | 원본 SimpleTrack 핵심 경로에는 없음 | covariance-aware hard gate |
| 빠른 차량 대응 | 예측 box와 GIoU 중심 | center/motion gate, observation velocity, miss 성장 |
| score 처리 | 단일 association/redundancy 중심 | high/low 분리, established joint match, low rescue |
| low-score birth | 별도 강한 제한 없음 | 금지 |
| class mismatch | 설정/구현에 따라 association 영향 가능 | association에서 무시, 출력 class는 10-hit hysteresis |
| birth 표시 | `min_hits` 중심 | 6-frame window 내 3 high hits 전까지 숨김 |
| 누락 이후 | age/redundancy로 유지 후 삭제 | lost와 hidden dormant를 분리해 ghost 없이 re-ID |
| 매칭 순서 | 전역 assignment 중심 | miss-age cascade + stage별 threshold |
| geometry 선택 | GIoU3D 중심 | tuned 3D와 tuned BEV profile 전환 가능 |


## 6. 성능 비교

### 6.1 최신 연속 detection / anchor GT 평가

평가 조건:

- RAPA-R post-NMS detection: **5,000 연속 프레임**, 384.562초
- 사람이 검수한 GT anchor: **348 프레임**, 6,472 boxes
- persistent GT: 3회 이상 등장한 trajectory
- 세 tracker에 동일한 raw `[boxes, scores, labels, timestamp]` stream 전달
- detector capture threshold: `0.2` (저장 stream 자체의 하한)
- SimpleTrack: adapter/config 기본 정책 사용
- RoadTrack V3: 자체 low-score threshold `0.27`, input NMS `0.001`, 3-of-6 confirmation 사용

| Tracker | HOTA ↑ | DetA ↑ | AssA ↑ | MOTA ↑ | IDF1 ↑ | IDSW ↓ | Frag ↓ | Precision ↑ | Recall ↑ | Runtime ms/continuous frame ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SimpleTrack native | 0.63907 | 0.70376 | 0.58150 | 0.64198 | 0.67631 | 1,009 | **84** | 0.88595 | 0.92495 | 30.71 |
| RoadTrack V3 `3d_optimized` native | **0.69661** | 0.77317 | **0.62847** | **0.68746** | **0.71068** | **960** | 86 | **0.91844** | 0.92560 | **24.94** |
| RoadTrack V3 `bev_optimized` native | 0.69585 | **0.77352** | 0.62683 | 0.68648 | 0.70757 | 964 | 89 | 0.91747 | **0.92641** | 27.34 |

해석:

- `3d_optimized`가 종합 1위였다.
- SimpleTrack native 대비 HOTA `+0.05754`, MOTA `+0.04548`, IDF1 `+0.03437`, IDSW `-49`였다.
- SimpleTrack은 fragmentation이 2회 적었지만 FP가 `733`으로 3D의 `506`보다 많았고, 그 결과 precision과 DetA가 낮았다.
- `bev_optimized`는 DetA와 recall이 아주 조금 높았지만, 3D 모드가 AssA/MOTA/IDF1 및 ID continuity에서 더 좋았다.
- 따라서 현재 기본값은 `3d_optimized`가 타당하며, BEV 모드는 z/height noise가 다른 bag에서 역전되는지 확인하는 비교 모드로 유지한다.


### 지표 해석

- **DetA**: detection과 GT가 얼마나 잘 대응되는지.
- **AssA**: 대응된 객체의 identity 연결이 얼마나 일관적인지.
- **HOTA**: detection과 association 품질을 함께 보는 균형 지표.
- **MOTA**: FP, FN, ID switch를 종합한 전통적 MOT 지표.
- **IDF1**: 전체 시간 동안 올바른 identity로 설명한 비율.
- **IDSW**: 동일 GT 객체에 tracker ID가 바뀐 횟수.
- **Frag**: 한 GT trajectory가 중간에 끊겼다가 다시 잡힌 횟수. 새 ID로 바뀌지 않아도 관측 구간이 끊기면 fragmentation이 될 수 있으므로 IDSW와 동일하지 않다.

## 7. 실행 방법

```bash
source /opt/ros/humble/setup.bash
source /root/workspace/poc_ws/install/setup.bash
source /root/workspace/DFAI_mmdet3d/venv/bin/activate

export ROS_DOMAIN_ID=98
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

cd /root/workspace/DFAI_mmdet3d
./inference_roadtrack_v3.sh
```

모드 전환:

```bash
ROADTRACK_V3_MODE=3d_optimized ./inference_roadtrack_v3.sh
ROADTRACK_V3_MODE=bev_optimized ./inference_roadtrack_v3.sh
```

Production adapter/CLI는 위 두 모드만 허용한다. 과거 실험용 metric을 production 파라미터와 잘못 혼합하는 일을 막기 위한 제한이다.

## 8. 주요 설정 위치

```text
src/rapa_tracking/roadtrack_v3/
├── README.md
├── adapter.py
├── tracker.py
├── configs/
│   ├── bosch_roadtrack_v3.yaml        # 현재 tuned production 설정
│   └── baseline_v2_equivalent.yaml    # V2-equivalent 출발점
├── track/                             # 분리형 filter 및 lifecycle
├── geometry/                          # rotated 3D/BEV geometry
└── tests/
```

튜닝 시 우선순위:

1. `input_nms.iou_threshold`: cross-class duplicate 제거 강도
2. `low/high/new_track_score_threshold`: 유지, 확인, birth의 score 경계
3. `profiles.*.association.*`: mode별 gate와 `max_cost`
4. `observation_velocity.*`: 위치 차분 속도의 반영 강도
5. `confirmation` 및 `lifecycle`: noise birth와 누락 유지 시간
6. `class_update_min_hits`: class 변경 hysteresis

한 번에 여러 축을 바꾸면 원인을 해석하기 어렵다. 동일 detection cache와 동일 GT를 사용해 한 그룹씩 ablation하는 것이 좋다.

상세 매칭 진단이 필요할 때:

```yaml
debug:
  record_match_details: true
```

이 옵션으로 center, GIoU, yaw, size, Mahalanobis 및 stage별 match 정보를 기록할 수 있다. 장시간 실시간 실행에서는 로그 비용을 고려해 기본값은 `false`이다.

## 9. Radar velocity 실험 상태

Scalar Doppler, multi-LOS WLS, radial KF/association gate를 실험 경로로 구현했지만 production 기본값은 꺼져 있다.

```yaml
radar:
  enabled: false
  kf_update: false
  association_gate: false
```

현재 통합 PointCloud2는 각 point를 만든 radar sensor의 원점/LOS transform을 보존하지 않는다. 따라서 `radial_velocity × LOS`를 완전한 Cartesian velocity로 간주할 수 없으며, edited slice에서는 radial KF/gate가 HOTA와 AssA를 낮추고 ID switch를 늘렸다. 향후 WLS를 사용하려면 각 point의 source radar transform, 다양한 LOS, 품질/variance를 함께 보존해야 한다.

## 10. 테스트

```bash
cd /root/workspace/DFAI_mmdet3d
source venv/bin/activate
cd /root/workspace/rapa_tracking
/root/workspace/DFAI_mmdet3d/venv/bin/python -m pytest -q \
  src/rapa_tracking/roadtrack_v3/tests
```

테스트 범위에는 동일 box, non-overlap 거리 순서, yaw `0/π` 동치, Mahalanobis gate, strict 1:1 Hungarian, high-speed zero-IoU continuity, lifecycle/class hysteresis 등이 포함된다.

## 11. 알려진 한계

1. 현재 파라미터는 Bosch/RAPA-R와 내부 GT slice에 강하게 최적화되어 있다.
2. ego-motion을 사용하지 않은 평가는 자차의 급격한 회전/가감속 장면을 충분히 대표하지 못한다.
3. CV model은 장시간 누락 중 급가감속·차선 변경을 완전히 예측할 수 없다.
4. appearance embedding을 쓰지 않으므로 기하와 운동이 매우 비슷한 밀집 차량의 장기 occlusion에는 한계가 있다.
5. tracker 입력 NMS `0.001`은 aggressive하므로 다른 detector에서는 가까운 정상 객체의 recall을 해칠 수 있다.
6. 최신 348-anchor GT는 완전 독립적인 연속 수동 GT가 아니므로 순위의 절대값보다 동일 조건의 상대 비교로 해석해야 한다.
