# RoadTrack V2 audit and V3 starting point

RoadTrack V3 was copied from frozen V2 on 2026-08-20. Before copying, V2 was
verified against its starting hashes:

- `roadtrack_v2.py`: `0c8b56bfd5a20e0d53a917837e80dfba7cec996e0e47d2e942cbae45097e73ee`
- `bosch_roadtrack_v2.yaml`: `c722875f1d2e342275119cc042334cadf38d4db561677ec31e9bad6f357860d2`

## Actual V2 architecture

- Planar position: decoupled 4-state CV `[x,y,vx,vy]`.
- Vertical: independent CV `[z,vz]`.
- Yaw: independent `[yaw,yaw_rate]` filter with pi-symmetric residual.
- Shape: independent EMA over length/width/height.
- Velocity: latent KF velocity plus bounded observation-displacement blend.
- Association: strict Hungarian, lifecycle cascade, class-independent matching,
  position Mahalanobis hard gate and switchable 3D GIoU/DIoU/ODIoU/hybrid cost.
- Score stages: high and low; low cannot birth or confirm, but currently performs
  a noise-inflated full box update.
- Lifecycle: tentative/confirmed/lost/dormant/dead with real timestamp timeouts.
- Class: association-independent, ten-consecutive-match hysteresis; ID persists.
- Ego: optional 2D world transform in the adapter; core has no ROS dependency.
- Preprocess: class-agnostic rotated 3D NMS after detector per-class NMS.
- Organization: 1,045-line single implementation file; refactoring is needed.

## Requirement classification

| Requirement | V2 status |
|---|---|
| Decoupled CV, z, yaw, shape | ALREADY_IMPLEMENTED |
| Real dt, Joseph update, yaw wrap | ALREADY_IMPLEMENTED |
| Hungarian 1:1 and cascade | ALREADY_IMPLEMENTED |
| Mahalanobis position hard gate | ALREADY_IMPLEMENTED |
| 3D GIoU/DIoU/ODIoU | ALREADY_IMPLEMENTED |
| High/low association and lifecycle | ALREADY_IMPLEMENTED |
| Class hysteresis and ego transform | ALREADY_IMPLEMENTED |
| Monolithic state comparison | NEEDS_IMPLEMENTATION as an ablation |
| CA spatial model | NEEDS_IMPLEMENTATION |
| BEV GIoU and Ro_GDIoU | NEEDS_IMPLEMENTATION |
| Low full/position-only/none semantics | PARTIALLY_IMPLEMENTED |
| Radar point assignment and robust Doppler | NEEDS_IMPLEMENTATION |
| Scalar radial KF measurement | NEEDS_IMPLEMENTATION |
| Observable Cartesian WLS and fallback | NEEDS_IMPLEMENTATION |
| Radar association gates | NEEDS_IMPLEMENTATION |
| HOTA/AssA evaluator | NEEDS_IMPLEMENTATION in the original evaluator |

V2, SimpleTrack and MCTrack are frozen baselines. All implementation and tuning
after the exact-copy check occurs only inside this V3 package.
