"""Direct renderer for internal tracker outputs.

This is not a ROS adapter or subscriber. The inference node calls ``publish``
directly after the selected tracker backend updates. It publishes separate mesh and
lane topics while the normal tracking topic remains untouched. Lane-follow
changes only a local copy of the boxes; tracker/Kalman state is never changed.
"""

import functools
import math
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import numpy as np
import yaml


class _CorsAssetHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "public, max-age=3600")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def log_message(self, _format, *_args):
        pass


def _quaternion_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quaternion_from_rpy_deg(values):
    roll, pitch, yaw = (math.radians(float(value)) for value in values)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class SimpleTrackVisualizationRenderer:
    def __init__(self, node, ros, logger, config_path, topic_suffix="",
                 start_asset_server=True):
        self.node = node
        self.Marker = ros["Marker"]
        self.MarkerArray = ros["MarkerArray"]
        self.Point = ros["Point"]
        self.logger = logger
        self.config_path = Path(config_path).resolve()
        with self.config_path.open("r", encoding="utf-8") as stream:
            self.cfg = yaml.safe_load(stream)["tracking_visualization"]

        self.asset_root = (self.config_path.parent / self.cfg["asset_directory"]).resolve()
        self.public_base_url = str(self.cfg["asset_public_base_url"]).rstrip("/")
        self.models = {group: self._load_model(group) for group in ("small", "large")}
        self.lane_boundaries = self._compute_lane_boundaries()
        self.lane_centers = [
            0.5 * (self.lane_boundaries[i] + self.lane_boundaries[i + 1])
            for i in range(len(self.lane_boundaries) - 1)
        ]
        self.lane_follow_state = {}
        self.motion_state = {}
        self.visual_header = None
        self.visual_start = {}
        self.visual_target = {}
        self.visual_animation_started = 0.0
        self.last_source_arrival = None
        self.visual_interval = 1.0 / float(
            self.cfg["interpolation_source_rate_hint_hz"])
        self.visual_lifetime = 1.0
        if start_asset_server:
            self._start_asset_server()
        topic_suffix = str(topic_suffix).strip("/")
        mesh_topic = str(self.cfg["mesh_topic"]).rstrip("/")
        lanes_topic = str(self.cfg["lanes_topic"]).rstrip("/")
        if topic_suffix:
            mesh_topic = f"{mesh_topic}/{topic_suffix}"
            lanes_topic = f"{lanes_topic}/{topic_suffix}"
        self.mesh_publisher = node.create_publisher(
            self.MarkerArray, mesh_topic, 10)
        self.lane_publisher = node.create_publisher(
            self.MarkerArray, lanes_topic, 10)
        publish_hz = float(self.cfg["interpolation_publish_hz"])
        if publish_hz <= 0.0:
            raise ValueError("interpolation_publish_hz must be positive")
        self.visual_timer = node.create_timer(
            1.0 / publish_hz, self._publish_interpolated)
        logger.info(
            "Integrated tracking visualization enabled: "
            f"lane_follow={self.cfg['use_lane_visual_follow']}, "
            f"interpolation={publish_hz:.1f}Hz, "
            f"mesh_topic={mesh_topic}, lanes_topic={lanes_topic}, "
            f"assets={self.public_base_url}")

    def _load_model(self, group):
        filename = str(self.cfg[f"{group}_mesh"])
        path = (self.asset_root / filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Tracking mesh asset does not exist: {path}")
        return {
            "url": f"{self.public_base_url}/{quote(filename)}",
            "dimensions": [float(v) for v in self.cfg[f"{group}_native_dimensions"]],
            "dimension_order": [int(v) for v in self.cfg[f"{group}_dimension_order"]],
            "offset_q": _quaternion_from_rpy_deg(self.cfg[f"{group}_mesh_rpy_deg"]),
            "z_offset": float(self.cfg[f"{group}_mesh_z_offset"]),
        }

    def _start_asset_server(self):
        handler = functools.partial(_CorsAssetHandler, directory=str(self.asset_root))
        address, port = str(self.cfg["asset_bind_address"]), int(self.cfg["asset_server_port"])
        try:
            self.asset_server = ThreadingHTTPServer((address, port), handler)
        except OSError as error:
            raise RuntimeError(
                f"Cannot start integrated mesh asset server on {address}:{port}: {error}") from error
        threading.Thread(target=self.asset_server.serve_forever, daemon=True).start()

    def _compute_lane_boundaries(self):
        count = int(self.cfg["lanes_num"])
        ego_lane = int(self.cfg["ego_lane_num"])
        width = float(self.cfg["lane_width"])
        if count <= 0 or not 1 <= ego_lane <= count:
            raise ValueError("Require lanes_num > 0 and 1 <= ego_lane_num <= lanes_num")
        leftmost = 0.5 * width + (ego_lane - 1) * width
        return [leftmost - i * width for i in range(count + 1)]

    def _apply_lane_visual_follow(self, raw_boxes, track_ids):
        boxes = np.asarray(raw_boxes, dtype=np.float32).copy()
        if not bool(self.cfg["use_lane_visual_follow"]):
            return boxes
        lane_half = 0.5 * float(self.cfg["lane_width"])
        interpolation = float(self.cfg["lane_visual_gain"])
        damping = float(self.cfg["lane_damping_gain"])
        live_ids = set()
        for index, (box, track_id) in enumerate(zip(boxes, track_ids)):
            track_id = int(track_id)
            live_ids.add(track_id)
            yaw = float(box[6])
            if abs(yaw) > float(self.cfg["lane_follow_yaw_disable_th"]):
                if bool(self.cfg["lane_follow_reset_state_on_disable"]):
                    self.lane_follow_state.pop(track_id, None)
                continue

            y_reference = float(box[1])
            # Do not pull an object that is already outside the rendered road
            # into the nearest outer lane. Lane following is a visualization
            # aid only for objects whose centres lie between the two outer
            # boundary lines.
            if bool(self.cfg.get("lane_follow_only_inside_boundaries", True)):
                road_y_min = min(self.lane_boundaries)
                road_y_max = max(self.lane_boundaries)
                if y_reference < road_y_min or y_reference > road_y_max:
                    self.lane_follow_state.pop(track_id, None)
                    continue
            nearest = min(self.lane_centers, key=lambda center: abs(y_reference - center))
            offset = y_reference - nearest
            state = self.lane_follow_state.setdefault(track_id, {
                "last_y": y_reference, "lane_center": nearest,
                "lane_change": False, "counter": 0,
            })
            if abs(offset) <= lane_half + float(self.cfg["lane_tolerance_low"]):
                target = (1.0 - interpolation) * y_reference + interpolation * nearest
                boxes[index, 1] = state["last_y"] + damping * (target - state["last_y"])
                state.update(last_y=float(boxes[index, 1]), lane_center=nearest,
                             lane_change=False)
                continue
            if (not state["lane_change"]
                    and abs(offset) > lane_half + float(self.cfg["lane_tolerance_high"])):
                state.update(lane_change=True, lane_center=nearest, counter=10)
            if state["lane_change"]:
                target = ((1.0 - interpolation) * y_reference
                          + interpolation * float(state["lane_center"]))
                boxes[index, 1] = state["last_y"] + damping * (target - state["last_y"])
                state["last_y"] = float(boxes[index, 1])
                if abs(y_reference - float(state["lane_center"])) < lane_half * 0.4:
                    state["counter"] -= 1
                    if state["counter"] <= 0:
                        state["lane_change"] = False
        for track_id in list(self.lane_follow_state):
            if track_id not in live_ids:
                self.lane_follow_state.pop(track_id, None)
        return boxes

    def _mesh_group(self, box):
        """Choose the visualization mesh from bbox length, not class label."""
        length = float(box[3])
        return ("large" if length >= float(self.cfg["large_mesh_min_length"])
                else "small")

    def _motion_color(self, x, y):
        distance = math.hypot(float(x), float(y))
        stops = self.cfg["same_direction_color_stops"]
        if distance <= float(stops[0]["distance"]):
            return [float(value) for value in stops[0]["color"]]
        for start, end in zip(stops, stops[1:]):
            end_distance = float(end["distance"])
            if distance <= end_distance:
                start_distance = float(start["distance"])
                ratio = ((distance - start_distance)
                         / max(end_distance - start_distance, 1e-6))
                # Smoothstep has zero slope at each colour stop, avoiding
                # visible hard transitions while keeping the intended hues.
                ratio = ratio * ratio * (3.0 - 2.0 * ratio)
                return [float(a) * (1.0 - ratio) + float(b) * ratio
                        for a, b in zip(start["color"], end["color"])]
        return [float(value) for value in stops[-1]["color"]]

    @staticmethod
    def _set_lifetime(marker, seconds):
        marker.lifetime.sec = int(seconds)
        marker.lifetime.nanosec = int((seconds % 1.0) * 1_000_000_000)

    def _make_mesh(self, header, box, label, track_id, class_names, lifetime, alpha_scale):
        x, y, z, dx, dy, dz, yaw = [float(v) for v in box[:7]]
        group = self._mesh_group(box)
        model = self.models[group]
        marker = self.Marker()
        marker.header, marker.ns, marker.id = header, f"simpletrack/mesh/{group}", int(track_id)
        marker.type, marker.action = self.Marker.MESH_RESOURCE, self.Marker.ADD
        marker.mesh_resource, marker.mesh_use_embedded_materials = model["url"], False
        dimensions = (dx, dy, dz)
        marker.scale.x, marker.scale.y, marker.scale.z = [
            max(dimensions[model["dimension_order"][i]] / model["dimensions"][i], 1e-5)
            for i in range(3)
        ]
        yaw_q = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
        result_q = _quaternion_multiply(yaw_q, model["offset_q"])
        marker.pose.orientation.x, marker.pose.orientation.y = result_q[0], result_q[1]
        marker.pose.orientation.z, marker.pose.orientation.w = result_q[2], result_q[3]
        marker.pose.position.x, marker.pose.position.y = x, y
        marker.pose.position.z = z + model["z_offset"]
        marker.color.r, marker.color.g, marker.color.b = self._motion_color(x, y)
        marker.color.a = float(self.cfg["mesh_alpha"]) * float(alpha_scale)
        self._set_lifetime(marker, lifetime)
        return marker

    def _collision_risk_percent(self, box, motion_velocity):
        """Return a display-only constant-velocity CRI in the range [0, 100].

        The published frame is ``ego_vehicle``: x is forward, y is lateral,
        box layout is [x, y, bottom-z, length, width, height, yaw], and velocity
        is [vx, vy, vz] in the same axes. The index combines swept expanded-box
        TTC, closest-point-of-approach clearance, and immediate proximity. It
        is a normalized warning index, not a calibrated collision probability.
        """
        if not bool(self.cfg["draw_collision_risk"]):
            return 0.0
        velocity = np.asarray(motion_velocity, dtype=np.float64).reshape(-1)
        if len(velocity) < 2 or not np.all(np.isfinite(velocity[:2])):
            return 0.0
        x, y, _, length, width, _, yaw = [float(v) for v in box[:7]]
        ego_dims = [float(v) for v in self.cfg["ego_dimensions"]]
        relative_position = np.asarray([
            x - float(self.cfg["ego_x"]),
            y - float(self.cfg["ego_y"]),
        ], dtype=np.float64)
        relative_velocity = velocity[:2]

        cosine, sine = abs(math.cos(yaw)), abs(math.sin(yaw))
        target_half_x = 0.5 * (cosine * length + sine * width)
        target_half_y = 0.5 * (sine * length + cosine * width)
        margin = float(self.cfg["collision_risk_margin_m"])
        half_extent = np.asarray([
            0.5 * ego_dims[0] + target_half_x + margin,
            0.5 * ego_dims[1] + target_half_y + margin,
        ], dtype=np.float64)
        if np.all(np.abs(relative_position) <= half_extent):
            return 100.0

        horizon = max(float(self.cfg["collision_risk_horizon_sec"]), 1e-3)
        enter, leave = 0.0, horizon
        collision_course = True
        for position, speed, extent in zip(
                relative_position, relative_velocity, half_extent):
            if abs(float(speed)) < 1e-6:
                if abs(float(position)) > float(extent):
                    collision_course = False
                    break
                continue
            first = (-float(extent) - float(position)) / float(speed)
            second = (float(extent) - float(position)) / float(speed)
            enter = max(enter, min(first, second))
            leave = min(leave, max(first, second))
            if leave < enter:
                collision_course = False
                break

        time_power = float(self.cfg["collision_risk_time_power"])
        if collision_course and leave >= enter and enter <= horizon:
            urgency = max(0.0, 1.0 - enter / horizon) ** time_power
            return float(np.clip(100.0 * urgency, 0.0, 100.0))

        current_clearance_vector = np.maximum(
            np.abs(relative_position) - half_extent, 0.0)
        current_clearance = float(np.linalg.norm(current_clearance_vector))
        proximity_distance = max(
            float(self.cfg["collision_risk_proximity_distance_m"]), 1e-3)
        proximity = max(0.0, 1.0 - current_clearance / proximity_distance)
        proximity *= float(self.cfg["collision_risk_proximity_max_fraction"])

        speed_squared = float(relative_velocity @ relative_velocity)
        closing = -float(relative_position @ relative_velocity)
        near_miss = 0.0
        if speed_squared > 1e-6 and closing > 0.0:
            closest_time = float(np.clip(closing / speed_squared, 0.0, horizon))
            closest_position = (
                relative_position + relative_velocity * closest_time)
            closest_clearance = float(np.linalg.norm(np.maximum(
                np.abs(closest_position) - half_extent, 0.0)))
            sigma = max(
                float(self.cfg["collision_risk_near_miss_sigma_m"]), 1e-3)
            spatial = math.exp(-0.5 * (closest_clearance / sigma) ** 2)
            temporal = max(0.0, 1.0 - closest_time / horizon) ** time_power
            near_miss = (spatial * temporal
                         * float(self.cfg[
                             "collision_risk_near_miss_max_fraction"]))
        return float(np.clip(100.0 * max(proximity, near_miss), 0.0, 100.0))

    def _make_velocity_arrow(self, header, box, motion_velocity, track_id,
                             lifetime, alpha_scale):
        if not bool(self.cfg["draw_velocity_arrows"]):
            return None
        velocity = np.asarray(motion_velocity, dtype=np.float64).reshape(-1)
        if len(velocity) < 2 or not np.all(np.isfinite(velocity[:2])):
            return None
        speed = float(np.linalg.norm(velocity[:2]))
        if speed < float(self.cfg["velocity_arrow_min_speed_mps"]):
            return None
        arrow_length = min(
            speed * float(self.cfg["velocity_arrow_horizon_sec"]),
            float(self.cfg["velocity_arrow_max_length_m"]))
        direction = velocity[:2] / max(speed, 1e-6)
        x, y, z, _, _, height, _ = [float(v) for v in box[:7]]
        arrow_z = (z + max(height * 0.5, 0.1)
                   + float(self.cfg["velocity_arrow_z_offset"]))
        marker = self.Marker()
        marker.header, marker.ns = header, "simpletrack/velocity"
        marker.id = int(track_id) + 4_000_000
        marker.type, marker.action = self.Marker.ARROW, self.Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.points = [
            self.Point(x=x, y=y, z=arrow_z),
            self.Point(
                x=x + float(direction[0]) * arrow_length,
                y=y + float(direction[1]) * arrow_length,
                z=arrow_z),
        ]
        marker.scale.x = float(self.cfg["velocity_arrow_shaft_diameter"])
        marker.scale.y = float(self.cfg["velocity_arrow_head_diameter"])
        marker.scale.z = float(self.cfg["velocity_arrow_head_length"])
        color = [float(v) for v in self.cfg["velocity_arrow_color"]]
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = float(self.cfg["velocity_arrow_alpha"]) * float(alpha_scale)
        self._set_lifetime(marker, lifetime)
        return marker

    def _make_text(self, header, box, track_id, risk_percent, lifetime):
        x, y, z, _, _, dz, _ = [float(v) for v in box[:7]]
        marker = self.Marker()
        marker.header, marker.ns = header, "simpletrack/ids"
        marker.id = int(track_id) + 1_000_000
        marker.type, marker.action = self.Marker.TEXT_VIEW_FACING, self.Marker.ADD
        marker.pose.position.x, marker.pose.position.y = x, y
        marker.pose.position.z = z + max(dz * 0.5, 0.1) + float(self.cfg["track_id_z_offset"])
        marker.pose.orientation.w = 1.0
        marker.scale.z = float(self.cfg["track_id_font_size"])
        marker.color.r, marker.color.g, marker.color.b = [float(v) for v in self.cfg["track_id_color"]]
        marker.color.a = float(self.cfg["track_id_alpha"])
        marker.text = f"{int(track_id)} CRI {int(round(risk_percent))}%"
        self._set_lifetime(marker, lifetime)
        return marker

    def _make_lanes(self, header, lifetime):
        positive = [v for v in self.lane_boundaries if v > 0.0]
        negative = [v for v in self.lane_boundaries if v < 0.0]
        highlighted = set(([min(positive)] if positive else []) + ([max(negative)] if negative else []))
        output = []
        for index, y in enumerate(self.lane_boundaries):
            is_ego = y in highlighted
            marker = self.Marker()
            marker.header = header
            marker.ns = "road/ego_lane_boundaries" if is_ego else "road/lane_boundaries"
            marker.id, marker.type, marker.action = 2_000_000 + index, self.Marker.LINE_STRIP, self.Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(self.cfg["ego_lane_line_width"] if is_ego else self.cfg["line_width"])
            if is_ego:
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = (0.15, 1.0, 0.15, 1.0)
            else:
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = (1.0, 1.0, 1.0, 1.0)
            marker.points = [
                self.Point(x=float(self.cfg["line_x_min"]), y=float(y), z=float(self.cfg["line_z"])),
                self.Point(x=float(self.cfg["line_x_max"]), y=float(y), z=float(self.cfg["line_z"])),
            ]
            self._set_lifetime(marker, lifetime)
            output.append(marker)
        return output

    def _make_ego(self, header, lifetime):
        dims = [float(v) for v in self.cfg["ego_dimensions"]]
        box = np.asarray([self.cfg["ego_x"], self.cfg["ego_y"], self.cfg["ego_z"],
                          dims[0], dims[1], dims[2], 0.0], dtype=np.float32)
        marker = self._make_mesh(header, box, 0, 3_000_000, ["small"], lifetime, 1.0)
        marker.ns = "road/ego_vehicle"
        ego_color = [float(value) for value in self.cfg["ego_color"]]
        marker.color.r, marker.color.g, marker.color.b = ego_color
        marker.color.a = 1.0
        return marker

    def render(self, header, raw_boxes, scores, labels, track_ids, class_names,
               lifetime, alpha_scales=None, motion_velocities=None,
               apply_lane_follow=True):
        del scores  # Kept in the call contract for future label/score styling.
        boxes = (self._apply_lane_visual_follow(raw_boxes, track_ids)
                 if apply_lane_follow
                 else np.asarray(raw_boxes, dtype=np.float32))
        if alpha_scales is None:
            alpha_scales = np.ones(len(boxes), dtype=np.float32)
        if motion_velocities is None:
            motion_velocities = np.zeros((len(boxes), 3), dtype=np.float32)
        output = self.MarkerArray()
        for box, label, track_id, alpha_scale, motion_velocity in zip(
                boxes, labels, track_ids, alpha_scales, motion_velocities):
            # Reverse-heading tracks stay available on the raw tracking topic
            # but are intentionally omitted from the mesh visualization.
            if math.cos(float(box[6])) < 0.0:
                continue
            output.markers.append(self._make_mesh(
                header, box, label, track_id, class_names, lifetime, alpha_scale))
            risk_percent = self._collision_risk_percent(box, motion_velocity)
            output.markers.append(self._make_text(
                header, box, track_id, risk_percent, lifetime))
            arrow = self._make_velocity_arrow(
                header, box, motion_velocity, track_id, lifetime, alpha_scale)
            if arrow is not None:
                output.markers.append(arrow)
        if bool(self.cfg["draw_ego_vehicle"]):
            output.markers.append(self._make_ego(header, lifetime))
        return output

    @staticmethod
    def _interpolate_box(start, target, ratio):
        result = np.asarray(start, dtype=np.float32).copy()
        target = np.asarray(target, dtype=np.float32)
        result[:6] += (target[:6] - result[:6]) * ratio
        yaw_delta = (float(target[6]) - float(result[6]) + math.pi) % (
            2.0 * math.pi) - math.pi
        result[6] = float(result[6]) + yaw_delta * ratio
        return result

    def _current_visual_state(self, now):
        if not self.visual_target:
            return {}
        elapsed = max(0.0, now - self.visual_animation_started)
        ratio = min(elapsed / max(self.visual_interval, 1e-6), 1.0)
        ratio = ratio * ratio * (3.0 - 2.0 * ratio)
        state = {}
        for track_id, target in self.visual_target.items():
            start = self.visual_start.get(track_id, target)
            state[track_id] = dict(target)
            state[track_id]["box"] = self._interpolate_box(
                start["box"], target["box"], ratio)
            state[track_id]["alpha"] = (
                float(start["alpha"]) * (1.0 - ratio)
                + float(target["alpha"]) * ratio)
            state[track_id]["motion_velocity"] = (
                np.asarray(start["motion_velocity"], np.float32) * (1.0 - ratio)
                + np.asarray(target["motion_velocity"], np.float32) * ratio)
        return state

    def _resolve_motion_velocities(self, raw_boxes, track_ids,
                                   motion_velocities, source_dt):
        raw_boxes = np.asarray(raw_boxes, dtype=np.float32)
        explicit = motion_velocities is not None
        if explicit:
            motion_velocities = np.asarray(
                motion_velocities, dtype=np.float32).reshape(-1, 3)
        live_ids = set()
        output = []
        for index, (raw_box, track_id) in enumerate(zip(raw_boxes, track_ids)):
            track_id = int(track_id)
            live_ids.add(track_id)
            previous = self.motion_state.get(track_id)
            candidate = motion_velocities[index] if explicit else None
            if candidate is not None and np.all(np.isfinite(candidate)):
                measured = np.asarray(candidate, np.float32)
            elif previous is not None and not explicit:
                measured = (
                    raw_box[:3] - previous["raw_box"][:3]
                ) / max(float(source_dt), 1e-3)
            elif previous is not None:
                measured = previous["velocity"]
            else:
                measured = np.zeros(3, np.float32)
            speed = float(np.linalg.norm(measured[:2]))
            max_speed = float(self.cfg["motion_velocity_max_mps"])
            if speed > max_speed > 0.0:
                measured = np.asarray(measured, np.float32).copy()
                measured[:2] *= max_speed / speed
            if previous is None:
                resolved = np.asarray(measured, np.float32)
            else:
                weight = float(self.cfg["motion_velocity_smoothing_alpha"])
                resolved = (
                    (1.0 - weight) * previous["velocity"]
                    + weight * measured).astype(np.float32)
            self.motion_state[track_id] = {
                "raw_box": np.asarray(raw_box, np.float32).copy(),
                "velocity": resolved.copy(),
            }
            output.append(resolved)
        for track_id in list(self.motion_state):
            if track_id not in live_ids:
                self.motion_state.pop(track_id, None)
        return (np.stack(output) if output
                else np.empty((0, 3), dtype=np.float32))

    def _publish_interpolated(self):
        if self.visual_header is None:
            return
        state = self._current_visual_state(time.monotonic())
        ordered = sorted(state.items())
        if ordered:
            boxes = np.stack([record["box"] for _, record in ordered])
            scores = np.asarray(
                [record["score"] for _, record in ordered], np.float32)
            labels = np.asarray(
                [record["label"] for _, record in ordered], np.int64)
            track_ids = np.asarray([track_id for track_id, _ in ordered], np.int64)
            alpha_scales = np.asarray(
                [record["alpha"] for _, record in ordered], np.float32)
            motion_velocities = np.stack([
                record["motion_velocity"] for _, record in ordered])
        else:
            boxes = np.empty((0, 7), np.float32)
            scores = np.empty((0,), np.float32)
            labels = np.empty((0,), np.int64)
            track_ids = np.empty((0,), np.int64)
            alpha_scales = np.empty((0,), np.float32)
            motion_velocities = np.empty((0, 3), np.float32)
        self.mesh_publisher.publish(self.render(
            self.visual_header, boxes, scores, labels, track_ids,
            self.visual_class_names, self.visual_lifetime,
            alpha_scales=alpha_scales,
            motion_velocities=motion_velocities,
            apply_lane_follow=False))
        lane_output = self.MarkerArray()
        if bool(self.cfg["draw_road_lines"]):
            lane_output.markers.extend(self._make_lanes(
                self.visual_header, self.visual_lifetime))
        self.lane_publisher.publish(lane_output)

    def publish(self, header, raw_boxes, scores, labels, track_ids,
                class_names, lifetime, alpha_scales=None,
                motion_velocities=None):
        now = time.monotonic()
        if alpha_scales is None:
            alpha_scales = np.ones(len(raw_boxes), dtype=np.float32)
        source_dt = self.visual_interval
        if self.last_source_arrival is not None:
            measured = now - self.last_source_arrival
            minimum = float(self.cfg["interpolation_min_duration_sec"])
            maximum = float(self.cfg["interpolation_max_duration_sec"])
            measured = min(max(measured, minimum), maximum)
            weight = float(self.cfg["interpolation_interval_ema_weight"])
            self.visual_interval = (
                (1.0 - weight) * self.visual_interval + weight * measured)
            source_dt = measured
        raw_boxes = np.asarray(raw_boxes, dtype=np.float32)
        resolved_velocities = self._resolve_motion_velocities(
            raw_boxes, track_ids, motion_velocities, source_dt)
        boxes = self._apply_lane_visual_follow(raw_boxes, track_ids)
        current = self._current_visual_state(now)
        target = {}
        for box, score, label, track_id, alpha, velocity in zip(
                boxes, scores, labels, track_ids,
                alpha_scales, resolved_velocities):
            target[int(track_id)] = {
                "box": np.asarray(box, np.float32).copy(),
                "score": float(score), "label": int(label),
                "alpha": float(alpha),
                "motion_velocity": np.asarray(velocity, np.float32).copy(),
            }
        self.last_source_arrival = now
        self.visual_start = {
            track_id: current.get(track_id, record)
            for track_id, record in target.items()}
        self.visual_target = target
        self.visual_animation_started = now
        self.visual_header = header
        self.visual_class_names = class_names
        self.visual_lifetime = float(lifetime)

    def close(self):
        if hasattr(self, "visual_timer"):
            self.visual_timer.cancel()
        if hasattr(self, "asset_server"):
            self.asset_server.shutdown()
            self.asset_server.server_close()
