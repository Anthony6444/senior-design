print("SCRIPT STARTED - MOTION CAPTURE WITH LEFT/RIGHT IDENTITY TRACKING")

import sys
from pathlib import Path
import cv2
import numpy as np
import math
import csv
import json
from ultralytics import YOLO
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

try:
    from scipy.signal import savgol_filter
except ImportError:
    print("ERROR: scipy is required for smoothed velocity/acceleration.")
    print("Install it with: pip install scipy")
    sys.exit(1)


# =====================================================
# Savitzky-Golay smoothing settings
# =====================================================

SAVGOL_WINDOW_LENGTH = 9   # Must be odd. At 20 Hz, 9 frames = 0.45 seconds.
SAVGOL_POLYORDER = 3

# Approximate camera vertical field of view used for torso-width depth estimate.
# Adjust later if you calibrate the webcam.
CAMERA_VERTICAL_FOV_DEG = 60.0


class LowPassFilter:
    def __init__(self):
        self.s = None

    def apply(self, x, alpha):
        if self.s is None:
            self.s = x
            return x
        self.s = alpha * x + (1 - alpha) * self.s
        return self.s


class OneEuroFilter:
    def __init__(self, freq, min_cutoff=0.001, beta=0.01, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_filter = LowPassFilter()
        self.x_filter = LowPassFilter()

    def alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def apply(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x

        dx = (x - self.x_prev) * self.freq
        self.x_prev = x

        edx = self.dx_filter.apply(dx, self.alpha(self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(edx)

        return self.x_filter.apply(x, self.alpha(cutoff))


HEAD_WEIGHT = 0.08
SHOULDERS_WEIGHT = 0.20
HIPS_WEIGHT = 0.45
ARMS_WEIGHT = 0.10
LEGS_WEIGHT = 0.22

TOTAL_WEIGHT = max(
    HEAD_WEIGHT + SHOULDERS_WEIGHT + HIPS_WEIGHT + ARMS_WEIGHT + LEGS_WEIGHT,
    1e-6,
)


if len(sys.argv) < 3:
    print("Usage: python Mocap.py <video_path> <target_person_idx>")
    print("target_person_idx: 0 = leader/left-start person, 1 = follower/right-start person")
    sys.exit(1)

video_path = Path(sys.argv[1])

if not video_path.exists():
    print(f"Video not found: {video_path}")
    sys.exit(1)

target_person_idx = int(sys.argv[2])

if target_person_idx not in [0, 1]:
    print("target_person_idx must be 0 or 1")
    sys.exit(1)

# Optional GRF arguments passed by main.py:
#   argv[3] = "1" if GRF is selected, otherwise "0"
#   argv[4] = target body weight in lbf
grf_enabled = False
target_weight_lbf = 0.0

if len(sys.argv) >= 4:
    grf_enabled = str(sys.argv[3]).strip() == "1"

if len(sys.argv) >= 5:
    try:
        target_weight_lbf = max(float(sys.argv[4]), 0.0)
    except (TypeError, ValueError):
        target_weight_lbf = 0.0


def compute_center_of_mass(joints):
    lm = mp_pose.PoseLandmark

    try:
        head = joints[lm.NOSE]

        shoulders = np.mean(
            [joints[lm.LEFT_SHOULDER], joints[lm.RIGHT_SHOULDER]],
            axis=0,
        )

        hips = np.mean(
            [joints[lm.LEFT_HIP], joints[lm.RIGHT_HIP]],
            axis=0,
        )

        arms = np.mean(
            [
                joints[lm.LEFT_ELBOW],
                joints[lm.RIGHT_ELBOW],
                joints[lm.LEFT_WRIST],
                joints[lm.RIGHT_WRIST],
            ],
            axis=0,
        )

        legs = np.mean(
            [
                joints[lm.LEFT_KNEE],
                joints[lm.RIGHT_KNEE],
                joints[lm.LEFT_ANKLE],
                joints[lm.RIGHT_ANKLE],
            ],
            axis=0,
        )

        weighted_sum = (
            head * HEAD_WEIGHT
            + shoulders * SHOULDERS_WEIGHT
            + hips * HIPS_WEIGHT
            + arms * ARMS_WEIGHT
            + legs * LEGS_WEIGHT
        )

        return weighted_sum / TOTAL_WEIGHT

    except KeyError:
        return np.array([0.0, 0.0, 0.0])


def compute_com_confidence(landmarks):
    lm = mp_pose.PoseLandmark

    try:
        head_conf = landmarks[lm.NOSE].visibility

        shoulders_conf = np.mean(
            [
                landmarks[lm.LEFT_SHOULDER].visibility,
                landmarks[lm.RIGHT_SHOULDER].visibility,
            ]
        )

        hips_conf = np.mean(
            [
                landmarks[lm.LEFT_HIP].visibility,
                landmarks[lm.RIGHT_HIP].visibility,
            ]
        )

        arms_conf = np.mean(
            [
                landmarks[lm.LEFT_ELBOW].visibility,
                landmarks[lm.RIGHT_ELBOW].visibility,
                landmarks[lm.LEFT_WRIST].visibility,
                landmarks[lm.RIGHT_WRIST].visibility,
            ]
        )

        legs_conf = np.mean(
            [
                landmarks[lm.LEFT_KNEE].visibility,
                landmarks[lm.RIGHT_KNEE].visibility,
                landmarks[lm.LEFT_ANKLE].visibility,
                landmarks[lm.RIGHT_ANKLE].visibility,
            ]
        )

        weighted_conf = (
            head_conf * HEAD_WEIGHT
            + shoulders_conf * SHOULDERS_WEIGHT
            + hips_conf * HIPS_WEIGHT
            + arms_conf * ARMS_WEIGHT
            + legs_conf * LEGS_WEIGHT
        )

        return weighted_conf / TOTAL_WEIGHT

    except (KeyError, AttributeError):
        return 0.0


MAX_PEOPLE = 2

tracked_people = {
    0: None,
    1: None,
}


def box_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def local_to_global_pixel(local_point, box):
    x1, y1, x2, y2 = box

    global_x_px = local_point[0] * (x2 - x1) + x1
    global_y_px = local_point[1] * (y2 - y1) + y1

    return np.array([global_x_px, global_y_px, local_point[2]])


def pixel_to_global_normalized(global_point_px, frame_width, frame_height):
    return np.array(
        [
            global_point_px[0] / frame_width,
            global_point_px[1] / frame_height,
            global_point_px[2],
        ]
    )


def pixel_to_relative_feet(global_point_px, origin_px, pixels_per_foot):
    if origin_px is None or pixels_per_foot is None:
        return None

    x_ft = (global_point_px[0] - origin_px[0]) / pixels_per_foot

    # Image Y increases downward, so invert Y so upward is positive.
    y_ft = -(global_point_px[1] - origin_px[1]) / pixels_per_foot

    return np.array([x_ft, y_ft, global_point_px[2]])


def landmark_global_xy(landmarks, landmark_id, box):
    x1, y1, x2, y2 = box
    landmark = landmarks[landmark_id]

    return np.array(
        [
            landmark.x * (x2 - x1) + x1,
            landmark.y * (y2 - y1) + y1,
        ]
    )


def distance_between_landmarks(landmarks, landmark_a, landmark_b, box):
    point_a = landmark_global_xy(landmarks, landmark_a, box)
    point_b = landmark_global_xy(landmarks, landmark_b, box)

    return float(np.linalg.norm(point_a - point_b))


def compute_body_widths_px(landmarks, box):
    lm = mp_pose.PoseLandmark

    shoulder_width_px = distance_between_landmarks(
        landmarks,
        lm.LEFT_SHOULDER,
        lm.RIGHT_SHOULDER,
        box,
    )

    hip_width_px = distance_between_landmarks(
        landmarks,
        lm.LEFT_HIP,
        lm.RIGHT_HIP,
        box,
    )

    valid_widths = []

    if shoulder_width_px > 1:
        valid_widths.append(shoulder_width_px)

    if hip_width_px > 1:
        valid_widths.append(hip_width_px)

    if not valid_widths:
        torso_width_px = None
    else:
        torso_width_px = float(np.mean(valid_widths))

    return shoulder_width_px, hip_width_px, torso_width_px


def compute_shin_lengths_px(landmarks, box):
    """
    Returns apparent left/right shin lengths in pixels.

    These are used as a sanity check for the torso-width Z estimate.
    If torso width changes a lot but shin lengths do not change similarly,
    the apparent torso width change is probably caused by body rotation/posture,
    not true camera-depth movement.
    """
    lm = mp_pose.PoseLandmark

    try:
        left_shin_px = distance_between_landmarks(
            landmarks,
            lm.LEFT_KNEE,
            lm.LEFT_ANKLE,
            box,
        )

        right_shin_px = distance_between_landmarks(
            landmarks,
            lm.RIGHT_KNEE,
            lm.RIGHT_ANKLE,
            box,
        )

        valid_shins = []

        if left_shin_px > 1:
            valid_shins.append(left_shin_px)

        if right_shin_px > 1:
            valid_shins.append(right_shin_px)

        if not valid_shins:
            average_shin_px = None
        else:
            average_shin_px = float(np.mean(valid_shins))

        return left_shin_px, right_shin_px, average_shin_px

    except Exception:
        return None, None, None



def estimate_torso_z_feet(
    current_torso_width_px,
    reference_torso_width_px,
    reference_torso_width_ft,
    focal_length_px,
):
    """
    Approximate camera-depth displacement from torso-width perspective scaling.

    Positive output means the torso appears farther from the camera than
    in the reference frame. Negative output means it appears closer.

    This is still an approximation because torso rotation and pose changes
    also change apparent torso width.
    """
    reference_distance_ft = compute_reference_distance_feet(
        reference_torso_width_px,
        reference_torso_width_ft,
        focal_length_px,
    )

    if current_torso_width_px is None or reference_distance_ft is None:
        return None

    if current_torso_width_px <= 1:
        return None

    current_distance_ft = (
        focal_length_px * reference_torso_width_ft / current_torso_width_px
    )

    z_raw = current_distance_ft - reference_distance_ft
    return z_raw


def apply_z_sanity_check(
    raw_z_torso_ft,
    current_torso_width_px,
    reference_torso_width_px,
    current_avg_shin_px,
    reference_avg_shin_px,
    previous_z_torso_ft,
):
    """
    Prevents torso-width depth from exploding during turns.

    True movement toward/away from the camera should change most apparent body
    segment sizes somewhat together. If torso width ratio and shin length ratio
    disagree strongly, the torso width change is likely caused by yaw rotation
    or posture instead of true camera-depth movement.
    """
    if raw_z_torso_ft is None:
        return previous_z_torso_ft

    if (
        current_torso_width_px is None
        or reference_torso_width_px is None
        or current_avg_shin_px is None
        or reference_avg_shin_px is None
        or reference_torso_width_px <= 1
        or reference_avg_shin_px <= 1
    ):
        return raw_z_torso_ft

    if previous_z_torso_ft is None:
        return raw_z_torso_ft

    torso_ratio = current_torso_width_px / reference_torso_width_px
    shin_ratio = current_avg_shin_px / reference_avg_shin_px

    ratio_disagreement = abs(torso_ratio - shin_ratio)
    z_jump = abs(raw_z_torso_ft - previous_z_torso_ft)

    disagreement_threshold = 0.45
    max_single_frame_z_jump_ft = 4.0

    # Loosened rejection:
    #   - A shin/torso ratio disagreement alone no longer deletes the point.
    #   - Reject only large single-frame Z jumps.
    #   - This prevents most/all COM rows from disappearing during normal turns.
    if z_jump > max_single_frame_z_jump_ft:
        return None

    return raw_z_torso_ft



def compute_reference_distance_feet(
    reference_torso_width_px,
    reference_torso_width_ft,
    focal_length_px,
):
    if (
        reference_torso_width_px is None
        or reference_torso_width_ft is None
        or focal_length_px is None
    ):
        return None

    if reference_torso_width_px <= 1:
        return None

    return focal_length_px * reference_torso_width_ft / reference_torso_width_px


def perspective_correct_relative_feet(
    global_point_px,
    origin_px,
    frame_width,
    frame_height,
    focal_length_px,
    reference_distance_ft,
    z_torso_ft_approx,
):
    """
    Approximate perspective-corrected X/Y coordinates relative to the
    reference origin using the estimated torso-depth displacement.

    X_WORLD_FT_APPROX and Y_WORLD_FT_APPROX are still approximate, but they
    reduce the false X/Y motion caused by perspective when the dancer moves
    toward or away from the camera.
    """
    if (
        global_point_px is None
        or origin_px is None
        or focal_length_px is None
        or reference_distance_ft is None
        or z_torso_ft_approx is None
    ):
        return None

    current_distance_ft = reference_distance_ft + z_torso_ft_approx

    if current_distance_ft <= 0:
        return None

    cx = frame_width / 2.0
    cy = frame_height / 2.0

    # Convert image coordinates into approximate camera-plane coordinates.
    current_x_ft = (global_point_px[0] - cx) * current_distance_ft / focal_length_px
    current_y_ft = (cy - global_point_px[1]) * current_distance_ft / focal_length_px

    # The origin is the leader left heel in the reference frame, so it uses
    # the reference distance.
    origin_x_ft = (origin_px[0] - cx) * reference_distance_ft / focal_length_px
    origin_y_ft = (cy - origin_px[1]) * reference_distance_ft / focal_length_px

    return np.array(
        [
            current_x_ft - origin_x_ft,
            current_y_ft - origin_y_ft,
        ]
    )


def get_valid_savgol_window(point_count):
    if point_count < 5:
        return None

    window = min(SAVGOL_WINDOW_LENGTH, point_count)

    if window % 2 == 0:
        window -= 1

    if window <= SAVGOL_POLYORDER:
        window = SAVGOL_POLYORDER + 2
        if window % 2 == 0:
            window += 1

    if window > point_count:
        return None

    return window



def interpolate_missing_z_values(csv_rows):
    """
    Offline interpolation of rejected Z-COM frames.

    Rejected Z frames are stored as blank during frame processing. This function
    fills those blank Z_TORSO_FT_APPROX values before velocity/acceleration and
    GRF calculations.

    Important behavior:
      - X/Y rows are not deleted when Z is rejected.
      - If at least two valid Z values exist, blanks are linearly interpolated.
      - If fewer than two valid Z values exist, raw torso-width Z is used as a
        fallback so GRF does not disappear completely.
    """
    com_rows_by_person = {}

    for row in csv_rows:
        if row["TYPE"] != "COM":
            continue

        person_id = row["PERSON"]

        if person_id not in com_rows_by_person:
            com_rows_by_person[person_id] = []

        com_rows_by_person[person_id].append(row)

    for person_id, rows in com_rows_by_person.items():
        rows.sort(key=lambda r: r["FRAME"])

        z_values = []
        raw_values = []

        for row in rows:
            z = row.get("Z_TORSO_FT_APPROX", "")
            raw_z = row.get("Z_TORSO_FT_RAW", "")

            z_values.append(np.nan if z == "" else float(z))
            raw_values.append(np.nan if raw_z == "" else float(raw_z))

        z_values = np.array(z_values, dtype=float)
        raw_values = np.array(raw_values, dtype=float)

        valid = ~np.isnan(z_values)
        raw_valid = ~np.isnan(raw_values)
        indices = np.arange(len(z_values))

        if np.sum(valid) >= 2:
            z_filled = np.interp(
                indices,
                indices[valid],
                z_values[valid],
            )
        elif np.sum(raw_valid) >= 2:
            # Fallback: use raw Z if the rejection rules removed too much data.
            z_filled = np.interp(
                indices,
                indices[raw_valid],
                raw_values[raw_valid],
            )
        elif np.sum(valid) == 1:
            z_filled = np.full(len(z_values), z_values[valid][0], dtype=float)
        elif np.sum(raw_valid) == 1:
            z_filled = np.full(len(z_values), raw_values[raw_valid][0], dtype=float)
        else:
            z_filled = np.zeros(len(z_values), dtype=float)

        for i, row in enumerate(rows):
            row["Z_TORSO_FT_APPROX"] = z_filled[i]



GRAVITY_FT_S2 = 32.174


def compute_total_required_external_force_lbf(weight_lbf, ax_ft_s2, ay_ft_s2, az_ft_s2):
    """
    Estimate total non-gravity external force required at the whole-body COM.

    Coordinate convention:
      +X = camera/global horizontal right
      +Y = vertical up
      +Z = approximate depth/top-view axis

    Feet + hands must provide:
      F = [m*ax, W + m*ay, m*az]

    where m = W/g in slugs, acceleration is ft/s^2, and force is lbf.
    """
    try:
        weight_lbf = max(float(weight_lbf), 0.0)
        ax_ft_s2 = float(ax_ft_s2)
        ay_ft_s2 = float(ay_ft_s2)
        az_ft_s2 = float(az_ft_s2)
    except (TypeError, ValueError):
        return 0.0, 0.0, 0.0

    if weight_lbf <= 0.0:
        return 0.0, 0.0, 0.0

    mass_slugs = weight_lbf / GRAVITY_FT_S2

    grf_x_lbf = mass_slugs * ax_ft_s2
    grf_y_lbf = weight_lbf + (mass_slugs * ay_ft_s2)
    grf_z_lbf = mass_slugs * az_ft_s2

    return grf_x_lbf, grf_y_lbf, grf_z_lbf


def compute_total_grf_columns(csv_rows, weight_lbf, enabled):
    """
    Adds total required external force columns to COM rows.

    main.py later subtracts hand forces from this total, then solves left/right
    foot GRFs using force balance and moment balance.
    """
    for row in csv_rows:
        row["GRF_X_LBF"] = ""
        row["GRF_Y_LBF"] = ""
        row["GRF_Z_LBF"] = ""

    if not enabled or weight_lbf <= 0.0:
        return

    for row in csv_rows:
        if row.get("TYPE") != "COM":
            continue

        try:
            ax = float(row["COM_AX_FT_S2"])
            ay = float(row["COM_AY_FT_S2"])
            az = float(row["COM_AZ_REL_S2"])
        except (KeyError, TypeError, ValueError):
            continue

        grf_x, grf_y, grf_z = compute_total_required_external_force_lbf(
            weight_lbf,
            ax,
            ay,
            az,
        )

        row["GRF_X_LBF"] = grf_x
        row["GRF_Y_LBF"] = grf_y
        row["GRF_Z_LBF"] = grf_z


def compute_smoothed_com_kinematics(csv_rows, fps):
    dt = 1.0 / fps

    com_rows_by_person = {}

    for row in csv_rows:
        if row["TYPE"] != "COM":
            continue

        x_source = row.get("X_WORLD_FT_APPROX", row.get("X_REL_FT", ""))
        y_source = row.get("Y_WORLD_FT_APPROX", row.get("Y_REL_FT", ""))
        z_source = row.get("Z_TORSO_FT_APPROX", row.get("Z_REL_LOCAL", ""))

        if x_source == "" or y_source == "" or z_source == "":
            continue

        person_id = row["PERSON"]

        if person_id not in com_rows_by_person:
            com_rows_by_person[person_id] = []

        com_rows_by_person[person_id].append(row)

    for person_id, rows in com_rows_by_person.items():
        rows.sort(key=lambda r: r["FRAME"])

        point_count = len(rows)
        window = get_valid_savgol_window(point_count)

        if window is None:
            print(f"WARNING: Not enough COM points to smooth person {person_id}.")
            continue

        polyorder = min(SAVGOL_POLYORDER, window - 1)

        x = np.array(
            [
                float(row.get("X_WORLD_FT_APPROX", row["X_REL_FT"]))
                if row.get("X_WORLD_FT_APPROX", "") != ""
                else float(row["X_REL_FT"])
                for row in rows
            ],
            dtype=float,
        )

        y = np.array(
            [
                float(row.get("Y_WORLD_FT_APPROX", row["Y_REL_FT"]))
                if row.get("Y_WORLD_FT_APPROX", "") != ""
                else float(row["Y_REL_FT"])
                for row in rows
            ],
            dtype=float,
        )

        z = np.array(
            [
                float(row.get("Z_TORSO_FT_APPROX", row["Z_REL_LOCAL"]))
                if row.get("Z_TORSO_FT_APPROX", "") != ""
                else float(row["Z_REL_LOCAL"])
                for row in rows
            ],
            dtype=float,
        )

        print(
            f"Applying Savitzky-Golay smoothing to person {person_id}: "
            f"window={window}, polyorder={polyorder}"
        )

        vx = savgol_filter(x, window_length=window, polyorder=polyorder, deriv=1, delta=dt, mode="interp")
        vy = savgol_filter(y, window_length=window, polyorder=polyorder, deriv=1, delta=dt, mode="interp")
        vz = savgol_filter(z, window_length=window, polyorder=polyorder, deriv=1, delta=dt, mode="interp")

        ax = savgol_filter(x, window_length=window, polyorder=polyorder, deriv=2, delta=dt, mode="interp")
        ay = savgol_filter(y, window_length=window, polyorder=polyorder, deriv=2, delta=dt, mode="interp")
        az = savgol_filter(z, window_length=window, polyorder=polyorder, deriv=2, delta=dt, mode="interp")

        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)

        for i, row in enumerate(rows):
            row["COM_VX_FT_S"] = vx[i]
            row["COM_VY_FT_S"] = vy[i]
            row["COM_VZ_REL_S"] = vz[i]
            row["COM_AX_FT_S2"] = ax[i]
            row["COM_AY_FT_S2"] = ay[i]
            row["COM_AZ_REL_S2"] = az[i]
            row["COM_SPEED_REL_S"] = speed[i]
            row["COM_ACCEL_MAG_REL_S2"] = accel_mag[i]


def get_top_candidate_boxes(boxes, max_people=2):
    return sorted(boxes, key=box_area, reverse=True)[:max_people]


def assign_people_left_right_once(boxes):
    assigned = {}

    if not boxes:
        return assigned

    boxes = get_top_candidate_boxes(boxes, MAX_PEOPLE)
    boxes = sorted(boxes, key=lambda b: box_center(b)[0])

    if len(boxes) >= 1:
        assigned[0] = boxes[0]

    if len(boxes) >= 2:
        assigned[1] = boxes[1]

    return assigned


def box_identity_score(current_box, previous_box, size_weight=80.0):
    """
    Scores how well a current YOLO box matches a previous tracked identity.

    Lower score = better match.

    Components:
      1) Center distance in pixels.
      2) Box-area change penalty.

    The area penalty helps resist identity swaps when one dancer is much larger
    or closer to the camera than the other. It is intentionally secondary so
    normal toward/away motion does not dominate the tracking.
    """
    if current_box is None or previous_box is None:
        return float("inf")

    center_distance = float(
        np.linalg.norm(box_center(current_box) - box_center(previous_box))
    )

    current_area = max(float(box_area(current_box)), 1.0)
    previous_area = max(float(box_area(previous_box)), 1.0)

    # Symmetric area-change penalty:
    #   same size -> 0
    #   2x or 1/2x -> about 0.693 * size_weight
    area_change_penalty = abs(math.log(current_area / previous_area))

    return center_distance + (size_weight * area_change_penalty)


def assign_people_by_previous_center(boxes, previous_people):
    assigned = {}

    if not boxes:
        return assigned

    boxes = get_top_candidate_boxes(boxes, MAX_PEOPLE)

    if previous_people[0] is None and previous_people[1] is None:
        return assign_people_left_right_once(boxes)

    # If both previous identities exist and two boxes are detected, compare the
    # two complete assignment pairings:
    #
    #   Option A: leader->box0, follower->box1
    #   Option B: leader->box1, follower->box0
    #
    # This avoids the old leader-first behavior where person 0 could claim the
    # best remaining box before person 1 was considered.
    if (
        len(boxes) >= 2
        and previous_people[0] is not None
        and previous_people[1] is not None
    ):
        box_a = boxes[0]
        box_b = boxes[1]

        score_a = (
            box_identity_score(box_a, previous_people[0])
            + box_identity_score(box_b, previous_people[1])
        )

        score_b = (
            box_identity_score(box_b, previous_people[0])
            + box_identity_score(box_a, previous_people[1])
        )

        if score_a <= score_b:
            assigned[0] = box_a
            assigned[1] = box_b
        else:
            assigned[0] = box_b
            assigned[1] = box_a

        return assigned

    # Fallback for one-box detections or when only one previous identity exists:
    # use the same center+area score, but still assign conservatively from the
    # available detections.
    available_boxes = boxes.copy()

    for person_id in [0, 1]:
        previous_box = previous_people[person_id]

        if previous_box is None or not available_boxes:
            continue

        best_box = min(
            available_boxes,
            key=lambda b: box_identity_score(b, previous_box),
        )

        assigned[person_id] = best_box
        available_boxes.remove(best_box)

    for person_id in [0, 1]:
        if person_id not in assigned and available_boxes:
            remaining_sorted = sorted(available_boxes, key=lambda b: box_center(b)[0])
            assigned[person_id] = remaining_sorted[0]
            available_boxes.remove(remaining_sorted[0])

    return assigned


print("Loading YOLO...")
yolo = YOLO("yolov8n.pt")
print("YOLO loaded")
yolo.fuse()

print("Loading MediaPipe Pose...")
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
print("MediaPipe Pose loaded")

mp_draw = mp_drawing

landmark_spec = mp_draw.DrawingSpec(
    color=(0, 255, 0),
    thickness=2,
    circle_radius=2,
)

connection_spec = mp_draw.DrawingSpec(
    color=(255, 0, 0),
    thickness=2,
)


cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print(f"Could not open video: {video_path}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps > 1 else 20.0

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video opened: {video_path}")
print(f"Resolution: {w} x {h}")
print(f"FPS: {fps}")

people_filters = {}

for person_id in range(MAX_PEOPLE):
    people_filters[person_id] = {}

    for joint_id in range(33):
        people_filters[person_id][joint_id] = {
            "x": OneEuroFilter(freq=fps, min_cutoff=1.5, beta=0.05),
            "y": OneEuroFilter(freq=fps, min_cutoff=1.5, beta=0.05),
            "z": OneEuroFilter(freq=fps, min_cutoff=1.5, beta=0.05),
        }

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

calibration_path = output_dir / "calibration.json"
calibration_data = None
origin_px = None
pixels_per_foot = None

target_torso_width_px_ref = None
target_torso_width_ft_ref = None
target_shoulder_width_px_ref = None
target_hip_width_px_ref = None
target_left_shin_px_ref = None
target_right_shin_px_ref = None
target_avg_shin_px_ref = None
focal_length_px = None
reference_distance_ft = None
previous_z_torso_by_person = {}
previous_z_for_xy_by_person = {}

if calibration_path.exists():
    with open(calibration_path, "r") as f:
        calibration_data = json.load(f)

    origin_px = calibration_data.get("origin_px")
    pixels_per_foot = calibration_data.get("pixels_per_foot")

    target_torso_width_px_ref = calibration_data.get("target_torso_width_px_ref")
    target_torso_width_ft_ref = calibration_data.get("target_torso_width_ft_ref")
    target_shoulder_width_px_ref = calibration_data.get("target_shoulder_width_px_ref")
    target_hip_width_px_ref = calibration_data.get("target_hip_width_px_ref")
    target_left_shin_px_ref = calibration_data.get("target_left_shin_px_ref")
    target_right_shin_px_ref = calibration_data.get("target_right_shin_px_ref")
    target_avg_shin_px_ref = calibration_data.get("target_avg_shin_px_ref")

    calibration_frame_height = calibration_data.get("frame_height", h)
    focal_length_px = calibration_frame_height / (
        2.0 * math.tan(math.radians(CAMERA_VERTICAL_FOV_DEG) / 2.0)
    )

    reference_distance_ft = compute_reference_distance_feet(
        target_torso_width_px_ref,
        target_torso_width_ft_ref,
        focal_length_px,
    )

    print(f"Loaded calibration: {calibration_path}")
    print(f"Origin px: {origin_px}")
    print(f"Pixels per foot: {pixels_per_foot}")
    print(f"Target torso width ref px: {target_torso_width_px_ref}")
    print(f"Target torso width ref ft: {target_torso_width_ft_ref}")
    print(f"Assumed focal length px: {focal_length_px:.2f}")
    if reference_distance_ft is not None:
        print(f"Reference torso distance ft: {reference_distance_ft:.3f}")
else:
    print("WARNING: calibration.json not found. Feet coordinates will be blank.")

output_video_path = output_dir / "pose_output.mp4"
output_csv_path = output_dir / "pose_3d_output.csv"

print(f"Creating output video: {output_video_path}")
out = cv2.VideoWriter(
    str(output_video_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    float(fps),
    (w, h),
)

if not out.isOpened():
    print("Could not create output video.")
    cap.release()
    pose.close()
    sys.exit(1)

csv_fieldnames = [
    "TYPE",
    "FRAME",
    "TIME_SEC",
    "PERSON",
    "JOINT",
    "X_GLOBAL_PX",
    "Y_GLOBAL_PX",
    "Z_LOCAL",
    "X_GLOBAL_NORM",
    "Y_GLOBAL_NORM",
    "X_REL_FT",
    "Y_REL_FT",
    "Z_REL_LOCAL",
    "X_WORLD_FT_APPROX",
    "Y_WORLD_FT_APPROX",
    "SHOULDER_WIDTH_PX",
    "HIP_WIDTH_PX",
    "TORSO_WIDTH_PX",
    "LEFT_SHIN_PX",
    "RIGHT_SHIN_PX",
    "AVG_SHIN_PX",
    "Z_TORSO_FT_RAW",
    "Z_TORSO_FT_APPROX",
    "COM_VX_FT_S",
    "COM_VY_FT_S",
    "COM_VZ_REL_S",
    "COM_AX_FT_S2",
    "COM_AY_FT_S2",
    "COM_AZ_REL_S2",
    "COM_SPEED_REL_S",
    "COM_ACCEL_MAG_REL_S2",
    "GRF_X_LBF",
    "GRF_Y_LBF",
    "GRF_Z_LBF",
    "CONFIDENCE",
]

csv_rows = []

frame_index = 0
print("Starting frame loop...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    time_sec = frame_index / fps
    time_str = f"{time_sec:.3f}"

    detections = yolo(frame, classes=[0], verbose=False)[0]

    people_coords = []

    if detections.boxes is not None:
        for b in detections.boxes:
            if float(b.conf[0]) < 0.4:
                continue

            x1, y1, x2, y2 = map(int, b.xyxy[0])
            people_coords.append((x1, y1, x2, y2))

    assigned_people = assign_people_by_previous_center(
        people_coords,
        tracked_people,
    )

    for person_id, box in assigned_people.items():
        tracked_people[person_id] = box

    for person_id in sorted(assigned_people.keys()):
        x1, y1, x2, y2 = assigned_people[person_id]

        crop = frame[
            max(0, y1):min(h, y2),
            max(0, x1):min(w, x2),
        ]

        if crop.size == 0:
            continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if not res.pose_landmarks:
            continue

        joints = {}

        for joint_id, lm in enumerate(res.pose_landmarks.landmark):
            smoothed_x = people_filters[person_id][joint_id]["x"].apply(lm.x)
            smoothed_y = people_filters[person_id][joint_id]["y"].apply(lm.y)
            smoothed_z = people_filters[person_id][joint_id]["z"].apply(lm.z)

            joints[joint_id] = np.array(
                [
                    smoothed_x,
                    smoothed_y,
                    smoothed_z,
                ]
            )

            lm.x = smoothed_x
            lm.y = smoothed_y
            lm.z = smoothed_z

        com_local = compute_center_of_mass(joints)
        com_confidence = compute_com_confidence(res.pose_landmarks.landmark)

        com_global_px = local_to_global_pixel(
            com_local,
            (x1, y1, x2, y2),
        )

        com_global_norm = pixel_to_global_normalized(
            com_global_px,
            w,
            h,
        )

        com_relative_ft = pixel_to_relative_feet(
            com_global_px,
            origin_px,
            pixels_per_foot,
        )

        shoulder_width_px, hip_width_px, torso_width_px = compute_body_widths_px(
            res.pose_landmarks.landmark,
            (x1, y1, x2, y2),
        )

        left_shin_px, right_shin_px, avg_shin_px = compute_shin_lengths_px(
            res.pose_landmarks.landmark,
            (x1, y1, x2, y2),
        )

        z_torso_ft_raw = estimate_torso_z_feet(
            torso_width_px,
            target_torso_width_px_ref,
            target_torso_width_ft_ref,
            focal_length_px,
        )

        z_torso_ft_approx = apply_z_sanity_check(
            z_torso_ft_raw,
            torso_width_px,
            target_torso_width_px_ref,
            avg_shin_px,
            target_avg_shin_px_ref,
            previous_z_torso_by_person.get(person_id),
        )

        if z_torso_ft_approx is not None:
            previous_z_torso_by_person[person_id] = z_torso_ft_approx
            previous_z_for_xy_by_person[person_id] = z_torso_ft_approx

        # If Z is rejected, keep Z blank for later interpolation, but still use
        # the previous valid Z to compute X/Y. This prevents Page 4 camera-view
        # and top-view plots from disappearing due to blank X_WORLD/Y_WORLD rows.
        z_for_world_xy = z_torso_ft_approx

        if z_for_world_xy is None:
            z_for_world_xy = previous_z_for_xy_by_person.get(person_id, 0.0)

        world_xy_approx = perspective_correct_relative_feet(
            com_global_px,
            origin_px,
            w,
            h,
            focal_length_px,
            reference_distance_ft,
            z_for_world_xy,
        )

        csv_rows.append(
            {
                "TYPE": "COM",
                "FRAME": frame_index,
                "TIME_SEC": time_str,
                "PERSON": person_id,
                "JOINT": 99,
                "X_GLOBAL_PX": com_global_px[0],
                "Y_GLOBAL_PX": com_global_px[1],
                "Z_LOCAL": com_global_px[2],
                "X_GLOBAL_NORM": com_global_norm[0],
                "Y_GLOBAL_NORM": com_global_norm[1],
                "X_REL_FT": com_relative_ft[0] if com_relative_ft is not None else "",
                "Y_REL_FT": com_relative_ft[1] if com_relative_ft is not None else "",
                "Z_REL_LOCAL": com_relative_ft[2] if com_relative_ft is not None else "",
                "X_WORLD_FT_APPROX": world_xy_approx[0] if world_xy_approx is not None else "",
                "Y_WORLD_FT_APPROX": world_xy_approx[1] if world_xy_approx is not None else "",
                "SHOULDER_WIDTH_PX": shoulder_width_px,
                "HIP_WIDTH_PX": hip_width_px,
                "TORSO_WIDTH_PX": torso_width_px if torso_width_px is not None else "",
                "LEFT_SHIN_PX": left_shin_px if left_shin_px is not None else "",
                "RIGHT_SHIN_PX": right_shin_px if right_shin_px is not None else "",
                "AVG_SHIN_PX": avg_shin_px if avg_shin_px is not None else "",
                "Z_TORSO_FT_RAW": z_torso_ft_raw if z_torso_ft_raw is not None else "",
                "Z_TORSO_FT_APPROX": z_torso_ft_approx if z_torso_ft_approx is not None else "",
                "COM_VX_FT_S": "",
                "COM_VY_FT_S": "",
                "COM_VZ_REL_S": "",
                "COM_AX_FT_S2": "",
                "COM_AY_FT_S2": "",
                "COM_AZ_REL_S2": "",
                "COM_SPEED_REL_S": "",
                "COM_ACCEL_MAG_REL_S2": "",
                "GRF_X_LBF": "",
                "GRF_Y_LBF": "",
                "GRF_Z_LBF": "",
                "CONFIDENCE": com_confidence,
            }
        )

        for joint_id, lm in enumerate(res.pose_landmarks.landmark):
            conf = lm.visibility if hasattr(lm, "visibility") else 1.0

            joint_local = np.array([lm.x, lm.y, lm.z])

            joint_global_px = local_to_global_pixel(
                joint_local,
                (x1, y1, x2, y2),
            )

            joint_global_norm = pixel_to_global_normalized(
                joint_global_px,
                w,
                h,
            )

            joint_relative_ft = pixel_to_relative_feet(
                joint_global_px,
                origin_px,
                pixels_per_foot,
            )

            csv_rows.append(
                {
                    "TYPE": "JOINT",
                    "FRAME": frame_index,
                    "TIME_SEC": time_str,
                    "PERSON": person_id,
                    "JOINT": joint_id,
                    "X_GLOBAL_PX": joint_global_px[0],
                    "Y_GLOBAL_PX": joint_global_px[1],
                    "Z_LOCAL": joint_global_px[2],
                    "X_GLOBAL_NORM": joint_global_norm[0],
                    "Y_GLOBAL_NORM": joint_global_norm[1],
                    "X_REL_FT": joint_relative_ft[0] if joint_relative_ft is not None else "",
                    "Y_REL_FT": joint_relative_ft[1] if joint_relative_ft is not None else "",
                    "Z_REL_LOCAL": joint_relative_ft[2] if joint_relative_ft is not None else "",
                    "X_WORLD_FT_APPROX": "",
                    "Y_WORLD_FT_APPROX": "",
                    "SHOULDER_WIDTH_PX": "",
                    "HIP_WIDTH_PX": "",
                    "TORSO_WIDTH_PX": "",
                    "LEFT_SHIN_PX": "",
                    "RIGHT_SHIN_PX": "",
                    "AVG_SHIN_PX": "",
                    "Z_TORSO_FT_RAW": "",
                    "Z_TORSO_FT_APPROX": "",
                    "COM_VX_FT_S": "",
                    "COM_VY_FT_S": "",
                    "COM_VZ_REL_S": "",
                    "COM_AX_FT_S2": "",
                    "COM_AY_FT_S2": "",
                    "COM_AZ_REL_S2": "",
                    "COM_SPEED_REL_S": "",
                    "COM_ACCEL_MAG_REL_S2": "",
                    "GRF_X_LBF": "",
                    "GRF_Y_LBF": "",
                    "GRF_Z_LBF": "",
                    "CONFIDENCE": conf,
                }
            )

        mp_draw.draw_landmarks(
            crop,
            res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_spec,
            connection_spec,
        )

        com_x = int(com_global_px[0])
        com_y = int(com_global_px[1])

        color = (0, 0, 255) if person_id == target_person_idx else (200, 200, 200)

        cv2.circle(
            frame,
            (com_x, com_y),
            8,
            color,
            -1,
        )

        label = "Leader" if person_id == 0 else "Follower"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            frame,
            f"{label} P{person_id} | Time: {time_str}s",
            (10, 30 + (person_id * 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        frame[
            max(0, y1):min(h, y2),
            max(0, x1):min(w, x2),
        ] = crop

    out.write(frame)
    frame_index += 1

    if frame_index % 20 == 0:
        print(f"Processed {frame_index} frames...")

print("Interpolating rejected Z-COM frames...")
interpolate_missing_z_values(csv_rows)

print("Computing smoothed COM velocity and acceleration...")
compute_smoothed_com_kinematics(csv_rows, fps)

print("Computing total required external force / GRF columns...")
compute_total_grf_columns(csv_rows, target_weight_lbf, grf_enabled)

print(f"Creating CSV: {output_csv_path}")
with open(output_csv_path, "w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

cap.release()
out.release()
pose.close()

print("DONE - Processing complete.")
print(f"Output video: {output_video_path}")
print(f"Output CSV: {output_csv_path}")
