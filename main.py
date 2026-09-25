import sys
import subprocess
from pathlib import Path
import json
import csv
import threading
import time
import math
import re

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

import cv2
import numpy as np
from ultralytics import YOLO
from mediapipe.python.solutions import pose as mp_pose

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from PySide6.QtCore import QFile, QTimer, Qt
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QBrush, QPen, QTransform
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PySide6.QtWidgets import QWidget, QVBoxLayout

BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CAMERA_INDEX = 1
TARGET_PERSON_IDX = 0  # 0 = leader/left-start person, 1 = follower/right-start person

HANDHOLD_FIXED_PORT = "COM8"
OPENHAND_FIXED_PORT = "COM9"

LEFT_FOOT_FIXED_PORT = "COM6"
RIGHT_FOOT_FIXED_PORT = "COM3"


def load_ui(ui_path):
    ui_file = QFile(str(ui_path))
    if not ui_file.open(QFile.ReadOnly):
        raise FileNotFoundError(f"Could not open UI file: {ui_path}")

    loader = QUiLoader()
    widget = loader.load(ui_file)
    ui_file.close()

    if widget is None:
        raise RuntimeError(f"Could not load UI file: {ui_path}")

    return widget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dance Mechanics System")

        self.stack = QStackedWidget()
        container = QWidget()

        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.stack)

        self.setCentralWidget(container)
        self.stack.setFixedSize(1500, 750)

        self.page1 = load_ui(UI_DIR / "page1_Configuration.ui")
        self.page2 = load_ui(UI_DIR / "page2_Initialization.ui")
        self.page3 = load_ui(UI_DIR / "page3_Record.ui")
        self.page4 = load_ui(UI_DIR / "page4_Analysis.ui")

        self.stack.addWidget(self.page1)
        self.stack.addWidget(self.page2)
        self.stack.addWidget(self.page3)
        self.stack.addWidget(self.page4)

        self.mocap_selected = False
        self.hands_selected = False
        self.feet_selected = False
        self.grf_selected = False
        self.com_selected = False

        self.mocap_synced = False
        self.hand_synced = False
        self.feet_synced = False

        # Hand gauge state. Handhold is implemented first; open-hand is reserved
        # for a near-identical second device later.
        self.handhold_port = None
        self.openhand_port = None
        self.hand_ports = {}
        self.hand_ports_path = OUTPUT_DIR / "hand_ports.json"

        self.hand_logging_active = False
        self.hand_stop_event = threading.Event()
        self.hand_threads = []
        self.hand_recording_start_time = None
        self.handhold_csv_path = OUTPUT_DIR / "handheld_data.csv"
        self.openhand_csv_path = OUTPUT_DIR / "openhand_data.csv"

        self.handhold_latest = {
            "device_time": "",
            "l1": 0.0,
            "l2": 0.0,
            "l3": 0.0,
            "l4": 0.0,
            "sum": 0.0,
            "x_component": 0.0,
            "y_component": 0.0,
            "z_component": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "resultant": 0.0,
            "signed_resultant": 0.0,
            "vertical_component": 0.0,
            "lateral_component": 0.0,
        }

        self.openhand_latest = {
            "device_time": "",
            "l1": 0.0,
            "l2": 0.0,
            "l3": 0.0,
            "l4": 0.0,
            "sum": 0.0,
            "x_component": 0.0,
            "y_component": 0.0,
            "z_component": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "resultant": 0.0,
            "signed_resultant": 0.0,
            "vertical_component": 0.0,
            "lateral_component": 0.0,
        }

        self.handhold_live_serial = None
        self.openhand_live_serial = None
        self.hand_live_timer = QTimer(self)
        self.hand_live_timer.timeout.connect(self.update_page2_hand_live_values)

        self.hand_record_live_timer = QTimer(self)
        self.hand_record_live_timer.timeout.connect(self.update_page3_hand_live_values)

        self.target_person_idx = 0  # 0 = leader, 1 = follower

        self.countdown_value = 5
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)

        self.reference_countdown_value = 5
        self.reference_countdown_timer = QTimer(self)
        self.reference_countdown_timer.timeout.connect(self.update_reference_countdown)

        self.video_output_path = OUTPUT_DIR / "mocap_raw_capture.mp4"
        self.analysis_video_path = OUTPUT_DIR / "mocap_analysis_trimmed.mp4"
        self.processed_video_path = OUTPUT_DIR / "pose_output.mp4"
        self.processed_csv_path = OUTPUT_DIR / "pose_3d_output.csv"
        self.playback_video_path = self.video_output_path

        self.reference_frame_path = OUTPUT_DIR / "reference_frame.jpg"
        self.calibration_path = OUTPUT_DIR / "calibration.json"
        self.current_preview_frame = None

        self.preview_camera = None
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.update_page2_preview)

        self.record_camera = None
        self.video_writer = None
        self.record_timer = QTimer(self)
        self.record_timer.timeout.connect(self.update_recording_frame)

        self.left_foot_port = None
        self.right_foot_port = None
        self.foot_ports_path = OUTPUT_DIR / "foot_ports.json"
        self.foot_logging_active = False
        self.foot_stop_event = threading.Event()
        self.foot_threads = []

        # Shared time origin for all devices in one trial.
        # Foot and hand CSV PC_TIME_SEC values use this same clock so Page 4
        # can align all devices with the trimmed mocap video.
        self.recording_session_start_time = None

        self.foot_recording_start_time = None
        self.left_foot_csv_path = OUTPUT_DIR / "left_foot_data.csv"
        self.right_foot_csv_path = OUTPUT_DIR / "right_foot_data.csv"

        self.playback_camera = None
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.update_playback_frame)
        self.playback_total_frames = 0
        self.playback_user_dragging = False
        self.playback_playing = False
        self.playback_fps = 20.0
        self.playback_interval_ms = 50
        self.analysis_time_offset_sec = 0.0

        # Hand gauges currently show a consistent fixed delay relative to mocap.
        # Negative shifts hand-gauge data earlier on Page 4.
        self.hand_manual_sync_offset_sec = -2.00
        self.hand_manual_time_scale = 0.75

        # COP currently shows a consistent fixed delay relative to mocap.
        # Negative shifts COP data earlier on Page 4.
        self.cop_manual_sync_offset_sec = -2.00
        self.cop_manual_time_scale = 0.75

        # Page 2 live foot sensor visualization
        self.left_foot_outline_path = UI_DIR / "images" / "footOutlinLeft.jpg"
        self.right_foot_outline_path = UI_DIR / "images" / "footOutlinRight.jpg"

        self.left_foot_outline_pixmap = QPixmap(str(self.left_foot_outline_path))
        self.right_foot_outline_pixmap = QPixmap(str(self.right_foot_outline_path))

        self.left_foot_live_serial = None
        self.right_foot_live_serial = None

        self.left_foot_latest_fsr = [0.0, 0.0, 0.0, 0.0]
        self.right_foot_latest_fsr = [0.0, 0.0, 0.0, 0.0]

        # Raw untared FSR values are kept so tare can be captured while foot is elevated.
        self.left_foot_latest_raw_fsr = [0.0, 0.0, 0.0, 0.0]
        self.right_foot_latest_raw_fsr = [0.0, 0.0, 0.0, 0.0]

        # Tare baselines subtracted from live display, recording CSV, and COP.
        self.left_foot_tare = [0.0, 0.0, 0.0, 0.0]
        self.right_foot_tare = [0.0, 0.0, 0.0, 0.0]

        self.foot_live_timer = QTimer(self)
        self.foot_live_timer.timeout.connect(self.update_page2_foot_live_views)

        # Page 3 live foot visualization during recording.
        # This uses the latest values read by the foot logging threads,
        # so it does not open duplicate serial connections.
        self.page3_foot_live_timer = QTimer(self)
        self.page3_foot_live_timer.timeout.connect(self.update_page3_foot_live_views)

        # COM plot data loaded from outputs/pose_3d_output.csv
        self.com_frames = []
        self.com_x = []
        self.com_y = []
        self.com_z = []
        self.camera_plot_limits = None
        self.top_plot_limits = None

        # Pre-rendered COM plot playback frames
        self.top_view_plot_frames = {}
        self.camera_view_plot_frames = {}
        self.plot_available_frames = []

        # Page 4 COP playback data
        self.left_cop_samples = []
        self.right_cop_samples = []
        self.cop_playback_total_frames = 0

        # Pre-rendered Page 4 COP foot animation frames
        self.left_cop_plot_frames = {}
        self.right_cop_plot_frames = {}
        self.foot_landmarks_by_frame = {}
        self.hand_landmarks_by_frame = {}

        # Page 4 hand force plot data
        self.handhold_samples = []
        self.openhand_samples = []
        self.page4_current_time_sec = 0.0

        # Static Page 4 hand plot pixmaps.
        # Playback draws only the moving cursor line on top, which is much faster
        # than re-rendering matplotlib for every frame.
        self.hand_static_plot_pixmaps = {}

        # Page 4 GRF placeholder values.
        # For now these display the total-body GRF in the left-foot GRF LCDs
        # after Mocap.py analysis. Right-foot GRF LCDs remain zero placeholders.
        self.latest_total_grf = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }
        self.grf_by_frame = {}
        self.grf_available_frames = []

        # Mocap foot landmarks loaded from pose_3d_output.csv.
        # Used to map COP from heel toward foot-index/toe direction for
        # per-foot GRF moment arms.
        self.foot_landmarks_by_frame = {}

        # Selected upper-limb landmarks for external hand-force moment arms.
        # Stored as: hand_landmarks_by_frame[frame][person_id][joint_id] = [x_ft, y_ft]
        self.hand_landmarks_by_frame = {}

        self.load_page1_logo()
        self.initialize_status_indicators()
        self.setup_initial_button_states()
        self.set_recording_indicator(False)
        self.update_page1_dependent_options()
        self.update_page1_start_button_state()
        self.style_play_pause_button()
        self.clear_com_plots()
        self.clear_page2_foot_live_views()
        self.clear_page3_foot_live_views()

        self.page1.MoCapCheckBox.toggled.connect(self.update_page1_dependent_options)
        self.page1.MoCapCheckBox.toggled.connect(self.update_page1_start_button_state)
        self.page1.HandGaugeCheckBox.toggled.connect(self.update_page1_start_button_state)
        self.page1.COPCheckBox.toggled.connect(self.update_page1_start_button_state)
        self.page1.GRFCheckBox.toggled.connect(self.update_page1_start_button_state)
        self.page1.COMTrackCheckBox.toggled.connect(self.update_page1_start_button_state)
        self.page1.nextButton.clicked.connect(self.go_to_initialization)

        self.page2.Back.clicked.connect(self.go_to_configuration)
        self.page2.Record.clicked.connect(self.go_to_recording)
        self.page2.CaptureReference.clicked.connect(self.start_reference_countdown)
        self.page2.MocapSync.clicked.connect(self.perform_mocap_calibration)
        self.page2.AnalyzeL.toggled.connect(self.handle_leader_analyze_toggled)
        self.page2.AnalyzeF.toggled.connect(self.handle_follower_analyze_toggled)
        self.page2.HandSync.clicked.connect(self.sync_hands)
        self.page2.FeetSync.clicked.connect(self.sync_feet)

        # Optional Page 2 handhold tare button. Recommended object names:
        # HHTare, HandholdTare, or HandTare
        for hh_tare_name in ["HHTare", "HandholdTare", "HandTare"]:
            if hasattr(self.page2, hh_tare_name):
                getattr(self.page2, hh_tare_name).clicked.connect(self.tare_handhold)
                break

        # Optional Page 2 open-hand tare button. Recommended object names:
        # OHTare, OpenHandTare, OpenhandTare, or OpenTare
        for oh_tare_name in ["OHTare", "OpenHandTare", "OpenhandTare", "OpenTare"]:
            if hasattr(self.page2, oh_tare_name):
                getattr(self.page2, oh_tare_name).clicked.connect(self.tare_openhand)
                break

        # Optional Page 2 tare buttons. Recommended object names:
        # LeftFootTare and RightFootTare
        if hasattr(self.page2, "LeftFootTare"):
            self.page2.LeftFootTare.clicked.connect(self.tare_left_foot)

        if hasattr(self.page2, "RightFootTare"):
            self.page2.RightFootTare.clicked.connect(self.tare_right_foot)

        # Shoe size is required before recording feet data.
        # Feet Sync remains available, but Record updates when ShoeSizeInput is edited.
        if hasattr(self.page2, "ShoeSizeInput"):
            if hasattr(self.page2.ShoeSizeInput, "textChanged"):
                self.page2.ShoeSizeInput.textChanged.connect(self.update_record_button_state)

            if hasattr(self.page2.ShoeSizeInput, "editingFinished"):
                self.page2.ShoeSizeInput.editingFinished.connect(self.update_record_button_state)

            if hasattr(self.page2.ShoeSizeInput, "valueChanged"):
                self.page2.ShoeSizeInput.valueChanged.connect(self.update_record_button_state)

        self.page3.Analyze.clicked.connect(self.go_to_analysis)
        self.page3.Back.clicked.connect(self.go_to_initialization)
        self.page3.Stop.clicked.connect(self.stop_recording)

        self.page4.NewTrial.clicked.connect(self.go_to_initialization)
        self.page4.ReconfigureSys.clicked.connect(self.go_to_configuration)
        self.page4.PlayPause.clicked.connect(self.toggle_playback)

        self.page4.horizontalSlider.sliderPressed.connect(self.playback_slider_pressed)
        self.page4.horizontalSlider.sliderReleased.connect(self.playback_slider_released)
        self.page4.horizontalSlider.valueChanged.connect(self.playback_slider_changed)

        self.stack.setCurrentIndex(0)
        self.showMaximized()

    def style_play_pause_button(self):
        self.page4.PlayPause.setText("▶")
        self.page4.PlayPause.setStyleSheet(
            """
            QPushButton {
                background-color: #9A9A9A;
                color: black;
                border-radius: 8px;
                font-size: 24px;
                font-weight: bold;
                padding: 4px;
            }

            QPushButton:hover {
                background-color: #B0B0B0;
            }

            QPushButton:pressed {
                background-color: #7A7A7A;
            }
            """
        )

    def cv_frame_to_pixmap(self, frame, target_label):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width

        q_image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(q_image)
        return pixmap.scaled(
            target_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

    def matplotlib_figure_to_pixmap(self, figure):
        canvas = FigureCanvasAgg(figure)
        canvas.draw()

        width, height = canvas.get_width_height()
        buffer = np.asarray(canvas.buffer_rgba())

        q_image = QImage(
            buffer.data,
            width,
            height,
            QImage.Format_RGBA8888,
        )

        return QPixmap.fromImage(q_image.copy())

    def draw_axis_indicator(self, frame):
        h, w = frame.shape[:2]

        origin = (55, h - 45)

        x_end = (105, h - 45)
        y_end = (55, h - 95)
        z_end = (25, h - 20)

        # X axis: right
        cv2.arrowedLine(frame, origin, x_end, (255, 255, 255), 2, tipLength=0.25)
        cv2.putText(frame, "X", (110, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Y axis: up
        cv2.arrowedLine(frame, origin, y_end, (255, 255, 255), 2, tipLength=0.25)
        cv2.putText(frame, "Y", (45, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Z axis: approximate depth
        cv2.arrowedLine(frame, origin, z_end, (255, 255, 255), 2, tipLength=0.25)
        cv2.putText(frame, "Z", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame


    def set_foot_sync_status(self, message):
        print(message)

        # Optional QLabel/QTextEdit/QPlainTextEdit support.
        # Recommended Page 2 object name: FootSyncStatus
        for name in ["FootSyncStatus", "FeetSyncStatus", "FootStatusText", "FeetStatusText"]:
            if hasattr(self.page2, name):
                widget = getattr(self.page2, name)

                widget.setStyleSheet("border: none; background: transparent; font-style: italic; color: rgb(220, 220, 220);")

                if hasattr(widget, "append"):
                    widget.append(message)
                elif hasattr(widget, "setPlainText"):
                    current = widget.toPlainText()
                    widget.setPlainText((current + "\n" + message).strip())
                elif hasattr(widget, "setText"):
                    current = widget.text()
                    if current:
                        widget.setText(current + "\n" + message)
                    else:
                        widget.setText(message)

                QApplication.processEvents()
                return

        QApplication.processEvents()

    def clear_foot_sync_status(self):
        for name in ["FootSyncStatus", "FeetSyncStatus", "FootStatusText", "FeetStatusText"]:
            if hasattr(self.page2, name):
                widget = getattr(self.page2, name)

                if hasattr(widget, "clear"):
                    widget.clear()
                elif hasattr(widget, "setPlainText"):
                    widget.setPlainText("")
                elif hasattr(widget, "setText"):
                    widget.setText("")

                QApplication.processEvents()
                return

    def get_shoe_size_value(self, edit):
        if edit is None:
            return None

        # Supports QSpinBox / QDoubleSpinBox.
        if hasattr(edit, "value"):
            try:
                value = float(edit.value())
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass

        # Supports QLineEdit.
        if hasattr(edit, "text"):
            text = edit.text().strip().replace(",", ".")

            if text == "":
                return None

            try:
                value = float(text)
            except ValueError:
                return None

            if value > 0:
                return value

        return None

    def has_valid_shoe_size(self):
        if hasattr(self.page2, "ShoeSizeInput"):
            return self.get_shoe_size_value(self.page2.ShoeSizeInput) is not None

        return False

    def update_feet_sync_button_state(self):
        if not hasattr(self.page2, "FeetSync"):
            return

        feet_group_enabled = True

        if hasattr(self.page2, "FeetGroupBox"):
            feet_group_enabled = self.page2.FeetGroupBox.isEnabled()

        can_sync_feet = feet_group_enabled and self.has_valid_shoe_size()
        self.page2.FeetSync.setEnabled(can_sync_feet)

        if not can_sync_feet:
            self.set_indicator_pair(
                self.page2.LeftFootRed,
                self.page2.LeftFootGreen,
                False,
            )
            self.set_indicator_pair(
                self.page2.RightFootRed,
                self.page2.RightFootGreen,
                False,
            )

    def apply_foot_tare(self, fsr_values, side):
        if fsr_values is None or len(fsr_values) != 4:
            return [0.0, 0.0, 0.0, 0.0]

        tare = self.left_foot_tare if side == "left" else self.right_foot_tare

        corrected = []
        for value, baseline in zip(fsr_values, tare):
            try:
                corrected_value = float(value) - float(baseline)
            except (TypeError, ValueError):
                corrected_value = 0.0

            corrected.append(max(corrected_value, 0.0))

        return corrected

    def tare_left_foot(self):
        self.left_foot_tare = [float(v) for v in self.left_foot_latest_raw_fsr]
        self.left_foot_latest_fsr = self.apply_foot_tare(
            self.left_foot_latest_raw_fsr,
            "left",
        )
        self.update_page2_foot_live_views()

    def tare_right_foot(self):
        self.right_foot_tare = [float(v) for v in self.right_foot_latest_raw_fsr]
        self.right_foot_latest_fsr = self.apply_foot_tare(
            self.right_foot_latest_raw_fsr,
            "right",
        )
        self.update_page2_foot_live_views()

    def clear_page2_foot_live_views(self):
        if hasattr(self.page2, "LeftFootLiveView"):
            self.page2.LeftFootLiveView.setScaledContents(False)
            self.page2.LeftFootLiveView.setText("Left foot live view")
            self.page2.LeftFootLiveView.setPixmap(QPixmap())

        if hasattr(self.page2, "RightFootLiveView"):
            self.page2.RightFootLiveView.setScaledContents(False)
            self.page2.RightFootLiveView.setText("Right foot live view")
            self.page2.RightFootLiveView.setPixmap(QPixmap())

    def parse_foot_sensor_values(self, line):
        line = str(line).strip()

        if not line:
            return None

        if "WAITING" in line or "SYNC" in line or "ACK" in line or "Time" in line:
            return None

        # Device format:
        # HH:MM:SS.mmm, FSR1, FSR2, FSR3, FSR4
        parts = [p.strip() for p in line.replace("\t", ",").split(",")]

        if len(parts) < 5:
            return None

        fsr_parts = parts[-4:]

        try:
            values = [float(v) for v in fsr_parts]
        except ValueError:
            return None

        if len(values) != 4:
            return None

        return values

    def get_dot_color_for_fsr_value(self, value):
        # Observed values can exceed 1023, so use a higher practical range.
        # The dot starts lighting around 30 and saturates around 1800.
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0

        min_active = 30.0
        max_active = 1800.0

        if value < min_active:
            return QColor(80, 80, 80)

        ratio = (value - min_active) / (max_active - min_active)
        ratio = max(0.0, min(ratio, 1.0))

        red = 255
        green = int(60 + 195 * ratio)
        blue = int(40 * (1.0 - ratio))

        return QColor(red, green, blue)

    def draw_foot_fsr_overlay(self, base_pixmap, fsr_values, side, target_label):
        label_width = max(target_label.width(), 200)
        label_height = max(target_label.height(), 200)

        canvas = QPixmap(label_width, label_height)
        canvas.fill(QColor("black"))

        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)

        # Keep outline comfortably inside label borders.
        outline_scale = 1.00
        max_outline_width = int(label_width * outline_scale)
        max_outline_height = int(label_height * outline_scale)

        if not base_pixmap.isNull():
            scaled_foot = base_pixmap.scaled(
                max_outline_width,
                max_outline_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

            # If the right-foot image appears as a left foot, mirror it in code.
            if side == "right":
                scaled_foot = scaled_foot.transformed(
                    QTransform().scale(-1, 1)
                )

            x_offset = int((label_width - scaled_foot.width()) / 2)
            y_offset = int((label_height - scaled_foot.height()) / 2)

            painter.drawPixmap(x_offset, y_offset, scaled_foot)
            draw_width = scaled_foot.width()
            draw_height = scaled_foot.height()
        else:
            x_offset = int(label_width * (1.0 - outline_scale) / 2)
            y_offset = int(label_height * (1.0 - outline_scale) / 2)
            draw_width = int(label_width * outline_scale)
            draw_height = int(label_height * outline_scale)

        # Approximate normalized FSR locations on the displayed foot image.
        #
        # Ball-of-foot dots moved inward slightly so they are not too wide.
        #
        # Left mapping:
        # FSR1 heel, FSR2 outside ball, FSR3 inside ball, FSR4 toe
        #
        # Right mapping:
        # FSR1 heel, FSR2 outside ball, FSR3 toe, FSR4 inside ball
        if side == "left":
            fsr_locations = {
                0: (0.50, 0.82),  # heel
                1: (0.38, 0.39),  # outside ball
                2: (0.62, 0.39),  # inside ball
                3: (0.52, 0.15),  # toe
            }
        else:
            fsr_locations = {
                0: (0.50, 0.82),  # heel
                1: (0.62, 0.39),  # outside ball
                2: (0.52, 0.15),  # toe
                3: (0.38, 0.39),  # inside ball
            }

        radius = max(9, int(min(draw_width, draw_height) * 0.04))

        if fsr_values is None or len(fsr_values) != 4:
            fsr_values = [0.0, 0.0, 0.0, 0.0]

        for index, value in enumerate(fsr_values):
            x_norm, y_norm = fsr_locations[index]

            x = int(x_offset + x_norm * draw_width)
            y = int(y_offset + y_norm * draw_height)

            color = self.get_dot_color_for_fsr_value(value)

            painter.setPen(QPen(QColor("white"), 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)

            painter.setPen(QPen(QColor("white"), 2))
            painter.drawText(x + radius + 3, y + 5, f"{index + 1}")

        painter.end()
        return canvas

    def read_latest_fsr_from_serial(self, ser_obj, previous_values, side_name):
        latest_values = previous_values

        if ser_obj is None:
            return latest_values

        side = "left" if side_name.lower().startswith("left") else "right"

        try:
            for _ in range(5):
                raw = ser_obj.readline()

                if not raw:
                    continue

                line = raw.decode(errors="ignore").strip()
                values = self.parse_foot_sensor_values(line)

                if values is None:
                    continue

                try:
                    raw_values = [float(v) for v in values[:4]]
                except (TypeError, ValueError):
                    continue

                if len(raw_values) != 4:
                    continue

                if side == "left":
                    self.left_foot_latest_raw_fsr = raw_values
                else:
                    self.right_foot_latest_raw_fsr = raw_values

                corrected_values = self.apply_foot_tare(raw_values, side)
                latest_values = corrected_values

                if max(corrected_values) > 30:
                    print(f"{side_name} FSR live: {corrected_values}")

        except Exception as exc:
            print(f"WARNING: {side_name} foot live read failed: {exc}")

        return latest_values

    def update_page2_foot_live_views(self):
        self.left_foot_latest_fsr = self.read_latest_fsr_from_serial(
            self.left_foot_live_serial,
            self.left_foot_latest_fsr,
            "Left",
        )

        self.right_foot_latest_fsr = self.read_latest_fsr_from_serial(
            self.right_foot_live_serial,
            self.right_foot_latest_fsr,
            "Right",
        )

        if hasattr(self.page2, "LeftFootLiveView"):
            pixmap = self.draw_foot_fsr_overlay(
                self.left_foot_outline_pixmap,
                self.left_foot_latest_fsr,
                "left",
                self.page2.LeftFootLiveView,
            )
            self.page2.LeftFootLiveView.setScaledContents(False)
            self.page2.LeftFootLiveView.setText("")
            self.page2.LeftFootLiveView.setPixmap(pixmap)
            self.page2.LeftFootLiveView.repaint()

        if hasattr(self.page2, "RightFootLiveView"):
            pixmap = self.draw_foot_fsr_overlay(
                self.right_foot_outline_pixmap,
                self.right_foot_latest_fsr,
                "right",
                self.page2.RightFootLiveView,
            )
            self.page2.RightFootLiveView.setScaledContents(False)
            self.page2.RightFootLiveView.setText("")
            self.page2.RightFootLiveView.setPixmap(pixmap)
            self.page2.RightFootLiveView.repaint()

    def clear_page3_foot_live_views(self):
        if hasattr(self.page3, "LeftFootLiveView"):
            self.page3.LeftFootLiveView.setScaledContents(False)
            self.page3.LeftFootLiveView.setText("Left foot live view")
            self.page3.LeftFootLiveView.setPixmap(QPixmap())

        if hasattr(self.page3, "RightFootLiveView"):
            self.page3.RightFootLiveView.setScaledContents(False)
            self.page3.RightFootLiveView.setText("Right foot live view")
            self.page3.RightFootLiveView.setPixmap(QPixmap())

    def update_page3_foot_live_views(self):
        if not self.feet_selected:
            self.clear_page3_foot_live_views()
            return

        if hasattr(self.page3, "LeftFootLiveView"):
            pixmap = self.draw_foot_fsr_overlay(
                self.left_foot_outline_pixmap,
                self.left_foot_latest_fsr,
                "left",
                self.page3.LeftFootLiveView,
            )
            self.page3.LeftFootLiveView.setScaledContents(False)
            self.page3.LeftFootLiveView.setText("")
            self.page3.LeftFootLiveView.setPixmap(pixmap)
            self.page3.LeftFootLiveView.repaint()

        if hasattr(self.page3, "RightFootLiveView"):
            pixmap = self.draw_foot_fsr_overlay(
                self.right_foot_outline_pixmap,
                self.right_foot_latest_fsr,
                "right",
                self.page3.RightFootLiveView,
            )
            self.page3.RightFootLiveView.setScaledContents(False)
            self.page3.RightFootLiveView.setText("")
            self.page3.RightFootLiveView.setPixmap(pixmap)
            self.page3.RightFootLiveView.repaint()

    def start_page3_foot_live_preview(self):
        if not self.feet_selected:
            self.clear_page3_foot_live_views()
            return

        self.update_page3_foot_live_views()
        self.page3_foot_live_timer.start(50)

    def stop_page3_foot_live_preview(self):
        self.page3_foot_live_timer.stop()

    def stop_page2_foot_live_preview(self):
        self.foot_live_timer.stop()

        for ser_obj in [self.left_foot_live_serial, self.right_foot_live_serial]:
            if ser_obj is not None:
                try:
                    ser_obj.close()
                except Exception:
                    pass

        self.left_foot_live_serial = None
        self.right_foot_live_serial = None

    def start_page2_foot_live_preview(self):
        if serial is None:
            print("ERROR: pyserial is not installed. Run: pip install pyserial")
            return

        self.stop_page2_foot_live_preview()

        self.left_foot_latest_fsr = [0.0, 0.0, 0.0, 0.0]
        self.right_foot_latest_fsr = [0.0, 0.0, 0.0, 0.0]
        self.left_foot_latest_raw_fsr = [0.0, 0.0, 0.0, 0.0]
        self.right_foot_latest_raw_fsr = [0.0, 0.0, 0.0, 0.0]

        # These ports are saved by sync_feet().
        if hasattr(self, "foot_ports"):
            left_port = self.foot_ports.get("left")
            right_port = self.foot_ports.get("right")
        else:
            left_port = None
            right_port = None

        try:
            if left_port:
                self.left_foot_live_serial = serial.Serial(left_port, 115200, timeout=0.05)
                self.left_foot_live_serial.write(f"SYNC:{time.time()}\n".encode())
                print(f"Started Page 2 left foot live view on {left_port}")

            if right_port:
                self.right_foot_live_serial = serial.Serial(right_port, 115200, timeout=0.05)
                self.right_foot_live_serial.write(f"SYNC:{time.time()}\n".encode())
                print(f"Started Page 2 right foot live view on {right_port}")

        except Exception as exc:
            print(f"ERROR: Could not start Page 2 foot live preview: {exc}")
            self.stop_page2_foot_live_preview()
            return

        self.update_page2_foot_live_views()
        self.foot_live_timer.start(50)

    def load_page1_logo(self):
        logo_path = UI_DIR / "images" / "SCS.png"

        if not logo_path.exists():
            print(f"WARNING: Logo file not found: {logo_path}")
            return

        pixmap = QPixmap(str(logo_path))
        if pixmap.isNull():
            print(f"WARNING: Logo could not be loaded: {logo_path}")
            return

        self.page1.SCSlogo.setPixmap(pixmap)
        self.page1.SCSlogo.setScaledContents(True)

    def setup_initial_button_states(self):
        self.page2.Record.setEnabled(False)

        self.page3.Analyze.setEnabled(False)
        self.page3.Back.setEnabled(False)
        self.page3.Stop.setEnabled(False)
        self.page3.Countdown.setText("")

        self.style_page3_action_buttons(stopped=False)

    def update_page1_start_button_state(self):
        any_selected = (
            self.page1.MoCapCheckBox.isChecked()
            or self.page1.HandGaugeCheckBox.isChecked()
            or self.page1.COPCheckBox.isChecked()
            or self.page1.GRFCheckBox.isChecked()
            or self.page1.COMTrackCheckBox.isChecked()
        )

        self.page1.nextButton.setEnabled(any_selected)

    def update_page1_dependent_options(self):
        mocap_on = self.page1.MoCapCheckBox.isChecked()

        self.page1.GRFCheckBox.setEnabled(mocap_on)
        self.page1.COMTrackCheckBox.setEnabled(mocap_on)

        if not mocap_on:
            self.page1.GRFCheckBox.setChecked(False)
            self.page1.COMTrackCheckBox.setChecked(False)

            disabled_style = """
            QCheckBox {
                color: #777777;
                font-size: 16px;
                font-weight: bold;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #777777;
                background-color: #333333;
            }
            """
            self.page1.GRFCheckBox.setStyleSheet(disabled_style)
            self.page1.COMTrackCheckBox.setStyleSheet(disabled_style)
        else:
            enabled_style = """
            QCheckBox {
                color: white;
                font-size: 16px;
                font-weight: bold;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid white;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #00aa00;
                border: 2px solid white;
            }
            """
            self.page1.GRFCheckBox.setStyleSheet(enabled_style)
            self.page1.COMTrackCheckBox.setStyleSheet(enabled_style)

    def style_page3_action_buttons(self, stopped):
        if stopped:
            analyze_style = """
            QPushButton {
                background-color: #1E88E5;
                color: white;
                border-radius: 8px;
                padding: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            """

            back_style = """
            QPushButton {
                background-color: #666666;
                color: white;
                border-radius: 8px;
                padding: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #777777;
            }
            """
        else:
            analyze_style = """
            QPushButton {
                background-color: #333333;
                color: #777777;
                border-radius: 8px;
                padding: 6px;
                font-weight: bold;
            }
            """

            back_style = """
            QPushButton {
                background-color: #222222;
                color: #666666;
                border-radius: 8px;
                padding: 6px;
                font-weight: bold;
            }
            """

        self.page3.Analyze.setStyleSheet(analyze_style)
        self.page3.Back.setStyleSheet(back_style)

    def read_page1_selections(self):
        self.mocap_selected = self.page1.MoCapCheckBox.isChecked()
        self.hands_selected = self.page1.HandGaugeCheckBox.isChecked()
        self.feet_selected = self.page1.COPCheckBox.isChecked()

        self.grf_selected = self.page1.GRFCheckBox.isChecked() and self.mocap_selected
        self.com_selected = self.page1.COMTrackCheckBox.isChecked() and self.mocap_selected

    def apply_system_configuration_to_page2(self):
        self.read_page1_selections()

        self.set_section_enabled(self.page2.MoCapSetupGroupBox, self.mocap_selected, show_border=True)
        self.set_section_enabled(self.page2.HandGroupBox, self.hands_selected, show_border=True)
        self.set_section_enabled(self.page2.FeetGroupBox, self.feet_selected, show_border=True)

    def apply_system_configuration_to_page4(self):
        self.read_page1_selections()

        self.set_section_enabled(self.page4.MoCapSetupGroupBox, self.mocap_selected or self.feet_selected)
        self.set_section_enabled(self.page4.HandGroupBox, self.hands_selected)
        self.set_section_enabled(self.page4.COMGroupBox, self.com_selected)

        self.set_section_enabled(self.page4.FeetGroupBox, self.feet_selected)

        self.set_tab_enabled(self.page4.Feettabs, self.page4.COPtab, self.feet_selected)
        self.set_tab_enabled(
            self.page4.Feettabs,
            self.page4.GRFtab,
            self.feet_selected and self.grf_selected,
        )

    def set_section_enabled(self, groupbox, enabled, show_border=False):
        groupbox.setEnabled(enabled)

        if show_border:
            enabled_group_style = """
                QGroupBox {
                    color: white;
                    border: 2px solid white;
                    border-radius: 8px;
                    margin-top: 12px;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
            """

            disabled_group_style = """
                QGroupBox {
                    color: #777777;
                    border: 2px solid #777777;
                    border-radius: 8px;
                    margin-top: 12px;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
            """
        else:
            enabled_group_style = "QGroupBox { color: white; }"
            disabled_group_style = "QGroupBox { color: #777777; }"

        if enabled:
            groupbox.setStyleSheet(
                enabled_group_style
                + """
                QLabel { color: white; }
                QPushButton { color: white; }
                QLineEdit { color: white; }
                QCheckBox { color: white; }
                QTabWidget { color: white; }
                """
            )
        else:
            groupbox.setStyleSheet(
                disabled_group_style
                + """
                QLabel { color: #777777; }
                QPushButton {
                    color: #777777;
                    background-color: #333333;
                }
                QLineEdit {
                    color: #777777;
                    background-color: #333333;
                }
                QCheckBox { color: #777777; }
                QTabWidget { color: #777777; }
                """
            )

    def set_tab_enabled(self, tab_widget, tab_page, enabled):
        index = tab_widget.indexOf(tab_page)
        if index == -1:
            print("WARNING: Could not find tab page in tab widget.")
            return
        tab_widget.setTabEnabled(index, enabled)

    def initialize_status_indicators(self):
        self.set_indicator_pair(self.page2.MoCapRed, self.page2.MoCapGreen, False)

        self.set_indicator_pair(self.page2.HandholdRed, self.page2.HandholdGreen, False)
        self.set_indicator_pair(self.page2.OpenHandRed, self.page2.OpenHandGreen, False)

        self.set_indicator_pair(self.page2.LeftFootRed, self.page2.LeftFootGreen, False)
        self.set_indicator_pair(self.page2.RightFootRed, self.page2.RightFootGreen, False)

    def set_indicator_pair(self, red_label, green_label, active):
        if active:
            red_color = "#444444"
            green_color = "green"
        else:
            red_color = "red"
            green_color = "#444444"

        red_label.setStyleSheet(
            f"""
            background-color: {red_color};
            border-radius: 10px;
            """
        )

        green_label.setStyleSheet(
            f"""
            background-color: {green_color};
            border-radius: 10px;
            """
        )

    def set_recording_indicator(self, active):
        if active:
            dot_color = "red"
            text_color = "white"
        else:
            dot_color = "#444444"
            text_color = "#777777"

        self.page3.RecordingRed.setStyleSheet(
            f"""
            background-color: {dot_color};
            border-radius: 10px;
            """
        )

        self.page3.RecordinginProgress.setStyleSheet(
            f"""
            color: {text_color};
            """
        )

    def start_page2_preview(self):
        if not self.mocap_selected:
            self.page2.VideoFeed.setText("Mocap not selected")
            return

        self.stop_page4_playback()
        self.stop_mocap_recording()

        self.preview_camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        if not self.preview_camera.isOpened():
            self.page2.VideoFeed.setText("ERROR: Could not open preview camera")
            print("ERROR: Could not open Page 2 preview camera.")
            return

        self.preview_timer.start(40)

    def update_page2_preview(self):
        if self.preview_camera is None:
            return

        success, frame = self.preview_camera.read()
        if not success:
            return

        self.current_preview_frame = frame.copy()

        self.page2.VideoFeed.setPixmap(self.cv_frame_to_pixmap(frame, self.page2.VideoFeed))

    def start_reference_countdown(self):
        self.reference_countdown_value = 5

        self.page2.CaptureReference.setEnabled(False)
        self.page2.CaptureReference.setText("5")

        self.reference_countdown_timer.start(1000)

    def update_reference_countdown(self):
        self.reference_countdown_value -= 1

        if self.reference_countdown_value > 0:
            self.page2.CaptureReference.setText(str(self.reference_countdown_value))
        elif self.reference_countdown_value == 0:
            self.page2.CaptureReference.setText("HOLD")
        else:
            self.reference_countdown_timer.stop()
            self.capture_reference_frame()
            self.page2.CaptureReference.setText("Capture Reference")
            self.page2.CaptureReference.setEnabled(True)

    def capture_reference_frame(self):
        if self.current_preview_frame is None:
            print("No preview frame available to capture.")
            self.page2.VideoFeed.setText("No preview frame available")
            return

        success = cv2.imwrite(
            str(self.reference_frame_path),
            self.current_preview_frame,
        )

        if success:
            print(f"Reference frame saved: {self.reference_frame_path}")
        else:
            print("ERROR: Could not save reference frame.")

    def reset_mocap_sync_state(self):
        self.mocap_synced = False

        self.set_indicator_pair(
            self.page2.MoCapRed,
            self.page2.MoCapGreen,
            False,
        )

        self.update_record_button_state()

    def handle_leader_analyze_toggled(self, checked):
        if checked:
            self.page2.AnalyzeF.blockSignals(True)
            self.page2.AnalyzeF.setChecked(False)
            self.page2.AnalyzeF.blockSignals(False)

            self.target_person_idx = 0
            self.reset_mocap_sync_state()

    def handle_follower_analyze_toggled(self, checked):
        if checked:
            self.page2.AnalyzeL.blockSignals(True)
            self.page2.AnalyzeL.setChecked(False)
            self.page2.AnalyzeL.blockSignals(False)

            self.target_person_idx = 1
            self.reset_mocap_sync_state()

    def get_height_feet(self, feet_edit, inch_edit):
        feet_text = feet_edit.text().strip()
        inch_text = inch_edit.text().strip()

        if feet_text == "":
            return None

        try:
            feet = float(feet_text)
            inches = float(inch_text) if inch_text != "" else 0
        except ValueError:
            return None

        return feet + (inches / 12.0)

    def perform_mocap_calibration(self):
        if not self.reference_frame_path.exists():
            print("ERROR: Reference frame not found.")
            return

        leader_height_ft = self.get_height_feet(
            self.page2.HeightLfeet,
            self.page2.HeightLinch,
        )

        follower_height_ft = self.get_height_feet(
            self.page2.HeightFfeet,
            self.page2.HeightFinch,
        )

        leader_selected = self.page2.AnalyzeL.isChecked()
        follower_selected = self.page2.AnalyzeF.isChecked()

        if not leader_selected and not follower_selected:
            print("ERROR: Select either leader or follower for analysis.")
            return

        if leader_selected:
            self.target_person_idx = 0

        if follower_selected:
            self.target_person_idx = 1

        if leader_height_ft is None:
            print("ERROR: Leader height missing. Leader height is required for calibration origin.")
            return

        if self.target_person_idx == 1 and follower_height_ft is None:
            print("ERROR: Follower height missing. Follower height is required when analyzing follower.")
            return

        print("Loading calibration image...")

        frame = cv2.imread(str(self.reference_frame_path))

        if frame is None:
            print("ERROR: Could not load reference frame.")
            return

        h, w = frame.shape[:2]

        print("Running YOLO calibration detection...")

        yolo = YOLO("yolov8n.pt")

        detections = yolo(
            frame,
            classes=[0],
            verbose=False,
        )[0]

        people_boxes = []

        if detections.boxes is not None:
            for b in detections.boxes:
                conf = float(b.conf[0])

                if conf < 0.4:
                    continue

                x1, y1, x2, y2 = map(int, b.xyxy[0])
                people_boxes.append((x1, y1, x2, y2))

        if len(people_boxes) == 0:
            print("ERROR: No dancers detected.")
            return

        people_boxes = sorted(
            people_boxes,
            key=lambda b: ((b[0] + b[2]) / 2.0),
        )

        # If follower is selected for analysis, both dancers must be visible
        # in the reference frame so PERSON 1 can be identified later.
        if self.target_person_idx == 1 and len(people_boxes) < 2:
            print("ERROR: Follower selected, but only one dancer detected in reference frame.")
            print("Please capture a reference image with both leader and follower visible.")
            return

        # Calibration origin is always leader, so leader must be detected.
        if len(people_boxes) < 1:
            print("ERROR: Leader not detected for calibration.")
            return

        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
        )

        def process_reference_box(box, label):
            x1, y1, x2, y2 = box

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                print(f"ERROR: Invalid {label} crop.")
                return None

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if not result.pose_landmarks:
                print(f"ERROR: No pose detected for {label}.")
                return None

            return result.pose_landmarks.landmark

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

        leader_box = people_boxes[0]
        leader_landmarks = process_reference_box(leader_box, "leader")

        if leader_landmarks is None:
            pose.close()
            return

        lm = mp_pose.PoseLandmark

        left_heel = leader_landmarks[lm.LEFT_HEEL]
        left_ankle = leader_landmarks[lm.LEFT_ANKLE]

        heel_visibility = getattr(left_heel, "visibility", 0)

        x1, y1, x2, y2 = leader_box

        if heel_visibility > 0.5:
            origin_local_x = left_heel.x
            origin_local_y = left_heel.y
            print("Using leader LEFT_HEEL as origin")
        else:
            print("Leader LEFT_HEEL low confidence, using LEFT_ANKLE")
            origin_local_x = left_ankle.x
            origin_local_y = left_ankle.y

        origin_global_x = origin_local_x * (x2 - x1) + x1
        origin_global_y = origin_local_y * (y2 - y1) + y1

        leader_person_height_px = y2 - y1
        pixels_per_foot = leader_person_height_px / leader_height_ft

        target_box = people_boxes[self.target_person_idx]
        target_height_ft = (
            leader_height_ft if self.target_person_idx == 0 else follower_height_ft
        )

        if self.target_person_idx == 0:
            target_landmarks = leader_landmarks
        else:
            target_landmarks = process_reference_box(target_box, "follower")
            if target_landmarks is None:
                pose.close()
                return

        target_x1, target_y1, target_x2, target_y2 = target_box
        target_person_height_px = target_y2 - target_y1
        target_pixels_per_foot = target_person_height_px / target_height_ft

        shoulder_width_px_ref = distance_between_landmarks(
            target_landmarks,
            lm.LEFT_SHOULDER,
            lm.RIGHT_SHOULDER,
            target_box,
        )

        hip_width_px_ref = distance_between_landmarks(
            target_landmarks,
            lm.LEFT_HIP,
            lm.RIGHT_HIP,
            target_box,
        )

        valid_widths = []

        if shoulder_width_px_ref > 1:
            valid_widths.append(shoulder_width_px_ref)

        if hip_width_px_ref > 1:
            valid_widths.append(hip_width_px_ref)

        if not valid_widths:
            print("ERROR: Could not measure reference shoulder/hip width.")
            pose.close()
            return

        torso_width_px_ref = float(np.mean(valid_widths))
        torso_width_ft_ref = torso_width_px_ref / target_pixels_per_foot

        left_shin_px_ref = distance_between_landmarks(
            target_landmarks,
            lm.LEFT_KNEE,
            lm.LEFT_ANKLE,
            target_box,
        )

        right_shin_px_ref = distance_between_landmarks(
            target_landmarks,
            lm.RIGHT_KNEE,
            lm.RIGHT_ANKLE,
            target_box,
        )

        valid_shins = []

        if left_shin_px_ref > 1:
            valid_shins.append(left_shin_px_ref)

        if right_shin_px_ref > 1:
            valid_shins.append(right_shin_px_ref)

        avg_shin_px_ref = float(np.mean(valid_shins)) if valid_shins else 1.0

        calibration_data = {
            "reference_subject": "leader",
            "selected_target": "leader" if self.target_person_idx == 0 else "follower",
            "target_person_idx": int(self.target_person_idx),

            "origin_px": [
                float(origin_global_x),
                float(origin_global_y),
            ],

            "origin_norm": [
                float(origin_global_x / w),
                float(origin_global_y / h),
            ],

            "pixels_per_foot": float(pixels_per_foot),
            "leader_height_ft": float(leader_height_ft),
            "height_ft": float(leader_height_ft),

            "target_height_ft": float(target_height_ft),
            "target_pixels_per_foot": float(target_pixels_per_foot),
            "target_person_height_px": float(target_person_height_px),

            "target_shoulder_width_px_ref": float(shoulder_width_px_ref),
            "target_hip_width_px_ref": float(hip_width_px_ref),
            "target_torso_width_px_ref": float(torso_width_px_ref),
            "target_torso_width_ft_ref": float(torso_width_ft_ref),
            "target_left_shin_px_ref": float(left_shin_px_ref),
            "target_right_shin_px_ref": float(right_shin_px_ref),
            "target_avg_shin_px_ref": float(avg_shin_px_ref),

            "frame_width": int(w),
            "frame_height": int(h),

            "reference_image": str(self.reference_frame_path),
        }

        with open(self.calibration_path, "w") as f:
            json.dump(calibration_data, f, indent=4)

        print(f"Calibration saved: {self.calibration_path}")
        print(f"Selected target: {calibration_data['selected_target']}")
        print(f"Target torso width ref px: {torso_width_px_ref:.2f}")
        print(f"Target torso width ref ft: {torso_width_ft_ref:.3f}")

        self.mocap_synced = True

        self.set_indicator_pair(
            self.page2.MoCapRed,
            self.page2.MoCapGreen,
            True,
        )

        self.update_record_button_state()

        pose.close()

    def stop_page2_preview(self):
        self.preview_timer.stop()
        self.reference_countdown_timer.stop()
        self.stop_page2_foot_live_preview()
        self.stop_page2_hand_live_preview()

        if hasattr(self.page2, "CaptureReference"):
            self.page2.CaptureReference.setText("Capture Reference")
            self.page2.CaptureReference.setEnabled(True)

        if self.preview_camera is not None:
            self.preview_camera.release()
            self.preview_camera = None

    def start_countdown(self):
        self.countdown_value = 5
        self.page3.Countdown.setText("5")
        self.countdown_timer.start(1000)

    def update_countdown(self):
        self.countdown_value -= 1

        if self.countdown_value > 0:
            self.page3.Countdown.setText(str(self.countdown_value))
        elif self.countdown_value == 0:
            self.page3.Countdown.setText("DANCE!")
        else:
            self.countdown_timer.stop()
            self.page3.Countdown.setText("")

    def begin_recording_session(self):
        # Start a fresh shared clock for this trial. All recorded devices use
        # this common time base before Page 4 applies the mocap trim offset.
        self.recording_session_start_time = time.time()
        self.analysis_time_offset_sec = 0.0

        self.page3.Stop.setEnabled(True)
        self.page3.Analyze.setEnabled(False)
        self.page3.Back.setEnabled(False)

        self.style_page3_action_buttons(stopped=False)
        self.set_recording_indicator(True)

        if self.mocap_selected:
            self.start_mocap_recording()
        else:
            self.page3.VideoFeed.setText("No mocap selected")

        if self.feet_selected:
            self.start_foot_logging()
            self.start_page3_foot_live_preview()
        else:
            self.clear_page3_foot_live_views()

        if self.hands_selected:
            self.start_hand_logging()

        self.start_countdown()

    def start_mocap_recording(self):
        self.record_camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        if not self.record_camera.isOpened():
            print("ERROR: Could not open webcam.")
            self.page3.VideoFeed.setText("ERROR: Could not open webcam")
            self.stop_recording()
            return

        # Request 20 Hz from the camera. Some webcams may ignore this, but the
        # Qt recording timer and VideoWriter below are both locked to 20 Hz.
        self.record_camera.set(cv2.CAP_PROP_FPS, 20)

        width = int(self.record_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.record_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            str(self.video_output_path),
            fourcc,
            fps,
            (width, height),
        )

        if not self.video_writer.isOpened():
            print("ERROR: Could not create video writer.")
            self.page3.VideoFeed.setText("ERROR: Could not create video file")
            self.stop_recording()
            return

        print(f"Recording mocap video to: {self.video_output_path}")
        # Match the VideoWriter FPS. 50 ms = 20 Hz.
        # This prevents the recorded video timeline from stretching relative to
        # the sensor timelines.
        self.record_timer.start(int(round(1000.0 / fps)))

    def update_recording_frame(self):
        if self.record_camera is None:
            return

        success, frame = self.record_camera.read()
        if not success:
            return

        if self.video_writer is not None:
            self.video_writer.write(frame)

        self.page3.VideoFeed.setPixmap(self.cv_frame_to_pixmap(frame, self.page3.VideoFeed))

    def stop_mocap_recording(self):
        self.record_timer.stop()

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if self.record_camera is not None:
            self.record_camera.release()
            self.record_camera = None

    def parse_device_time_to_seconds(self, device_time):
        # Supports:
        #   seconds as a number, e.g. "12.345"
        #   milliseconds as a number, e.g. "12345"
        #   HH:MM:SS.mmm, e.g. "00:00:12.345"
        #   MM:SS.mmm, e.g. "00:12.345"
        #
        # ESP32-style millis() values are common. If a plain numeric value is
        # very large, treat it as milliseconds and convert to seconds.
        #
        # Returns None if the format is not parseable.
        try:
            text = str(device_time).strip()

            if text == "":
                return None

            try:
                value = float(text)

                # If the value is too large to be a normal trial time in seconds,
                # it is probably milliseconds from millis().
                if value > 300.0:
                    return value / 1000.0

                return value
            except ValueError:
                pass

            parts = text.split(":")

            if len(parts) == 3:
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600.0 + minutes * 60.0 + seconds

            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60.0 + seconds

        except Exception:
            return None

        return None


    def get_synced_sample_time_sec(
        self,
        device_time,
        device_zero_time_sec,
        sync_pc_time,
        recording_start_time,
        fallback_recording_start_time,
    ):
        # Preferred timestamp:
        #   sample_time = PC time when SYNC was sent relative to trial start
        #                 + device elapsed time since first post-sync sample.
        #
        # This avoids lag/stretch from serial buffering because sample time is
        # based on the device clock instead of Python's receive time.
        device_time_sec = self.parse_device_time_to_seconds(device_time)

        if device_time_sec is not None and device_zero_time_sec is not None:
            if recording_start_time is None:
                recording_start_time = fallback_recording_start_time

            sync_offset_sec = max(0.0, float(sync_pc_time) - float(recording_start_time))
            elapsed_device_sec = max(0.0, float(device_time_sec) - float(device_zero_time_sec))
            return sync_offset_sec + elapsed_device_sec

        return time.time() - fallback_recording_start_time


    def parse_foot_sensor_line(self, line):
        parts = [part.strip() for part in line.replace("\t", ",").split(",")]

        if len(parts) < 5:
            return None

        try:
            fsr_values = [float(value) for value in parts[1:5]]
        except ValueError:
            return None

        return parts[0], fsr_values

    def identify_foot_device_on_port(self, port_name):
        if serial is None:
            return None

        # Foot sensors are fixed to known Bluetooth COM ports.
        # This avoids slow scanning of every available COM port.
        if port_name == LEFT_FOOT_FIXED_PORT:
            expected_side = "left"
        elif port_name == RIGHT_FOOT_FIXED_PORT:
            expected_side = "right"
        else:
            return None

        try:
            ser = serial.Serial(port_name, 115200, timeout=0.2)
            time.sleep(0.5)

            detected_side = None
            start = time.time()

            try:
                ser.write(f"SYNC:{time.time()}\n".encode())
            except Exception:
                pass

            while time.time() - start < 2.0:
                raw = ser.readline()

                if not raw:
                    continue

                line = raw.decode(errors="ignore").strip()
                upper_line = line.upper()

                if "WAITING_FOR_SYNC_RIGHT" in upper_line or "RIGHT" in upper_line:
                    detected_side = "right"
                    break

                if "WAITING_FOR_SYNC_LEFT" in upper_line or "LEFT" in upper_line:
                    detected_side = "left"
                    break

                # Some firmware may stream numeric FSR data without a side label.
                # Since the ports are fixed, accept numeric-looking foot data
                # as the expected side for that COM port.
                parsed_values = self.parse_foot_sensor_values(line)

                if parsed_values is not None:
                    detected_side = expected_side
                    break

            if detected_side is None:
                detected_side = expected_side

            ser.close()

            if detected_side == expected_side:
                print(f"Synced {detected_side} foot on {port_name}")
                return detected_side

            print(
                f"WARNING: {port_name} responded as {detected_side}, "
                f"but expected {expected_side}."
            )
            return None

        except Exception as exc:
            print(f"Could not check {port_name}: {exc}")
            return None

    def start_foot_logging(self):
        if not self.feet_selected:
            return

        if self.left_foot_port is None or self.right_foot_port is None:
            print("ERROR: Foot sensors are not synced. Cannot start foot logging.")
            return

        if serial is None:
            print("ERROR: pyserial is not installed. Run: pip install pyserial")
            return

        self.stop_foot_logging()

        self.foot_stop_event.clear()
        self.foot_logging_active = True
        self.foot_recording_start_time = (
            self.recording_session_start_time
            if self.recording_session_start_time is not None
            else time.time()
        )
        self.foot_threads = []

        self.left_foot_latest_fsr = [0.0, 0.0, 0.0, 0.0]
        self.right_foot_latest_fsr = [0.0, 0.0, 0.0, 0.0]

        foot_configs = [
            ("left", self.left_foot_port, self.left_foot_csv_path),
            ("right", self.right_foot_port, self.right_foot_csv_path),
        ]

        for side, port_name, csv_path in foot_configs:
            thread = threading.Thread(
                target=self.foot_logger_loop,
                args=(side, port_name, csv_path),
                daemon=True,
            )
            thread.start()
            self.foot_threads.append(thread)

        print("Started foot sensor logging.")

    def foot_logger_loop(self, side, port_name, csv_path):
        try:
            ser = serial.Serial(port_name, 115200, timeout=0.5)
            time.sleep(0.2)

            try:
                ser.reset_input_buffer()
                ser.reset_output_buffer()
            except Exception:
                pass

            sync_pc_time = time.time()
            sync_message = f"SYNC:{sync_pc_time}\n"
            ser.write(sync_message.encode())
            print(f"Sent recording sync to {side} foot on {port_name}")

            device_zero_time_sec = None

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "PC_TIME_SEC",
                    "DEVICE_TIME",
                    "FSR1",
                    "FSR2",
                    "FSR3",
                    "FSR4",
                ])

                while not self.foot_stop_event.is_set():
                    raw = ser.readline()

                    if not raw:
                        continue

                    line = raw.decode(errors="ignore").strip()

                    if not line:
                        continue

                    if "WAITING" in line or "SYNC" in line or "LEFT" in line or "RIGHT" in line:
                        continue

                    parsed = self.parse_foot_sensor_line(line)

                    if parsed is None:
                        continue

                    device_time, fsr_values = parsed

                    device_time_sec = self.parse_device_time_to_seconds(device_time)

                    if device_time_sec is not None and device_zero_time_sec is None:
                        device_zero_time_sec = device_time_sec

                    raw_values = [float(v) for v in fsr_values]
                    corrected_values = self.apply_foot_tare(raw_values, side)

                    if side == "left":
                        self.left_foot_latest_raw_fsr = raw_values
                        self.left_foot_latest_fsr = corrected_values
                    elif side == "right":
                        self.right_foot_latest_raw_fsr = raw_values
                        self.right_foot_latest_fsr = corrected_values

                    pc_time_sec = self.get_synced_sample_time_sec(
                        device_time,
                        device_zero_time_sec,
                        sync_pc_time,
                        self.recording_session_start_time,
                        self.foot_recording_start_time,
                    )

                    writer.writerow([
                        f"{pc_time_sec:.6f}",
                        device_time,
                        corrected_values[0],
                        corrected_values[1],
                        corrected_values[2],
                        corrected_values[3],
                    ])

            ser.close()
            print(f"Stopped {side} foot logging: {csv_path}")

        except Exception as exc:
            print(f"ERROR logging {side} foot sensor on {port_name}: {exc}")

    def stop_foot_logging(self):
        if not self.foot_logging_active and not self.foot_threads:
            return

        self.foot_stop_event.set()

        for thread in self.foot_threads:
            thread.join(timeout=2.0)

        self.foot_threads = []
        self.foot_logging_active = False
        print("Foot sensor logging stopped.")

    def create_analysis_video_after_countdown(self):
        # Recording starts as soon as Page 3 begins, while the countdown text is running.
        # The dancer starts when the countdown text says "DANCE!".
        #
        # This makes a temporary video for mocap analysis starting at the DANCE moment.
        # The raw recording is not overwritten.
        #
        # Timing rule:
        #   - All device CSVs use the shared raw-recording time base.
        #   - Successful trim sets self.analysis_time_offset_sec to countdown_trim_seconds.
        #   - Page 4 loaders subtract self.analysis_time_offset_sec from COP,
        #     handhold, and open-hand sample times.
        #   - If analysis falls back to raw video, offset is reset to 0.0.
        countdown_trim_seconds = 4.0

        if not self.video_output_path.exists():
            self.analysis_time_offset_sec = 0.0
            return self.video_output_path

        capture = cv2.VideoCapture(str(self.video_output_path))

        if not capture.isOpened():
            print("WARNING: Could not open raw video for countdown trimming. Using full raw video.")
            self.analysis_time_offset_sec = 0.0
            return self.video_output_path

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1e-6:
            fps = 20.0

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(round(countdown_trim_seconds * fps))

        if total_frames <= start_frame + 1:
            print("WARNING: Video is too short to trim countdown. Using full raw video.")
            capture.release()
            self.analysis_time_offset_sec = 0.0
            return self.video_output_path

        if self.analysis_video_path.exists():
            self.analysis_video_path.unlink()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self.analysis_video_path),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            print("WARNING: Could not create trimmed analysis video. Using full raw video.")
            capture.release()
            self.analysis_time_offset_sec = 0.0
            return self.video_output_path

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        written_frames = 0

        while True:
            success, frame = capture.read()

            if not success:
                break

            writer.write(frame)
            written_frames += 1

        writer.release()
        capture.release()

        if written_frames <= 0:
            print("WARNING: Trimmed analysis video had no frames. Using full raw video.")
            self.analysis_time_offset_sec = 0.0
            return self.video_output_path

        self.analysis_time_offset_sec = countdown_trim_seconds

        print(
            f"Created trimmed mocap analysis video: {self.analysis_video_path} "
            f"(skipped first {countdown_trim_seconds:.1f} sec / {start_frame} frames)"
        )

        return self.analysis_video_path

    def get_selected_target_weight_lbf(self):
        # Supports common weight input object names without requiring UI edits.
        # Leader is used when AnalyzeL is selected; follower is used when AnalyzeF is selected.
        if self.target_person_idx == 0:
            candidate_names = [
                "WeightL",
                "WeightLeader",
                "LeaderWeight",
                "WeightLBSL",
                "WeightLLBS",
                "WeightLbsL",
                "WeightLbsLeader",
            ]
        else:
            candidate_names = [
                "WeightF",
                "WeightFollower",
                "FollowerWeight",
                "WeightLBSF",
                "WeightFLBS",
                "WeightLbsF",
                "WeightLbsFollower",
            ]

        for name in candidate_names:
            if not hasattr(self.page2, name):
                continue

            widget = getattr(self.page2, name)

            try:
                if hasattr(widget, "value"):
                    value = float(widget.value())
                    if value > 0:
                        return value

                if hasattr(widget, "text"):
                    text = widget.text().strip()
                    if text:
                        value = float(text)
                        if value > 0:
                            return value
            except (TypeError, ValueError):
                continue

        return None

    def read_latest_total_grf_from_csv(self):
        self.latest_total_grf = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }
        self.grf_by_frame = {}
        self.grf_available_frames = []

        if not self.grf_selected:
            return

        if not self.processed_csv_path.exists():
            return

        latest_row = None

        try:
            with open(self.processed_csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    if row.get("TYPE") != "COM":
                        continue

                    try:
                        person_id = int(float(row.get("PERSON", "-1")))
                    except ValueError:
                        continue

                    if person_id != self.target_person_idx:
                        continue

                    if row.get("GRF_X_LBF", "") == "":
                        continue

                    try:
                        frame_number = int(float(row.get("FRAME", "0")))
                        grf_x = float(row.get("GRF_X_LBF", 0.0))
                        grf_y = float(row.get("GRF_Y_LBF", 0.0))
                        grf_z = float(row.get("GRF_Z_LBF", 0.0))
                    except (TypeError, ValueError):
                        continue

                    self.grf_by_frame[frame_number] = {
                        "x": grf_x,
                        "y": grf_y,
                        "z": grf_z,
                    }

                    latest_row = self.grf_by_frame[frame_number]

            self.grf_available_frames = sorted(self.grf_by_frame.keys())

            if latest_row is not None:
                self.latest_total_grf = latest_row

            print(f"Loaded {len(self.grf_available_frames)} GRF frame values.")

        except Exception as exc:
            print(f"WARNING: Could not read GRF values from CSV: {exc}")

    def set_numeric_display_value(self, widget_name, value):
        if not hasattr(self.page4, widget_name):
            return

        widget = getattr(self.page4, widget_name)

        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0

        if hasattr(widget, "display"):
            widget.display(float(f"{value:.2f}"))
        elif hasattr(widget, "setText"):
            widget.setText(f"{value:.2f}")

    def get_grf_for_frame(self, frame_number):
        if not self.grf_available_frames:
            return {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            }

        frame_number = int(max(0, frame_number))

        selected_frame = None

        for frame in self.grf_available_frames:
            if frame <= frame_number:
                selected_frame = frame
            else:
                break

        if selected_frame is None:
            selected_frame = self.grf_available_frames[0]

        return self.grf_by_frame.get(
            selected_frame,
            {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            },
        )

    def get_nearest_com_state_for_frame(self, frame_number):
        # Returns approximate COM [x, y, z] in feet for the selected frame.
        if not self.com_frames:
            return np.array([0.0, 3.0, 0.0], dtype=float)

        selected_index = 0

        for index, frame in enumerate(self.com_frames):
            if frame <= frame_number:
                selected_index = index
            else:
                break

        try:
            return np.array(
                [
                    float(self.com_x[selected_index]),
                    float(self.com_y[selected_index]),
                    float(self.com_z[selected_index]),
                ],
                dtype=float,
            )
        except (TypeError, ValueError, IndexError):
            return np.array([0.0, 3.0, 0.0], dtype=float)


    def get_nearest_foot_landmarks_for_frame(self, frame_number):
        if not self.foot_landmarks_by_frame:
            return {}

        frame_number = int(max(0, frame_number))
        available_frames = sorted(self.foot_landmarks_by_frame.keys())
        selected_frame = available_frames[0]

        for frame in available_frames:
            if frame <= frame_number:
                selected_frame = frame
            else:
                break

        return self.foot_landmarks_by_frame.get(selected_frame, {})


    def get_com_y_for_frame(self, frame_number):
        com_xyz = self.get_nearest_com_state_for_frame(frame_number)
        return max(abs(float(com_xyz[1])), 1.0)


    def get_cop_sample_for_time(self, samples, time_sec):
        if not samples:
            return None

        index = self.get_cop_index_for_time(samples, time_sec)

        if index < 0 or index >= len(samples):
            return None

        return samples[index].get("cop")


    def get_foot_contact_state_for_frame(self, frame_number):
        # Contact is determined from calibrated FSR total force inside the COP
        # samples. This prevents assigning GRF to an unloaded foot.
        time_sec = float(frame_number) / max(float(self.playback_fps), 1.0)

        left_cop = self.get_cop_sample_for_time(self.left_cop_samples, time_sec)
        right_cop = self.get_cop_sample_for_time(self.right_cop_samples, time_sec)

        contact_threshold_lbf = 3.0

        left_load = 0.0
        right_load = 0.0

        if left_cop is not None:
            try:
                left_load = max(float(left_cop.get("total_force", 0.0)), 0.0)
            except (TypeError, ValueError):
                left_load = 0.0

        if right_cop is not None:
            try:
                right_load = max(float(right_cop.get("total_force", 0.0)), 0.0)
            except (TypeError, ValueError):
                right_load = 0.0

        return {
            "left_contact": left_load >= contact_threshold_lbf,
            "right_contact": right_load >= contact_threshold_lbf,
            "left_load": left_load,
            "right_load": right_load,
        }


    def get_contact_point_relative_to_com_placeholder(self, side, frame_number):
        # Approximate 3D COP/contact point relative to whole-body COM, in feet.
        #
        # Method:
        #   1) Use mocap heel landmark as the origin.
        #   2) Use heel -> foot_index as the direction toward the ball/toe.
        #   3) Place the calculated COP along that global direction.
        #   4) Clip only after global placement: COP cannot land behind the heel
        #      along the global heel-to-toe axis.
        #
        # Limitation:
        #   The current mocap joint CSV provides X/Y joint coordinates, not a
        #   true calibrated foot depth coordinate, so this maps the camera-view
        #   heel-to-toe vector into the X/Z ground plane as a placeholder.
        time_sec = float(frame_number) / max(float(self.playback_fps), 1.0)

        if side == "left":
            samples = self.left_cop_samples
            heel_joint_id = 29
            foot_index_joint_id = 31
            fallback_x_ft = -0.35
        else:
            samples = self.right_cop_samples
            heel_joint_id = 30
            foot_index_joint_id = 32
            fallback_x_ft = 0.35

        com_xyz = self.get_nearest_com_state_for_frame(frame_number)
        com_height_ft = max(abs(float(com_xyz[1])), 1.0)

        cop = self.get_cop_sample_for_time(samples, time_sec)

        # Default to ball-of-foot if COP is missing.
        cop_x_in = 0.0
        cop_y_in = 9.0 / 2.54
        heel_y_in = 23.0 / 2.54

        if cop is not None:
            try:
                cop_x_in = float(cop.get("cop_x_in", 0.0))
                cop_y_in = float(cop.get("cop_y_in", cop_y_in))

                foot_length_in = float(cop.get("foot_length_in", 10.0))
                reference_foot_length_in = 25.5 / 2.54
                scale = foot_length_in / reference_foot_length_in
                heel_y_in = (23.0 / 2.54) * scale
            except (TypeError, ValueError):
                cop_x_in = 0.0
                cop_y_in = 9.0 / 2.54
                heel_y_in = 23.0 / 2.54

        # In the local COP frame, larger y is closer to heel and smaller y is
        # closer to toe. Do not clip here; clip after global placement.
        heel_to_cop_ft = (heel_y_in - cop_y_in) / 12.0
        medial_lateral_offset_ft = cop_x_in / 12.0

        landmarks = self.get_nearest_foot_landmarks_for_frame(frame_number)
        heel_xy = landmarks.get(heel_joint_id)
        foot_index_xy = landmarks.get(foot_index_joint_id)

        if heel_xy is not None and foot_index_xy is not None:
            direction_xy = foot_index_xy - heel_xy
            direction_norm = float(np.linalg.norm(direction_xy))

            if direction_norm > 1e-6:
                direction_xy = direction_xy / direction_norm
            else:
                direction_xy = np.array([0.0, -1.0], dtype=float)

            # Perpendicular direction for medial/lateral COP shift in image plane.
            perp_xy = np.array([-direction_xy[1], direction_xy[0]], dtype=float)

            heel_global_xz = np.array(
                [
                    float(heel_xy[0]),
                    float(com_xyz[2]),
                ],
                dtype=float,
            )

            # Map camera-view foot direction into ground-plane X/Z.
            foot_forward_global = np.array(
                [
                    float(direction_xy[0]),
                    -float(direction_xy[1]),
                ],
                dtype=float,
            )

            forward_norm = float(np.linalg.norm(foot_forward_global))

            if forward_norm > 1e-6:
                foot_forward_global = foot_forward_global / forward_norm
            else:
                foot_forward_global = np.array([0.0, 1.0], dtype=float)

            foot_lateral_global = np.array(
                [
                    float(perp_xy[0]),
                    -float(perp_xy[1]),
                ],
                dtype=float,
            )

            lateral_norm = float(np.linalg.norm(foot_lateral_global))

            if lateral_norm > 1e-6:
                foot_lateral_global = foot_lateral_global / lateral_norm
            else:
                foot_lateral_global = np.array([1.0, 0.0], dtype=float)

            candidate_contact_xz = (
                heel_global_xz
                + heel_to_cop_ft * foot_forward_global
                + medial_lateral_offset_ft * foot_lateral_global
            )

            # Global-frame clipping: if COP lands behind heel along the global
            # heel-to-toe direction, move it to the heel line while preserving
            # the medial/lateral offset.
            heel_to_candidate = candidate_contact_xz - heel_global_xz
            forward_projection_ft = float(np.dot(heel_to_candidate, foot_forward_global))

            if forward_projection_ft < 0.0:
                candidate_contact_xz = (
                    heel_global_xz
                    + medial_lateral_offset_ft * foot_lateral_global
                )

            contact_x_ft = float(candidate_contact_xz[0])
            contact_z_ft = float(candidate_contact_xz[1])

        else:
            # Fallback if foot landmarks are missing.
            contact_x_ft = float(com_xyz[0]) + fallback_x_ft + medial_lateral_offset_ft
            contact_z_ft = float(com_xyz[2]) + max(0.0, heel_to_cop_ft)

        contact_y_ft = 0.0

        r_x = contact_x_ft - float(com_xyz[0])
        r_y = contact_y_ft - float(com_height_ft)
        r_z = contact_z_ft - float(com_xyz[2])

        return np.array([r_x, r_y, r_z], dtype=float)


    def get_nearest_hand_landmarks_for_frame(self, frame_number):
        if not self.hand_landmarks_by_frame:
            return {}

        frame_number = int(max(0, frame_number))
        available_frames = sorted(self.hand_landmarks_by_frame.keys())
        selected_frame = available_frames[0]

        for frame in available_frames:
            if frame <= frame_number:
                selected_frame = frame
            else:
                break

        return self.hand_landmarks_by_frame.get(selected_frame, {})


    def get_time_sample_for_frame(self, samples, frame_number):
        if not samples:
            return None

        time_sec = float(frame_number) / max(float(self.playback_fps), 1.0)

        best_index = 0
        best_diff = abs(float(samples[0].get("time_sec", 0.0)) - time_sec)

        for index, sample in enumerate(samples[1:], start=1):
            sample_time = float(sample.get("time_sec", 0.0))
            diff = abs(sample_time - time_sec)

            if diff < best_diff:
                best_diff = diff
                best_index = index
            elif sample_time > time_sec and diff > best_diff:
                break

        return samples[best_index]


    def landmark_xy_to_global_xyz(self, landmark_xy, com_xyz):
        # Current mocap CSV provides X/Y joint coordinates. Use the current COM
        # depth as a placeholder Z for hand moment arms until true hand depth is
        # available.
        return np.array(
            [
                float(landmark_xy[0]),
                float(landmark_xy[1]),
                float(com_xyz[2]),
            ],
            dtype=float,
        )


    def build_handhold_basis(self, lead_left_wrist_xy, follow_right_wrist_xy):
        # Primary axis: lead left wrist -> follow right wrist.
        direction_xy = follow_right_wrist_xy - lead_left_wrist_xy
        direction_norm = float(np.linalg.norm(direction_xy))

        if direction_norm > 1e-6:
            primary_axis = np.array(
                [
                    float(direction_xy[0]),
                    float(direction_xy[1]),
                    0.0,
                ],
                dtype=float,
            )
            primary_axis = primary_axis / float(np.linalg.norm(primary_axis))
        else:
            primary_axis = np.array([1.0, 0.0, 0.0], dtype=float)

        global_up = np.array([0.0, 1.0, 0.0], dtype=float)

        # Lateral axis is perpendicular to the handhold axis and vertical.
        lateral_axis = np.cross(primary_axis, global_up)
        lateral_norm = float(np.linalg.norm(lateral_axis))

        if lateral_norm > 1e-6:
            lateral_axis = lateral_axis / lateral_norm
        else:
            lateral_axis = np.array([0.0, 0.0, 1.0], dtype=float)

        # Vertical-ish axis is perpendicular to primary and lateral. This allows
        # the wrist-to-wrist pitch to affect the vertical component direction.
        vertical_axis = np.cross(lateral_axis, primary_axis)
        vertical_norm = float(np.linalg.norm(vertical_axis))

        if vertical_norm > 1e-6:
            vertical_axis = vertical_axis / vertical_norm
        else:
            vertical_axis = global_up.copy()

        return primary_axis, vertical_axis, lateral_axis


    def get_handhold_external_force_and_point(self, frame_number, com_xyz):
        # HH external force:
        #   The handhold force is applied at the midpoint between the
        #   leader's left wrist and follower's right wrist.
        #
        # Sign convention for x_component:
        #   Positive force points away from the selected/analyzed dancer.
        #   target 0 leader   -> positive along leader wrist -> follower wrist
        #   target 1 follower -> positive along follower wrist -> leader wrist
        #
        # Vertical/lateral components still use the handhold-based basis.
        sample = self.get_time_sample_for_frame(self.handhold_samples, frame_number)

        if sample is None:
            return None, None

        landmarks_by_person = self.get_nearest_hand_landmarks_for_frame(frame_number)

        lead = landmarks_by_person.get(0, {})
        follow = landmarks_by_person.get(1, {})

        lead_left_wrist = lead.get(15)
        follow_right_wrist = follow.get(16)

        if lead_left_wrist is None or follow_right_wrist is None:
            return None, None

        primary_axis, vertical_axis, lateral_axis = self.build_handhold_basis(
            lead_left_wrist,
            follow_right_wrist,
        )

        if self.target_person_idx == 0:
            away_from_selected_axis = primary_axis
        else:
            away_from_selected_axis = -primary_axis

        force = (
            float(sample.get("x_component", 0.0)) * away_from_selected_axis
            + float(sample.get("vertical_component", 0.0)) * vertical_axis
            + float(sample.get("lateral_component", 0.0)) * lateral_axis
        )

        application_xy = 0.5 * (lead_left_wrist + follow_right_wrist)
        point = self.landmark_xy_to_global_xyz(application_xy, com_xyz)

        return force, point


    def get_openhand_external_force_and_point(self, frame_number, com_xyz):
        # OH external force:
        #   Force is applied at the leader's right palm, approximated from
        #   RIGHT_WRIST, RIGHT_INDEX, and RIGHT_PINKY.
        #
        # Direction approximation:
        #   True palm normal would come from a 3D cross product of:
        #       wrist -> index knuckle
        #       wrist -> pinky knuckle
        #
        #   Current pose CSV stores X/Y for these hand landmarks, so for now the
        #   palm normal is approximated in the camera/global X-Y plane as a
        #   vector perpendicular to the index-pinky line.
        #
        # Sign convention for GRF analysis:
        #   target 0 leader   -> force points toward the leader's palm
        #   target 1 follower -> force points away from the leader's palm
        #
        # Moment arm:
        #   Always from the selected dancer's COM to the leader palm point.
        sample = self.get_time_sample_for_frame(self.openhand_samples, frame_number)

        if sample is None:
            return None, None

        landmarks_by_person = self.get_nearest_hand_landmarks_for_frame(frame_number)
        lead = landmarks_by_person.get(0, {})

        right_wrist = lead.get(16)
        right_pinky = lead.get(18)
        right_index = lead.get(20)

        if right_wrist is None or right_pinky is None or right_index is None:
            return None, None

        # Approximate palm point from available pose landmarks.
        # This is closer to the palm than using wrist alone.
        palm_xy = (right_wrist + right_index + right_pinky) / 3.0

        hand_width_xy = right_index - right_pinky
        width_norm = float(np.linalg.norm(hand_width_xy))

        if width_norm > 1e-6:
            hand_width_xy = hand_width_xy / width_norm
        else:
            hand_width_xy = np.array([1.0, 0.0], dtype=float)

        # Candidate palm normal in the camera/global X-Y plane.
        normal_xy = np.array(
            [
                -float(hand_width_xy[1]),
                float(hand_width_xy[0]),
            ],
            dtype=float,
        )

        normal_norm = float(np.linalg.norm(normal_xy))

        if normal_norm > 1e-6:
            normal_xy = normal_xy / normal_norm
        else:
            normal_xy = np.array([0.0, 1.0], dtype=float)

        # Define "toward the leader palm" using wrist -> palm direction.
        wrist_to_palm_xy = palm_xy - right_wrist
        wrist_to_palm_norm = float(np.linalg.norm(wrist_to_palm_xy))

        if wrist_to_palm_norm > 1e-6:
            toward_palm_xy = wrist_to_palm_xy / wrist_to_palm_norm

            # Flip candidate normal so it points toward the leader's palm.
            if float(np.dot(normal_xy, toward_palm_xy)) < 0.0:
                normal_xy = -normal_xy

        toward_palm_axis = np.array(
            [
                float(normal_xy[0]),
                float(normal_xy[1]),
                0.0,
            ],
            dtype=float,
        )

        if self.target_person_idx == 0:
            force_axis = toward_palm_axis
        else:
            force_axis = -toward_palm_axis

        # Open hand currently uses one dominant normal-force magnitude.
        # Use signed_resultant if available, otherwise x_component.
        force_magnitude = float(sample.get("signed_resultant", sample.get("x_component", 0.0)))
        force = force_magnitude * force_axis

        point = self.landmark_xy_to_global_xyz(palm_xy, com_xyz)

        return force, point


    def get_external_hand_force_and_moment_for_frame(self, frame_number):
        # Returns total non-ground hand force and moment about COM.
        # These are external forces on the selected dancer. They are included in
        # the foot GRF balance as:
        #   FL + FR + F_hand = total_required_external_force
        #   rLxFL + rRxFR + rHxF_hand = 0
        com_xyz = self.get_nearest_com_state_for_frame(frame_number)

        total_hand_force = np.array([0.0, 0.0, 0.0], dtype=float)
        total_hand_moment = np.array([0.0, 0.0, 0.0], dtype=float)

        if self.hands_selected:
            for force, point in [
                self.get_handhold_external_force_and_point(frame_number, com_xyz),
                self.get_openhand_external_force_and_point(frame_number, com_xyz),
            ]:
                if force is None or point is None:
                    continue

                r_hand = point - com_xyz
                total_hand_force += force
                total_hand_moment += np.cross(r_hand, force)

        return total_hand_force, total_hand_moment


    def solve_per_foot_grf_placeholder(self, total_grf, frame_number):
        # Solve 6 unknowns:
        #   [LFx, LFy, LFz, RFx, RFy, RFz]
        #
        # Equations during double support:
        #   FL + FR = total external GRF demand
        #   rL x FL + rR x FR = 0
        #
        # Hand forces are intentionally not included yet.
        try:
            total_force = np.array(
                [
                    float(total_grf.get("x", 0.0)),
                    float(total_grf.get("y", 0.0)),
                    float(total_grf.get("z", 0.0)),
                ],
                dtype=float,
            )
        except (TypeError, ValueError):
            total_force = np.array([0.0, 0.0, 0.0], dtype=float)

        hand_force, hand_moment = self.get_external_hand_force_and_moment_for_frame(frame_number)

        # The feet only need to supply the remaining external force after hand
        # forces are included.
        foot_force_demand = total_force - hand_force

        contact = self.get_foot_contact_state_for_frame(frame_number)
        left_contact = contact["left_contact"]
        right_contact = contact["right_contact"]

        zero_grf = {"x": 0.0, "y": 0.0, "z": 0.0}

        if not left_contact and not right_contact:
            return zero_grf.copy(), zero_grf.copy()

        if left_contact and not right_contact:
            return {
                "x": float(foot_force_demand[0]),
                "y": max(0.0, float(foot_force_demand[1])),
                "z": float(foot_force_demand[2]),
            }, zero_grf.copy()

        if right_contact and not left_contact:
            return zero_grf.copy(), {
                "x": float(foot_force_demand[0]),
                "y": max(0.0, float(foot_force_demand[1])),
                "z": float(foot_force_demand[2]),
            }

        r_left = self.get_contact_point_relative_to_com_placeholder("left", frame_number)
        r_right = self.get_contact_point_relative_to_com_placeholder("right", frame_number)

        lx, ly, lz = r_left
        rx, ry, rz = r_right

        matrix = np.array(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],

                # Moment balance about COM: r x F = 0
                [0.0, -lz, ly, 0.0, -rz, ry],
                [lz, 0.0, -lx, rz, 0.0, -rx],
                [-ly, lx, 0.0, -ry, rx, 0.0],
            ],
            dtype=float,
        )

        # Moment equations move the known hand moment to the RHS:
        #   rLxFL + rRxFR = -rHxFhand
        rhs = np.array(
            [
                foot_force_demand[0],
                foot_force_demand[1],
                foot_force_demand[2],
                -hand_moment[0],
                -hand_moment[1],
                -hand_moment[2],
            ],
            dtype=float,
        )

        try:
            solution = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(matrix, rhs, rcond=None)[0]

        left = {
            "x": float(solution[0]),
            "y": max(0.0, float(solution[1])),
            "z": float(solution[2]),
        }

        right = {
            "x": float(solution[3]),
            "y": max(0.0, float(solution[4])),
            "z": float(solution[5]),
        }

        # If the zero-moment solve gives an impossible negative vertical force,
        # fall back to FSR-load split while preserving total force direction.
        if solution[1] < -1e-6 or solution[4] < -1e-6:
            left_fraction = 0.5
            right_fraction = 0.5

            left_load = max(float(contact.get("left_load", 0.0)), 0.0)
            right_load = max(float(contact.get("right_load", 0.0)), 0.0)
            load_sum = left_load + right_load

            if load_sum > 1e-6:
                left_fraction = left_load / load_sum
                right_fraction = right_load / load_sum

            left = {
                "x": float(foot_force_demand[0] * left_fraction),
                "y": max(0.0, float(foot_force_demand[1] * left_fraction)),
                "z": float(foot_force_demand[2] * left_fraction),
            }

            right = {
                "x": float(foot_force_demand[0] * right_fraction),
                "y": max(0.0, float(foot_force_demand[1] * right_fraction)),
                "z": float(foot_force_demand[2] * right_fraction),
            }

        return left, right


    def update_page4_grf_placeholder_values(self, frame_number=0):
        # Display estimated per-foot GRFs using COP/FSR contact state and a
        # placeholder 6-equation force/moment-balance split.
        if not self.grf_selected:
            return

        current_grf = self.get_grf_for_frame(frame_number)
        left_grf, right_grf = self.solve_per_foot_grf_placeholder(
            current_grf,
            frame_number,
        )

        self.set_numeric_display_value("LGRFX", left_grf.get("x", 0.0))
        self.set_numeric_display_value("LGRFY", left_grf.get("y", 0.0))
        self.set_numeric_display_value("LGRFZ", left_grf.get("z", 0.0))

        self.set_numeric_display_value("RGRFX", right_grf.get("x", 0.0))
        self.set_numeric_display_value("RGRFY", right_grf.get("y", 0.0))
        self.set_numeric_display_value("RGRFZ", right_grf.get("z", 0.0))

    def run_mocap_analysis(self):
        if not self.mocap_selected:
            self.playback_video_path = self.video_output_path
            return

        if not self.video_output_path.exists():
            print("ERROR: Raw mocap video not found.")
            self.playback_video_path = self.video_output_path
            return

        analysis_input_video_path = self.create_analysis_video_after_countdown()

        if self.processed_video_path.exists():
            self.processed_video_path.unlink()

        if self.processed_csv_path.exists():
            self.processed_csv_path.unlink()

        print("Running mocap analysis...")

        self.page3.Analyze.setText("Analyzing...")
        self.page3.Analyze.setEnabled(False)
        QApplication.processEvents()

        grf_flag = "1" if self.grf_selected else "0"
        grf_weight_lbf = self.get_selected_target_weight_lbf()

        if self.grf_selected and grf_weight_lbf is None:
            print("WARNING: GRF selected but target weight was not found. Using 0 lb placeholder.")
            grf_weight_lbf = 0.0

        result = subprocess.run(
            [
                sys.executable,
                "Mocap.py",
                str(analysis_input_video_path),
                str(self.target_person_idx),
                grf_flag,
                str(grf_weight_lbf if grf_weight_lbf is not None else 0.0),
            ],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )

        print(result.stdout)

        if result.stderr:
            print("Mocap.py stderr:")
            print(result.stderr)

        self.page3.Analyze.setText("Analyze")
        self.page3.Analyze.setEnabled(True)

        if result.returncode == 0 and self.processed_video_path.exists():
            print("Mocap analysis complete.")
            self.playback_video_path = self.processed_video_path
        else:
            print("Mocap analysis failed or output missing. Falling back to raw video.")
            self.playback_video_path = self.video_output_path
            self.analysis_time_offset_sec = 0.0

    def get_current_shoe_gender(self):
        # Optional future support if you add a gender selector.
        # Supported object names: ShoeGenderInput, ShoeGender, ShoeGenderCombo
        for name in ["ShoeGenderInput", "ShoeGender", "ShoeGenderCombo"]:
            if hasattr(self.page2, name):
                widget = getattr(self.page2, name)

                try:
                    if hasattr(widget, "currentText"):
                        gender_text = widget.currentText().strip().lower()
                    elif hasattr(widget, "text"):
                        gender_text = widget.text().strip().lower()
                    else:
                        continue
                except Exception:
                    continue

                if "women" in gender_text or "female" in gender_text:
                    return "women"

                if "men" in gender_text or "male" in gender_text:
                    return "men"

        # Current UI has only ShoeSizeInput, so default to men's chart.
        return "men"

    def shoe_size_to_foot_length_inches(self, shoe_size, gender=None):
        # Length size guide from the supplied table.
        # Values are foot/shoe length in inches.
        men_length_in = {
            5.0: 9.0,
            5.5: 9.2,
            6.0: 9.4,
            6.5: 9.5,
            7.0: 9.7,
            7.5: 9.8,
            8.0: 10.0,
            8.5: 10.2,
            9.0: 10.4,
            9.5: 10.5,
            10.0: 10.7,
            10.5: 10.9,
            11.0: 11.0,
            11.5: 11.2,
            12.0: 11.3,
        }

        women_length_in = {
            6.0: 9.0,
            6.5: 9.2,
            7.0: 9.4,
            7.5: 9.5,
            8.0: 9.7,
            8.5: 9.8,
            9.0: 10.0,
            9.5: 10.2,
            10.0: 10.4,
            10.5: 10.5,
            11.0: 10.7,
            11.5: 10.9,
            12.0: 11.0,
            12.5: 11.2,
            13.0: 11.3,
        }

        try:
            shoe_size = float(shoe_size)
        except (TypeError, ValueError):
            return 10.0

        gender = gender or self.get_current_shoe_gender()
        chart = women_length_in if gender == "women" else men_length_in

        # Snap to the closest half-size in the chosen chart.
        closest_size = min(chart.keys(), key=lambda size: abs(size - shoe_size))
        return chart[closest_size]

    def get_current_shoe_size(self):
        if hasattr(self.page2, "ShoeSizeInput"):
            return self.get_shoe_size_value(self.page2.ShoeSizeInput)

        if hasattr(self.page2, "ShoeSize"):
            return self.get_shoe_size_value(self.page2.ShoeSize)

        return None

    def calibrate_fsr_to_force(self, raw_value, side, fsr_index):
        try:
            raw = float(raw_value)
        except (TypeError, ValueError):
            return 0.0

        if raw <= 0:
            return 0.0

        if side == "left":
            calibration = {
                0: (0.005747, -1.408),  # FSR1 heel
                1: (0.008772,  0.482),  # FSR2 outside ball
                2: (0.005319, -2.819),  # FSR3 inside ball
                3: (0.009804, -5.245),  # FSR4 toe
            }
        else:
            calibration = {
                0: (0.007042,  0.810),  # FSR1 heel
                1: (0.004950, -0.965),  # FSR2 outside ball
                2: (0.010870, -3.696),  # FSR3 toe
                3: (0.017857, -4.643),  # FSR4 inside ball
            }

        slope, intercept = calibration[fsr_index]
        force = slope * raw + intercept

        return max(force, 0.0)

    def compute_cop_from_fsr_values(self, fsr_values, side):
        if fsr_values is None or len(fsr_values) != 4:
            return None

        shoe_size = self.get_current_shoe_size()
        shoe_gender = self.get_current_shoe_gender()
        foot_length_in = self.shoe_size_to_foot_length_inches(shoe_size, shoe_gender)

        # GitHub reference geometry converted from cm to inches:
        # original reference foot length = 25.5 cm = ~10.039 in
        # heel y = 23.0 cm = 9.055 in
        # ball y = 9.0 cm = 3.543 in
        # toe  y = 3.5 cm = 1.378 in
        # ball x = +/-2.3 cm = +/-0.906 in
        reference_foot_length_in = 25.5 / 2.54
        scale = foot_length_in / reference_foot_length_in

        heel_y = (23.0 / 2.54) * scale
        ball_y = (9.0 / 2.54) * scale
        toe_y = (3.5 / 2.54) * scale
        ball_x = (2.3 / 2.54) * scale

        if side == "left":
            # FSR1 = heel, FSR2 = outside ball, FSR3 = inside ball, FSR4 = toe
            sensor_positions = [
                (0.0, heel_y),
                (-ball_x, ball_y),
                (ball_x, ball_y),
                (0.0, toe_y),
            ]
        else:
            # FSR1 = heel, FSR2 = outside ball, FSR3 = toe, FSR4 = inside ball
            sensor_positions = [
                (0.0, heel_y),
                (ball_x, ball_y),
                (0.0, toe_y),
                (-ball_x, ball_y),
            ]

        forces = [
            self.calibrate_fsr_to_force(fsr_values[i], side, i)
            for i in range(4)
        ]

        total_force = sum(forces)

        if total_force <= 1e-6:
            return None

        cop_x_in = sum(forces[i] * sensor_positions[i][0] for i in range(4)) / total_force
        cop_y_in = sum(forces[i] * sensor_positions[i][1] for i in range(4)) / total_force

        # Convert physical COP inches to normalized drawing coordinates on the foot outline.
        # These limits intentionally match the visible foot area used by the FSR overlay,
        # not the full QLabel image box. This keeps COP points inside the actual foot.
        #
        # Approximate visible foot outline bounds:
        #   toe/top     ≈ 0.15
        #   heel/bottom ≈ 0.82
        #   left edge   ≈ 0.32
        #   right edge  ≈ 0.68
        #
        # y_in: toe is small, heel is large. Drawing y: toe near top, heel near bottom.
        foot_x_center = 0.50
        foot_half_width_norm = 0.18
        foot_top_norm = 0.15
        foot_bottom_norm = 0.82

        # Ball half-width in this code is about 0.906 in, and should map near 0.18 normalized.
        x_norm = foot_x_center + (cop_x_in / max(ball_x, 0.25)) * foot_half_width_norm

        # Map physical y directly between toe and heel drawing positions.
        toe_y_in = toe_y
        heel_y_in = heel_y
        y_ratio = (cop_y_in - toe_y_in) / max(heel_y_in - toe_y_in, 1.0)
        y_norm = foot_top_norm + y_ratio * (foot_bottom_norm - foot_top_norm)

        x_norm = max(foot_x_center - foot_half_width_norm, min(x_norm, foot_x_center + foot_half_width_norm))
        y_norm = max(foot_top_norm, min(y_norm, foot_bottom_norm))

        return {
            "cop_x_in": cop_x_in,
            "cop_y_in": cop_y_in,
            "foot_length_in": foot_length_in,
            "shoe_gender": shoe_gender,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "total_force": total_force,
        }

    def load_cop_samples_for_page4(self):
        self.left_cop_samples = self.load_single_foot_cop_samples(self.left_foot_csv_path, "left")
        self.right_cop_samples = self.load_single_foot_cop_samples(self.right_foot_csv_path, "right")

        max_time = 0.0

        if self.left_cop_samples:
            max_time = max(max_time, self.left_cop_samples[-1]["time_sec"])

        if self.right_cop_samples:
            max_time = max(max_time, self.right_cop_samples[-1]["time_sec"])

        self.cop_playback_total_frames = max(1, int(max_time * max(float(self.playback_fps), 1.0)) + 1)

        print(
            f"Loaded COP samples | left={len(self.left_cop_samples)} | right={len(self.right_cop_samples)} | frames={self.cop_playback_total_frames}"
        )

    def load_single_foot_cop_samples(self, csv_path, side):
        samples = []

        if not csv_path.exists():
            print(f"COP CSV not found for {side}: {csv_path}")
            return samples

        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    try:
                        time_sec = (
                            (
                                float(row["PC_TIME_SEC"])
                                - self.analysis_time_offset_sec
                            )
                            * self.cop_manual_time_scale
                            + self.cop_manual_sync_offset_sec
                        )

                        if time_sec < 0:
                            continue

                        fsr_values = [
                            float(row["FSR1"]),
                            float(row["FSR2"]),
                            float(row["FSR3"]),
                            float(row["FSR4"]),
                        ]
                    except (KeyError, TypeError, ValueError):
                        continue

                    cop = self.compute_cop_from_fsr_values(fsr_values, side)

                    samples.append(
                        {
                            "time_sec": time_sec,
                            "fsr_values": fsr_values,
                            "cop": cop,
                        }
                    )

        except Exception as exc:
            print(f"ERROR loading {side} COP samples: {exc}")

        return samples

    def get_cop_index_for_time(self, samples, time_sec):
        if not samples:
            return -1

        best_index = 0
        best_diff = abs(samples[0]["time_sec"] - time_sec)

        for index, sample in enumerate(samples[1:], start=1):
            diff = abs(sample["time_sec"] - time_sec)
            if diff < best_diff:
                best_diff = diff
                best_index = index
            elif sample["time_sec"] > time_sec and diff > best_diff:
                break

        return best_index

    def draw_page4_cop_overlay(self, side, samples, sample_index, target_label):
        base_pixmap = self.left_foot_outline_pixmap if side == "left" else self.right_foot_outline_pixmap

        label_width = max(target_label.width(), 200)
        label_height = max(target_label.height(), 200)

        canvas = QPixmap(label_width, label_height)
        canvas.fill(QColor("black"))

        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)

        outline_scale = 1.00
        max_outline_width = int(label_width * outline_scale)
        max_outline_height = int(label_height * outline_scale)

        if not base_pixmap.isNull():
            scaled_foot = base_pixmap.scaled(
                max_outline_width,
                max_outline_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

            if side == "right":
                scaled_foot = scaled_foot.transformed(QTransform().scale(-1, 1))

            x_offset = int((label_width - scaled_foot.width()) / 2)
            y_offset = int((label_height - scaled_foot.height()) / 2)
            painter.drawPixmap(x_offset, y_offset, scaled_foot)

            draw_width = scaled_foot.width()
            draw_height = scaled_foot.height()
        else:
            x_offset = int(label_width * (1.0 - outline_scale) / 2)
            y_offset = int(label_height * (1.0 - outline_scale) / 2)
            draw_width = int(label_width * outline_scale)
            draw_height = int(label_height * outline_scale)

        # Draw previous 3 valid COP points faintly, plus current point solid.
        trail_points = []

        if sample_index >= 0:
            start_index = max(0, sample_index - 3)

            for index in range(start_index, sample_index + 1):
                cop = samples[index].get("cop")
                if cop is None:
                    continue

                x = int(x_offset + cop["x_norm"] * draw_width)
                y = int(y_offset + cop["y_norm"] * draw_height)
                trail_points.append((x, y, index == sample_index))

        if len(trail_points) >= 2:
            for i in range(len(trail_points) - 1):
                x1, y1, _ = trail_points[i]
                x2, y2, _ = trail_points[i + 1]
                alpha_color = QColor(255, 255, 255, 90 + i * 35)
                painter.setPen(QPen(alpha_color, 2))
                painter.drawLine(x1, y1, x2, y2)

        for x, y, is_current in trail_points:
            if is_current:
                radius = max(8, int(min(draw_width, draw_height) * 0.045))
                color = QColor(255, 0, 0)
                border = QColor(255, 255, 255)
            else:
                radius = max(5, int(min(draw_width, draw_height) * 0.028))
                color = QColor(255, 255, 255, 120)
                border = QColor(255, 255, 255, 150)

            painter.setPen(QPen(border, 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)

        painter.end()
        return canvas

    def generate_all_cop_plot_frames(self):
        self.left_cop_plot_frames = {}
        self.right_cop_plot_frames = {}

        if not self.feet_selected:
            return

        total_frames = max(0, self.playback_total_frames)

        if total_frames <= 0:
            total_frames = max(0, self.cop_playback_total_frames)

        if total_frames <= 0:
            return

        print("Generating COP playback foot frames...")

        for frame_number in range(total_frames):
            time_sec = frame_number / max(float(self.playback_fps), 1.0)

            if hasattr(self.page4, "LeftFootAnimation"):
                left_index = self.get_cop_index_for_time(self.left_cop_samples, time_sec)
                self.left_cop_plot_frames[frame_number] = self.draw_page4_cop_overlay(
                    "left",
                    self.left_cop_samples,
                    left_index,
                    self.page4.LeftFootAnimation,
                )

            if hasattr(self.page4, "RightFootAnimation"):
                right_index = self.get_cop_index_for_time(self.right_cop_samples, time_sec)
                self.right_cop_plot_frames[frame_number] = self.draw_page4_cop_overlay(
                    "right",
                    self.right_cop_samples,
                    right_index,
                    self.page4.RightFootAnimation,
                )

        print(
            f"Generated COP playback frames | left={len(self.left_cop_plot_frames)} | right={len(self.right_cop_plot_frames)}"
        )

    def update_page4_cop_views(self, frame_number):
        if not self.feet_selected:
            return

        frame_number = int(max(0, frame_number))

        if hasattr(self.page4, "LeftFootAnimation"):
            left_pixmap = self.left_cop_plot_frames.get(frame_number)

            if left_pixmap is None:
                time_sec = frame_number / max(float(self.playback_fps), 1.0)
                left_index = self.get_cop_index_for_time(self.left_cop_samples, time_sec)
                left_pixmap = self.draw_page4_cop_overlay(
                    "left",
                    self.left_cop_samples,
                    left_index,
                    self.page4.LeftFootAnimation,
                )

            self.page4.LeftFootAnimation.setText("")
            self.page4.LeftFootAnimation.setPixmap(left_pixmap)
            self.page4.LeftFootAnimation.repaint()

        if hasattr(self.page4, "RightFootAnimation"):
            right_pixmap = self.right_cop_plot_frames.get(frame_number)

            if right_pixmap is None:
                time_sec = frame_number / max(float(self.playback_fps), 1.0)
                right_index = self.get_cop_index_for_time(self.right_cop_samples, time_sec)
                right_pixmap = self.draw_page4_cop_overlay(
                    "right",
                    self.right_cop_samples,
                    right_index,
                    self.page4.RightFootAnimation,
                )

            self.page4.RightFootAnimation.setText("")
            self.page4.RightFootAnimation.setPixmap(right_pixmap)
            self.page4.RightFootAnimation.repaint()

    def clear_com_plots(self):
        self.top_view_plot_frames = {}
        self.camera_view_plot_frames = {}
        self.plot_available_frames = []
        self.left_cop_plot_frames = {}
        self.right_cop_plot_frames = {}

        if hasattr(self.page4, "LeftFootAnimation_2"):
            self.page4.LeftFootAnimation_2.setText("Top View COM plot will appear here")
            self.page4.LeftFootAnimation_2.setPixmap(QPixmap())

        if hasattr(self.page4, "LeftFootAnimation_3"):
            self.page4.LeftFootAnimation_3.setText("Camera View COM plot will appear here")
            self.page4.LeftFootAnimation_3.setPixmap(QPixmap())

        if hasattr(self.page4, "LeftFootAnimation"):
            self.page4.LeftFootAnimation.setText("Left COP will appear here")
            self.page4.LeftFootAnimation.setPixmap(QPixmap())

        if hasattr(self.page4, "RightFootAnimation"):
            self.page4.RightFootAnimation.setText("Right COP will appear here")
            self.page4.RightFootAnimation.setPixmap(QPixmap())

    def load_com_data_for_plots(self):
        self.com_frames = []
        self.com_x = []
        self.com_y = []
        self.com_z = []
        self.camera_plot_limits = None
        self.top_plot_limits = None
        self.top_view_plot_frames = {}
        self.camera_view_plot_frames = {}
        self.plot_available_frames = []
        self.foot_landmarks_by_frame = {}
        self.hand_landmarks_by_frame = {}

        if not self.processed_csv_path.exists():
            print("COM CSV not found. Plots unavailable.")
            self.clear_com_plots()
            return

        try:
            with open(self.processed_csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    try:
                        person_id = int(float(row.get("PERSON", "-1")))
                    except ValueError:
                        continue

                    row_type = row.get("TYPE")

                    if row_type == "COM":
                        if person_id != self.target_person_idx:
                            continue

                        try:
                            frame = int(float(row["FRAME"]))
                            x = float(row["X_WORLD_FT_APPROX"])
                            y = float(row["Y_WORLD_FT_APPROX"])
                            z = float(row["Z_TORSO_FT_APPROX"])
                        except (KeyError, TypeError, ValueError):
                            continue

                        self.com_frames.append(frame)
                        self.com_x.append(x)
                        self.com_y.append(y)
                        self.com_z.append(z)

                    elif row_type == "JOINT":
                        try:
                            frame = int(float(row["FRAME"]))
                            joint_id = int(float(row["JOINT"]))

                            # Mocap.py writes camera/global relative joint positions.
                            # X_REL_FT/Y_REL_FT are available in the current CSV.
                            x_rel_ft = float(row["X_REL_FT"])
                            y_rel_ft = float(row["Y_REL_FT"])
                        except (KeyError, TypeError, ValueError):
                            continue

                        # MediaPipe Pose landmark IDs used here:
                        # 15 = LEFT_WRIST, 16 = RIGHT_WRIST
                        # 17 = LEFT_PINKY, 18 = RIGHT_PINKY
                        # 19 = LEFT_INDEX, 20 = RIGHT_INDEX
                        # 29 = LEFT_HEEL, 30 = RIGHT_HEEL
                        # 31 = LEFT_FOOT_INDEX, 32 = RIGHT_FOOT_INDEX.
                        relevant_hand_joint_ids = [15, 16, 17, 18, 19, 20]
                        relevant_foot_joint_ids = [29, 30, 31, 32]

                        if joint_id in relevant_hand_joint_ids:
                            if frame not in self.hand_landmarks_by_frame:
                                self.hand_landmarks_by_frame[frame] = {}

                            if person_id not in self.hand_landmarks_by_frame[frame]:
                                self.hand_landmarks_by_frame[frame][person_id] = {}

                            self.hand_landmarks_by_frame[frame][person_id][joint_id] = np.array(
                                [x_rel_ft, y_rel_ft],
                                dtype=float,
                            )

                        # Foot GRF moment arms use the currently analyzed target's feet.
                        if person_id != self.target_person_idx:
                            continue

                        if joint_id not in relevant_foot_joint_ids:
                            continue

                        if frame not in self.foot_landmarks_by_frame:
                            self.foot_landmarks_by_frame[frame] = {}

                        self.foot_landmarks_by_frame[frame][joint_id] = np.array(
                            [x_rel_ft, y_rel_ft],
                            dtype=float,
                        )

        except Exception as exc:
            print(f"ERROR loading COM CSV: {exc}")
            self.clear_com_plots()
            return

        if not self.com_frames:
            print("No calibrated COM data found for plots.")
            self.clear_com_plots()
            return

        self.camera_plot_limits = (-5, 15, 0, 4)
        self.top_plot_limits = (-5, 15, -3, 3)

        print(f"Loaded {len(self.com_frames)} COM points for plotting.")
        self.generate_all_plot_frames()

    def compute_plot_limits(self, xs, ys):
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        x_span = max(max_x - min_x, 0.1)
        y_span = max(max_y - min_y, 0.1)

        x_pad = x_span * 0.15
        y_pad = y_span * 0.15

        return (
            min_x - x_pad,
            max_x + x_pad,
            min_y - y_pad,
            max_y + y_pad,
        )


    def generate_all_plot_frames(self):
        self.top_view_plot_frames = {}
        self.camera_view_plot_frames = {}
        self.plot_available_frames = []

        if not self.com_frames:
            return

        print("Generating COM playback plot frames...")

        top_label_width = 400
        top_label_height = 300
        cam_label_width = 400
        cam_label_height = 300

        if hasattr(self.page4, "LeftFootAnimation_2"):
            top_label_width = max(self.page4.LeftFootAnimation_2.width(), 300)
            top_label_height = max(self.page4.LeftFootAnimation_2.height(), 100)

        if hasattr(self.page4, "LeftFootAnimation_3"):
            cam_label_width = max(self.page4.LeftFootAnimation_3.width(), 300)
            cam_label_height = max(self.page4.LeftFootAnimation_3.height(), 100)

        for index, frame_number in enumerate(self.com_frames):
            top_figure = Figure(
                figsize=(top_label_width / 100.0, top_label_height / 100.0),
                dpi=100,
                facecolor="black",
            )

            top_axes = top_figure.add_subplot(111)
            top_axes.set_facecolor("black")

            past_x = self.com_x[: index + 1]
            past_z = self.com_z[: index + 1]

            top_axes.plot(
                past_x,
                past_z,
                color="white",
                linewidth=1.5,
                alpha=0.85,
            )

            top_axes.scatter(
                past_x,
                past_z,
                facecolors="none",
                edgecolors="#B0B0B0",
                s=18,
                linewidths=0.8,
            )

            top_axes.scatter(
                [self.com_x[index]],
                [self.com_z[index]],
                color="red",
                s=42,
                zorder=5,
            )

            top_axes.set_title("Top View: X vs Approx Z", color="white", fontsize=8, pad=2)
            top_axes.set_xlabel("X (ft)", color="white", fontsize=7, labelpad=1)
            top_axes.set_ylabel("Z approx (ft)", color="white", fontsize=7, labelpad=1)
            top_axes.tick_params(axis="both", colors="white", labelsize=6)

            for spine in top_axes.spines.values():
                spine.set_color("white")

            if self.top_plot_limits is not None:
                x_min, x_max, y_min, y_max = self.top_plot_limits
                top_axes.set_xlim(x_min, x_max)
                top_axes.set_ylim(y_min, y_max)
                top_axes.set_aspect("equal", adjustable="box")

            top_axes.grid(True, color="#333333", linewidth=0.5)
            top_figure.tight_layout(pad=0.4)

            self.top_view_plot_frames[frame_number] = self.matplotlib_figure_to_pixmap(top_figure)

            cam_figure = Figure(
                figsize=(cam_label_width / 100.0, cam_label_height / 100.0),
                dpi=100,
                facecolor="black",
            )

            cam_axes = cam_figure.add_subplot(111)
            cam_axes.set_facecolor("black")

            past_x = self.com_x[: index + 1]
            past_y = self.com_y[: index + 1]

            cam_axes.plot(
                past_x,
                past_y,
                color="white",
                linewidth=1.5,
                alpha=0.85,
            )

            cam_axes.scatter(
                past_x,
                past_y,
                facecolors="none",
                edgecolors="#B0B0B0",
                s=18,
                linewidths=0.8,
            )

            cam_axes.scatter(
                [self.com_x[index]],
                [self.com_y[index]],
                color="red",
                s=42,
                zorder=5,
            )

            cam_axes.set_title("Camera View: Approx X vs Approx Y", color="white", fontsize=8, pad=2)
            cam_axes.set_xlabel("Approx X (ft)", color="white", fontsize=7, labelpad=1)
            cam_axes.set_ylabel("Approx Y (ft)", color="white", fontsize=7, labelpad=1)
            cam_axes.tick_params(axis="both", colors="white", labelsize=6)

            for spine in cam_axes.spines.values():
                spine.set_color("white")

            if self.camera_plot_limits is not None:
                x_min, x_max, y_min, y_max = self.camera_plot_limits
                cam_axes.set_xlim(x_min, x_max)
                cam_axes.set_ylim(y_min, y_max)
                cam_axes.set_aspect("equal", adjustable="box")

            cam_axes.grid(True, color="#333333", linewidth=0.5)
            cam_figure.tight_layout(pad=0.4)

            self.camera_view_plot_frames[frame_number] = self.matplotlib_figure_to_pixmap(cam_figure)

        self.plot_available_frames = sorted(self.top_view_plot_frames.keys())
        print(f"Generated {len(self.plot_available_frames)} COM plot playback frames.")
    def get_com_index_for_frame(self, frame_number):
        if not self.com_frames:
            return -1

        current_index = -1

        for index, frame in enumerate(self.com_frames):
            if frame <= frame_number:
                current_index = index
            else:
                break

        return current_index

    def update_com_plots(self, frame_number):
        if not self.plot_available_frames:
            return

        selected_frame = None

        for frame in self.plot_available_frames:
            if frame <= frame_number:
                selected_frame = frame
            else:
                break

        if selected_frame is None:
            return

        top_pixmap = self.top_view_plot_frames.get(selected_frame)
        cam_pixmap = self.camera_view_plot_frames.get(selected_frame)

        if top_pixmap is not None:
            self.page4.LeftFootAnimation_2.setPixmap(
                top_pixmap.scaled(
                    self.page4.LeftFootAnimation_2.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

        if cam_pixmap is not None:
            self.page4.LeftFootAnimation_3.setPixmap(
                cam_pixmap.scaled(
                    self.page4.LeftFootAnimation_3.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    def clear_hand_plots(self):
        self.hand_static_plot_pixmaps = {}

        if hasattr(self.page4, "Handheldplot"):
            self.page4.Handheldplot.setText("Handheld force plot will appear here")
            self.page4.Handheldplot.setPixmap(QPixmap())

        if hasattr(self.page4, "Openhandplot"):
            self.page4.Openhandplot.setText("Open hand force plot will appear here")
            self.page4.Openhandplot.setPixmap(QPixmap())

        for name, message in [
            ("Openprimaryplot", "Open hand compression/tension force plot will appear here"),
            ("Openverticalplot", "Open hand vertical force plot will appear here"),
            ("Openlateralplot", "Open hand lateral force plot will appear here"),
            ("openprimaryplot", "Open hand compression/tension force plot will appear here"),
            ("openverticalplot", "Open hand vertical force plot will appear here"),
            ("openlateralplot", "Open hand lateral force plot will appear here"),
        ]:
            if hasattr(self.page4, name):
                widget = getattr(self.page4, name)
                widget.setText(message)
                widget.setPixmap(QPixmap())

        if hasattr(self.page4, "primaryplot"):
            self.page4.primaryplot.setText("Handheld compression/tension force plot will appear here")
            self.page4.primaryplot.setPixmap(QPixmap())

        if hasattr(self.page4, "verticalplot"):
            self.page4.verticalplot.setText("Handheld vertical force plot will appear here")
            self.page4.verticalplot.setPixmap(QPixmap())

        if hasattr(self.page4, "lateralplot"):
            self.page4.lateralplot.setText("Handheld lateral force plot will appear here")
            self.page4.lateralplot.setPixmap(QPixmap())

    def load_single_hand_samples(self, csv_path, device_label):
        samples = []

        if not csv_path.exists():
            print(f"{device_label} CSV not found: {csv_path}")
            return samples

        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    try:
                        time_sec = (
                            (
                                float(row["PC_TIME_SEC"])
                                - self.analysis_time_offset_sec
                            )
                            * self.hand_manual_time_scale
                            + self.hand_manual_sync_offset_sec
                        )

                        if time_sec < 0:
                            continue

                        x_component = float(row.get("X_COMPONENT", 0.0))
                        y_component = float(row.get("Y_COMPONENT", 0.0))
                        z_component = float(row.get("Z_COMPONENT", 0.0))
                        raw_sum = float(row.get("SUM", x_component))
                        roll = float(row.get("ROLL", 0.0))
                        resultant = float(row.get("RESULTANT", 0.0))
                        signed_resultant = float(row.get("SIGNED_RESULTANT", 0.0))
                        vertical_component = float(row.get("VERTICAL_COMPONENT", y_component))
                        lateral_component = float(row.get("LATERAL_COMPONENT", z_component))
                    except (KeyError, TypeError, ValueError):
                        continue

                    samples.append(
                        {
                            "time_sec": time_sec,
                            "x_component": x_component,
                            "y_component": y_component,
                            "z_component": z_component,
                            "roll": roll,
                            "sum": raw_sum,
                            "resultant": resultant,
                            "signed_resultant": signed_resultant,
                            "vertical_component": vertical_component,
                            "lateral_component": lateral_component,
                        }
                    )

            print(f"Loaded {len(samples)} {device_label} force samples.")

        except Exception as exc:
            print(f"ERROR loading {device_label} samples: {exc}")
            return []

        return samples

    def load_openhand_samples_for_page4(self):
        self.openhand_samples = self.load_single_hand_samples(self.openhand_csv_path, "open hand")

    def load_handhold_samples_for_page4(self):
        self.handhold_samples = self.load_single_hand_samples(self.handhold_csv_path, "handhold")

        if not self.handhold_samples:
            self.clear_hand_plots()
            return

    def get_hand_plot_y_limits(self, values):
        clean_values = []

        for value in values:
            try:
                clean_values.append(float(value))
            except (TypeError, ValueError):
                continue

        if not clean_values:
            return (-1.0, 1.0)

        min_value = min(clean_values)
        max_value = max(clean_values)

        if abs(min_value) < 1e-9 and abs(max_value) < 1e-9:
            return (-1.0, 1.0)

        max_abs = max(abs(min_value), abs(max_value))
        pad = max(max_abs * 0.15, 0.5)

        return (-(max_abs + pad), max_abs + pad)

    def draw_hand_time_plot(self, target_label, times, values, title, ylabel, y_limits=None, cursor_time_sec=None):
        label_width = max(target_label.width(), 300)
        label_height = max(target_label.height(), 100)

        figure = Figure(
            figsize=(label_width / 100.0, label_height / 100.0),
            dpi=100,
            facecolor="black",
        )

        axes = figure.add_subplot(111)
        axes.set_facecolor("black")

        axes.plot(times, values, color="white", linewidth=1.5)
        axes.axhline(0, color="#666666", linewidth=0.8)

        if cursor_time_sec is not None and times:
            cursor_time_sec = max(min(float(cursor_time_sec), max(times)), min(times))
            axes.axvline(cursor_time_sec, color="red", linewidth=1.4)

        if y_limits is not None:
            axes.set_ylim(y_limits[0], y_limits[1])

        axes.set_title(title, color="white", fontsize=8, pad=2)
        axes.set_xlabel("Time (s)", color="white", fontsize=7, labelpad=1)
        axes.set_ylabel(ylabel, color="white", fontsize=7, labelpad=1)
        axes.tick_params(axis="both", colors="white", labelsize=6)

        for spine in axes.spines.values():
            spine.set_color("white")

        axes.grid(True, color="#333333", linewidth=0.5)
        figure.tight_layout(pad=0.4)

        return self.matplotlib_figure_to_pixmap(figure)

    def plot_hand_samples_to_widgets(self, samples, widget_names, title_prefix, y_limit_map=None, cursor_time_sec=None):
        if not samples:
            return

        if y_limit_map is None:
            y_limit_map = {}

        times = [sample["time_sec"] for sample in samples]
        sum_values = [sample["sum"] for sample in samples]
        primary_values = [sample["x_component"] for sample in samples]
        vertical_values = [sample["vertical_component"] for sample in samples]
        lateral_values = [sample["lateral_component"] for sample in samples]

        resultant_names, primary_names, vertical_names, lateral_names = widget_names

        for name in resultant_names:
            if hasattr(self.page4, name):
                widget = getattr(self.page4, name)
                pixmap = self.draw_hand_time_plot(
                    widget,
                    times,
                    sum_values,
                    f"{title_prefix}: Tension / Compression Force",
                    "Force",
                    y_limit_map.get("sum"),
                    cursor_time_sec,
                )
                widget.setText("")
                widget.setPixmap(pixmap)
                widget.repaint()

        for name in primary_names:
            if hasattr(self.page4, name):
                widget = getattr(self.page4, name)
                pixmap = self.draw_hand_time_plot(
                    widget,
                    times,
                    primary_values,
                    f"{title_prefix}: Primary Force",
                    "Force",
                    y_limit_map.get("primary"),
                    cursor_time_sec,
                )
                widget.setText("")
                widget.setPixmap(pixmap)
                widget.repaint()

        for name in vertical_names:
            if hasattr(self.page4, name):
                widget = getattr(self.page4, name)
                pixmap = self.draw_hand_time_plot(
                    widget,
                    times,
                    vertical_values,
                    f"{title_prefix}: Calibrated Vertical Force",
                    "Force",
                    y_limit_map.get("vertical"),
                    cursor_time_sec,
                )
                widget.setText("")
                widget.setPixmap(pixmap)
                widget.repaint()

        for name in lateral_names:
            if hasattr(self.page4, name):
                widget = getattr(self.page4, name)
                pixmap = self.draw_hand_time_plot(
                    widget,
                    times,
                    lateral_values,
                    f"{title_prefix}: Calibrated Lateral Force",
                    "Force",
                    y_limit_map.get("lateral"),
                    cursor_time_sec,
                )
                widget.setText("")
                widget.setPixmap(pixmap)
                widget.repaint()

    def render_hand_static_plot_pixmaps(self, samples, widget_names, title_prefix, y_limit_map=None):
        pixmaps = {}

        if not samples:
            return pixmaps

        if y_limit_map is None:
            y_limit_map = {}

        times = [sample["time_sec"] for sample in samples]
        sum_values = [sample["sum"] for sample in samples]
        primary_values = [sample["x_component"] for sample in samples]
        vertical_values = [sample["vertical_component"] for sample in samples]
        lateral_values = [sample["lateral_component"] for sample in samples]

        resultant_names, primary_names, vertical_names, lateral_names = widget_names

        for name in resultant_names:
            if hasattr(self.page4, name):
                widget = getattr(self.page4, name)
                pixmaps[name] = {
                    "pixmap": self.draw_hand_time_plot(
                        widget,
                        times,
                        sum_values,
                        f"{title_prefix}: Tension / Compression Force",
                        "Force",
                        y_limit_map.get("sum"),
                        None,
                    ),
                    "time_min": min(times),
                    "time_max": max(times),
                }

        for name in primary_names:
            if hasattr(self.page4, name):
                widget = getattr(self.page4, name)
                pixmaps[name] = {
                    "pixmap": self.draw_hand_time_plot(
                        widget,
                        times,
                        primary_values,
                        f"{title_prefix}: Primary Force",
                        "Force",
                        y_limit_map.get("primary"),
                        None,
                    ),
                    "time_min": min(times),
                    "time_max": max(times),
                }

        for name in vertical_names:
            if hasattr(self.page4, name):
                widget = getattr(self.page4, name)
                pixmaps[name] = {
                    "pixmap": self.draw_hand_time_plot(
                        widget,
                        times,
                        vertical_values,
                        f"{title_prefix}: Calibrated Vertical Force",
                        "Force",
                        y_limit_map.get("vertical"),
                        None,
                    ),
                    "time_min": min(times),
                    "time_max": max(times),
                }

        for name in lateral_names:
            if hasattr(self.page4, name):
                widget = getattr(self.page4, name)
                pixmaps[name] = {
                    "pixmap": self.draw_hand_time_plot(
                        widget,
                        times,
                        lateral_values,
                        f"{title_prefix}: Calibrated Lateral Force",
                        "Force",
                        y_limit_map.get("lateral"),
                        None,
                    ),
                    "time_min": min(times),
                    "time_max": max(times),
                }

        return pixmaps

    def get_page4_hand_y_limits(self):
        hh_y_limits = {}

        if self.handhold_samples:
            hh_sum_values = [sample["sum"] for sample in self.handhold_samples]
            hh_primary_values = [sample["x_component"] for sample in self.handhold_samples]
            hh_vertical_values = [sample["vertical_component"] for sample in self.handhold_samples]
            hh_lateral_values = [sample["lateral_component"] for sample in self.handhold_samples]

            hh_y_limits = {
                "sum": self.get_hand_plot_y_limits(hh_sum_values),
                "primary": self.get_hand_plot_y_limits(hh_primary_values),
                "vertical": self.get_hand_plot_y_limits(hh_vertical_values),
                "lateral": self.get_hand_plot_y_limits(hh_lateral_values),
            }

        return hh_y_limits

    def generate_hand_static_plot_pixmaps(self):
        self.hand_static_plot_pixmaps = {}

        if not self.hands_selected:
            return

        if not self.handhold_samples and not self.openhand_samples:
            return

        hh_y_limits = self.get_page4_hand_y_limits()

        print("Generating static hand plots...")

        self.hand_static_plot_pixmaps.update(
            self.render_hand_static_plot_pixmaps(
                self.handhold_samples,
                (
                    ["Handheldplot"],
                    ["primaryplot"],
                    ["verticalplot"],
                    ["lateralplot"],
                ),
                "Handhold",
                hh_y_limits,
            )
        )

        self.hand_static_plot_pixmaps.update(
            self.render_hand_static_plot_pixmaps(
                self.openhand_samples,
                (
                    ["Openhandplot"],
                    ["Openprimaryplot", "openprimaryplot"],
                    ["Openverticalplot", "openverticalplot"],
                    ["Openlateralplot", "openlateralplot"],
                ),
                "Open Hand",
                hh_y_limits,
            )
        )

        print(f"Generated {len(self.hand_static_plot_pixmaps)} static hand plots.")

    def draw_cursor_on_hand_pixmap(self, base_pixmap, time_min, time_max, cursor_time_sec):
        if base_pixmap is None or base_pixmap.isNull():
            return QPixmap()

        output = QPixmap(base_pixmap)

        if time_max <= time_min:
            return output

        cursor_time_sec = max(min(float(cursor_time_sec), float(time_max)), float(time_min))
        ratio = (cursor_time_sec - time_min) / (time_max - time_min)

        # Approximate the actual matplotlib plotting region inside the pixmap.
        # This avoids expensive matplotlib redraws and keeps the red cursor close
        # to the true x-axis location.
        left_margin = int(output.width() * 0.12)
        right_margin = int(output.width() * 0.04)
        top_margin = int(output.height() * 0.14)
        bottom_margin = int(output.height() * 0.22)

        x = int(left_margin + ratio * max(1, output.width() - left_margin - right_margin))
        y1 = top_margin
        y2 = max(y1 + 1, output.height() - bottom_margin)

        painter = QPainter(output)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("red"), 2))
        painter.drawLine(x, y1, x, y2)
        painter.end()

        return output

    def update_page4_hand_plot_frames(self, frame_number):
        if not self.hands_selected:
            return

        if not self.hand_static_plot_pixmaps:
            return

        cursor_time_sec = max(0.0, float(frame_number) / max(float(self.playback_fps), 1.0))

        for widget_name, info in self.hand_static_plot_pixmaps.items():
            if not hasattr(self.page4, widget_name):
                continue

            widget = getattr(self.page4, widget_name)
            pixmap = self.draw_cursor_on_hand_pixmap(
                info["pixmap"],
                info["time_min"],
                info["time_max"],
                cursor_time_sec,
            )

            widget.setText("")
            widget.setPixmap(pixmap)
            widget.repaint()

    def update_page4_hand_plots(self):
        if not self.hands_selected:
            self.clear_hand_plots()
            return

        if not self.handhold_samples and not self.openhand_samples:
            self.clear_hand_plots()
            return

        hh_y_limits = self.get_page4_hand_y_limits()
        cursor_time_sec = self.page4_current_time_sec

        self.plot_hand_samples_to_widgets(
            self.handhold_samples,
            (
                ["Handheldplot"],
                ["primaryplot"],
                ["verticalplot"],
                ["lateralplot"],
            ),
            "Handhold",
            hh_y_limits,
            cursor_time_sec,
        )

        self.plot_hand_samples_to_widgets(
            self.openhand_samples,
            (
                ["Openhandplot"],
                ["Openprimaryplot", "openprimaryplot"],
                ["Openverticalplot", "openverticalplot"],
                ["Openlateralplot", "openlateralplot"],
            ),
            "Open Hand",
            hh_y_limits,
            cursor_time_sec,
        )

    def update_page4_time_linked_plots(self, frame_number):
        self.page4_current_time_sec = max(0.0, float(frame_number) / max(float(self.playback_fps), 1.0))

        if self.hands_selected and (self.handhold_samples or self.openhand_samples):
            self.update_page4_hand_plot_frames(frame_number)

    def get_valid_video_fps(self, capture, fallback=20.0):
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        except Exception:
            fps = float(fallback)

        if fps is None or fps <= 1e-6 or math.isnan(fps) or math.isinf(fps):
            fps = float(fallback)

        return fps


    def start_page4_playback(self):
        self.stop_page2_preview()
        self.stop_mocap_recording()
        self.stop_page4_playback()
        self.page4_current_time_sec = 0.0

        if self.mocap_selected:
            self.load_com_data_for_plots()
        else:
            self.clear_com_plots()

        if self.feet_selected:
            self.load_cop_samples_for_page4()

        if self.hands_selected:
            self.load_handhold_samples_for_page4()
            self.load_openhand_samples_for_page4()
        else:
            self.clear_hand_plots()

        # Feet-only mode: no video, but keep slider/play enabled for COP animation.
        if not self.mocap_selected:
            self.page4.VideoFeed.setText("Mocap not selected")

            if self.feet_selected and self.cop_playback_total_frames > 1:
                self.playback_total_frames = self.cop_playback_total_frames
                self.playback_fps = 20.0
                self.playback_interval_ms = 50

                self.page4.horizontalSlider.setMinimum(0)
                self.page4.horizontalSlider.setMaximum(max(0, self.playback_total_frames - 1))
                self.page4.horizontalSlider.setValue(0)
                self.page4.horizontalSlider.setEnabled(True)

                self.playback_playing = False
                self.page4.PlayPause.setEnabled(True)
                self.page4.PlayPause.setText("▶")
                self.generate_all_cop_plot_frames()
                self.generate_hand_static_plot_pixmaps()
                self.update_page4_cop_views(0)
                self.update_page4_time_linked_plots(0)
            else:
                self.generate_hand_static_plot_pixmaps()
                self.update_page4_time_linked_plots(0)
                self.page4.horizontalSlider.setEnabled(False)
                self.page4.PlayPause.setEnabled(False)
                self.page4.PlayPause.setText("▶")

            return

        if not self.playback_video_path.exists():
            self.page4.VideoFeed.setText("No video found")
            self.page4.horizontalSlider.setEnabled(False)
            self.page4.PlayPause.setEnabled(False)
            self.page4.PlayPause.setText("▶")
            self.generate_hand_static_plot_pixmaps()
            self.update_page4_time_linked_plots(0)
            return

        self.playback_camera = cv2.VideoCapture(str(self.playback_video_path))

        if not self.playback_camera.isOpened():
            self.page4.VideoFeed.setText("ERROR: Could not open playback video")
            self.page4.horizontalSlider.setEnabled(False)
            self.page4.PlayPause.setEnabled(False)
            self.page4.PlayPause.setText("▶")
            self.generate_hand_static_plot_pixmaps()
            self.update_page4_time_linked_plots(0)
            return

        self.playback_total_frames = int(self.playback_camera.get(cv2.CAP_PROP_FRAME_COUNT))

        # Use the actual FPS stored in the playback video file instead of
        # assuming exactly 20 Hz. This keeps frame_number / playback_fps on the
        # same timeline as the video and reduces drift against COP/hand data.
        self.playback_fps = self.get_valid_video_fps(self.playback_camera, fallback=20.0)
        self.playback_interval_ms = max(1, int(round(1000.0 / self.playback_fps)))

        print(f"Playback FPS from video metadata: {self.playback_fps:.3f}")

        self.page4.horizontalSlider.setMinimum(0)
        self.page4.horizontalSlider.setMaximum(max(0, self.playback_total_frames - 1))
        self.page4.horizontalSlider.setValue(0)
        self.page4.horizontalSlider.setEnabled(True)

        self.playback_playing = False
        self.page4.PlayPause.setEnabled(True)
        self.page4.PlayPause.setText("▶")

        self.generate_hand_static_plot_pixmaps()
        self.show_playback_frame(0)

    def show_playback_frame(self, frame_number):
        if self.playback_camera is None:
            return

        self.playback_camera.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = self.playback_camera.read()

        if not success:
            return

        frame = self.draw_axis_indicator(frame)
        self.page4.VideoFeed.setPixmap(self.cv_frame_to_pixmap(frame, self.page4.VideoFeed))

        self.page4.horizontalSlider.blockSignals(True)
        self.page4.horizontalSlider.setValue(frame_number)
        self.page4.horizontalSlider.blockSignals(False)

        self.update_com_plots(frame_number)
        self.update_page4_cop_views(frame_number)
        self.update_page4_time_linked_plots(frame_number)
        self.update_page4_grf_placeholder_values(frame_number)

        self.playback_camera.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def toggle_playback(self):
        if self.playback_camera is None and not (self.feet_selected and self.playback_total_frames > 1):
            return

        self.playback_playing = not self.playback_playing

        if self.playback_playing:
            self.page4.PlayPause.setText("⏸")
            self.playback_timer.start(self.playback_interval_ms)
        else:
            self.page4.PlayPause.setText("▶")
            self.playback_timer.stop()

    def update_playback_frame(self):
        # Feet-only COP playback mode: advance slider and COP overlays without video.
        if self.playback_camera is None:
            if self.feet_selected and self.playback_total_frames > 1:
                current_frame = self.page4.horizontalSlider.value() + 1

                if current_frame >= self.playback_total_frames:
                    self.playback_timer.stop()
                    self.playback_playing = False
                    self.page4.PlayPause.setText("▶")
                    current_frame = self.playback_total_frames - 1

                if not self.playback_user_dragging:
                    self.page4.horizontalSlider.blockSignals(True)
                    self.page4.horizontalSlider.setValue(current_frame)
                    self.page4.horizontalSlider.blockSignals(False)

                self.update_page4_cop_views(current_frame)
                self.update_page4_time_linked_plots(current_frame)
                self.update_page4_grf_placeholder_values(current_frame)

            return

        success, frame = self.playback_camera.read()

        if not success:
            self.playback_timer.stop()
            self.playback_playing = False
            self.page4.PlayPause.setText("▶")

            final_frame = max(0, self.playback_total_frames - 1)
            self.show_playback_frame(final_frame)
            return

        current_frame = int(self.playback_camera.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        current_frame = max(0, current_frame)

        if not self.playback_user_dragging:
            self.page4.horizontalSlider.blockSignals(True)
            self.page4.horizontalSlider.setValue(current_frame)
            self.page4.horizontalSlider.blockSignals(False)

        frame = self.draw_axis_indicator(frame)
        self.page4.VideoFeed.setPixmap(self.cv_frame_to_pixmap(frame, self.page4.VideoFeed))
        self.update_com_plots(current_frame)
        self.update_page4_cop_views(current_frame)
        self.update_page4_time_linked_plots(current_frame)
        self.update_page4_grf_placeholder_values(current_frame)

    def playback_slider_pressed(self):
        self.playback_user_dragging = True

    def playback_slider_released(self):
        self.playback_user_dragging = False
        self.seek_playback(self.page4.horizontalSlider.value())

    def playback_slider_changed(self, value):
        if self.playback_user_dragging:
            self.seek_playback(value)

    def seek_playback(self, frame_number):
        # Feet-only COP seek mode.
        if self.playback_camera is None:
            if self.feet_selected:
                was_playing = self.playback_playing
                self.playback_timer.stop()
                self.update_page4_cop_views(frame_number)
                self.update_page4_time_linked_plots(frame_number)
                self.update_page4_grf_placeholder_values(frame_number)

                self.page4.horizontalSlider.blockSignals(True)
                self.page4.horizontalSlider.setValue(frame_number)
                self.page4.horizontalSlider.blockSignals(False)

                if was_playing:
                    self.playback_timer.start(self.playback_interval_ms)

            return

        was_playing = self.playback_playing

        self.playback_timer.stop()
        self.show_playback_frame(frame_number)

        if was_playing:
            self.playback_timer.start(self.playback_interval_ms)

    def stop_page4_playback(self):
        self.playback_timer.stop()
        self.playback_playing = False

        if hasattr(self.page4, "PlayPause"):
            self.page4.PlayPause.setText("▶")

        if self.playback_camera is not None:
            self.playback_camera.release()
            self.playback_camera = None

    def sync_mocap(self):
        if not self.mocap_selected:
            return

        self.mocap_synced = True
        self.set_indicator_pair(self.page2.MoCapRed, self.page2.MoCapGreen, True)
        self.update_record_button_state()

    def set_hand_sync_status(self, message):
        print(message)

        for name in ["HHSyncStatus", "HandSyncStatus", "HandsSyncStatus", "HandStatusText"]:
            if hasattr(self.page2, name):
                widget = getattr(self.page2, name)

                widget.setStyleSheet("border: none; background: transparent; font-style: italic; color: rgb(220, 220, 220);")

                if hasattr(widget, "append"):
                    widget.append(message)
                elif hasattr(widget, "setPlainText"):
                    current = widget.toPlainText()
                    widget.setPlainText((current + "\n" + message).strip())
                elif hasattr(widget, "setText"):
                    current = widget.text()
                    if current:
                        widget.setText(current + "\n" + message)
                    else:
                        widget.setText(message)

                QApplication.processEvents()
                return

        QApplication.processEvents()

    def clear_hand_sync_status(self):
        for name in ["HHSyncStatus", "HandSyncStatus", "HandsSyncStatus", "HandStatusText"]:
            if hasattr(self.page2, name):
                widget = getattr(self.page2, name)

                if hasattr(widget, "clear"):
                    widget.clear()
                elif hasattr(widget, "setPlainText"):
                    widget.setPlainText("")
                elif hasattr(widget, "setText"):
                    widget.setText("")

                QApplication.processEvents()
                return

    def parse_handhold_sensor_line(self, line):
        line = str(line).strip()

        if not line:
            return None

        if "WAITING" in line or "SYNC" in line or "ACK" in line or "READY" in line:
            return None

        parts = [p.strip() for p in line.replace("\t", ",").split(",")]

        if len(parts) < 9:
            return None

        if parts[0].upper() != "HANDHELD":
            return None

        try:
            device_time = parts[1]
            l1 = float(parts[2])
            l2 = float(parts[3])
            l3 = float(parts[4])
            l4 = -float(parts[5])
            total_sum = -float(parts[6])
            roll = float(parts[7])
            pitch = float(parts[8])
        except (TypeError, ValueError):
            return None

        # Device-relative force components:
        # X = L1 + L2, Y = L3, Z = L4.
        # X only displays/uses a value if both X-axis load cells are reading.
        # This avoids one disconnected/dead load cell producing a misleading X value.
        x_component = -1* (l1 + l2) if (abs(l1) > 1e-6 and abs(l2) > 1e-6) else 0.0
        y_component = l3
        z_component = l4

        # HH load-cell-axis calibration from the 2.4 lb rotated-device calibration trial.
        #
        # Page 2 is used as a load-cell verification page:
        #   X display = L1 + L2 tension/compression axis
        #   Y display = calibrated LC3 axis
        #   Z display = calibrated LC4 axis
        #
        # These are NOT roll-transformed on Page 2. They let you press/load
        # each physical sensing direction and verify that it reads about 2.4 lb.
        #
        # The signed axis calibrations below are chosen so the four 2.4 lb
        # calibration holds read positive, since all four were physically
        # upward-force tests after rotating the device.
        # Updated signed Page 2 calibration from observed 2.4 lb checks:
        #
        # Page 2 mapping:
        #   Y display = LC4 / vertical axis
        #   Z display = LC3 / lateral axis
        #
        # Desired signs:
        #   vertical up   = positive
        #   vertical down = negative
        #   lateral left  = positive
        #   lateral right = negative
        #
        # Observed with previous code:
        #   vertical up    read +1.70  -> scale by 2.4 / 1.70
        #   vertical down  read +1.66  -> scale by 2.4 / 1.66 and preserve negative sign
        #   lateral right  read +1.29  -> scale by 2.4 / 1.29 and make negative
        #   lateral left   read +0.94  -> scale by 2.4 / 0.94 and keep positive
        #
        # LC3 raw sign observed:
        #   right  -> positive raw LC3, should display negative
        #   left   -> negative raw LC3, should display positive
        # Updated from latest Page 2 check:
        #   left  read +1.81 -> target +2.40, so increase left branch
        #   right read -4.84 -> target -2.40, so reduce right branch
        if y_component >= 0:
            lc3_axis_component = -0.92 * y_component
        else:
            lc3_axis_component = -3.38 * y_component

        # Updated from latest Page 2 check:
        #   up   read -2.57 -> target +2.40, so flip sign and rescale
        #   down read +2.37 -> target -2.40, so flip sign and rescale
        if z_component >= 0:
            lc4_axis_component = -0.76 * z_component
        else:
            lc4_axis_component = -1.19 * z_component

        # Keep Page 2 as a local load-cell verification display, but save
        # Page 3/Page 4/CSV as global interpreted vertical/lateral force.
        #
        # Page 2:
        #   LC4 sign changes when you rotate 180 degrees.
        #   LC3 sign changes when you rotate +/-90 degrees.
        #
        # Page 4:
        #   those rotated cases should all become "upward" if the physical
        #   applied force was upward.
        roll_rad = math.radians(roll)

        if abs(lc4_axis_component) >= abs(lc3_axis_component):
            # LC4-dominant vertical case:
            # roll near 0 and 180 should be corrected by -cos(roll).
            vertical_component = lc4_axis_component * (-math.cos(roll_rad))
            lateral_component = lc3_axis_component
        else:
            # LC3-dominant vertical case:
            # roll near +90 and -90 should be corrected by -sin(roll).
            vertical_component = lc3_axis_component * (-math.sin(roll_rad))
            lateral_component = lc4_axis_component

        resultant = math.sqrt(
            (x_component ** 2)
            + (vertical_component ** 2)
            + (lateral_component ** 2)
        )

        # Preserve primary-force direction for the resultant.
        # Positive/negative sign comes from X = L1 + L2.
        if abs(x_component) > 1e-6:
            signed_resultant = math.copysign(resultant, x_component)
        else:
            signed_resultant = 0.0

        return {
            "device_time": device_time,
            "l1": l1,
            "l2": l2,
            "l3": l3,
            "l4": l4,
            "sum": total_sum,
            "x_component": x_component,

            # Page 2 HH load-cell verification fields:
            #   y_component = calibrated LC4 vertical axis
            #   z_component = calibrated LC3 lateral axis
            "y_component": lc4_axis_component,
            "z_component": lc3_axis_component,

            # Named transformed components are still saved for Page 3/Page 4/CSV.
            "roll": roll,
            "pitch": pitch,
            "resultant": resultant,
            "signed_resultant": signed_resultant,
            "vertical_component": vertical_component,
            "lateral_component": lateral_component,
        }

    def parse_openhand_sensor_line(self, line):
        line = str(line).strip()

        if not line:
            return None

        if "WAITING" in line or "SYNC" in line or "ACK" in line or "READY" in line:
            return None

        parts = [p.strip() for p in line.replace("\t", ",").split(",")]

        if len(parts) < 6:
            return None

        device_tag = parts[0].upper().replace(" ", "").replace("_", "")

        if device_tag not in ["OPENHAND", "HANDOPEN", "OPEN"]:
            return None

        # Preferred full format:
        # OPENHAND,time,L1,L2,L3,L4,SUM,roll,pitch
        if len(parts) >= 9:
            try:
                device_time = parts[1]
                l1 = float(parts[2])
                l2 = float(parts[3])
                l3 = float(parts[4])
                l4 = -float(parts[5])
                total_sum = -float(parts[6])
                roll = float(parts[7])
                pitch = float(parts[8])
            except (TypeError, ValueError):
                return None

            fake_handhold_line = f"HANDHELD,{device_time},{l1},{l2},{l3},{-l4},{-total_sum},{roll},{pitch}"
            return self.parse_handhold_sensor_line(fake_handhold_line)

        # Two-load-cell fallback format:
        # OPENHAND,time,L1,L2,SUM,roll,pitch
        if len(parts) >= 7:
            try:
                device_time = parts[1]
                l1 = float(parts[2])
                l2 = float(parts[3])
                total_sum = -float(parts[4])
                roll = float(parts[5])
                pitch = float(parts[6])
            except (TypeError, ValueError):
                return None
        else:
            return None

        x_component = -1.2 * (l1 + l2) if (abs(l1) > 1e-6 and abs(l2) > 1e-6) else 0.0
        vertical_component = 0.0
        lateral_component = 0.0
        resultant = abs(x_component)
        signed_resultant = math.copysign(resultant, x_component) if abs(x_component) > 1e-6 else 0.0

        return {
            "device_time": device_time,
            "l1": l1,
            "l2": l2,
            "l3": 0.0,
            "l4": 0.0,
            "sum": total_sum,
            "x_component": x_component,
            "y_component": vertical_component,
            "z_component": lateral_component,
            "roll": roll,
            "pitch": pitch,
            "resultant": resultant,
            "signed_resultant": signed_resultant,
            "vertical_component": vertical_component,
            "lateral_component": lateral_component,
        }

    def identify_hand_device_on_port(self, port_name):
        if serial is None:
            return None

        # Handhold and open-hand are fixed to known Bluetooth COM ports.
        # This avoids scanning COM3/foot ports.
        if port_name not in [HANDHOLD_FIXED_PORT, OPENHAND_FIXED_PORT]:
            return None

        try:
            ser = serial.Serial(port_name, 115200, timeout=0.2)
            time.sleep(0.5)

            detected_device = None
            start = time.time()

            try:
                ser.write(f"SYNC:{time.time()}\n".encode())
            except Exception:
                pass

            while time.time() - start < 2.0:
                raw = ser.readline()

                if not raw:
                    continue

                line = raw.decode(errors="ignore").strip()

                parsed_handhold = self.parse_handhold_sensor_line(line)
                if parsed_handhold is not None:
                    detected_device = "handheld"
                    self.handhold_latest = parsed_handhold
                    break

                parsed_openhand = self.parse_openhand_sensor_line(line)
                if parsed_openhand is not None:
                    detected_device = "openhand"
                    self.openhand_latest = parsed_openhand
                    break

            ser.close()

            if detected_device == "handheld":
                print(f"Synced handhold gauge on {port_name}")

            if detected_device == "openhand":
                print(f"Synced open hand gauge on {port_name}")

            return detected_device

        except Exception as exc:
            print(f"Could not check {port_name}: {exc}")
            return None

    def sync_hands(self):
        if not self.hands_selected:
            return

        if serial is None or list_ports is None:
            self.clear_hand_sync_status()
            self.set_hand_sync_status("ERROR: pyserial is not installed. Run: pip install pyserial")
            self.hand_synced = False
            self.update_record_button_state()
            return

        self.clear_hand_sync_status()
        self.set_hand_sync_status("Searching for hand gauges...")

        self.hand_synced = False
        self.handhold_port = None
        self.openhand_port = None
        self.hand_ports = {}

        self.set_indicator_pair(self.page2.HandholdRed, self.page2.HandholdGreen, False)
        self.set_indicator_pair(self.page2.OpenHandRed, self.page2.OpenHandGreen, False)

        detected_device = self.identify_hand_device_on_port(HANDHOLD_FIXED_PORT)

        if detected_device == "handheld":
            self.handhold_port = HANDHOLD_FIXED_PORT
            self.set_indicator_pair(self.page2.HandholdRed, self.page2.HandholdGreen, True)

        detected_device = self.identify_hand_device_on_port(OPENHAND_FIXED_PORT)

        if detected_device == "openhand":
            self.openhand_port = OPENHAND_FIXED_PORT
            self.set_indicator_pair(self.page2.OpenHandRed, self.page2.OpenHandGreen, True)

        # Handhold remains required. Open-hand records/displays if found.
        self.hand_synced = self.handhold_port is not None

        if self.hand_synced:
            self.hand_ports = {
                "handheld": self.handhold_port,
            }

            if self.openhand_port is not None:
                self.hand_ports["openhand"] = self.openhand_port

            with open(self.hand_ports_path, "w") as f:
                json.dump(self.hand_ports, f, indent=4)

            self.clear_hand_sync_status()

            if self.openhand_port is not None:
                self.set_hand_sync_status("Handhold and open hand gauges found.")
            else:
                self.set_hand_sync_status("Handhold gauge found.")

            self.start_page2_hand_live_preview()
        else:
            self.clear_hand_sync_status()
            self.set_hand_sync_status("No handhold gauge found.")

        self.update_record_button_state()


    def sync_feet(self):
        if not self.feet_selected:
            return

        if serial is None or list_ports is None:
            self.clear_foot_sync_status()
            self.set_foot_sync_status("ERROR: pyserial is not installed. Run: pip install pyserial")
            self.feet_synced = False
            self.update_record_button_state()
            return

        self.clear_foot_sync_status()
        self.set_foot_sync_status("Searching for foot sensors on fixed ports...")

        self.feet_synced = False
        self.left_foot_port = None
        self.right_foot_port = None
        self.foot_ports = {}

        self.set_indicator_pair(self.page2.LeftFootRed, self.page2.LeftFootGreen, False)
        self.set_indicator_pair(self.page2.RightFootRed, self.page2.RightFootGreen, False)

        detected_left = self.identify_foot_device_on_port(LEFT_FOOT_FIXED_PORT)

        if detected_left == "left":
            self.left_foot_port = LEFT_FOOT_FIXED_PORT
            self.set_indicator_pair(
                self.page2.LeftFootRed,
                self.page2.LeftFootGreen,
                True,
            )

        detected_right = self.identify_foot_device_on_port(RIGHT_FOOT_FIXED_PORT)

        if detected_right == "right":
            self.right_foot_port = RIGHT_FOOT_FIXED_PORT
            self.set_indicator_pair(
                self.page2.RightFootRed,
                self.page2.RightFootGreen,
                True,
            )

        self.feet_synced = (
            self.left_foot_port is not None
            and self.right_foot_port is not None
        )

        if self.feet_synced:
            foot_ports = {
                "left": self.left_foot_port,
                "right": self.right_foot_port,
            }

            self.foot_ports = foot_ports

            with open(self.foot_ports_path, "w") as f:
                json.dump(foot_ports, f, indent=4)

            self.clear_foot_sync_status()
            self.set_foot_sync_status("Both feet sensors found.")
            self.start_page2_foot_live_preview()

        elif self.left_foot_port and not self.right_foot_port:
            self.clear_foot_sync_status()
            self.set_foot_sync_status("Only found left foot.")

        elif self.right_foot_port and not self.left_foot_port:
            self.clear_foot_sync_status()
            self.set_foot_sync_status("Only found right foot.")

        else:
            self.clear_foot_sync_status()
            self.set_foot_sync_status("No foot sensors found.")

        self.update_record_button_state()


    def tare_handhold(self):
        # Sends tare-all command to the handhold ESP32.
        # Firmware command expected: T = tare all load cells.
        try:
            if self.handhold_live_serial is not None:
                self.handhold_live_serial.write(b"T\n")
                print("Sent handhold tare command over live serial.")
                return

            if self.handhold_port and serial is not None:
                with serial.Serial(self.handhold_port, 115200, timeout=0.2) as ser:
                    time.sleep(0.2)
                    ser.write(b"T\n")
                    print("Sent handhold tare command over temporary serial.")
                return

            print("WARNING: Cannot tare handhold; handhold is not synced.")
        except Exception as exc:
            print(f"WARNING: Handhold tare failed: {exc}")

    def tare_openhand(self):
        # Sends tare-all command to the open-hand ESP32.
        # Firmware command expected: T = tare all load cells.
        try:
            if self.openhand_live_serial is not None:
                self.openhand_live_serial.write(b"T\n")
                print("Sent open-hand tare command over live serial.")
                return

            if self.openhand_port and serial is not None:
                with serial.Serial(self.openhand_port, 115200, timeout=0.2) as ser:
                    time.sleep(0.2)
                    ser.write(b"T\n")
                    print("Sent open-hand tare command over temporary serial.")
                return

            print("WARNING: Cannot tare open hand; open hand is not synced.")
        except Exception as exc:
            print(f"WARNING: Open-hand tare failed: {exc}")


    def stop_page2_hand_live_preview(self):
        self.hand_live_timer.stop()

        for ser_obj in [self.handhold_live_serial, self.openhand_live_serial]:
            if ser_obj is not None:
                try:
                    ser_obj.close()
                except Exception:
                    pass

        self.handhold_live_serial = None
        self.openhand_live_serial = None

    def start_page2_hand_live_preview(self):
        if serial is None:
            print("ERROR: pyserial is not installed. Run: pip install pyserial")
            return

        self.stop_page2_hand_live_preview()

        if not self.handhold_port and not self.openhand_port:
            return

        try:
            if self.handhold_port:
                self.handhold_live_serial = serial.Serial(self.handhold_port, 115200, timeout=0.05)
                self.handhold_live_serial.write(f"SYNC:{time.time()}\n".encode())
                print(f"Started Page 2 handhold live view on {self.handhold_port}")

            if self.openhand_port:
                self.openhand_live_serial = serial.Serial(self.openhand_port, 115200, timeout=0.05)
                self.openhand_live_serial.write(f"SYNC:{time.time()}\n".encode())
                print(f"Started Page 2 open hand live view on {self.openhand_port}")

        except Exception as exc:
            print(f"ERROR: Could not start Page 2 hand live preview: {exc}")
            self.stop_page2_hand_live_preview()
            return

        self.update_page2_hand_live_values()
        self.hand_live_timer.start(50)

    def read_latest_hand_serial(self, ser_obj, parser, latest_name):
        if ser_obj is None:
            return

        try:
            for _ in range(5):
                raw = ser_obj.readline()

                if not raw:
                    continue

                line = raw.decode(errors="ignore").strip()
                parsed = parser(line)

                if parsed is None:
                    continue

                setattr(self, latest_name, parsed)

        except Exception as exc:
            print(f"WARNING: Hand live read failed: {exc}")

    def display_hand_values_on_page2(self, latest, name_map):
        x_component = latest.get("x_component", 0.0)
        y_component = latest.get("y_component", 0.0)
        z_component = latest.get("z_component", 0.0)

        # Page 2 HH load-cell verification display:
        # X = calibrated L1 + L2 axis
        # Y = calibrated LC4 axis
        # Z = calibrated LC3 axis
        #
        # No roll/rotation transform is applied here.
        display_map = {
            name_map[0]: x_component,
            name_map[1]: y_component,
            name_map[2]: z_component,
        }

        for widget_name, value in display_map.items():
            if hasattr(self.page2, widget_name):
                widget = getattr(self.page2, widget_name)

                if hasattr(widget, "display"):
                    widget.display(float(f"{value:.2f}"))
                elif hasattr(widget, "setText"):
                    widget.setText(f"{value:.2f}")

    def update_page2_hand_live_values(self):
        self.read_latest_hand_serial(
            self.handhold_live_serial,
            self.parse_handhold_sensor_line,
            "handhold_latest",
        )

        self.read_latest_hand_serial(
            self.openhand_live_serial,
            self.parse_openhand_sensor_line,
            "openhand_latest",
        )

        # Page 2 HH display: x = L1+L2, y = calibrated LC3, z = calibrated LC4.
        self.display_hand_values_on_page2(
            self.handhold_latest,
            ("xHandholdLBS", "yHandholdLBS", "zHandholdLBS"),
        )

        # Page 2 open hand uses one LCD object: OpenHandLBS.
        open_signed_resultant = self.openhand_latest.get("signed_resultant", 0.0)

        if hasattr(self.page2, "OpenHandLBS"):
            widget = self.page2.OpenHandLBS

            if hasattr(widget, "display"):
                widget.display(float(f"{open_signed_resultant:.2f}"))
            elif hasattr(widget, "setText"):
                widget.setText(f"{open_signed_resultant:.2f}")

    def update_page3_hand_live_values(self):
        signed_resultant = self.handhold_latest.get("signed_resultant", 0.0)
        open_signed_resultant = self.openhand_latest.get("signed_resultant", 0.0)

        if hasattr(self.page3, "HandholdLBS"):
            widget = self.page3.HandholdLBS

            if hasattr(widget, "display"):
                widget.display(float(f"{signed_resultant:.2f}"))
            elif hasattr(widget, "setText"):
                widget.setText(f"{signed_resultant:.2f}")

        for widget_name in ["OpenHandLBS", "OpenhandLBS", "OpenLBS"]:
            if hasattr(self.page3, widget_name):
                widget = getattr(self.page3, widget_name)

                if hasattr(widget, "display"):
                    widget.display(float(f"{open_signed_resultant:.2f}"))
                elif hasattr(widget, "setText"):
                    widget.setText(f"{open_signed_resultant:.2f}")

    def start_hand_logging(self):
        if not self.hands_selected:
            return

        if self.handhold_port is None:
            print("ERROR: Handhold gauge is not synced. Cannot start hand logging.")
            return

        if serial is None:
            print("ERROR: pyserial is not installed. Run: pip install pyserial")
            return

        self.stop_page2_hand_live_preview()
        self.stop_hand_logging()

        self.hand_stop_event.clear()
        self.hand_logging_active = True
        self.hand_recording_start_time = (
            self.recording_session_start_time
            if self.recording_session_start_time is not None
            else time.time()
        )
        self.hand_threads = []

        thread = threading.Thread(
            target=self.hand_logger_loop,
            args=("handheld", self.handhold_port, self.handhold_csv_path),
            daemon=True,
        )
        thread.start()
        self.hand_threads.append(thread)

        if self.openhand_port is not None:
            thread = threading.Thread(
                target=self.hand_logger_loop,
                args=("openhand", self.openhand_port, self.openhand_csv_path),
                daemon=True,
            )
            thread.start()
            self.hand_threads.append(thread)

        print("Started hand gauge logging.")
        self.update_page3_hand_live_values()
        self.hand_record_live_timer.start(50)

    def hand_logger_loop(self, device_name, port_name, csv_path):
        try:
            if device_name == "handheld" and port_name != HANDHOLD_FIXED_PORT:
                return None

            if device_name == "openhand" and port_name != OPENHAND_FIXED_PORT:
                return None

            ser = serial.Serial(port_name, 115200, timeout=0.2)
            time.sleep(0.2)

            try:
                ser.reset_input_buffer()
                ser.reset_output_buffer()
            except Exception:
                pass

            sync_pc_time = time.time()
            sync_message = f"SYNC:{sync_pc_time}\n"
            ser.write(sync_message.encode())
            print(f"Sent recording sync to {device_name} gauge on {port_name}")

            device_zero_time_sec = None

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "PC_TIME_SEC",
                    "DEVICE_TIME",
                    "L1",
                    "L2",
                    "L3",
                    "L4",
                    "SUM",
                    "X_COMPONENT",
                    "Y_COMPONENT",
                    "Z_COMPONENT",
                    "ROLL",
                    "PITCH",
                    "RESULTANT",
                    "SIGNED_RESULTANT",
                    "VERTICAL_COMPONENT",
                    "LATERAL_COMPONENT",
                ])

                while not self.hand_stop_event.is_set():
                    raw = ser.readline()

                    if not raw:
                        continue

                    line = raw.decode(errors="ignore").strip()

                    if not line:
                        continue

                    if device_name == "openhand":
                        parsed = self.parse_openhand_sensor_line(line)
                    else:
                        parsed = self.parse_handhold_sensor_line(line)

                    if parsed is None:
                        continue

                    if device_name == "openhand":
                        self.openhand_latest = parsed
                    else:
                        self.handhold_latest = parsed

                    device_time_sec = self.parse_device_time_to_seconds(parsed["device_time"])

                    if device_time_sec is not None and device_zero_time_sec is None:
                        device_zero_time_sec = device_time_sec

                    pc_time_sec = self.get_synced_sample_time_sec(
                        parsed["device_time"],
                        device_zero_time_sec,
                        sync_pc_time,
                        self.recording_session_start_time,
                        self.hand_recording_start_time,
                    )

                    writer.writerow([
                        f"{pc_time_sec:.6f}",
                        parsed["device_time"],
                        parsed["l1"],
                        parsed["l2"],
                        parsed["l3"],
                        parsed["l4"],
                        parsed["sum"],
                        parsed["x_component"],
                        parsed["y_component"],
                        parsed["z_component"],
                        parsed["roll"],
                        parsed["pitch"],
                        parsed["resultant"],
                        parsed["signed_resultant"],
                        parsed["vertical_component"],
                        parsed["lateral_component"],
                    ])

            ser.close()
            print(f"Stopped {device_name} hand gauge logging: {csv_path}")

        except Exception as exc:
            print(f"ERROR logging {device_name} hand gauge on {port_name}: {exc}")

    def stop_hand_logging(self):
        self.hand_record_live_timer.stop()

        if not self.hand_logging_active and not self.hand_threads:
            return

        self.hand_stop_event.set()

        for thread in self.hand_threads:
            thread.join(timeout=2.0)

        self.hand_threads = []
        self.hand_logging_active = False
        print("Hand gauge logging stopped.")

    def update_record_button_state(self):
        if not (
            self.mocap_selected
            or self.hands_selected
            or self.feet_selected
            or self.grf_selected
            or self.com_selected
        ):
            self.page2.Record.setEnabled(False)
            return

        mocap_ready = (not self.mocap_selected) or self.mocap_synced
        hands_ready = (not self.hands_selected) or self.hand_synced
        feet_ready = (not self.feet_selected) or (
            self.feet_synced and self.has_valid_shoe_size()
        )

        self.page2.Record.setEnabled(mocap_ready and hands_ready and feet_ready)

    def go_to_configuration(self):
        self.stop_page2_preview()
        self.stop_page4_playback()
        self.stack.setCurrentIndex(0)
        self.update_page1_dependent_options()

    def go_to_initialization(self):
        self.stop_page4_playback()
        self.apply_system_configuration_to_page2()

        self.stack.setCurrentIndex(1)

        self.mocap_synced = False
        self.hand_synced = False
        self.feet_synced = False
        self.handhold_port = None
        self.openhand_port = None
        self.left_foot_tare = [0.0, 0.0, 0.0, 0.0]
        self.right_foot_tare = [0.0, 0.0, 0.0, 0.0]

        self.initialize_status_indicators()
        self.update_record_button_state()

        self.clear_page2_foot_live_views()
        self.start_page2_preview()

    def go_to_recording(self):
        self.stop_page2_preview()
        self.clear_page3_foot_live_views()
        self.stack.setCurrentIndex(2)
        self.begin_recording_session()

    def stop_recording(self):
        self.countdown_timer.stop()
        self.page3.Countdown.setText("")

        if self.mocap_selected:
            self.stop_mocap_recording()

        if self.feet_selected:
            self.stop_foot_logging()

        if self.hands_selected:
            self.stop_hand_logging()

        self.stop_page3_foot_live_preview()

        self.page3.Stop.setEnabled(False)
        self.page3.Analyze.setEnabled(True)
        self.page3.Back.setEnabled(True)

        self.style_page3_action_buttons(stopped=True)
        self.set_recording_indicator(False)

    def go_to_analysis(self):
        if self.mocap_selected:
            self.run_mocap_analysis()

        self.read_latest_total_grf_from_csv()
        self.apply_system_configuration_to_page4()
        self.stack.setCurrentIndex(3)
        self.start_page4_playback()
        self.update_page4_grf_placeholder_values(0)

    def closeEvent(self, event):
        self.stop_page2_preview()
        self.stop_page3_foot_live_preview()
        self.stop_mocap_recording()
        self.stop_foot_logging()
        self.stop_hand_logging()
        self.stop_page4_playback()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
