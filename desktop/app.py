from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = PROJECT_ROOT / "main.py"

PIPELINE_STEPS = [
    "all",
    "preprocess",
    "asr",
    "translate",
    "tts",
    "postprocess",
    "subtitles",
    "metrics",
    "prepare_finetune",
]

FORCE_STEPS = ["", *PIPELINE_STEPS]
MT_PROVIDERS = ["", "hf", "gemini", "openai", "openrouter", "groq", "cerebras", "openai_compatible"]
MT_STRATEGIES = ["", "per-segment", "sentence-boundary-aware"]
MT_STYLES = ["", "standard", "academic", "casual", "news", "compact"]
OPENAI_RESPONSE_FORMATS = ["", "json_schema", "json_object", "none"]
TTS_PROVIDERS = ["", "xtts", "elevenlabs"]
SUBTITLE_MODES = ["", "soft", "hard", "both"]
SMART_SYNC_MODES = ["", "1", "0"]
SMART_SYNC_PROVIDERS = ["", "gemini", "openai", "openrouter", "groq", "cerebras", "openai_compatible"]

KNOWN_ARTIFACTS = [
    ("final_video.mp4", "Финальное видео"),
    ("final_mix.wav", "Финальный микс"),
    ("final_dubbing.wav", "Дубляж"),
    ("run_report.md", "Отчет"),
    ("metrics.json", "Метрики"),
    ("tts_config.json", "TTS config"),
    ("elevenlabs_voice.json", "ElevenLabs voice"),
    ("translation_quality.json", "Качество перевода"),
    ("segments.json", "ASR segments"),
    ("translated_segments.json", "Перевод"),
    ("translated_segments.clean.json", "Очищенный перевод"),
    ("temp/speaker_profile.json", "Speaker profile"),
    ("temp/speaker_ref.wav", "Speaker ref"),
    ("temp/vocals.wav", "Vocals"),
    ("temp/background.wav", "Background"),
    ("subtitles", "Субтитры"),
    ("subtitles/subtitles_manifest.json", "Subtitles manifest"),
]

ONLINE_MT_MODEL = "openai/gpt-5.4-mini"
LOCAL_MT_MODEL = "facebook/nllb-200-distilled-1.3B"


def sanitize_job_name(name: str) -> str:
    normalized = re.sub(r"[^\w.-]+", "_", name.strip(), flags=re.UNICODE)
    normalized = normalized.strip("._-")
    return normalized or "job"


def job_name_from_video(video_path: str) -> str:
    stem = Path(video_path).stem if video_path else "job"
    if stem.lower().startswith("video_") and len(stem) > len("video_"):
        stem = stem[len("video_") :]
    return sanitize_job_name(stem)


def combo_value(combo: QComboBox) -> str:
    data = combo.currentData()
    return "" if data is None else str(data)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Dubbing Desktop")
        self.resize(1220, 780)

        self.settings = QSettings("video-dubbing", "desktop")
        self.process: QProcess | None = None
        self.process_kind = ""
        self.last_output_dir: Path | None = None

        self._build_ui()
        self._restore_settings()
        self._update_buttons()
        self.refresh_artifacts()

    def _build_ui(self) -> None:
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Automatic Video Dubbing")
        title_font = QFont()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title.setFont(title_font)
        self.status_label = QLabel("Готово")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.status_label)
        root_layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_controls_panel())
        splitter.addWidget(self._build_output_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 800])
        root_layout.addWidget(splitter, 1)

        self.setCentralWidget(root)

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)

        preset_group = QGroupBox("Быстрые профили")
        preset_layout = QGridLayout(preset_group)
        self.online_preset_button = QPushButton("Online current")
        self.local_preset_button = QPushButton("Local fallback")
        self.online_preset_button.clicked.connect(self.apply_online_preset)
        self.local_preset_button.clicked.connect(self.apply_local_fallback_preset)
        preset_layout.addWidget(self.online_preset_button, 0, 0)
        preset_layout.addWidget(self.local_preset_button, 0, 1)
        layout.addWidget(preset_group)

        input_group = QGroupBox("Задание")
        input_form = QFormLayout(input_group)
        input_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.python_edit = QLineEdit(sys.executable)
        python_browse = QPushButton("...")
        python_browse.setFixedWidth(34)
        python_browse.clicked.connect(self.browse_python)
        python_row = QHBoxLayout()
        python_row.addWidget(self.python_edit, 1)
        python_row.addWidget(python_browse)
        input_form.addRow("Python", python_row)

        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText(r"C:\path\to\video.mp4")
        self.video_edit.textChanged.connect(self._on_video_changed)
        video_browse = QPushButton("...")
        video_browse.setFixedWidth(34)
        video_browse.clicked.connect(self.browse_video)
        video_row = QHBoxLayout()
        video_row.addWidget(self.video_edit, 1)
        video_row.addWidget(video_browse)
        input_form.addRow("Видео", video_row)

        self.job_edit = QLineEdit()
        self.job_edit.setPlaceholderText("по имени видео")
        self.job_edit.textChanged.connect(self.refresh_artifacts)
        input_form.addRow("Job name", self.job_edit)

        self.step_combo = QComboBox()
        for step in PIPELINE_STEPS:
            self.step_combo.addItem(step, step)
        input_form.addRow("Шаг", self.step_combo)

        self.test_checkbox = QCheckBox("Test mode")
        self.test_checkbox.stateChanged.connect(self.refresh_artifacts)
        self.resume_checkbox = QCheckBox("Resume")
        self.resume_checkbox.setChecked(True)
        self.resume_checkbox.stateChanged.connect(self._update_force_enabled)
        self.skip_metrics_checkbox = QCheckBox("Без метрик")
        self.skip_metrics_checkbox.setChecked(True)
        self.skip_metrics_checkbox.setToolTip("Для Step = all завершить пайплайн после субтитров")
        self.step_combo.currentTextChanged.connect(self._update_skip_metrics_enabled)
        self.force_combo = QComboBox()
        self._fill_combo(self.force_combo, FORCE_STEPS, empty_label="не пересчитывать")
        self._update_force_enabled()

        mode_row = QHBoxLayout()
        mode_row.addWidget(self.test_checkbox)
        mode_row.addWidget(self.resume_checkbox)
        mode_row.addWidget(self.skip_metrics_checkbox)
        mode_row.addStretch(1)
        input_form.addRow("Режим", mode_row)
        input_form.addRow("Force step", self.force_combo)
        self._update_skip_metrics_enabled()

        layout.addWidget(input_group)

        mt_group = QGroupBox("Перевод")
        mt_form = QFormLayout(mt_group)
        self.mt_provider_combo = QComboBox()
        self._fill_combo(self.mt_provider_combo, MT_PROVIDERS, empty_label="config.py")
        self.mt_model_edit = QLineEdit()
        self.mt_model_edit.setPlaceholderText("config.py")
        self.mt_strategy_combo = QComboBox()
        self._fill_combo(self.mt_strategy_combo, MT_STRATEGIES, empty_label="config.py")
        self.mt_style_combo = QComboBox()
        self._fill_combo(self.mt_style_combo, MT_STYLES, empty_label="config.py")
        self.mt_profile_edit = QLineEdit()
        self.mt_profile_edit.setPlaceholderText("interview, lecture, podcast...")
        self.mt_corrections_edit = QLineEdit()
        self.mt_corrections_edit.setPlaceholderText("03 -> o3 | term -> термин")
        mt_form.addRow("Provider", self.mt_provider_combo)
        mt_form.addRow("Model", self.mt_model_edit)
        mt_form.addRow("Strategy", self.mt_strategy_combo)
        mt_form.addRow("Style", self.mt_style_combo)
        mt_form.addRow("Profile", self.mt_profile_edit)
        mt_form.addRow("ASR fixes", self.mt_corrections_edit)
        layout.addWidget(mt_group)

        openai_group = QGroupBox("OpenAI-compatible / RouterAI")
        openai_form = QFormLayout(openai_group)
        self.mt_openai_api_key_env_edit = QLineEdit()
        self.mt_openai_api_key_env_edit.setPlaceholderText("OPENAI_API_KEY / ROUTERAI_API_KEY")
        self.mt_openai_base_url_edit = QLineEdit()
        self.mt_openai_base_url_edit.setPlaceholderText("https://.../v1")
        self.mt_openai_response_format_combo = QComboBox()
        self._fill_combo(self.mt_openai_response_format_combo, OPENAI_RESPONSE_FORMATS, empty_label="config.py")
        openai_form.addRow("API key env", self.mt_openai_api_key_env_edit)
        openai_form.addRow("Base URL", self.mt_openai_base_url_edit)
        openai_form.addRow("Response format", self.mt_openai_response_format_combo)
        layout.addWidget(openai_group)

        tts_group = QGroupBox("Голос и субтитры")
        tts_form = QFormLayout(tts_group)
        self.tts_provider_combo = QComboBox()
        self._fill_combo(self.tts_provider_combo, TTS_PROVIDERS, empty_label="config.py")
        self.elevenlabs_voice_id_edit = QLineEdit()
        self.elevenlabs_voice_id_edit.setPlaceholderText("готовый voice_id")
        self.elevenlabs_voice_name_edit = QLineEdit()
        self.elevenlabs_voice_name_edit.setPlaceholderText("имя нового clone")
        self.elevenlabs_no_clone_checkbox = QCheckBox("Не создавать voice clone")
        self.subtitle_mode_combo = QComboBox()
        self._fill_combo(self.subtitle_mode_combo, SUBTITLE_MODES, empty_label="config.py")
        self.subtitle_original_checkbox = QCheckBox("Субтитры из original_text")
        tts_form.addRow("TTS", self.tts_provider_combo)
        tts_form.addRow("Voice ID", self.elevenlabs_voice_id_edit)
        tts_form.addRow("Voice name", self.elevenlabs_voice_name_edit)
        tts_form.addRow("", self.elevenlabs_no_clone_checkbox)
        tts_form.addRow("Subtitles", self.subtitle_mode_combo)
        tts_form.addRow("", self.subtitle_original_checkbox)
        layout.addWidget(tts_group)

        smart_group = QGroupBox("SmartSync")
        smart_form = QFormLayout(smart_group)
        self.smart_sync_mode_combo = QComboBox()
        self._fill_combo(self.smart_sync_mode_combo, SMART_SYNC_MODES, empty_label="config.py")
        self.smart_sync_mode_combo.setItemText(1, "enabled")
        self.smart_sync_mode_combo.setItemText(2, "disabled")
        self.smart_sync_provider_combo = QComboBox()
        self._fill_combo(self.smart_sync_provider_combo, SMART_SYNC_PROVIDERS, empty_label="config.py")
        self.smart_sync_model_edit = QLineEdit()
        self.smart_sync_model_edit.setPlaceholderText("config.py")
        self.smart_sync_api_key_env_edit = QLineEdit()
        self.smart_sync_api_key_env_edit.setPlaceholderText("GROQ_API_KEY / OPENAI_API_KEY")
        self.smart_sync_base_url_edit = QLineEdit()
        self.smart_sync_base_url_edit.setPlaceholderText("https://.../v1")
        self.smart_sync_max_rewrites_edit = QLineEdit()
        self.smart_sync_max_rewrites_edit.setPlaceholderText("config.py")
        self.smart_sync_candidates_edit = QLineEdit()
        self.smart_sync_candidates_edit.setPlaceholderText("config.py")
        smart_form.addRow("Mode", self.smart_sync_mode_combo)
        smart_form.addRow("Provider", self.smart_sync_provider_combo)
        smart_form.addRow("Model", self.smart_sync_model_edit)
        smart_form.addRow("API key env", self.smart_sync_api_key_env_edit)
        smart_form.addRow("Base URL", self.smart_sync_base_url_edit)
        smart_form.addRow("Max rewrites", self.smart_sync_max_rewrites_edit)
        smart_form.addRow("Candidates", self.smart_sync_candidates_edit)
        layout.addWidget(smart_group)

        action_group = QFrame()
        action_layout = QGridLayout(action_group)
        action_layout.setContentsMargins(0, 0, 0, 0)
        self.doctor_button = QPushButton("Doctor")
        self.config_button = QPushButton("Show config")
        self.run_button = QPushButton("Run")
        self.stop_button = QPushButton("Stop")
        self.open_output_button = QPushButton("Open output")
        self.open_video_button = QPushButton("Open video")
        self.open_report_button = QPushButton("Open report")
        self.doctor_button.clicked.connect(self.run_doctor)
        self.config_button.clicked.connect(self.show_config)
        self.run_button.clicked.connect(self.run_pipeline)
        self.stop_button.clicked.connect(self.stop_process)
        self.open_output_button.clicked.connect(self.open_output_dir)
        self.open_video_button.clicked.connect(lambda: self.open_artifact("final_video.mp4"))
        self.open_report_button.clicked.connect(lambda: self.open_artifact("run_report.md"))
        action_layout.addWidget(self.doctor_button, 0, 0)
        action_layout.addWidget(self.config_button, 0, 1)
        action_layout.addWidget(self.run_button, 1, 0)
        action_layout.addWidget(self.stop_button, 1, 1)
        action_layout.addWidget(self.open_output_button, 2, 0)
        action_layout.addWidget(self.open_video_button, 2, 1)
        action_layout.addWidget(self.open_report_button, 3, 0, 1, 2)
        layout.addWidget(action_group)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        scroll.setMaximumWidth(540)
        scroll.setWidget(panel)
        return scroll

    def _build_output_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setLineWrapMode(QTextEdit.NoWrap)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        self.log_edit.setFont(mono)

        self.artifacts_list = QListWidget()
        self.artifacts_list.setMaximumHeight(190)
        self.artifacts_list.itemDoubleClicked.connect(self._open_artifact_item)

        clear_button = QPushButton("Clear log")
        clear_button.clicked.connect(self.log_edit.clear)
        refresh_button = QPushButton("Refresh artifacts")
        refresh_button.clicked.connect(self.refresh_artifacts)
        output_buttons = QHBoxLayout()
        output_buttons.addWidget(QLabel("Артефакты"))
        output_buttons.addStretch(1)
        output_buttons.addWidget(refresh_button)
        output_buttons.addWidget(clear_button)

        layout.addLayout(output_buttons)
        layout.addWidget(self.artifacts_list)
        layout.addWidget(self.log_edit, 1)
        return panel

    def _fill_combo(self, combo: QComboBox, values: list[str], *, empty_label: str) -> None:
        for value in values:
            combo.addItem(empty_label if value == "" else value, value)

    def _restore_settings(self) -> None:
        self.python_edit.setText(str(self.settings.value("python", sys.executable)))
        self.video_edit.setText(str(self.settings.value("video", "")))
        self.job_edit.setText(str(self.settings.value("job", "")))
        self.test_checkbox.setChecked(self.settings.value("test", False, type=bool))
        self.resume_checkbox.setChecked(self.settings.value("resume", True, type=bool))
        self.skip_metrics_checkbox.setChecked(self.settings.value("skip_metrics", True, type=bool))
        self._set_combo_value(self.step_combo, str(self.settings.value("step", "all")))
        self._set_combo_value(self.mt_provider_combo, str(self.settings.value("mt_provider", "")))
        self.mt_model_edit.setText(str(self.settings.value("mt_model", "")))
        self._set_combo_value(self.mt_strategy_combo, str(self.settings.value("mt_strategy", "")))
        self._set_combo_value(self.mt_style_combo, str(self.settings.value("mt_style", "")))
        self.mt_profile_edit.setText(str(self.settings.value("mt_profile", "")))
        self.mt_corrections_edit.setText(str(self.settings.value("mt_corrections", "")))
        self.mt_openai_api_key_env_edit.setText(
            str(self.settings.value("mt_openai_api_key_env", os.getenv("MT_OPENAI_API_KEY_ENV", "")))
        )
        self.mt_openai_base_url_edit.setText(
            str(self.settings.value("mt_openai_base_url", os.getenv("MT_OPENAI_BASE_URL", "")))
        )
        self._set_combo_value(
            self.mt_openai_response_format_combo,
            str(self.settings.value("mt_openai_response_format", "")),
        )
        self._set_combo_value(self.tts_provider_combo, str(self.settings.value("tts_provider", "")))
        self.elevenlabs_voice_id_edit.setText(str(self.settings.value("elevenlabs_voice_id", "")))
        self.elevenlabs_voice_name_edit.setText(str(self.settings.value("elevenlabs_voice_name", "")))
        self.elevenlabs_no_clone_checkbox.setChecked(self.settings.value("elevenlabs_no_clone", False, type=bool))
        self._set_combo_value(self.subtitle_mode_combo, str(self.settings.value("subtitle_mode", "")))
        self.subtitle_original_checkbox.setChecked(self.settings.value("subtitle_original", False, type=bool))
        self._set_combo_value(self.smart_sync_mode_combo, str(self.settings.value("smart_sync_mode", "")))
        self._set_combo_value(self.smart_sync_provider_combo, str(self.settings.value("smart_sync_provider", "")))
        self.smart_sync_model_edit.setText(str(self.settings.value("smart_sync_model", "")))
        self.smart_sync_api_key_env_edit.setText(
            str(self.settings.value("smart_sync_api_key_env", os.getenv("SMART_SYNC_API_KEY_ENV", "")))
        )
        self.smart_sync_base_url_edit.setText(
            str(self.settings.value("smart_sync_base_url", os.getenv("SMART_SYNC_BASE_URL", "")))
        )
        self.smart_sync_max_rewrites_edit.setText(str(self.settings.value("smart_sync_max_rewrites", "")))
        self.smart_sync_candidates_edit.setText(str(self.settings.value("smart_sync_candidates", "")))

    def _save_settings(self) -> None:
        self.settings.setValue("python", self.python_edit.text().strip())
        self.settings.setValue("video", self.video_edit.text().strip())
        self.settings.setValue("job", self.job_edit.text().strip())
        self.settings.setValue("test", self.test_checkbox.isChecked())
        self.settings.setValue("resume", self.resume_checkbox.isChecked())
        self.settings.setValue("skip_metrics", self.skip_metrics_checkbox.isChecked())
        self.settings.setValue("step", self.step_combo.currentText())
        self.settings.setValue("mt_provider", combo_value(self.mt_provider_combo))
        self.settings.setValue("mt_model", self.mt_model_edit.text().strip())
        self.settings.setValue("mt_strategy", combo_value(self.mt_strategy_combo))
        self.settings.setValue("mt_style", combo_value(self.mt_style_combo))
        self.settings.setValue("mt_profile", self.mt_profile_edit.text().strip())
        self.settings.setValue("mt_corrections", self.mt_corrections_edit.text().strip())
        self.settings.setValue("mt_openai_api_key_env", self.mt_openai_api_key_env_edit.text().strip())
        self.settings.setValue("mt_openai_base_url", self.mt_openai_base_url_edit.text().strip())
        self.settings.setValue("mt_openai_response_format", combo_value(self.mt_openai_response_format_combo))
        self.settings.setValue("tts_provider", combo_value(self.tts_provider_combo))
        self.settings.setValue("elevenlabs_voice_id", self.elevenlabs_voice_id_edit.text().strip())
        self.settings.setValue("elevenlabs_voice_name", self.elevenlabs_voice_name_edit.text().strip())
        self.settings.setValue("elevenlabs_no_clone", self.elevenlabs_no_clone_checkbox.isChecked())
        self.settings.setValue("subtitle_mode", combo_value(self.subtitle_mode_combo))
        self.settings.setValue("subtitle_original", self.subtitle_original_checkbox.isChecked())
        self.settings.setValue("smart_sync_mode", combo_value(self.smart_sync_mode_combo))
        self.settings.setValue("smart_sync_provider", combo_value(self.smart_sync_provider_combo))
        self.settings.setValue("smart_sync_model", self.smart_sync_model_edit.text().strip())
        self.settings.setValue("smart_sync_api_key_env", self.smart_sync_api_key_env_edit.text().strip())
        self.settings.setValue("smart_sync_base_url", self.smart_sync_base_url_edit.text().strip())
        self.settings.setValue("smart_sync_max_rewrites", self.smart_sync_max_rewrites_edit.text().strip())
        self.settings.setValue("smart_sync_candidates", self.smart_sync_candidates_edit.text().strip())

    def _set_combo_value(self, combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def apply_online_preset(self) -> None:
        self._set_combo_value(self.mt_provider_combo, "openai_compatible")
        self.mt_model_edit.setText(ONLINE_MT_MODEL)
        self._set_combo_value(self.mt_strategy_combo, "sentence-boundary-aware")
        self._set_combo_value(self.mt_style_combo, "compact")
        self._set_combo_value(self.tts_provider_combo, "elevenlabs")
        self._set_combo_value(self.subtitle_mode_combo, "hard")
        self._set_combo_value(self.mt_openai_response_format_combo, "json_schema")
        if not self.mt_openai_api_key_env_edit.text().strip():
            self.mt_openai_api_key_env_edit.setText("OPENAI_API_KEY")
        self._set_combo_value(self.smart_sync_mode_combo, "1")
        self._set_combo_value(self.smart_sync_provider_combo, "openai_compatible")
        self.smart_sync_model_edit.setText(ONLINE_MT_MODEL)
        if not self.smart_sync_api_key_env_edit.text().strip():
            self.smart_sync_api_key_env_edit.setText(self.mt_openai_api_key_env_edit.text().strip() or "OPENAI_API_KEY")
        if not self.smart_sync_base_url_edit.text().strip() and self.mt_openai_base_url_edit.text().strip():
            self.smart_sync_base_url_edit.setText(self.mt_openai_base_url_edit.text().strip())
        if not self.smart_sync_max_rewrites_edit.text().strip():
            self.smart_sync_max_rewrites_edit.setText("1")
        if not self.smart_sync_candidates_edit.text().strip():
            self.smart_sync_candidates_edit.setText("3")

    def apply_local_fallback_preset(self) -> None:
        self._set_combo_value(self.mt_provider_combo, "hf")
        self.mt_model_edit.setText(LOCAL_MT_MODEL)
        self._set_combo_value(self.mt_strategy_combo, "per-segment")
        self._set_combo_value(self.mt_style_combo, "standard")
        self._set_combo_value(self.tts_provider_combo, "xtts")
        self._set_combo_value(self.smart_sync_mode_combo, "0")

    def browse_python(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите python.exe",
            str(Path(self.python_edit.text()).parent if self.python_edit.text() else PROJECT_ROOT),
            "Python executable (python.exe);;All files (*)",
        )
        if path:
            self.python_edit.setText(path)

    def browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео",
            str(PROJECT_ROOT / "data" / "input"),
            "Video files (*.mp4 *.mkv *.mov *.avi *.webm);;All files (*)",
        )
        if path:
            self.video_edit.setText(path)
            if not self.job_edit.text().strip():
                self.job_edit.setText(job_name_from_video(path))

    def _on_video_changed(self) -> None:
        self.refresh_artifacts()

    def _update_force_enabled(self) -> None:
        self.force_combo.setEnabled(self.resume_checkbox.isChecked())

    def _update_skip_metrics_enabled(self) -> None:
        self.skip_metrics_checkbox.setEnabled(self.step_combo.currentText() == "all")

    def _update_buttons(self) -> None:
        running = self.process is not None and self.process.state() != QProcess.NotRunning
        for button in (
            self.online_preset_button,
            self.local_preset_button,
            self.doctor_button,
            self.config_button,
            self.run_button,
        ):
            button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.open_output_button.setEnabled(not running)
        self.open_video_button.setEnabled(not running)
        self.open_report_button.setEnabled(not running)

    def run_doctor(self) -> None:
        self.start_process("doctor", self._build_command("doctor"))

    def show_config(self) -> None:
        self.start_process("config", self._build_command("config"))

    def run_pipeline(self) -> None:
        self.start_process("pipeline", self._build_command("pipeline"))

    def _build_command(self, kind: str) -> list[str]:
        python = self.python_edit.text().strip() or sys.executable
        args = [python, str(MAIN_SCRIPT)]

        if kind == "doctor":
            args.append("--doctor")
        elif kind == "config":
            args.append("--show-config")
        else:
            args.extend(["--step", self.step_combo.currentText()])

        video = self.video_edit.text().strip().strip('"')
        if video:
            args.extend(["--video", video])

        job = self.job_edit.text().strip()
        if job:
            args.extend(["--job-name", sanitize_job_name(job)])

        if self.test_checkbox.isChecked():
            args.append("--test")
        if (
            kind == "pipeline"
            and self.step_combo.currentText() == "all"
            and self.skip_metrics_checkbox.isChecked()
        ):
            args.append("--skip-metrics")
        if self.resume_checkbox.isChecked() and kind == "pipeline":
            args.append("--resume")
            force_step = combo_value(self.force_combo)
            if force_step:
                args.extend(["--force-step", force_step])

        self._append_combo_arg(args, "--mt-provider", self.mt_provider_combo)
        self._append_text_arg(args, "--mt-model", self.mt_model_edit)
        self._append_combo_arg(args, "--mt-strategy", self.mt_strategy_combo)
        self._append_combo_arg(args, "--mt-style", self.mt_style_combo)
        self._append_text_arg(args, "--mt-profile", self.mt_profile_edit)
        for correction in self._split_corrections(self.mt_corrections_edit.text()):
            args.extend(["--mt-asr-correction", correction])

        self._append_combo_arg(args, "--tts-provider", self.tts_provider_combo)
        self._append_text_arg(args, "--elevenlabs-voice-id", self.elevenlabs_voice_id_edit)
        self._append_text_arg(args, "--elevenlabs-voice-name", self.elevenlabs_voice_name_edit)
        if self.elevenlabs_no_clone_checkbox.isChecked():
            args.append("--elevenlabs-no-clone")

        self._append_combo_arg(args, "--subtitle-mode", self.subtitle_mode_combo)
        if self.subtitle_original_checkbox.isChecked():
            args.append("--subtitle-original")

        return args

    def _append_combo_arg(self, args: list[str], flag: str, combo: QComboBox) -> None:
        value = combo_value(combo)
        if value:
            args.extend([flag, value])

    def _append_text_arg(self, args: list[str], flag: str, edit: QLineEdit) -> None:
        value = edit.text().strip()
        if value:
            args.extend([flag, value])

    def _split_corrections(self, text: str) -> list[str]:
        if not text.strip():
            return []
        return [part.strip() for part in text.split("|") if part.strip()]

    def start_process(self, kind: str, command: list[str]) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Процесс уже запущен", "Дождитесь завершения текущей команды или остановите ее.")
            return

        python = Path(command[0])
        if not python.exists() and "\\" in command[0]:
            QMessageBox.critical(self, "Python не найден", command[0])
            return
        if not MAIN_SCRIPT.exists():
            QMessageBox.critical(self, "main.py не найден", str(MAIN_SCRIPT))
            return

        video = self.video_edit.text().strip().strip('"')
        if video and not Path(video).exists():
            QMessageBox.critical(self, "Видео не найдено", video)
            return

        self._save_settings()
        self.process_kind = kind
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self.process.setProgram(command[0])
        self.process.setArguments(command[1:])
        self.process.setProcessEnvironment(self._build_process_environment())
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)

        self.log_edit.clear()
        self.append_log("$ " + " ".join(f'"{part}"' if " " in part else part for part in command) + "\n\n")
        self.status_label.setText("Запущено: " + kind)
        self._update_buttons()
        self.process.start()

    def _build_process_environment(self) -> QProcessEnvironment:
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")

        mt_api_key_env = self.mt_openai_api_key_env_edit.text().strip()
        mt_base_url = self.mt_openai_base_url_edit.text().strip()
        smart_api_key_env = self.smart_sync_api_key_env_edit.text().strip()
        smart_base_url = self.smart_sync_base_url_edit.text().strip()

        self._insert_text_env(env, "MT_OPENAI_API_KEY_ENV", self.mt_openai_api_key_env_edit)
        self._insert_text_env(env, "MT_OPENAI_BASE_URL", self.mt_openai_base_url_edit)
        self._insert_combo_env(env, "MT_OPENAI_RESPONSE_FORMAT", self.mt_openai_response_format_combo)
        self._insert_combo_env(env, "SMART_SYNC_ENABLED", self.smart_sync_mode_combo)
        self._insert_combo_env(env, "SMART_SYNC_PROVIDER", self.smart_sync_provider_combo)
        self._insert_text_env(env, "SMART_SYNC_MODEL_NAME", self.smart_sync_model_edit)
        self._insert_text_env(env, "SMART_SYNC_API_KEY_ENV", self.smart_sync_api_key_env_edit)
        self._insert_text_env(env, "SMART_SYNC_BASE_URL", self.smart_sync_base_url_edit)
        self._insert_text_env(env, "SMART_SYNC_MAX_REWRITES", self.smart_sync_max_rewrites_edit)
        self._insert_text_env(env, "SMART_SYNC_CANDIDATES", self.smart_sync_candidates_edit)

        if mt_api_key_env and not smart_api_key_env and combo_value(self.smart_sync_provider_combo) == "openai_compatible":
            env.insert("SMART_SYNC_API_KEY_ENV", mt_api_key_env)
        if mt_base_url and not smart_base_url and combo_value(self.smart_sync_provider_combo) == "openai_compatible":
            env.insert("SMART_SYNC_BASE_URL", mt_base_url)
        return env

    def _insert_text_env(self, env: QProcessEnvironment, name: str, edit: QLineEdit) -> None:
        value = edit.text().strip()
        if value:
            env.insert(name, value)

    def _insert_combo_env(self, env: QProcessEnvironment, name: str, combo: QComboBox) -> None:
        value = combo_value(combo)
        if value:
            env.insert(name, value)

    def _read_stdout(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._consume_output(text)

    def _read_stderr(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        self._consume_output(text)

    def _consume_output(self, text: str) -> None:
        self.append_log(text)
        output_match = re.search(r"Выходная:\s*(.+)", text)
        if output_match:
            self.last_output_dir = Path(output_match.group(1).strip())
            self.refresh_artifacts()

    def append_log(self, text: str) -> None:
        cursor = self.log_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.log_edit.setTextCursor(cursor)
        self.log_edit.ensureCursorVisible()

    def _process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        status = "завершено" if exit_code == 0 and exit_status == QProcess.NormalExit else "ошибка"
        self.append_log(f"\n[{self.process_kind}: {status}, exit code {exit_code}]\n")
        self.status_label.setText(f"{self.process_kind}: {status}")
        self.refresh_artifacts()
        self._update_buttons()

    def _process_error(self, error) -> None:  # noqa: ANN001 - PySide enum varies by version.
        self.append_log(f"\n[QProcess error: {error}]\n")
        self.status_label.setText("Ошибка запуска")
        self._update_buttons()

    def stop_process(self) -> None:
        if self.process is None or self.process.state() == QProcess.NotRunning:
            return
        self.process.terminate()
        if not self.process.waitForFinished(3000):
            self.process.kill()
        self.status_label.setText("Остановлено")
        self._update_buttons()

    def current_job_name(self) -> str:
        explicit = self.job_edit.text().strip()
        if explicit:
            return sanitize_job_name(explicit)
        return job_name_from_video(self.video_edit.text().strip())

    def current_output_dir(self) -> Path:
        if self.last_output_dir is not None:
            return self.last_output_dir
        base = PROJECT_ROOT / "data" / ("test" if self.test_checkbox.isChecked() else "output")
        return base / self.current_job_name()

    def refresh_artifacts(self) -> None:
        output_dir = self.current_output_dir()
        self.artifacts_list.clear()
        header = QListWidgetItem(f"Output: {output_dir}")
        header.setFlags(header.flags() & ~Qt.ItemIsSelectable)
        self.artifacts_list.addItem(header)
        for relative_path, label in KNOWN_ARTIFACTS:
            path = output_dir / relative_path
            status = "OK" if path.exists() else "--"
            item = QListWidgetItem(f"{status:>2}  {label}: {relative_path}")
            item.setData(Qt.UserRole, str(path))
            if path.exists():
                item.setToolTip(str(path))
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self.artifacts_list.addItem(item)

    def _open_artifact_item(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if path:
            self.open_path(Path(path))

    def open_output_dir(self) -> None:
        self.open_path(self.current_output_dir())

    def open_artifact(self, relative_path: str) -> None:
        self.open_path(self.current_output_dir() / relative_path)

    def open_path(self, path: Path) -> None:
        if not path.exists():
            QMessageBox.information(self, "Файл не найден", str(path))
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def closeEvent(self, event) -> None:  # noqa: ANN001 - Qt event object.
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            answer = QMessageBox.question(
                self,
                "Остановить процесс?",
                "Пайплайн еще выполняется. Остановить его и закрыть приложение?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.stop_process()
        self._save_settings()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Video Dubbing Desktop")
    app.setOrganizationName("video-dubbing")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
