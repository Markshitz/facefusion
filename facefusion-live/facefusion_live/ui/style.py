DARK_QSS = """
* { font-family: "Segoe UI", "Inter", sans-serif; font-size: 10pt; color: #e5e7eb; }
QMainWindow, QWidget { background: #16181d; }
QTabWidget::pane { border: 0; }
QTabBar::tab { background: transparent; padding: 10px 22px; color: #9ca3af; font-weight: 600; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #f9fafb; border-bottom: 2px solid #6366f1; }
QTabBar::tab:hover { color: #f9fafb; }
QGroupBox { border: 1px solid #262a33; border-radius: 8px; margin-top: 14px; padding: 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #a5b4fc; }
QPushButton, QToolButton { background: #232733; border: 1px solid #313644; border-radius: 6px; padding: 7px 14px; }
QPushButton:hover, QToolButton:hover { background: #2b3040; }
QPushButton:disabled { color: #4b5563; background: #1c1f27; }
QPushButton#primary { background: #6366f1; border: 0; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #7577f5; }
QPushButton#primary:disabled { background: #2e3050; color: #6b7280; }
QPushButton#danger { background: #dc2626; border: 0; color: white; font-weight: 600; }
QToolButton:checked { background: #312e81; border: 2px solid #22c55e; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: #1f232c; border: 1px solid #313644; border-radius: 6px; padding: 5px 8px; }
QComboBox QAbstractItemView { background: #1f232c; selection-background-color: #4338ca; }
QSlider::groove:horizontal { height: 4px; background: #313644; border-radius: 2px; }
QSlider::handle:horizontal { background: #818cf8; width: 14px; margin: -6px 0; border-radius: 7px; }
QSlider::sub-page:horizontal { background: #6366f1; border-radius: 2px; }
QProgressBar { background: #1f232c; border: 0; border-radius: 6px; height: 14px; text-align: center; }
QProgressBar::chunk { background: #6366f1; border-radius: 6px; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #6b7280; border-radius: 4px; background: #1f232c; }
QCheckBox::indicator:checked { background: #6366f1; border-color: #a5b4fc; }
QLabel#muted { color: #9ca3af; }
QLabel#stat { color: #a5b4fc; font-family: "Consolas", monospace; font-size: 10pt; }
QLabel#banner { background: #3b2f0b; color: #fde68a; border-radius: 6px; padding: 8px; }
QLabel#error { color: #fca5a5; }
QTableWidget { background: #1a1d24; gridline-color: #262a33; border: 0; }
QHeaderView::section { background: #1f232c; border: 0; padding: 6px; color: #9ca3af; }
QScrollArea { border: 0; }
QStatusBar { background: #111318; color: #9ca3af; }
"""
