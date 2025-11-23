import sys
import csv
import numpy as np
from scipy.io.wavfile import write
from scipy.fft import fft, fftfreq
from scipy import signal
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QComboBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)
import pyqtgraph as pg

class Generator:
    def __init__(self, fs, duration):
        self.fs = fs
        self.duration = duration
        self.t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        self.y = np.zeros_like(self.t)

    def set_params(self, fs, duration):
        self.fs = fs
        self.duration = duration
        self.t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    def sine(self, f, A):
        self.y = A * np.sin(2 * np.pi * f * self.t)
        return self.t, self.y

    def square(self, f, A):
        self.y = A * signal.square(2 * np.pi * f * self.t)
        return self.t, self.y

    def sawtooth(self, f, A):
        self.y = A * signal.sawtooth(2 * np.pi * f * self.t)
        return self.t, self.y

    def triangle(self, f, A):
        self.y = A * signal.sawtooth(2 * np.pi * f * self.t, width=1)
        return self.t, self.y

    def white_noise(self, A):
        self.y = np.random.uniform(-A, A, len(self.t))
        return self.t, self.y

    def get_fft(self):
        N = len(self.t)
        dt = 1 / self.fs
        yf = 2.0 / N * np.abs(fft(self.y)[0:N // 2])
        xf = fftfreq(N, d=dt)[0:N // 2]
        return xf, yf

    def save_wav(self, filename):
        scaled = np.int16(self.y / np.max(np.abs(self.y)) * 32767)
        write(filename, int(self.fs), scaled)
        print("wav saved to ", filename)

    def save_signal_csv(self, filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Amplitude'])
            for t_val, y_val in zip(self.t, self.y):
                writer.writerow([t_val, y_val])
        print("signal saved to ", filename)

    def save_fft_csv(self, filename):
        xf, yf = self.get_fft()
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frequency', 'Magnitude'])
            for f_val, m_val in zip(xf, yf):
                writer.writerow([f_val, m_val])
        print("fft saved to ", filename)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.gen = Generator(44100, 1.0)
        self.initUI()
        self.update_data()

    def initUI(self):
        self.setWindowTitle('Generator Sygnałów i Analiza FFT')
        self.setGeometry(100, 100, 1200, 800)

        main_layout = QVBoxLayout()

        controls_layout = QHBoxLayout()
        
        self.combo_type = QComboBox()
        self.combo_type.addItems(['Sine', 'Square', 'Sawtooth', 'Triangle', 'WhiteNoise'])
        self.combo_type.currentIndexChanged.connect(self.update_data)
        controls_layout.addWidget(QLabel("Typ:"))
        controls_layout.addWidget(self.combo_type)

        self.spin_freq = QDoubleSpinBox()
        self.spin_freq.setRange(1, 20000)
        self.spin_freq.setValue(440)
        self.spin_freq.setPrefix("f: ")
        self.spin_freq.setSuffix(" Hz")
        self.spin_freq.valueChanged.connect(self.update_data)
        controls_layout.addWidget(self.spin_freq)

        self.spin_amp = QDoubleSpinBox()
        self.spin_amp.setRange(0.1, 1.0)
        self.spin_amp.setSingleStep(0.1)
        self.spin_amp.setValue(1.0)
        self.spin_amp.setPrefix("A: ")
        self.spin_amp.valueChanged.connect(self.update_data)
        controls_layout.addWidget(self.spin_amp)

        self.spin_fs = QDoubleSpinBox()
        self.spin_fs.setRange(1000, 96000)
        self.spin_fs.setValue(44100)
        self.spin_fs.setPrefix("Fs: ")
        self.spin_fs.valueChanged.connect(self.update_data)
        controls_layout.addWidget(self.spin_fs)

        self.spin_dur = QDoubleSpinBox()
        self.spin_dur.setRange(0.1, 10.0)
        self.spin_dur.setValue(1.0)
        self.spin_dur.setPrefix("T: ")
        self.spin_dur.setSuffix(" s")
        self.spin_dur.valueChanged.connect(self.update_data)
        controls_layout.addWidget(self.spin_dur)

        btn_wav = QPushButton("Zapisz WAV")
        btn_wav.clicked.connect(lambda: self.gen.save_wav("output.wav"))
        controls_layout.addWidget(btn_wav)

        btn_csv_sig = QPushButton("Zapisz CSV (Sygnał)")
        btn_csv_sig.clicked.connect(lambda: self.gen.save_signal_csv("signal.csv"))
        controls_layout.addWidget(btn_csv_sig)

        btn_csv_fft = QPushButton("Zapisz CSV (FFT)")
        btn_csv_fft.clicked.connect(lambda: self.gen.save_fft_csv("fft.csv"))
        controls_layout.addWidget(btn_csv_fft)

        main_layout.addLayout(controls_layout)

        graphs_layout = QHBoxLayout()

        self.plot_time = pg.PlotWidget(title="Przebieg czasowy")
        self.plot_time.setLabel('left', 'Amplituda')
        self.plot_time.setLabel('bottom', 'Czas [s]')
        self.plot_time.showGrid(x=True, y=True)
        graphs_layout.addWidget(self.plot_time)

        self.plot_fft = pg.PlotWidget(title="Transformata Fouriera")
        self.plot_fft.setLabel('left', 'Amplituda')
        self.plot_fft.setLabel('bottom', 'Częstotliwość [Hz]')
        self.plot_fft.showGrid(x=True, y=True)
        graphs_layout.addWidget(self.plot_fft)

        main_layout.addLayout(graphs_layout)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Czas [s]', 'Wartość'])
        # Cała szerokość, zamiast małej tabelki
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        main_layout.addWidget(self.table)

        self.setLayout(main_layout)

    def update_data(self):
        fs = self.spin_fs.value()
        dur = self.spin_dur.value()
        freq = self.spin_freq.value()
        amp = self.spin_amp.value()
        sig_type = self.combo_type.currentText()

        self.gen.set_params(fs, dur)

        if sig_type == 'Sine':
            t, y = self.gen.sine(freq, amp)
        elif sig_type == 'Square':
            t, y = self.gen.square(freq, amp)
        elif sig_type == 'Sawtooth':
            t, y = self.gen.sawtooth(freq, amp)
        elif sig_type == 'Triangle':
            t, y = self.gen.triangle(freq, amp)
        else:
            t, y = self.gen.white_noise(amp)

        self.plot_time.clear()
        self.plot_time.plot(t, y, pen='y')
        self.plot_time.setXRange(0, 0.01)

        xf, yf = self.gen.get_fft()
        self.plot_fft.clear()
        self.plot_fft.plot(xf, yf, pen='r')
        self.plot_fft.setXRange(0, 1000)

        self.table.setRowCount(100)
        for i in range(100):
            self.table.setItem(i, 0, QTableWidgetItem(f"{t[i]:.6f}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{y[i]:.6f}"))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
