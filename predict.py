import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Ishlatilayotgan qurilma:", device)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # 28x28 -> 14x14
            nn.Conv2d(16, 32, 3, 1, 1),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # 14x14 -> 7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("simplecnn_state.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === PyQt GUI ===
CANVAS_SIZE = 400
MNIST_SIZE = 28

class DrawCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(CANVAS_SIZE, CANVAS_SIZE)
        self.pixmap = QtGui.QPixmap(self.size())
        self.pixmap.fill(QtGui.QColor('black'))
        self.last_pos = None
        self.pen_width = 24
        self.pen_color = QtGui.QColor('white')
        self.setCursor(QtCore.Qt.CrossCursor)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.last_pos = event.pos()
            self._draw_point(self.last_pos)

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton and self.last_pos is not None:
            current = event.pos()
            self._draw_line(self.last_pos, current)
            self.last_pos = current

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.last_pos = None

    def _draw_point(self, pos):
        painter = QtGui.QPainter(self.pixmap)
        pen = QtGui.QPen(self.pen_color, self.pen_width, QtCore.Qt.SolidLine,
                         QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawPoint(pos)
        self.update()

    def _draw_line(self, p1, p2):
        painter = QtGui.QPainter(self.pixmap)
        pen = QtGui.QPen(self.pen_color, self.pen_width, QtCore.Qt.SolidLine,
                         QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(p1, p2)
        self.update()

    def clear(self):
        self.pixmap.fill(QtGui.QColor('black'))
        self.update()

    def get_image(self):
        qimg = self.pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB32)
        w, h = qimg.width(), qimg.height()
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
        rgb = arr[..., :3][:, :, ::-1]
        pil = Image.fromarray(rgb)
        gray = pil.convert('L')
        return gray

    def get_mnist_array(self):
        gray = self.get_image()
        small = gray.resize((MNIST_SIZE, MNIST_SIZE), Image.LANCZOS)
        arr = np.array(small).astype(np.float32) / 255.0
        return arr

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MNIST Drawer + Predictor")
        self.canvas = DrawCanvas(self)
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setFixedSize(140, 140)
        self.preview_label.setStyleSheet("background: black; border: 1px solid #444;")
        self.pred_label = QtWidgets.QLabel("Natija: —")
        self.pred_label.setStyleSheet("font-size: 18px; color: white; background: #333; padding: 6px; border-radius: 6px;")
        self.pred_label.setAlignment(QtCore.Qt.AlignCenter)
        self.pred_label.setFixedHeight(40)
        self._build_ui()

    def _build_ui(self):
        btn_clear = QtWidgets.QPushButton("Clear")
        btn_clear.clicked.connect(self.canvas.clear)

        btn_predict = QtWidgets.QPushButton("Predict")
        btn_predict.clicked.connect(self.predict_digit)

        controls = QtWidgets.QVBoxLayout()
        controls.addWidget(self.preview_label)
        controls.addWidget(self.pred_label)
        controls.addSpacing(6)
        controls.addWidget(btn_predict)
        controls.addWidget(btn_clear)
        controls.addStretch(1)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addLayout(controls)

        self.setLayout(layout)
        self.setStyleSheet("background-color: #222; color: white;")

    def predict_digit(self):
        arr = self.canvas.get_mnist_array()
        img = Image.fromarray((arr * 255).astype(np.uint8))

        qt_img = self.pil2pixmap(img.resize((140, 140), Image.NEAREST))
        self.preview_label.setPixmap(qt_img)

        # Tensorga o‘tkazish
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(dim=1).item()

        self.pred_label.setText(f"Natija: {pred}")

    def pil2pixmap(self, im):
        if im.mode != 'RGB':
            im = im.convert('RGB')
        data = im.tobytes("raw", "RGB")
        qimg = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        return pix

def main():
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
