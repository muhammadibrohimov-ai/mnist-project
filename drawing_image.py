import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image

CANVAS_SIZE = 280  # ko'rinadigan canvas o'lchami (kattalik)
MNIST_SIZE = 28    # chiqadigan MNIST o'lchami

class DrawCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(CANVAS_SIZE, CANVAS_SIZE)
        self.pixmap = QtGui.QPixmap(self.size())
        self.pixmap.fill(QtGui.QColor('black'))  # fon qora (MNIST oq rangda yozadi)
        self.last_pos = None
        self.pen_width = 24  # qalinlik (MNIST uchun katta va zich bo'lishi yaxshi)
        self.pen_color = QtGui.QColor('white')  # oq yozuv (raqam)
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
        pen = QtGui.QPen(self.pen_color, self.pen_width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawPoint(pos)
        self.update()

    def _draw_line(self, p1, p2):
        painter = QtGui.QPainter(self.pixmap)
        pen = QtGui.QPen(self.pen_color, self.pen_width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(p1, p2)
        self.update()

    def clear(self):
        self.pixmap.fill(QtGui.QColor('black'))
        self.update()

    def get_image(self):
        """QPixmap -> PIL.Image (grayscale)"""
        qimg = self.pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB32)
        w = qimg.width()
        h = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
        # RGB32: arr[...,0:3] are B,G,R
        rgb = arr[..., :3][:, :, ::-1]  # convert BGR -> RGB
        pil = Image.fromarray(rgb)
        gray = pil.convert('L')  # grayscale
        return gray

    def get_mnist_array(self):
        """28x28 normalized array (0..1) with MNIST-style (white foreground on black background)"""
        gray = self.get_image()
        # Smooth & resize with ANTIALIAS then invert to get 0 background, 1 foreground like MNIST (optional)
        small = gray.resize((MNIST_SIZE, MNIST_SIZE), Image.LANCZOS)
        arr = np.array(small).astype(np.float32)
        # Normalize 0..1 and invert (MNIST digits are white on black; we'll keep 0 background)
        arr = arr / 255.0
        # Optionally invert so digit is high values (1.0) on black (0.0). Current: white stroke -> 1.0
        # Many ML pipelines expect 0..1 with 1=white stroke. Keep as is.
        return arr

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MNIST Drawer — PyQt5")
        self.canvas = DrawCanvas(self)
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setFixedSize(140, 140)
        self.preview_label.setStyleSheet("background: black; border: 1px solid #444;")
        self._build_ui()
        # Timer to refresh preview (so it's responsive while drawing)
        self.timer = QtCore.QTimer()
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.update_preview)
        self.timer.start()

    def _build_ui(self):
        btn_clear = QtWidgets.QPushButton("Clear")
        btn_clear.clicked.connect(self.canvas.clear)

        btn_save_png = QtWidgets.QPushButton("Save PNG")
        btn_save_png.clicked.connect(self.save_png)

        btn_save_npy = QtWidgets.QPushButton("Save .npy (28x28)")
        btn_save_npy.clicked.connect(self.save_npy)

        btn_zoom = QtWidgets.QPushButton("Preview 28×28 (show)")
        btn_zoom.clicked.connect(self.update_preview)

        controls = QtWidgets.QVBoxLayout()
        controls.addWidget(self.preview_label)
        controls.addSpacing(6)
        controls.addWidget(btn_zoom)
        controls.addWidget(btn_save_png)
        controls.addWidget(btn_save_npy)
        controls.addWidget(btn_clear)
        controls.addStretch(1)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addLayout(controls)

        # bottom: info label
        self.info = QtWidgets.QLabel("Chizish: sichqoncha bilan. Qalinlik: {} px".format(self.canvas.pen_width))
        v = QtWidgets.QVBoxLayout()
        v.addLayout(layout)
        v.addWidget(self.info)
        self.setLayout(v)

    def update_preview(self):
        mnist_arr = self.canvas.get_mnist_array()
        # Resize for preview display (scale up to fit preview_label) while preserving contrast
        img = Image.fromarray((mnist_arr * 255).astype(np.uint8))
        img = img.resize((self.preview_label.width(), self.preview_label.height()), Image.NEAREST)
        # Convert to QPixmap
        qt_img = self.pil2pixmap(img)
        self.preview_label.setPixmap(qt_img)

    def pil2pixmap(self, im):
        if im.mode != 'RGB':
            im = im.convert('RGB')
        data = im.tobytes("raw", "RGB")
        qimg = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        return pix

    def save_png(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save PNG", "", "PNG Files (*.png)")
        if path:
            img = self.canvas.get_image()
            img.save(path, format='PNG')
            QtWidgets.QMessageBox.information(self, "Saved", f"PNG saqlandi: {path}")

    def save_npy(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save .npy", "", "NumPy files (*.npy)")
        if path:
            arr = self.canvas.get_mnist_array()
            # optionally invert or normalize differently depending on your training pipeline
            np.save(path, arr)
            QtWidgets.QMessageBox.information(self, "Saved", f".npy saqlandi: {path}\nshape={arr.shape}, min={arr.min():.3f}, max={arr.max():.3f}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
