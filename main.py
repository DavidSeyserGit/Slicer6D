# gui.py
import sys
import os
import numpy as np
import pyvista as pv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFrame, QGridLayout,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QLabel, QSplitter, QStatusBar
)
from PyQt5.QtCore import Qt
from pyvistaqt import QtInteractor

import deformer  # our deformation module

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Print Overhang Deformer")
        self.resize(1200, 800)

        # Model data
        self.original_mesh = None
        self.deformed_mesh = None
        self.overhang_faces = None
        self.print_bed = None

        # Central widget & layout
        central = QWidget()
        self.setCentralWidget(central)
        hl = QHBoxLayout(central)

        # Splitter: controls | viewer
        splitter = QSplitter(Qt.Horizontal)
        hl.addWidget(splitter)

        # --- Control panel
        ctrl = QWidget()
        ctrl.setMinimumWidth(300)
        ctrl.setMaximumWidth(400)
        vctrl = QVBoxLayout(ctrl)

        self._build_file_group(vctrl)
        self._build_view_group(vctrl)
        self._build_orientation_group(vctrl)
        self._build_overhang_group(vctrl)
        self._build_deform_group(vctrl)

        vctrl.addStretch()
        splitter.addWidget(ctrl)

        # --- Viewer
        viewer_frame = QFrame()
        viewer_layout = QVBoxLayout(viewer_frame)
        self.plotter = QtInteractor(viewer_frame)
        viewer_layout.addWidget(self.plotter)
        splitter.addWidget(viewer_frame)

        # Initial plotter setup
        self.plotter.set_background([0.2, 0.2, 0.2])
        self.plotter.add_axes(interactive=True, line_width=3)
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.plotter.add_text("Load a model to begin", font_size=16,
                              position="upper_left")

    def _build_file_group(self, parent):
        gb = QGroupBox("File")
        l = QVBoxLayout(gb)
        btn = QPushButton("Load STL")
        btn.clicked.connect(self.load_model)
        l.addWidget(btn)
        hb = QHBoxLayout()
        b1 = QPushButton("Export Orig")
        b1.clicked.connect(self.export_original)
        b2 = QPushButton("Export Def")
        b2.clicked.connect(self.export_deformed)
        hb.addWidget(b1); hb.addWidget(b2)
        l.addLayout(hb)
        parent.addWidget(gb)

    def _build_view_group(self, parent):
        gb = QGroupBox("View")
        g = QGridLayout(gb)
        bxy = QPushButton("Top XY"); bxy.clicked.connect(lambda: self.plotter.view_xy())
        bxz = QPushButton("Front XZ"); bxz.clicked.connect(lambda: self.plotter.view_xz())
        byz = QPushButton("Side YZ"); byz.clicked.connect(lambda: self.plotter.view_yz())
        bis = QPushButton("Iso"); bis.clicked.connect(lambda: self.plotter.view_isometric())
        g.addWidget(bxy,0,0); g.addWidget(bxz,0,1)
        g.addWidget(byz,1,0); g.addWidget(bis,1,1)
        self.wf_cb = QCheckBox("Wireframe"); self.wf_cb.stateChanged.connect(self.update_display)
        self.ed_cb = QCheckBox("Show Edges"); self.ed_cb.setChecked(True)
        self.ed_cb.stateChanged.connect(self.update_display)
        g.addWidget(self.wf_cb,2,0); g.addWidget(self.ed_cb,2,1)
        parent.addWidget(gb)

    def _build_orientation_group(self, parent):
        gb = QGroupBox("Print Orientation")
        l = QVBoxLayout(gb)
        h = QHBoxLayout()
        h.addWidget(QLabel("Base:"))
        self.orient_combo = QComboBox()
        self.orient_combo.addItems([
            "Bottom (XY)", "Front (XZ)", "Side (YZ)",
            "Top (-XY)", "Back (-XZ)", "Other (-YZ)"
        ])
        h.addWidget(self.orient_combo)
        l.addLayout(h)
        btn = QPushButton("Set Orientation")
        btn.clicked.connect(self.set_orientation)
        l.addWidget(btn)
        self.bed_cb = QCheckBox("Show Bed"); self.bed_cb.setChecked(True)
        self.bed_cb.stateChanged.connect(self.update_display)
        l.addWidget(self.bed_cb)
        parent.addWidget(gb)

    def _build_overhang_group(self, parent):
        gb = QGroupBox("Overhang Detection")
        l = QVBoxLayout(gb)
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Angle:"))
        self.ang_sb = QSpinBox(); self.ang_sb.setRange(20,90); self.ang_sb.setValue(45)
        self.ang_sb.setSuffix("°"); h1.addWidget(self.ang_sb)
        l.addLayout(h1)
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Support Dist:"))
        self.sup_sb = QDoubleSpinBox(); self.sup_sb.setRange(0.1,20.0)
        self.sup_sb.setValue(5.0); self.sup_sb.setSuffix("mm")
        self.sup_sb.setSingleStep(0.5); h2.addWidget(self.sup_sb)
        l.addLayout(h2)
        self.ray_cb = QCheckBox("Use Ray Trace"); self.ray_cb.setChecked(True)
        l.addWidget(self.ray_cb)
        btn = QPushButton("Detect Overhangs"); btn.clicked.connect(self.detect_overhangs)
        l.addWidget(btn)
        parent.addWidget(gb)

    def _build_deform_group(self, parent):
        gb = QGroupBox("Deformation")
        l = QVBoxLayout(gb)
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Strength:"))
        self.str_sb = QSpinBox(); self.str_sb.setRange(1,100); self.str_sb.setValue(20)
        self.str_sb.setSuffix("%"); h1.addWidget(self.str_sb)
        l.addLayout(h1)
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Smooth iters:"))
        self.smo_sb = QSpinBox(); self.smo_sb.setRange(0,10); self.smo_sb.setValue(3)
        h2.addWidget(self.smo_sb)
        l.addLayout(h2)
        hb = QHBoxLayout()
        b1 = QPushButton("Deform"); b1.clicked.connect(self.deform_model)
        b2 = QPushButton("Reset"); b2.clicked.connect(self.reset_model)
        hb.addWidget(b1); hb.addWidget(b2)
        l.addLayout(hb)
        parent.addWidget(gb)

    def load_model(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open STL", "", "STL Files (*.stl)")
        if not fn: return
        self.original_mesh = pv.read(fn)
        self.deformed_mesh = None
        self.overhang_faces = None
        self.print_bed = deformer.determine_print_bed(self.original_mesh)
        self.display_original()
        self.status.showMessage(f"Loaded {os.path.basename(fn)}")

    def display_original(self):
        self.plotter.clear()
        mesh = self.deformed_mesh if self.deformed_mesh is not None else self.original_mesh
        if self.wf_cb.isChecked():
            self.plotter.add_mesh(mesh, style='wireframe', color='white')
        else:
            self.plotter.add_mesh(mesh,
                show_edges=self.ed_cb.isChecked(), color='white')
        self.plotter.add_axes(interactive=True, line_width=3)
        if self.bed_cb.isChecked(): self._draw_bed()
        self.plotter.reset_camera()

    def _draw_bed(self):
        if self.print_bed is None: return
        nb, pt = self.print_bed
        bnds = self.original_mesh.bounds
        xs, ys = bnds[1]-bnds[0], bnds[3]-bnds[2]
        c = ((bnds[0]+bnds[1])/2,(bnds[2]+bnds[3])/2,bnds[4])
        plane = pv.Plane(center=c, direction=nb,
            i_size=xs*1.2, j_size=ys*1.2)
        self.plotter.add_mesh(plane, color='lightblue', opacity=0.3)

    def set_orientation(self):
        if self.original_mesh is None: return
        txt = self.orient_combo.currentText()
        m = self.original_mesh.copy()
        if txt=="Front (XZ)": m.rotate_x(90)
        elif txt=="Side (YZ)": m.rotate_y(90)
        elif txt=="Top (-XY)": m.rotate_x(180)
        elif txt=="Back (-XZ)": m.rotate_x(-90)
        elif txt=="Other (-YZ)": m.rotate_y(-90)
        self.original_mesh = m
        self.deformed_mesh = None
        self.overhang_faces = None
        self.print_bed = deformer.determine_print_bed(m)
        self.display_original()
        self.status.showMessage(f"Orientation set to {txt}")

    def detect_overhangs(self):
        """Detect overhang faces (angle‑only + optional ray filtering)."""
        if self.original_mesh is None:
            return

        angle = self.ang_sb.value()
        support_dist = self.sup_sb.value()
        use_ray      = self.ray_cb.isChecked()

        # 1) Pure angle‑based overhangs
        faces = deformer.detect_overhangs(
            self.original_mesh,
            angle_threshold=angle,
            print_bed=self.print_bed
        )

        # 2) Optional ray‑cast filter to keep only truly unsupported
        if use_ray and faces.size > 0:
            centers = self.original_mesh.cell_centers().points
            filtered = []
            for f in faces:
                c = centers[f]
                hits, _ = self.original_mesh.ray_trace(
                    c, [0, 0, -1], first_point=True
                )
                # keep if no hit or hit is farther than support_dist
                if len(hits) == 0 or np.linalg.norm(hits[0] - c) > support_dist:
                    filtered.append(f)
            faces = np.array(filtered, dtype=int)

        self.overhang_faces = faces

        # 3) Visualize
        self.display_original()
        if faces.size > 0:
            mesh_over = self.original_mesh.extract_cells(faces)
            self.plotter.add_mesh(mesh_over, color='red', opacity=0.8)
            self.plotter.add_text(
                f"{faces.size} overhangs", position='upper_left', font_size=14
            )
            self.status.showMessage(f"Detected {faces.size} overhangs")
        else:
            self.plotter.add_text(
                "No overhangs detected", position='upper_left', font_size=14
            )
            self.status.showMessage("No overhangs detected")

    def deform_model(self):
        """Iteratively deform until no overhangs remain."""
        if self.original_mesh is None:
            self.statusBar().showMessage("No model loaded")
            return

        # If no overhangs yet, run detection
        if self.overhang_faces is None or self.overhang_faces.size == 0:
            self.detect_overhangs()
            if self.overhang_faces is None or self.overhang_faces.size == 0:
                self.statusBar().showMessage("No overhangs to deform")
                return

        # Grab parameters
        angle   = self.ang_sb.value()
        strength= self.str_sb.value()
        support = self.sup_sb.value()
        use_ray = self.ray_cb.isChecked()
        max_iter= 10  # you can expose this if you like

        # Run the new deform_mesh signature
        self.deformed_mesh, modified_vertices = deformer.deform_mesh(
            self.original_mesh,
            angle_threshold   = angle,
            print_bed         = self.print_bed,
            support_distance  = support,
            use_ray           = use_ray,
            max_iterations    = max_iter
        )

        # Optional smoothing
        n_smooth = self.smo_sb.value()
        if n_smooth > 0:
            self.deformed_mesh.smooth(n_iter=n_smooth, inplace=True)

        # Redraw original (wireframe) + deformed (solid)
        self.plotter.clear()
        self.plotter.add_mesh(
            self.original_mesh,
            style='wireframe',
            color='white',
            opacity=0.3
        )
        self.plotter.add_mesh(
            self.deformed_mesh,
            show_edges=self.ed_cb.isChecked(),
            color='cyan'
        )
        self.plotter.add_axes(interactive=True, line_width=3)
        if self.bed_cb.isChecked():
            self._draw_bed()
        self.plotter.add_text(
            f"Deformed {len(modified_vertices)} vertices",
            position='upper_left',
            font_size=14
        )
        self.statusBar().showMessage(f"Deformed {len(modified_vertices)} vertices")

    def reset_model(self):
        self.deformed_mesh = None
        self.display_original()
        self.status.showMessage("Reset to original")

    def update_display(self):
        # simply redraw current state
        if self.deformed_mesh:
            self.deformed_mesh, _ = deformer.deform_mesh(
                self.original_mesh, self.overhang_faces, self.str_sb.value(),
                self.print_bed)
        self.display_original()

    def export_original(self):
        if not self.original_mesh: return
        fn, _ = QFileDialog.getSaveFileName(self,
            "Save Original", "", "STL (*.stl)")
        if fn: self.original_mesh.save(fn); self.status.showMessage(f"Saved {fn}")

    def export_deformed(self):
        if not self.deformed_mesh: return
        fn, _ = QFileDialog.getSaveFileName(self,
            "Save Deformed", "", "STL (*.stl)")
        if fn: self.deformed_mesh.save(fn); self.status.showMessage(f"Saved {fn}")

if __name__=="__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
