import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QAction, QMessageBox, QInputDialog,
    QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QTextEdit,QDialog
)
from PyQt5.QtCore import Qt

from romgeo_spg.spg_file import SPGFile
from romgeo_spg.plot import visualize_heatmap
from romgeo_spg.compare import compare_geoid_heights, compare_geodetic_shifts, compare_geoid_heights_interpolated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class SPGManagerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPG File Manager")
        self.setGeometry(100, 100, 1400, 800)
        self.spg: SPGFile = SPGFile()

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.export_actions = []
        self.preview_actions = []
        self.compare_action = None  # <-- Add this
        self.init_menu()
        self.init_tabs()
        self.update_export_preview_actions()

    def init_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        load_action = QAction("Load SPG", self)
        load_action.triggered.connect(self.load_spg)
        file_menu.addAction(load_action)

        save_action = QAction("Save SPG", self)
        save_action.triggered.connect(self.save_spg)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        show_structure_action = QAction("Show SPG Structure", self)
        show_structure_action.triggered.connect(self.show_spg_structure)
        file_menu.addAction(show_structure_action)

        file_menu.addSeparator()

        new_action = QAction("New SPG", self)
        new_action.triggered.connect(lambda: (
            self.spg.reset(),
            self.tree_widget.clear(),
            self.load_grids_to_tables(),
            self.show_edit_tree()
        ))
        new_action.setEnabled(False)  # Initially disabled, enable when needed
        file_menu.addAction(new_action)
        
        add_cnc_action = QAction("Add/Reset CNC Metadata", self)
        add_cnc_action.triggered.connect(
            lambda: (
            self.spg.append_cnc_metadata(),
            self.tree_widget.clear() if hasattr(self, "tree_widget") else None,
            self.show_edit_tree() if hasattr(self, "tree_widget") else None
            )
        )
        file_menu.addAction(add_cnc_action)

        file_menu.addSeparator()
        
        self.compare_action = QAction("Compare SPG", self)  # <-- Store as instance variable
        self.compare_action.triggered.connect(self.compare_spg)
        self.compare_action.setEnabled(False)  # Initially disabled
        file_menu.addAction(self.compare_action)
        
        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        file_menu.addSeparator()    

        import_menu = menubar.addMenu("Import")
        import_menu.addAction("Import X Grid CSV", lambda: self.import_grid_csv(self.geodetic_x_table))
        import_menu.addAction("Import Y Grid CSV", lambda: self.import_grid_csv(self.geodetic_y_table))
        import_menu.addAction("Import Z Grid CSV", lambda: self.import_grid_csv(self.geoid_table))

        export_menu = menubar.addMenu("Export")
        self.export_actions = [
            export_menu.addAction("Export X Grid CSV", lambda: self.export_grid_csv(self.geodetic_x_table)),
            export_menu.addAction("Export Y Grid CSV", lambda: self.export_grid_csv(self.geodetic_y_table)),
            export_menu.addAction("Export Z Grid CSV", lambda: self.export_grid_csv(self.geoid_table)),
            export_menu.addSeparator(),
            export_menu.addAction("Export X@Y Grid CSV", self.export_xy_grid_csv),
            export_menu.addSeparator(),
            export_menu.addAction("Export Surfer 6 GRT XY", self.export_legacy_xy_grt),
            export_menu.addAction("Export Surfer 6 GRT Z",  self.export_legacy_z_grt),
            export_menu.addSeparator(),
            export_menu.addAction("Export Legacy TXT XY", self.export_legacy_txt_xy),
            export_menu.addAction("Export Legacy TXT Z",  self.export_legacy_txt_z)
        ]

        preview_menu = menubar.addMenu("Preview")
        self.preview_actions = [
            preview_menu.addAction("Geodetic Shifts X",   lambda: self.visualize_geodetic_shifts('x')),
            preview_menu.addAction("Geodetic Shifts y",   lambda: self.visualize_geodetic_shifts('y')),
            preview_menu.addAction("Geodetic Shifts X*Y", lambda: self.visualize_geodetic_shifts('both')),
            preview_menu.addSeparator(),
            preview_menu.addAction("Geoid Heights Z", lambda: self.visualize_geoid_heights()),
            preview_menu.addSeparator(),
            preview_menu.addAction("Geodetic Shifts X (by index)",   lambda: self.visualize_geodetic_shifts('x', use_index=True)),
            preview_menu.addAction("Geodetic Shifts y (by index)",   lambda: self.visualize_geodetic_shifts('y', use_index=True)),
            preview_menu.addAction("Geodetic Shifts X*Y (by index)", lambda: self.visualize_geodetic_shifts('both', use_index=True)),
            preview_menu.addSeparator(),
            preview_menu.addAction("Geoid Heights Z (by index)", lambda: self.visualize_geoid_heights(use_index=True))
        ]

    def update_export_preview_actions(self):
        has_data = self.geodetic_x_table.rowCount() > 0 and self.geodetic_x_table.columnCount() > 0
        for action in self.export_actions:
            if isinstance(action, QAction):
                action.setEnabled(has_data)
        for action in self.preview_actions:
            if isinstance(action, QAction):
                action.setEnabled(has_data)
        if self.compare_action:
            self.compare_action.setEnabled(has_data)  # <-- Enable/disable Compare SPG

    def init_tabs(self):
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Key", "Value"])
        self.tree_widget.itemDoubleClicked.connect(self.edit_tree_item)
        self.tabs.addTab(self.tree_widget, "SPG Tree")

        self.geodetic_x_table = QTableWidget()
        self.tabs.addTab(self.geodetic_x_table, "Geodetic X Grid")

        self.geodetic_y_table = QTableWidget()
        self.tabs.addTab(self.geodetic_y_table, "Geodetic Y Grid")

        self.geoid_table = QTableWidget()
        self.tabs.addTab(self.geoid_table, "Geoid Heights Grid")

    def show_spg_structure(self):

        if self.spg:

            structure = self.spg.print_tree_structure()
            dlg = QDialog(self)
            dlg.setWindowTitle("SPG Structure")
            dlg.resize(800, 600)
            layout = QVBoxLayout(dlg)
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setPlainText(structure)
            layout.addWidget(text_edit)
            btn = QPushButton("Close")
            btn.clicked.connect(dlg.accept)
            layout.addWidget(btn)
            dlg.setLayout(layout)
            dlg.exec_()
        else:
            QMessageBox.warning(self, "No SPG Loaded", "Please load an SPG file first.")

    def load_spg(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open SPG File", "", "SPG Files (*.spg)")
        if file_path:
            try:
                self.spg = SPGFile(file_path)
                self.tree_widget.clear()
                self.load_grids_to_tables()
                #QMessageBox.information(self, "Loaded", f"Loaded SPG File: {file_path}")
                self.show_edit_tree()
                self.setWindowTitle(f"SPG File Manager - {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load SPG file: {str(e)}")

    def save_spg(self):
        if self.spg:
            self.save_grids_from_tables()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save SPG File", "", "SPG Files (*.spg)")
            if file_path:
                self.spg.save_spg(file_path)
                QMessageBox.information(self, "Saved", "SPG file saved successfully.")

    def load_grids_to_tables(self):
        # Use the new data access pattern
        x_grid = self.spg.data["grids"]["geodetic_shifts"]["grid"][0]
        y_grid = self.spg.data["grids"]["geodetic_shifts"]["grid"][1]
        z_grid = self.spg.data["grids"]["geoid_heights"]["grid"][0]
        self._load_grid_to_table(self.geodetic_x_table, x_grid)
        self._load_grid_to_table(self.geodetic_y_table, y_grid)
        self._load_grid_to_table(self.geoid_table, z_grid)
        self.update_export_preview_actions()  # <-- Add this

    def _load_grid_to_table(self, table, grid):
        rows, cols = grid.shape
        table.setRowCount(rows)
        table.setColumnCount(cols)
        for i in range(rows):
            for j in range(cols):
                value = str(grid[i, j])
                item = QTableWidgetItem(value)
                table.setItem(i, j, item)

    def save_grids_from_tables(self):
        def table_to_array(table):
            rows, cols = table.rowCount(), table.columnCount()
            arr = np.zeros((rows, cols), dtype=np.float32)
            for i in range(rows):
                for j in range(cols):
                    try:
                        arr[i, j] = float(table.item(i, j).text())
                    except:
                        arr[i, j] = 0.0
            return arr

        x_array = table_to_array(self.geodetic_x_table)
        y_array = table_to_array(self.geodetic_y_table)
        
        if x_array.shape != y_array.shape:
            QMessageBox.critical(self, "Shape Mismatch", "Geodetic X and Y grids must have the same shape.")
            return

        # Check and update geodetic metadata shape
        x_rows, x_cols = x_array.shape
        y_rows, y_cols = y_array.shape
        geo_rows, geo_cols = self.geoid_table.rowCount(), self.geoid_table.columnCount()

        # Geodetic metadata update shape
        gmeta = self.spg.data["grids"]["geodetic_shifts"]["metadata"]
        if gmeta.get("nrows") != x_rows or gmeta.get("ncols") != x_cols:
            reply = QMessageBox.question(
                self, "Update Geodetic Metadata",
                f"Geodetic grid shape ({x_rows}x{x_cols}) does not match metadata (nrows={gmeta.get('nrows')}, ncols={gmeta.get('ncols')}). Update metadata?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                gmeta["nrows"] = x_rows
                gmeta["ncols"] = x_cols

        # Geoid metadata update shape
        zmeta = self.spg.data["grids"]["geoid_heights"]["metadata"]
        if zmeta.get("nrows") != geo_rows or zmeta.get("ncols") != geo_cols:
            reply = QMessageBox.question(
                self, "Update Geoid Metadata",
                f"Geoid grid shape ({geo_rows}x{geo_cols}) does not match metadata (nrows={zmeta.get('nrows')}, ncols={zmeta.get('ncols')}). Update metadata?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                zmeta["nrows"] = geo_rows
                zmeta["ncols"] = geo_cols

        # Assign arrays to the correct place in the data structure
        self.spg.data["grids"]["geodetic_shifts"]["grid"] = np.stack([x_array, y_array], axis=0)
        self.spg.data["grids"]["geoid_heights"]["grid"][0] = np.reshape(table_to_array(self.geoid_table), (geo_rows, geo_cols))

        self.tree_widget.clear()
        self.show_edit_tree()
        self.update_export_preview_actions()  # <-- Add this

    def show_edit_tree(self):
        root = QTreeWidgetItem(["SPG Structure"])
        self.tree_widget.addTopLevelItem(root)
        self.populate_tree(root, self.spg.data)
        self.tree_widget.expandAll()
        self.tree_widget.resizeColumnToContents(0)
        self.tree_widget.resizeColumnToContents(1)
        max_width = 400
        if self.tree_widget.columnWidth(1) > max_width:
            self.tree_widget.setColumnWidth(1, max_width)

    def populate_tree(self, parent, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                item = QTreeWidgetItem([str(key)])
                parent.addChild(item)
                self.populate_tree(item, value)
            elif isinstance(value, np.ndarray):
                if key == "grid":
                    parent_key = parent.text(0).lower()
                    if "geodetic_shifts" in parent_key:
                        x_item = QTreeWidgetItem(["Geodetic X Grid", "[Open X Tab]"])
                        y_item = QTreeWidgetItem(["Geodetic Y Grid", "[Open Y Tab]"])
                        x_item.setData(1, Qt.UserRole, self.tabs.indexOf(self.geodetic_x_table))
                        y_item.setData(1, Qt.UserRole, self.tabs.indexOf(self.geodetic_y_table))
                        x_item.setForeground(1, Qt.blue)
                        y_item.setForeground(1, Qt.blue)
                        parent.addChild(x_item)
                        parent.addChild(y_item)
                    elif "geoid_heights" in parent_key:
                        z_item = QTreeWidgetItem(["Geoid Grid", "[Open Z Tab]"])
                        z_item.setData(1, Qt.UserRole, self.tabs.indexOf(self.geoid_table))
                        z_item.setForeground(1, Qt.blue)
                        parent.addChild(z_item)
                else:
                    item = QTreeWidgetItem([str(key), str(value.shape)])
                    parent.addChild(item)
            else:
                item = QTreeWidgetItem([str(key), str(value)])
                parent.addChild(item)

    def edit_tree_item(self, item, column):
        if column == 1 and item.data(1, Qt.UserRole) is not None:
            self.tabs.setCurrentIndex(item.data(1, Qt.UserRole))
            return
        if item.childCount() == 0 and column == 1:
            old_value = item.text(1)
            new_value, ok = QInputDialog.getMultiLineText(self, "Edit Value", f"New value for '{item.text(0)}':", old_value)
            if ok:
                item.setText(1, new_value)

    def import_grid_csv(self, table):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import CSV", "", "CSV Files (*.csv)")
        if file_path:
            df = pd.read_csv(file_path, header=None)
            table.setRowCount(df.shape[0])
            table.setColumnCount(df.shape[1])
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    table.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))
        self.update_export_preview_actions()  # <-- Add this

    def export_grid_csv(self, table):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV Files (*.csv)")
        if file_path:
            rows, cols = table.rowCount(), table.columnCount()
            data = [[table.item(i, j).text() if table.item(i, j) else "" for j in range(cols)] for i in range(rows)]
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, header=False)

    def export_xy_grid_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export XY Grid CSV", "", "CSV Files (*.csv)")
        if file_path:
            x_grid = self.spg.data["grids"]["geodetic_shifts"]["grid"][0]
            y_grid = self.spg.data["grids"]["geodetic_shifts"]["grid"][1]
            rows, cols = x_grid.shape
            data = [f"{x_grid[i, j]}@{y_grid[i, j]}" for i in range(rows) for j in range(cols)]
            pd.DataFrame(data).to_csv(file_path, index=False, header=False)

    def export_legacy_xy_grt(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Surfer 6 GRT (XY)", "", "GRT Files (*.grt)")
        if file_path:
            x_grid = self.spg.data["grids"]["geodetic_shifts"]["grid"][0]
            y_grid = self.spg.data["grids"]["geodetic_shifts"]["grid"][1]
            self._write_surfer_grd(file_path, x_grid, y_grid)

    def export_legacy_z_grt(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Surfer 6 GRT (Z)", "", "GRT Files (*.grt)")
        if file_path:
            z_grid = self.spg.data["grids"]["geoid_heights"]["grid"][0]
            self._write_surfer_grd(file_path, z_grid)

    def export_legacy_txt_z(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Legacy TXT Z", "", "TXT Files (*.txt)")
        if file_path and self.spg:
            z = self.spg.data["grids"]["geoid_heights"]["grid"][0]
            meta = self.spg.data["grids"]["geoid_heights"]["metadata"]
            release = self.spg.data.get("metadata", {}).get("release_date", "unknown")
            updated = self.spg.data.get("metadata", {}).get("valid_from", "unknown")

            minla = meta.get("minla", 0.0)
            maxla = meta.get("maxla", 0.0)
            minphi = meta.get("minphi", 0.0)
            maxphi = meta.get("maxphi", 0.0)
            stepla = meta.get("stepla", 0.0)
            stepphi = meta.get("stepphi", 0.0)

            flat_values = z.flatten()
            flat_values = np.where(np.isnan(flat_values), 999, flat_values)

            with open(file_path, "w") as f:
                f.write("SUBGRID: EGG97 --> QGeoid Romania\n")
                f.write("GRID PARINTE: NU\n")
                f.write(f"CREAT:   {release}\n")
                f.write(f"ACTUALIZAT:  {updated}\n")
                f.write("Minimum longitude (minLa):\n")
                f.write(f"{minla:.9f}\n")
                f.write("Maximum longitude (maxLa):\n")
                f.write(f"{maxla:.9f}\n")
                f.write("Minimum latitude (minPhi):\n")
                f.write(f"{minphi:.9f}\n")
                f.write("Maximum latitude (maxPhi):\n")
                f.write(f"{maxphi:.9f}\n")
                f.write("Longitude grid interval (stepLa):\n")
                f.write(f" {stepla:.9f}\n")
                f.write("Latitude grid interval (stepPhi):\n")
                f.write(f" {stepphi:.9f}\n")
                f.write("Number of grid shift values (rows x columns):\n")
                f.write(f"{flat_values.size}\n")
                f.write("Number of dimensions (1 for dZita - grid shift value):\n")
                f.write("1\n")
                f.write("Grid shift values (dZita) (columns: minLa-->maxLa; rows: minPhi-->maxPhi):\n")
                for v in flat_values:
                    f.write(f"  {v:.6f}\n")

    def export_legacy_txt_xy(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Legacy TXT XY", "", "TXT Files (*.txt)")
        if file_path and self.spg:
            x = self.spg.data["grids"]["geodetic_shifts"]["grid"][0]
            y = self.spg.data["grids"]["geodetic_shifts"]["grid"][1]
            meta = self.spg.data["grids"]["geodetic_shifts"]["metadata"]
            release = self.spg.data.get("metadata", {}).get("release_date", "unknown")
            updated = self.spg.data.get("metadata", {}).get("valid_from", "unknown")

            mine = meta.get("mine", 0.0)
            maxe = meta.get("maxe", 0.0)
            minn = meta.get("minn", 0.0)
            maxn = meta.get("maxn", 0.0)
            stepe = meta.get("stepe", 0.0)
            stepn = meta.get("stepn", 0.0)

            flat_x = x.flatten()
            flat_y = y.flatten()

            flat_x = np.where(np.isnan(flat_x), 999, flat_x)
            flat_y = np.where(np.isnan(flat_y), 999, flat_y)

            with open(file_path, "w") as f:
                f.write("SUBGRID: ETRS89 --> Krasovski42_Stereografic70  \n")
                f.write("GRID PARINTE: NU\n")
                f.write(f"CREAT:   {release}\n")
                f.write(f"ACTUALIZAT:  {updated}\n")
                f.write("Minimum East (minE):\n")
                f.write(f" {mine:.3f}\n")
                f.write("Maximum East (maxE):\n")
                f.write(f" {maxe:.3f}\n")
                f.write("Minimum North (minN):\n")
                f.write(f" {minn:.3f}\n")
                f.write("Maximum North (maxN):\n")
                f.write(f" {maxn:.3f}\n")
                f.write("East grid interval (stepE):\n")
                f.write(f"  {stepe:.3f}\n")
                f.write("North grid interval (stepN):\n")
                f.write(f"  {stepn:.3f}\n")
                f.write("Number of grid shift values (rows x columns):\n")
                f.write(f"{flat_x.size}\n")
                f.write("Number of dimensions (2 for dEast and dNorth - grid shift values):\n")
                f.write("2\n")
                f.write("Grid shift values (dEast dNorth) (columns: minE-->maxE; rows: minN-->maxN):\n")
                for dx, dy in zip(flat_x, flat_y):
                    f.write(f" {dx:.6f}  {dy:.6f}\n")

    def _write_surfer_grd(self, file_path, *arrays):
        grid = arrays[0]
        nrows, ncols = grid.shape

        # Use the new data access pattern for metadata
        if np.array_equal(grid, self.spg.data["grids"]["geodetic_shifts"]["grid"][0]) or np.array_equal(grid, self.spg.data["grids"]["geodetic_shifts"]["grid"][1]):
            meta = self.spg.data["grids"]["geodetic_shifts"]["metadata"]
            minx, maxx = meta["mine"], meta["maxe"]
            miny, maxy = meta["minn"], meta["maxn"]
        elif np.array_equal(grid, self.spg.data["grids"]["geoid_heights"]["grid"][0]):
            meta = self.spg.data["grids"]["geoid_heights"]["metadata"]
            minx, maxx = meta["minla"], meta["maxla"]
            miny, maxy = meta["minphi"], meta["maxphi"]
        else:
            raise ValueError("Unknown grid passed to Surfer export")

        dx = (maxx - minx) / (ncols - 1) if ncols > 1 else 1
        dy = (maxy - miny) / (nrows - 1) if nrows > 1 else 1

        with open(file_path, "w") as f:
            f.write("DSAA\n")
            f.write(f"{ncols} {nrows}\n")
            f.write(f"{minx:.6f} {maxx:.6f}\n")
            f.write(f"{miny:.6f} {maxy:.6f}\n")
            f.write(f"{-9999.0:.6f} {9999.0:.6f}\n")
            for array in arrays:
                for row in array:
                    f.write(" ".join(f"{v:.6f}" if np.isfinite(v) else "9999" for v in row) + "\n")

    def visualize_geodetic_shifts(self, axis='both', **kwargs):
        if self.spg:
            # You may want to update this to use the new data structure if needed
            visualize_heatmap(self.spg, "geodetic", axis=axis, saveas=None, **kwargs)

    def visualize_geoid_heights(self, **kwargs):
        if self.spg:
            # You may want to update this to use the new data structure if needed
            visualize_heatmap(self.spg, 'geoid', saveas=None, **kwargs)

    def compare_spg(self):
        if not self.spg:
            QMessageBox.warning(self, "No SPG Loaded", "Please load an SPG file first.")
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "Open SPG File to Compare", "", "SPG Files (*.spg)")
        if not file_path:
            return
        try:
            other_spg = SPGFile(file_path)
            # Compare geoid heights
            geoid_diff = compare_geoid_heights(self.spg, other_spg)
            # Compare geodetic shifts
            geodetic_diff = compare_geodetic_shifts(self.spg, other_spg, mode='both')
            # Compare geoid heights interpolated
            geoid_interp_diff = compare_geoid_heights_interpolated(self.spg, other_spg)

            msg = (
                "Comparison Results:\n\n"
                f"Geoid Heights Difference:\n{geoid_diff}\n\n"
                f"Geodetic Shifts Difference:\n{geodetic_diff}\n\n"
                f"Geoid Heights Interpolated Difference:\n{geoid_interp_diff}\n"
            )

            dlg = QDialog(self)
            dlg.setWindowTitle("SPG Comparison Results")
            dlg.resize(800, 600)
            layout = QVBoxLayout(dlg)
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setPlainText(msg)
            layout.addWidget(text_edit)
            btn = QPushButton("Close")
            btn.clicked.connect(dlg.accept)
            layout.addWidget(btn)
            dlg.setLayout(layout)
            dlg.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to compare SPG files: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SPGManagerGUI()
    window.show()
    sys.exit(app.exec_())
