# https://stackoverflow.com/questions/58590199/how-to-show-folium-map-inside-a-pyqt5-gui
# https://stackoverflow.com/questions/57065370/how-do-i-make-a-split-window
# https://stackoverflow.com/questions/60437182/how-to-include-folium-map-into-pyqt5-application-window?noredirect=1&lq=1

# https://www.learnpyqt.com/courses/graphics-plotting/plotting-matplotlib/
# https://www.learnpyqt.com/courses/model-views/qtableview-modelviews-numpy-pandas/
# https://stackoverflow.com/questions/29734471/qtablewidget-current-selection-change-signal
# https://wiki.python.org/moin/PyQt/Reading%20selections%20from%20a%20selection%20model
# https://stackoverflow.com/questions/22577327/how-to-retrieve-the-selected-row-of-a-qtableview
#https://www.youtube.com/watch?v=Gpw6BygkUCw
from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets, uic, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from qtwidgets import PasswordEdit

import folium
import io
import sys

import wqxsde
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import pandas as pd


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=2.5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111, aspect='equal', frameon=False, xticks=[], yticks=[])
        super(MplCanvas, self).__init__(self.fig)
 

class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])

    def flags(self, index):
        """
        Make table editable.
        make first column non editable
        :param index:
        :return:
        """
        if index.column() > -1:
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable
        #elif index.column() == 1:
        #    return QtCore.Qt.DecorationRole
        else:
            return QtCore.Qt.ItemIsSelectable

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        """
        Edit data in table cells
        :param index:
        :param value:
        :param role:
        :return:
        """
        if index.isValid():
            selected_row = self._data.iloc[index.row()]
            selected_column = self._data.columns[index.column()]
            self._data.iloc[index.row(),index.column()] = value
            self.dataChanged.emit(index, index, (QtCore.Qt.DisplayRole, ))
            #ok = databaseOperations.update_existing(selected_row['_id'], selected_row)
            #if ok:
            #    return True
        return False

    def insertRows(self):
        row_count = len(self._data)
        self.beginInsertRows(QtCore.QModelIndex(), row_count, row_count)
        empty_data = {key: None for key in self._data.columns if not key=='_id'}

        self._data = self._data.append(empty_data, ignore_index=True)
        row_count += 1
        self.endInsertRows()
        return True

    def removeRows(self, position):
        row_count = self.rowCount()
        row_count -= 1
        self.beginRemoveRows(QtCore.QModelIndex(), row_count, row_count)
        row_id = position.row()
        document_id = self._data[row_id]['_id']
        #databaseOperations.remove_data(document_id)
        self.user_data.pop(row_id)
        self.endRemoveRows()

    def context_menu(self):
        menu = QtWidgets.QMenu()
        add_data = menu.addAction("Add New Data")
        add_data.setIcon(QtGui.QIcon(":/icons/images/add-icon.png"))
        add_data.triggered.connect(lambda: self.model.insertRows())
        if self.tableView.selectedIndexes():
            remove_data = menu.addAction("Remove Data")
            remove_data.setIcon(QtGui.QIcon(":/icons/images/remove.png"))
            remove_data.triggered.connect(lambda: self.model.removeRows(self.tableView.currentIndex()))
        cursor = QtGui.QCursor()
        menu.exec_(cursor.pos())

class InLineEditDelegate(QtWidgets.QItemDelegate):
    """
    Delegate is important for inline editing of cells
    """
    def createEditor(self, parent, option, index):
        return super(InLineEditDelegate, self).createEditor(parent, option, index)

    def setEditorData(self, editor, index):
        text = index.data(QtCore.Qt.EditRole) or index.data(QtCore.Qt.DisplayRole)
        editor.setText(str(text))

class LoginPage(QDialog):
    def __init__(self, *args, **kwargs):
        super(LoginPage, self).__init__(*args, **kwargs)
        uic.loadUi('sdelog.ui', self)

        self.userpw = self.pwbox.text()
        self.usernm = self.usernamebox.text()

        self.pickSaveDir.clicked.connect(self.getfile)

        self.buttonBox.accepted.connect(self.acceptbutt)
        #self.buttonBox.accepted.connect(wqxsde.SDEtoWQX(usernm, userpw, self.name))
        self.buttonBox.rejected.connect(self.reject)

    def acceptbutt(self):
        if self.name:
            print('yes')
            self.userpw = self.pwbox.text()
            self.usernm = self.usernamebox.text()
            self.chemdata = wqxsde.SDEtoWQX(self.usernm, self.userpw, self.name+"/test.csv")

        else:
            print('no', self.name, self.usernm)
        self.accept()
        #self.close()

    def getfile(self):
        self.name = QFileDialog.getExistingDirectory(self, 'Save File')
        print(self.name)

class WQPPage(QDialog):
    def __init__(self, *args, **kwargs):
        super(WQPPage, self).__init__(*args, **kwargs)
        uic.loadUi('wqpdialog.ui', self)

        #self.wqpdata = None
        #self.bbtoplat.valueChanged.connect()
        self.buttonBox.accepted.connect(self.acceptspin)
        # self.buttonBox.accepted.connect(wqxsde.SDEtoWQX(usernm, userpw, self.name))
        self.buttonBox.rejected.connect(self.reject)

    def acceptspin(self):
        tlat = self.bbtoplat.value()
        tlon = self.bbtoplon.value()
        blat = self.bbbotlat.value()
        blon = self.bbbotlon.value()

        print(f'{tlon},{tlat},{blon},{blat}')
        self.wqpdata = wqxsde.WQP(f'{tlon},{blat},{blon},{tlat}','bBox')

        self.accept()
        # self.close()


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page
        uic.loadUi('guiwindow.ui', self)

        filename = "G:/My Drive/Python/Pycharm/wqxsde/examples/piperdata.csv"
        df = pd.read_csv(filename)
        self.df = df.dropna(subset=['Latitude', 'Longitude'], how='any')

        # Setup Piper Diagram for plotting
        self.sc = MplCanvas(self.piperframe)
        toolbar = NavigationToolbar(self.sc, self.piperframe)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.sc)
        self.piperframe.setLayout(layout)

        self.actionFrom_SDE_requires_login.triggered.connect(self.executeLoginPage)
        self.actionDownload_Here.triggered.connect(self.executeWQPPage)
        #self.model = TableModel(self.df)
        #self.delegate = InLineEditDelegate()
        #self.ResultTableView.setModel(self.model)
        #self.ResultTableView.setItemDelegate(self.delegate)
        #self.ResultTableView.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        #self.ResultTableView.customContextMenuRequested.connect(self.context_menu)
        #self.selmodel = QItemSelectionModel(self.model)
        #self.ActivitiesTableView.setModel(self.model)
        #self.ResultTableView.clicked.connect(self.print_row)
        #self.ResultTableView.pressed.connect(self.print_row)

        self.graphresultsbutt.clicked.connect(self.add_selected_piper)
        self.addallpipbutt.clicked.connect(self.add_all_piper)
        self.clearpipbutt.clicked.connect(self.clear_piper)
        #self.selmodel.seChanged(self.print_row)
        #self.model.flags(QtCore.Qt.ItemIsEditable)
        #self.model.itemSelectionChanged.connect(self.print_row)

        #self.write_df_to_qtable(self.df, self.ActivitiesTableWidget)

        #self.ActivitiesTableWidget.itemSelectionChanged.connect(self.print_row)
        self.importsdebutt.clicked.connect(self.executeLoginPage)

    def executeWQPPage(self, s):
        self.wqpdlg = WQPPage(self)
        if self.wqpdlg.exec_():
            print("Success!")

            wqp = self.wqpdlg.wqpdata
            dfd = {}
            dfd['Result'] = wqp.results
            dfd['Station'] = wqp.stations
            dfd['Activity'] = wqp.activities

            self.add_data(dfd)
        else:
            print("Cancel!")

    def executeLoginPage(self, s):
        self.dlg = LoginPage(self)
        if self.dlg.exec_():
            print("Success!")
            #self.dlg.chemdata
            dfd = self.dlg.chemdata.ugs_tabs
            self.add_data(dfd)
        else:
            print("Cancel!")

    def add_data(self, dfd):
        if hasattr(MainWindow,'StationModel') and 'Station' in dfd.keys():
            self._data.append(dfd['Station'], ignore_index=True)
        else:
            self.StationModel = TableModel(dfd['Station'])
            self.StationTableView.setModel(self.StationModel)
            self.delegate = InLineEditDelegate()
            self.StationTableView.setItemDelegate(self.delegate)
            self.statselmodel = QItemSelectionModel(self.StationModel)
            self.map_data(model=self.StationModel, lat='latitude', lon='longitude')
        if hasattr(MainWindow,'ResultModel') and 'Result' in dfd.keys():
            self._data.append(dfd['Result'], ignore_index=True)
        else:
            self.ResultModel = TableModel(dfd['Result'])
            self.ResultTableView.setModel(self.ResultModel)
            self.delegate = InLineEditDelegate()
            self.ResultTableView.setItemDelegate(self.delegate)
            self.resselmodel = QItemSelectionModel(self.ResultModel)
        if hasattr(MainWindow,'ActivityModel') and 'Activity' in dfd.keys():
            self._data.append(dfd['Activity'], ignore_index=True)
        else:
            self.ActivityModel = TableModel(dfd['Activity'])
            self.ActivityTableView.setModel(self.ActivityModel)
            self.delegate = InLineEditDelegate()
            self.ActivityTableView.setItemDelegate(self.delegate)
            self.actselmodel = QItemSelectionModel(self.ActivityModel)


    # Takes a df and writes it to a qtable provided. df headers become qtable headers
    # https://stackoverflow.com/questions/31475965/fastest-way-to-populate-qtableview-from-pandas-data-frame
    def write_df_to_qtable(self, df, table):
        headers = list(df)
        table.setRowCount(df.shape[0])
        table.setColumnCount(df.shape[1])
        table.setHorizontalHeaderLabels(headers)

        # getting data from df is computationally costly so convert it to array first
        df_array = df.values
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                table.setItem(row, col, QTableWidgetItem(str(df_array[row, col])))

    def map_data(self, model=None, sellist=None, lat = 'Latitude',lon = 'Longitude'):

        if model:
            pass
        else:
            model = self.model

        m = folium.Map(location=[model._data[lat].mean(),
                                 model._data[lon].mean()],
                       tiles="Stamen Terrain", zoom_start=13)

        for i in model._data.index:
            tooltip = i

            if sellist and i in model._data.iloc[sellist, 2:-1].index:
                folium.Marker([model._data.loc[i, lat],
                               model._data.loc[i, lon]],
                              popup=f"<i>{i}</i>",
                              tooltip=tooltip,
                              icon=folium.Icon(color='red')
                              ).add_to(m)
            else:
                folium.Marker([model._data.loc[i, lat],
                               model._data.loc[i, lon]],
                              popup=f"<i>{i}</i>",
                              tooltip=tooltip,
                              icon=folium.Icon(color='blue')
                              ).add_to(m)

        sw = model._data[[lat, lon]].min().values.tolist()
        ne = model._data[[lat, lon]].max().values.tolist()

        m.fit_bounds([sw, ne])

        data = io.BytesIO()
        m.save(data, close_file=False)

        self.stationmap.setHtml(data.getvalue().decode())
        # w.resize(500, 250)
        self.stationmap.show()

    def add_selected_piper(self):
        self.sc.axes.cla()
        self.sc.axes = self.sc.fig.add_subplot(111, aspect='equal', frameon=False, xticks=[], yticks=[])
        items = self.ActivityTableView.selectionModel().selectedRows()
        #TODO Connect Results Stations and Activities with Selection
        #stats = self.StationTableView._data.
        sellist = [i.row() for i in items]
        arr =  self.ActivityModel._data.iloc[sellist][['Ca','Mg','Na','K','HCO3','CO3','Cl','SO4']].to_numpy()
        print(arr)
        arrays =[[arr,{"label": 'data', "facecolor": "red"}]]
        wqxsde.piper(arrays, "title", use_color=True,
                     fig=self.sc.fig, ax=self.sc.axes)
        self.sc.draw()
        self.map_data(sellist)

    def add_all_piper(self):
        markers = ["s", "o", "^", "v", "+", "x"]
        arrays = []
        for i, (label, group_df) in enumerate(self.ActivityModel
                                                      ._data.groupby("additional-field")):
            arr = group_df.iloc[:, 2:10].values
            arrays.append([arr, {"label": label,
                                 "marker": markers[i],
                                 "edgecolor": "k",
                                 "linewidth": 0.3,
                                 "facecolor": "none", }, ])
        #print(arrays)
        self.sc.axes = self.sc.fig.add_subplot(111, aspect='equal', frameon=False, xticks=[], yticks=[])
        rgb = wqxsde.piper(arrays, "title", use_color=True, fig=self.sc.fig, ax=self.sc.axes)
        self.sc.axes.legend()
        self.sc.draw()

    def clear_piper(self):
        self.sc.axes.cla()
        self.sc.draw()

    def context_menu(self):
        menu = QtWidgets.QMenu()
        add_data = menu.addAction("Add New Data")
        #add_data.setIcon(QtGui.QIcon(":/icons/images/add-icon.png"))
        add_data.triggered.connect(lambda: self.model.insertRows())
        if self.ResultTableView.selectedIndexes():
            remove_data = menu.addAction("Remove Data")
            #remove_data.setIcon(QtGui.QIcon(":/icons/images/remove.png"))
            remove_data.triggered.connect(lambda: self.model.removeRows(self.tableView.currentIndex()))
        cursor = QtGui.QCursor()
        menu.exec_(cursor.pos())



def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()