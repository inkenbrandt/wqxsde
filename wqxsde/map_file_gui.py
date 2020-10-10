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
import requests
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

        self.states = {'':0,'Alabama': 1, 'Alaska': 2, 'Arizona': 4, 'Arkansas': 5, 'California': 6, 'Colorado': 8,
                  'Connecticut': 9, 'Delaware': 10, 'Florida': 12, 'Georgia': 13, 'Hawaii': 15, 'Idaho': 16,
                  'Illinois': 17, 'Indiana': 18, 'Iowa': 19, 'Kansas': 20, 'Kentucky': 21, 'Louisiana': 22,
                  'Maine': 23, 'Maryland': 24, 'Massachusetts': 25, 'Michigan': 26, 'Minnesota': 27, 'Mississippi': 28,
                  'Missouri': 29, 'Montana': 30, 'Nebraska': 31, 'Nevada': 32, 'New Hampshire': 33, 'New Jersey': 34,
                  'New Mexico': 35, 'New York': 36, 'North Carolina': 37, 'North Dakota': 38, 'Ohio': 39, 'Oklahoma': 40,
                  'Oregon': 41, 'Pennsylvania': 42, 'Rhode Island': 44, 'South Carolina': 45, 'South Dakota': 46,
                  'Tennessee': 47, 'Texas': 48, 'Utah': 49, 'Vermont': 50, 'Virginia': 51, 'Washington': 53,
                  'West Virginia': 54, 'Wisconsin': 55, 'Wyoming': 56, 'American Samoa': 60, 'Guam': 66,
                  'Northern Mariana Islands': 69, 'Puerto Rico': 72, 'Virgin Islands': 78}

        self.statecombo.addItems(self.states.keys())
        self.statecombo.currentTextChanged.connect(self.get_county)

        self.punits = {'m': 0.000621371, 'km': 0.621371, 'mi': 1, 'ft': 0.000189394}
        self.distunitcombo.addItems(self.punits.keys())
        #self.distunitcombo.currentTextChanged.connect(self.get_dis)
        #self.buttonGroup.buttonToggled.connect(lambda: self.btnstate(self.buttonGroup))

    def get_dis(self, value):
        self.finaldis = self.punits[value]*self.distancespin.value()
        #self.wqpdata = wqxsde.WQP(f'{self.finaldis},{self.plat},{self.plon}', 'within')

    def acceptspin(self):
        if self.buttonGroup.checkedButton().text() == "DISTANCE FROM POINT":

            self.finaldis = self.punits[self.distunitcombo.currentText()] * self.distancespin.value()
            self.wqpdata = wqxsde.WQP([self.finaldis,self.wqpypnt.value(),self.wqpxpnt.value()], 'rad')
        elif self.buttonGroup.checkedButton().text() == "BBOX":
            tlat = self.bbtoplat.value()
            tlon = self.bbtoplon.value()
            blat = self.bbbotlat.value()
            blon = self.bbbotlon.value()

            print(f'{tlon},{tlat},{blon},{blat}')
            self.wqpdata = wqxsde.WQP(f'{tlon},{blat},{blon},{tlat}','bBox')
        elif self.buttonGroup.checkedButton().text() == "COUNTY":
            st = f"{self.states[self.statecombo.currentText()]:02d}"
            co = self.countyfips[self.countycombo.currentText()]
            self.wqpdata = wqxsde.WQP([st, co],'countyCd')
        self.accept()

    def get_county(self, value):
        i = self.states[value]
        urlcounties = f"https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/1/query?where=STATE+%3D+{i}&outFields=COUNTY%2CNAME&text=&objectIds=&time=&geometry=&geometryType=esriGeometryEnvelope&inSR=&spatialRel=esriSpatialRelIntersects&relationParam=&outFields=&returnGeometry=false&returnTrueCurves=false&maxAllowableOffset=&geometryPrecision=&outSR=&returnIdsOnly=false&returnCountOnly=false&returnZ=false&returnM=false&returnDistinctValues=false&returnExtentsOnly=false&rangeValues=&f=pjson"
        if i > 0:
            js = requests.get(urlcounties).json()['features']
            if len(js) > 0:
                self.countycombo.clear()
                df = pd.DataFrame(js)
                df['names'] = df['attributes'].apply(lambda x: x['NAME'],1)
                df['county'] = df['attributes'].apply(lambda x: x['COUNTY'], 1)
                self.countycombo.addItems(sorted(df['names'].values))
                df = df.set_index('names').drop(['attributes'], axis=1)
                self.countyfips = df.to_dict()['county']
        else:
            self.countycombo.clear()


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

        self.graphresultsbutt.clicked.connect(self.add_selected_piper)
        self.addallpipbutt.clicked.connect(self.add_all_piper)
        self.clearpipbutt.clicked.connect(self.clear_piper)

        self.importsdebutt.clicked.connect(self.executeLoginPage)
        #self.activityselection = self.ActivityTableView.selectionModel()
        #self.ActivityTableView.verticalHeader().sectionClicked.connect(self.testchange)
        #self.activityselection.modelChanged.connect(self.testchange)
        #self.ActivityTableView.clicked.connect(self.testchange)


    def resultsel(self, s):
        self.resultselection = self.ResultTableView.selectionModel()

        indexes = self.resultselection.selectedRows()
        indices = [i.row() for i in sorted(indexes)]
        print(self.ResultModel._data.iloc[indices,:])

    def stationsel(self,s):
        self.stationselection = self.StationTableView.selectionModel()

        indexes = self.stationselection.selectedRows(column=2)
        df = self.ResultModel._data
        role = Qt.DisplayRole
        dg = df[df['monitoringlocationid'].isin([self.StationModel.data(i, role) for i in indexes])]
        print(dg)
        mode = QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows
        for i in dg.index:
            self.ResultTableView.selectRow(i)
        #for i in dg.index:
        #    self.ResultTableView.selectRow(i)

    def activitysel(self, s):
        self.activityselection = self.ActivityTableView.selectionModel()

        indexes = self.activityselection.selectedRows()
        indices = [i.row() for i in sorted(indexes)]
        print(self.ActivityModel._data.iloc[indices,:])

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
            #self.activityselection.selectionChanged.connect(self.testchange)
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
        #TODO Prevent Duplication of Index
        if hasattr(MainWindow,'StationModel') and 'Station' in dfd.keys():
            self.StationModel._data.append(dfd['Station'], ignore_index=True)
        else:
            self.StationModel = TableModel(dfd['Station'])
            self.StationTableView.setModel(self.StationModel)
            self.stationproxymodel = QSortFilterProxyModel()
            self.stationproxymodel.setSourceModel(self.StationModel)
            self.StationTableView.setSortingEnabled(True)
            self.StationTableView.setModel(self.stationproxymodel)
            self.delegate = InLineEditDelegate()
            self.StationTableView.setItemDelegate(self.delegate)
            self.StationTableView.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.StationTableView.customContextMenuRequested.connect(self.context_menu)
            self.StationTableView.verticalHeader().sectionClicked.connect(self.stationsel)
            self.statselmodel = QItemSelectionModel(self.StationModel)
            self.map_data(lat='latitude', lon='longitude')
        if hasattr(MainWindow,'ResultModel') and 'Result' in dfd.keys():
            self.ResultModel._data.append(dfd['Result'], ignore_index=True)
        else:
            self.ResultModel = TableModel(dfd['Result'])
            self.ResultTableView.setModel(self.ResultModel)
            self.resultproxymodel = QSortFilterProxyModel()
            self.resultproxymodel.setSourceModel(self.ResultModel)
            self.ResultTableView.setSortingEnabled(True)
            self.ResultTableView.setModel(self.resultproxymodel)
            self.delegate = InLineEditDelegate()
            self.ResultTableView.setItemDelegate(self.delegate)
            self.ResultTableView.setItemDelegate(self.delegate)
            self.ResultTableView.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

            self.ResultTableView.verticalHeader().sectionClicked.connect(self.resultsel)
            self.resselmodel = QItemSelectionModel(self.ResultModel)


        if hasattr(MainWindow,'ActivityModel') and 'Activity' in dfd.keys():
            self.ActivityModel._data.append(dfd['Activity'], ignore_index=True)
        else:
            self.ActivityModel = TableModel(dfd['Activity'])
            self.ActivityTableView.setModel(self.ActivityModel)
            self.activityproxymodel = QSortFilterProxyModel()
            self.activityproxymodel.setSourceModel(self.ActivityModel)
            self.ActivityTableView.setSortingEnabled(True)
            self.ActivityTableView.setModel(self.activityproxymodel)
            self.delegate = InLineEditDelegate()
            self.ActivityTableView.setItemDelegate(self.delegate)
            self.ActivityTableView.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.ActivityTableView.setItemDelegate(self.delegate)

            self.ActivityTableView.verticalHeader().sectionClicked.connect(self.activitysel)
            self.actselmodel = QItemSelectionModel(self.ActivityModel)

        self.clrBycomboBox.clear()
        self.clrBycomboBox.addItems(self.ActivityModel._data.columns)
        self.grpbycombo.clear()
        self.grpbycombo.addItems(self.ActivityModel._data.columns)
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

    def map_data(self, sellist=None, lat = 'latitude',lon = 'longitude'):

        m = folium.Map(location=[self.StationModel._data[lat].mean(),
                                 self.StationModel._data[lon].mean()],
                       tiles="Stamen Terrain", zoom_start=13)

        icons = {'Spring':'star-empty', 'Well':'heart-empty', 'Stream':'tint','Land':'globe'}

        for i in self.StationModel._data.index:
            tooltip = i
            popup = f"""Name = {self.StationModel._data.loc[i,'locationname']}<br>Type = {self.StationModel._data.loc[i,'locationtype']}<br>ID = {self.StationModel._data.loc[i,'locationid']}"""
            if sellist and i in self.StationModel._data.iloc[sellist, 2:-1].index:
                folium.Marker([self.StationModel._data.loc[i, lat],
                               self.StationModel._data.loc[i, lon]],
                              popup=popup,
                              tooltip=tooltip,
                              icon=folium.Icon(color='red',
                                               icon=icons.get(self.StationModel._data.loc[i,'locationtype'],'bullseye'))
                              ).add_to(m)
            else:
                folium.Marker([self.StationModel._data.loc[i, lat],
                               self.StationModel._data.loc[i, lon]],
                              popup=popup,
                              tooltip=tooltip,
                              icon=folium.Icon(color='blue',
                                               icon=icons.get(self.StationModel._data.loc[i,'locationtype'],'bullseye'))
                              ).add_to(m)

        sw = self.StationModel._data[[lat, lon]].min().values.tolist()
        ne = self.StationModel._data[[lat, lon]].max().values.tolist()

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
        #TODO Add groupby functionality
        #stats = self.StationTableView._data.
        sellist = [i.row() for i in items]
        piperlist = ['Ca','Mg','Na','K','HCO3','CO3','Cl','SO4']
        arr =  self.ActivityModel._data.iloc[sellist][piperlist].dropna(subset=piperlist,axis=0,how='any').to_numpy()
        print(arr)
        arrays =[[arr,{"label": 'data', "facecolor": "red"}]]
        wqxsde.piper(arrays, "title", use_color=True,
                     fig=self.sc.fig, ax=self.sc.axes)
        self.sc.draw()
        self.map_data(sellist)

    def add_all_piper(self):
        markers = ["s", "o", "^", "v", "+", "x"]
        arrays = []
        for i, (label, group_df) in enumerate(self.ActivityModel._data.groupby("additional-field")):
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