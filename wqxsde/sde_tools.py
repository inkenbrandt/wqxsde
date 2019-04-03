import pandas as pd
import numpy as np
import arcpy

# define environ
enviro = "C:/Users/paulinkenbrandt/AppData/Roaming/Esri/Desktop10.6/ArcCatalog/Connection to DEFAULT@uggp.agrc.utah.gov.sde"
chem_table_name = "UGGP.UGGPADMIN.UGS_NGWMN_Monitoring_Phy_Chem_Results"
activities_table_name = "UGGP.UGGPADMIN.UGS_NGWMN_Monitoring_Phy_Chem_Activities"
wqx_results_filename = "G:/My Drive/WORK/NGWMN/Database/ResultsExport.xlsx"

def edit_table(df, fieldnames=None,
               sde_table="UGGP.UGGPADMIN.UGS_NGWMN_Monitoring_Phy_Chem_Results",
               enviro="C:/Users/paulinkenbrandt/AppData/Roaming/Esri/Desktop10.6/ArcCatalog/UGS_SDE.sde"):
    """
    this function will append rows to an existing SDE table from a pandas dataframe. It requires editing privledges.
    :param df: pandas dataframe with data you wish to append to SDE table
    :param fieldnames: names of fields you wish to import
    :param sde_table: name of sde table
    :param enviro: path to connection file of the SDE
    :return:
    """
    arcpy.env.workspace = enviro

    if len(fieldnames) > 0:
        pass
    else:
        fieldnames = df.columns

    read_descr = arcpy.Describe(sde_table)
    sde_field_names = []
    for field in read_descr.fields:
        sde_field_names.append(field.name)
    sde_field_names.remove('OBJECTID')

    for name in fieldnames:
        if name not in sde_field_names:
            fieldnames.remove(name)
            print("{:} not in {:} fieldnames!".format(name, sde_table))

    try:
        egdb_conn = arcpy.ArcSDESQLExecute(enviro)
        egdb_conn.startTransaction()
        print("Transaction started...")
        # Perform the update
        try:
            # build the sql query to pull the maximum object id
            sqlid = """SELECT max(OBJECTID) FROM {:};""".format(sde_table)
            objid = egdb_conn.execute(sqlid)

            subset = df[fieldnames]
            rowlist = subset.values.tolist()
            # build the insert sql to append to the table
            sqlbeg = "INSERT INTO {:}({:},OBJECTID)\nVALUES ".format(sde_table, ",".join(map(str, fieldnames)))
            sqlendlist = []

            for j in range(len(rowlist)):
                objid += 1
                strfill = ["("]
                # This loop deals with different data types and NULL values
                for k in range(len(rowlist[j])):
                    if pd.isna(rowlist[j][k]):
                        strfill.append("NULL")
                    elif isinstance(rowlist[j][k], (int, float)):
                        strfill.append("{:}".format(rowlist[j][k]))
                    else:
                        strfill.append("'{:}'".format(rowlist[j][k]))

                strfill.append(",{:})".format(objid))
                sqlendlist.append(",".join(map(str, strfill)))

            sqlend = "{:}".format(",".join(sqlendlist))
            sql = sqlbeg + sqlend

            egdb_return = egdb_conn.execute(sql)

            # If the update completed successfully, commit the changes.  If not, rollback.
            if egdb_return == True:
                egdb_conn.commitTransaction()
                print("Committed Transaction")
            else:
                egdb_conn.rollbackTransaction()
                print("Rolled back any changes.")
                print("++++++++++++++++++++++++++++++++++++++++\n")
        except Exception as err:
            print(err)
            egdb_return = False
        # Disconnect and exit
        del egdb_conn
    except Exception as err:
        print(err)


def fast_sde_to_df(enviro, table):
    """
    converts an SDE table to a Pandas Dataframe
    :param enviro: file location or sde connection file location of table
    :param table: table with chemistry data in SDE
    :return:
    """
    egdb_conn = arcpy.ArcSDESQLExecute(enviro)
    egdb_conn.startTransaction()
    # print("Transaction started...")
    # Perform the update
    sqlid = "SELECT * FROM {:};".format(table)

    #sqcols = """SELECT * FROM UGGP.UGGPADMIN.INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = N'{:}'"""

    sql = egdb_conn.execute(sqlid)
    # cols = egdb_conn.execute(sqcols)
    cols = [f.name for f in arcpy.ListFields(table)]
    return pd.DataFrame(sql, columns=cols)


def compare_sde_wqx(wqx_results_filename, enviro, chem_table_name, table_type='chem'):
    """
    compares unique rows in an SDE chem table to that of a WQX download
    :param wqx_results_filename: excel file with wqx results download
    :param enviro: file location or sde connection file location of table
    :param chem_table_name: table with chemistry data in SDE
    :return:
    """
    arcpy.env.workspace = enviro

    wqx_chem_table = pd.read_excel(wqx_results_filename)
    try:
        sde_chem_table = fast_sde_to_df(enviro, chem_table_name)
    except:
        sde_chem_table = table_to_pandas_dataframe(chem_table_name)

    if table_type == 'chem':
        wqx_chem_table['uniqueid'] = wqx_chem_table[['Monitoring Location ID', 'Activity ID', 'Characteristic']].apply(
            lambda x: "{:}-{:}-{:}".format(str(x[0]), str(x[1]), x[2]), 1)
        sde_chem_table['uniqueid'] = sde_chem_table[['MonitoringLocationID', 'ActivityID', 'CharacteristicName']].apply(
            lambda x: "{:}-{:}-{:}".format(str(x[0]), str(x[1]), x[2]), 1)
        wqx_chem_table = wqx_chem_table.set_index('uniqueid')
        sde_chem_table = sde_chem_table.set_index('uniqueid')
    else:
        sde_chem_table = sde_chem_table.set_index('LocationID')
        wqx_chem_table = wqx_chem_table.set_index('Monitoring Location ID')

    objtable = []

    for ind in sde_chem_table.index:
        if ind in wqx_chem_table.index:
            objtable.append(sde_chem_table.loc[ind, 'OBJECTID'])
    loc_dict = {}  # empty dictionary
    # iterate input table
    with arcpy.da.UpdateCursor(chem_table_name, ['OID@', 'inwqx']) as tcurs:
        for row in tcurs:
            # index location row[2]=SHAPE@X and row[1]=SHAPE@Y that matches index locations in dictionary
            if row[0] in objtable:
                row[1] = 1
                tcurs.updateRow(row)

    sde_chem_table = fast_sde_to_df(enviro, chem_table_name)
    return wqx_chem_table, sde_chem_table

def table_to_pandas_dataframe(table, field_names=None, query=None, sql_sn=(None, None)):
    """
    Load data into a Pandas Data Frame for subsequent analysis.
    :param table: Table readable by ArcGIS.
    :param field_names: List of fields.
    :param query: SQL query to limit results
    :param sql_sn: sort fields for sql; see http://pro.arcgis.com/en/pro-app/arcpy/functions/searchcursor.htm
    :return: Pandas DataFrame object.
    """
    # TODO Make fast with SQL
    # if field names are not specified
    if not field_names:
        field_names = get_field_names(table)
    # create a pandas data frame
    df = pd.DataFrame(columns=field_names)

    # use a search cursor to iterate rows
    with arcpy.da.SearchCursor(table, field_names, query, sql_clause=sql_sn) as search_cursor:
        # iterate the rows
        for row in search_cursor:
            # combine the field names and row items together, and append them
            df = df.append(dict(zip(field_names, row)), ignore_index=True)

    # return the pandas data frame
    return df


def get_field_names(table):
    read_descr = arcpy.Describe(table)
    field_names = []
    for field in read_descr.fields:
        field_names.append(field.name)
    field_names.remove('OBJECTID')
    return field_names


class ProcessStateLabText(object):

    def __init__(self, file_path, save_path, sample_matches_file, schema_file_path,
                 enviro="C:/Users/paulinkenbrandt/AppData/Roaming/Esri/Desktop10.6/ArcCatalog/Connection to DEFAULT@uggp.agrc.utah.gov.sde"):

        self.chem_table_name = "UGGP.UGGPADMIN.UGS_NGWMN_Monitoring_Phy_Chem_Results"
        self.activities_table_name = "UGGP.UGGPADMIN.UGS_NGWMN_Monitoring_Phy_Chem_Activities"
        self.stations_table_name = "UGGP.UGGPADMIN.UGS_NGWMN_Monitoring_Locations"

        self.save_folder = save_path
        self.schema_file_path = schema_file_path
        self.sample_matches_file = sample_matches_file

        self.state_lab_chem = pd.read_csv(file_path, sep="\t")

        self.param_explain = {'Fe': 'Iron', 'Mn': 'Manganese', 'Ca': 'Calcium',
                              'Mg': 'Magnesium', 'Na': 'Sodium',
                              'K': 'Potassium', 'HCO3': 'Bicarbonate',
                              'CO3': 'Carbonate', 'SO4': 'Sulfate',
                              'Cl': 'Chloride', 'F': 'Floride', 'NO3-N': 'Nitrate as Nitrogen',
                              'NO3': 'Nitrate', 'B': 'Boron', 'TDS': 'Total dissolved solids',
                              'Total Dissolved Solids': 'Total dissolved solids',
                              'Hardness': 'Total hardness', 'hard': 'Total hardness',
                              'Total Suspended Solids': 'Total suspended solids',
                              'Cond': 'Conductivity', 'pH': 'pH', 'Cu': 'Copper',
                              'Pb': 'Lead', 'Zn': 'Zinc', 'Li': 'Lithium', 'Sr': 'Strontium',
                              'Br': 'Bromide', 'I': 'Iodine', 'PO4': 'Phosphate', 'SiO2': 'Silica',
                              'Hg': 'Mercury', 'NO3+NO2-N': 'Nitrate + Nitrite as Nitrogen',
                              'As': 'Arsenic', 'Cd': 'Cadmium', 'Ag': 'Silver',
                              'Alk': 'Alkalinity, total', 'P': 'Phosphorous',
                              'Ba': 'Barium', 'DO': 'Dissolved oxygen',
                              'Q': 'Discharge', 'Temp': 'Temperature',
                              'Hard_CaCO3': 'Hardness as Calcium Carbonate',
                              'DTW': 'Depth to water',
                              'O18': 'Oxygen-18', '18O': 'Oxygen-18', 'D': 'Deuterium',
                              'd2H': 'Deuterium', 'C14': 'Carbon-14',
                              'C14err': 'Carbon-14 error', 'Trit_err': 'Tritium error',
                              'Meas_Alk': 'Alkalinity, total', 'Turb': 'Turbidity',
                              'TSS': 'Total suspended solids',
                              'C13': 'Carbon-13', 'Tritium': 'Tritium',
                              'S': 'Sulfur', 'density': 'density',
                              'Cr': 'Chromium', 'Se': 'Selenium',
                              'temp': 'Temperature', 'NO2': 'Nitrite',
                              'O18err': 'Oxygen-18 error', 'd2Herr': 'Deuterium error',
                              'NaK': 'Sodium + Potassium', 'Al': 'Aluminum',
                              'Be': 'Beryllium', 'Co': 'Cobalt',
                              'Mo': 'Molydenum', 'Ni': 'Nickel',
                              'V': 'Vanadium', 'SAR': 'Sodium absorption ratio',
                              'Hard': 'Total hardness', 'Free Carbon Dioxide': 'Carbon dioxide',
                              'CO2': 'Carbon dioxide'
                              }
        self.chemcols = {'Sample Number': 'ActivityID',
                         'Station ID': 'MonitoringLocationID',
                         'Sample Date': 'ActivityStartDate',
                         'Sample Time': 'ActivityStartTime',
                         'Sample Description': 'notes',
                         'Collector': 'personnel',
                         'Method Agency': 'ResultAnalyticalMethodContext',
                         'Method ID': 'ResultAnalyticalMethodID',
                         'Matrix Description': 'ResultSampleFraction',
                         'Result Value': 'resultvalue',
                         'Lower Report Limit': 'ResultDetecQuantLimitMeasure',
                         'Method Detect Limit': 'ResultDetecQuantLimitUnit',
                         'Units': 'ResultUnit',
                         'Analysis Date': 'AnalysisStartDate'}

        self.proj_name_matches = {'Arches Monitoring Wells': 'UAMW',
                                  'Bryce': 'UBCW',
                                  'Castle Valley': 'CAVW',
                                  'GSL Chem': 'GSLCHEM',
                                  'Juab Valley': 'UJVW',
                                  'Mills/Mona Wetlands': 'MMWET',
                                  'Monroe Septic': 'UMSW',
                                  'Ogden Valley': 'UOVW',
                                  'Round Valley': 'URVH',
                                  'Snake Valley': 'USVW', 'Snake Valley Wetlands': 'SVWET',
                                  'Tule Valley Wetlands': 'TVWET', 'UGS-NGWMN': 'UNGWMN',
                                  'WRI - Grouse Creek': 'UWRIG',
                                  'WRI - Montezuma': 'UWRIM',
                                  'WRI - Tintic Valley': 'UWRIT'}

    def run_calcs(self):
        matches_dict = self.get_sample_matches()
        state_lab_chem = self.state_lab_chem
        state_lab_chem['Station ID'] = state_lab_chem['Sample Number'].apply(lambda x: matches_dict.get(x), 1)
        state_lab_chem['ResultDetecQuantLimitType'] = 'Lower Reporting Limit'

        projectmatch = self.get_proj_match()
        state_lab_chem['ProjectID'] = state_lab_chem['Station ID'].apply(lambda x: projectmatch.get(x), 1)
        state_lab_chem['ProjectID'] = state_lab_chem['ProjectID'].apply(lambda x: self.proj_name_matches.get(x), 1)
        state_lab_chem['Matrix Description'] = state_lab_chem['Matrix Description'].apply(lambda x: self.ressampfr(x),
                                                                                          1)
        state_lab_chem['ResultDetectionCondition'] = state_lab_chem[['Problem Identifier', 'Result Code']].apply(
            lambda x: self.lssthn(x), 1)
        state_lab_chem['Sample Date'] = pd.to_datetime(state_lab_chem['Sample Date'].str.split(' ', expand=True)[0])
        state_lab_chem['Analysis Date'] = pd.to_datetime(state_lab_chem['Analysis Date'].str.split(' ', expand=True)[0])
        state_lab_chem = state_lab_chem.apply(lambda df: self.renamepar(df), 1)
        state_lab_chem = state_lab_chem.rename(columns=self.chemcols)
        chemgroups = self.get_group_names()
        state_lab_chem['characteristicgroup'] = state_lab_chem['CharacteristicName'].apply(lambda x: chemgroups.get(x),
                                                                                           1)
        unneeded_cols = ['Trip ID', 'Agency Bill Code',
                         'Test Comment', 'Result Comment', 'Sample Report Limit',
                         'Chain of Custody', 'Cost Code', 'Test Number',
                         'CAS Number', 'Project Name',
                         'Sample Received Date', 'Method Description', 'Param Description',
                         'Dilution Factor', 'Batch Number', 'Replicate Number',
                         'Sample Detect Limit', 'Problem Identifier', 'Result Code',
                         'Sample Type', 'Project Comment', 'Sample Comment']
        state_lab_chem = state_lab_chem.drop(unneeded_cols, axis=1)
        state_lab_chem['ResultValueType'] = 'Actual'
        state_lab_chem['ResultStatusID'] = 'Final'
        state_lab_chem['ResultAnalyticalMethodContext'] = state_lab_chem['ResultAnalyticalMethodContext'].apply(
            lambda x: 'APHA' if x == 'SM' else 'USEPA', 1)
        state_lab_chem['inwqx'] = 0
        unitdict = {'MG-L': 'mg/l', 'UG-L': 'ug/l', 'NONE': 'None', 'UMHOS-CM': 'uS/cm'}
        state_lab_chem['ResultUnit'] = state_lab_chem['ResultUnit'].apply(lambda x: unitdict.get(x, x), 1)
        state_lab_chem['ResultDetecQuantLimitUnit'] = state_lab_chem['ResultUnit']
        self.state_lab_chem = state_lab_chem
        self.save_it(self.save_folder)
        return state_lab_chem

    def pull_sde_stations(self):
        stations = table_to_pandas_dataframe(self.stations_table_name, field_names=['LocationID', 'QWNetworkName'])
        return stations

    def get_sample_matches(self):
        matches = pd.read_excel(self.sample_matches_file, 'UTGS_EDD_20190304')
        matches = matches[['Station ID', 'Sample Number']].drop_duplicates()
        matches['Station ID'] = matches['Station ID'].apply(lambda x: "{:.0f}".format(x), 1)
        matches_dict = matches[['Sample Number', 'Station ID']].set_index(['Sample Number']).to_dict()['Station ID']
        return matches_dict

    def get_proj_match(self):
        stations = self.pull_sde_stations()

        projectmatch = stations[['LocationID', 'QWNetworkName']].set_index('LocationID').to_dict()['QWNetworkName']

        return projectmatch

    def ressampfr(self, x):
        if str(x).strip() == 'Water, Filtered':
            return 'Dissolved'
        else:
            return 'Total'

    def lssthn(self, x):
        if x[0] == '<':
            return "Below Reporting Limit"
        elif x[0] == '>':
            return "Above Operating Range"
        elif x[1] == 'U' and pd.isna(x[0]):
            return "Not Detected"
        else:
            return None

    def renamepar(self, df):

        x = df['Param Description']
        x = str(x).strip()
        y = None

        if x in self.param_explain.keys():
            z = self.param_explain.get(x)

        if " as " in x:
            z = x.split(' as ')[0]
            y = x.split(' as ')[1]
        else:
            z = x

        if str(z).strip() == 'Alkalinity':
            z = 'Alkalinity, total'

        if y == 'Calcium Carbonate':
            y = 'as CaCO3'
        elif y == 'Carbonate':
            y = 'as CO3'
        elif y == 'Nitrogen':
            y = 'as N'
        elif z == 'Total Phosphate' and pd.isna(y):
            z = 'Orthophosphate'
            y = 'as PO4'
        df['CharacteristicName'] = z
        df['MethodSpeciation'] = y
        return df

    def check_chems(self, df, char_schema):
        missing_chem = []
        for chem in df['CharacteristicName'].unique():
            if chem not in char_schema['Name'].values:
                print(chem)
                missing_chem.append(chem)
        return missing_chem

    def get_group_names(self):
        char_schema = pd.read_excel(self.schema_file_path, "CHARACTERISTIC")
        chemgroups = char_schema[['Name', 'Group Name']].set_index(['Name']).to_dict()['Group Name']
        return chemgroups

    def save_it(self, savefolder):
        self.state_lab_chem.to_csv("{:}/state_lab_to_sde_{:%Y%m%d}.csv".format(savefolder, pd.datetime.today()))