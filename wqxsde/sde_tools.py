import pandas as pd
import numpy as np
import requests
import datetime
from sqlalchemy import create_engine

class SDEconnect(object):
    def __init__(self):
        self.engine = None
        self.user = None
        self.passoword = None

        self.tabnames = {'Result': "ugs_ngwmn_monitoring_phy_chem_results",
                         'Activity': "ugs_ngwmn_monitoring_phy_chem_activities",
                         'Station': "ugs_ngwmn_monitoring_locations"}
        self.ugs_tabs = {}
        self.fieldnames = {}
        self.fieldnames['Result'] = ['activityid', 'monitoringlocationid', 'resultanalyticalmethodcontext',
                                     'resultanalyticalmethodid',
                                     'resultsamplefraction',
                                     'resultvalue', 'detecquantlimitmeasure', 'resultdetecquantlimitunit', 'resultunit',
                                     'analysisstartdate', 'resultdetecquantlimittype', 'resultdetectioncondition',
                                     'characteristicname',
                                     'methodspeciation', 'characteristicgroup',
                                     'inwqx', 'created_user', 'last_edited_user', 'created_date', 'last_edited_date',
                                     'resultid']

        self.fieldnames['Station'] = ['locationid', 'locationname', 'locationtype', 'huc8', 'huc12',
                                      'triballandind', 'triballandname', 'latitude', 'longitude',
                                      'horizontalcollectionmethod', 'horizontalcoordrefsystem',
                                      'state', 'county',
                                      'verticalmeasure', 'verticalunit', 'verticalcoordrefsystem',
                                      'verticalcollectionmethod',
                                      'altlocationid', 'altlocationcontext',
                                      'welltype', 'welldepth', 'welldepthmeasureunit', 'aquifername']

        self.fieldnames['Activity'] = ['activityid', 'projectid', 'monitoringlocationid', 'activitystartdate',
                      'activitystarttime', 'notes', 'personnel', 'created_user', 'created_date', 'last_edited_user',
                      'last_edited_date']

    def start_engine(self, user, password, host='nrwugspgressp', port='5432', db='ugsgwp'):
        self.user = user
        self.password = password
        connstr = f"postgresql+psycopg2://{self.user}:{self.password}@{host}:{port}/{db}"
        self.engine = create_engine(connstr, pool_recycle=3600)

    def get_sde_tables(self):
        """
        Pulls tables from the UGS sde database
        :return:
        """
        try:
            for tab, nam in self.tabnames.items():
                sql = f"SELECT * FROM {nam:}"
                self.ugs_tabs[tab] = pd.read_sql(sql, self.engine)
        except:
            print("Please use .start_engine() to enter credentials")

    def get_group_names(self):
        # "https://cdxnodengn.epa.gov/cdx-srs-rest/"
        char_domains = "http://www.epa.gov/storet/download/DW_domainvalues.xls"
        char_schema = pd.read_excel(char_domains)
        self.char_schema = char_schema[['PK_ISN', 'REGISTRY_NAME', 'CHARACTERISTIC_GROUP_TYPE', 'SRS_ID', 'CAS_NUMBER']]
        self.chemgroups = \
            self.char_schema[['REGISTRY_NAME', 'CHARACTERISTIC_GROUP_TYPE']].set_index(['REGISTRY_NAME']).to_dict()[
                'CHARACTERISTIC_GROUP_TYPE']


class SDEtoWQX(SDEconnect):
    def __init__(self, savedir):
        # self.enviro = conn_file
        SDEconnect.__init__()
        self.savedir = savedir
        self.import_config_link = "https://cdx.epa.gov/WQXWeb/ImportConfigurationDetail.aspx?mode=import&impcfg_uid=6441"
        self.rename = {}
        self.rename['Station'] = {'MonitoringLocationIdentifier': 'locationid',
                                  'MonitoringLocationName': 'locationname',
                                  'MonitoringLocationTypeName': 'locationtype',
                                  'HUCEightDigitCode': 'huc8',
                                  'LatitudeMeasure': 'latitude',
                                  'LongitudeMeasure': 'longitude',
                                  'HorizontalCollectionMethodName': 'horizontalcollectionmethod',
                                  'HorizontalCoordinateReferenceSystemDatumName': 'horizontalcoordrefsystem',
                                  'VerticalMeasure/MeasureValue': 'verticalmeasure',
                                  'VerticalMeasure/MeasureUnitCode': 'verticalunit',
                                  'VerticalCollectionMethodName': 'verticalcollectionmethod',
                                  'VerticalCoordinateReferenceSystemDatumName': 'verticalcoordrefsystem',
                                  'StateCode': 'state',
                                  'CountyCode': 'county'}

        self.rename['Activity'] = {'ActivityIdentifier': 'activityid',
                                   'ProjectIdentifier': 'projectid',
                                   'MonitoringLocationIdentifier': 'monitoringlocationid',
                                   'ActivityStartDate': 'activitystartdate',
                                   'ActivityStartTime/Time': 'activitystarttime'}

        self.rename['Result'] = {'ActivityIdentifier': 'activityid',
                                 'MonitoringLocationIdentifier': 'monitoringlocationid',
                                 'ResultDetectionConditionText': 'resultdetectioncondition',
                                 'CharacteristicName': 'characteristicname',
                                 'ResultSampleFractionText': 'resultsamplefraction',
                                 'ResultMeasureValue': 'resultvalue',
                                 'ResultMeasure/MeasureUnitCode': 'resultunit',
                                 'MeasureQualifierCode': 'resultqualifier',
                                 'ResultAnalyticalMethod/MethodIdentifierContext': 'resultanalyticalmethodcontext',
                                 'ResultAnalyticalMethod/MethodName': 'resultanalyticalmethodid',
                                 'LaboratoryName': 'laboratoryname',
                                 'AnalysisStartDate': 'analysisstartdate',
                                 'DetectionQuantitationLimitTypeName': 'resultdetecquantlimittype',
                                 'DetectionQuantitationLimitMeasure/MeasureValue': 'detecquantlimitmeasure',
                                 'DetectionQuantitationLimitMeasure/MeasureUnitCode': 'resultdetecquantlimitunit'}

        self.wqp_tabs = {}
        self.ugs_to_upload = {}
        self.get_sde_tables()
        self.get_wqp_tables()
        self.prep_station_sde()

    def get_wqp_tables(self, **kwargs):
        """
        Pulls tables from the EPA/USGS Water Quality Portal website services
        :return:
        """
        kwargs['countrycode'] = 'US'
        kwargs['organization'] = 'UTAHGS'
        kwargs['mimeType'] = 'csv'
        kwargs['zip'] = 'no'
        kwargs['sorted'] = 'no'

        for res in self.tabnames.keys():
            base_url = f"https://www.waterqualitydata.us/data/{res}/search?"
            response_ob = requests.get(base_url, params=kwargs)
            self.wqp_tabs[res] = pd.read_csv(response_ob.url).dropna(how='all', axis=1).rename(columns=self.rename[res])

    def compare_sde_wqx(self):
        """
        compares unique rows in ugs SDE tables to those in EPA WQX
        """

        for tab in [self.wqp_tabs['Result'], self.ugs_tabs['Result']]:
            tab['uniqueid'] = tab[['monitoringlocationid', 'activityid', 'characteristicname']].apply(
                lambda x: "{:}-{:}-{:}".format(str(x[0]), str(x[1]), x[2]), 1)
            tab = tab.drop_duplicates(subset='uniqueid')

        for key, value in {'Result': 'uniqueid', 'Station': 'locationid', 'Activity': 'activityid'}.items():
            self.ugs_tabs[key]['inwqx'] = self.ugs_tabs[key][value].apply(
                lambda x: 1 if x in self.wqp_tabs[key].index else 0, 1)
            self.ugs_to_upload[key] = self.ugs_tabs[key][self.ugs_tabs[key]['inwqx'] == 0]

    def prep_station_sde(self):
        """

        :param sde_stat_table:
        :param save_dir:
        """
        self.ugs_to_upload['Station'] = self.ugs_to_upload['Station'][self.ugs_to_upload['Station']['Send'] == 1]
        self.ugs_to_upload['Station'] = self.ugs_to_upload['Station'].reset_index()
        self.ugs_to_upload['Station']['triballandind'] = 'No'
        self.ugs_to_upload['Station']['triballandname'] = None
        self.ugs_to_upload['Station'] = self.ugs_to_upload['Station'].apply(lambda x: self.get_context(x), 1)
        self.ugs_to_upload['Station'] = self.ugs_to_upload['Station'][self.get_stat_col_order()]
        self.ugs_to_upload['Station'] = self.ugs_to_upload['Station'][
            self.ugs_to_upload['Station']['locationtype'] != 'Atmosphere']
        self.ugs_to_upload['Station'] = self.ugs_to_upload['Station'].sort_values("LocationID")

    def save_file(self):
        self.sde_stat_import.to_csv(self.save_dir + "/stations_{:%Y%m%d}.csv".format(pd.datetime.today()), index=False)

    def get_context(self, df):
        if pd.isnull(df['usgs_id']):
            if pd.isnull(df['win']):
                if pd.isnull(df['wrnum']):
                    df['altlocationcontext'] = None
                    df['altlocationid'] = None
                else:
                    df['altLocationcontext'] = 'Utah Water Rights Number'
                    df['altlocationid'] = df['wrnum']
            else:
                df['altlocationcontext'] = 'Utah Well ID'
                df['altlocationid'] = df['win']
        else:
            df['altlocationcontext'] = 'usgs_id'
            df['altlocationid'] = df['usgs_id']
        return df


class EPAtoSDE(SDEconnect):

    def __init__(self, epa_file_path, save_path):
        """
        Class to prep. data from the US EPA lab to import into the EPA WQX
        :param user:
        :param file_path:
        :param save_path:
        :param schema_file_path:
        :param conn_path:
        """
        SDEconnect.__init__()
        self.save_folder = save_path
        self.epa_raw_data = pd.read_excel(epa_file_path)
        self.epa_rename = {'Laboratory': 'laboratoryname',
                           'LabNumber': 'activityid',
                           'SampleName': 'monitoringlocationid',
                           'Method': 'resultanalyticalmethodid',
                           'Analyte': 'characteristicname',
                           'ReportLimit': 'resultdetecquantlimitunit',
                           'Result': 'resultvalue',
                           'AnalyteQual': 'resultqualifier',
                           'AnalysisClass': 'resultsamplefraction',
                           'ReportLimit': 'detecquantlimitmeasure',
                           'Units': 'resultunit',
                           }

        self.epa_drop = ['Batch', 'Analysis', 'Analyst', 'CASNumber', 'Elevation', 'LabQual',
                         'Client', 'ClientMatrix', 'Dilution', 'SpkAmt', 'UpperLimit', 'Recovery',
                         'Surrogate', 'LowerLimit', 'Latitude', 'Longitude', 'SampleID', 'ProjectNumber',
                         'Sampled', 'Analyzed', 'PrepMethod', 'Prepped', 'Project']

        self.get_group_names()
        self.epa_data = self.run_calcs()

    def renamepar(self, df):
        x = df['characteristicname']
        pardict = {'Ammonia as N': ['Ammonia', 'as N'], 'Sulfate as SO4': ['Sulfate', 'as SO4'],
                   'Nitrate as N': ['Nitrate', 'as N'], 'Nitrite as N': ['Nitrite', 'as N'],
                   'Orthophosphate as P': ['Orthophosphate', 'as P']}
        if ' as' in x:
            df['characteristicname'] = pardict.get(x)[0]
            df['methodspeciation'] = pardict.get(x)[1]
        else:
            df['characteristicname'] = df['characteristicname']
            df['methodspeciation'] = None

        return df

    def hasless(self, df):
        if '<' in str(df['resultvalue']):
            df['resultvalue'] = None
            df['ResultDetectionCondition'] = 'Below Reporting Limit'
            df['ResultDetecQuantLimitType'] = 'Lower Reporting Limit'
        elif '>' in str(df['resultvalue']):
            df['resultvalue'] = None
            df['ResultDetectionCondition'] = 'Above Reporting Limit'
            df['ResultDetecQuantLimitType'] = 'Upper Reporting Limit'
        elif '[' in str(df['resultvalue']):
            df['resultvalue'] = pd.to_numeric(df['resultvalue'].split(" ")[0], errors='coerce')
            df['ResultDetecQuantLimitType'] = None
            df['ResultDetectionCondition'] = None
        else:
            df['resultvalue'] = pd.to_numeric(df['resultvalue'], errors='coerce')
            df['ResultDetecQuantLimitType'] = None
            df['ResultDetectionCondition'] = None
        return df

    def resqual(self, x):
        if pd.isna(x[1]) and x[0] == 'Below Reporting Limit':
            return 'BRL'
        elif pd.notnull(x[1]):
            return x[1]
        else:
            return None

    def filtmeth(self, x):
        if "EPA" in x:
            x = x.split(' ')[1]
        elif '/' in x:
            x = x.split('/')[0]
        else:
            x = x
        return x

    def chem_lookup(self, chem):
        url = f'https://cdxnodengn.epa.gov/cdx-srs-rest/substance/name/{chem}?qualifier=exact'
        rqob = requests.get(url).json()
        moleweight = float(rqob[0]['molecularWeight'])
        moleformula = rqob[0]['molecularFormula']
        casnumber = rqob[0]['currentCasNumber']
        epaname = rqob[0]['epaName']
        return [epaname, moleweight, moleformula, casnumber]

    def run_calcs(self):
        epa_raw_data = self.epa_raw_data
        epa_raw_data = epa_raw_data.rename(columns=self.epa_rename)
        epa_raw_data['resultsamplefraction'] = epa_raw_data['resultsamplefraction'].apply(
            lambda x: 'Total' if 'WET' else x, 1)
        epa_raw_data['personnel'] = None
        epa_raw_data = epa_raw_data.apply(lambda x: self.hasless(x), 1)
        epa_raw_data['resultanalyticalmethodid'] = epa_raw_data['resultanalyticalmethodid'].apply(
            lambda x: self.filtmeth(x), 1)
        epa_raw_data['resultanalyticalmethodcontext'] = 'USEPA'
        epa_raw_data['projectid'] = 'UNGWMN'
        epa_raw_data['resultqualifier'] = epa_raw_data[['resultdetectioncondition',
                                                        'resultqualifier']].apply(lambda x: self.resqual(x), 1)
        epa_raw_data['inwqx'] = 0
        epa_raw_data['notes'] = None
        epa_raw_data = epa_raw_data.apply(lambda x: self.renamepar(x), 1)
        epa_raw_data['resultid'] = epa_raw_data[['activityid', 'characteristicname']].apply(
            lambda x: str(x[0]) + '-' + str(x[1]), 1)
        epa_raw_data['activitystartdate'] = epa_raw_data['sampled'].apply(lambda x: "{:%Y-%m-%d}".format(x), 1)
        epa_raw_data['activitystarttime'] = epa_raw_data['sampled'].apply(lambda x: "{:%H:%M}".format(x), 1)
        epa_raw_data['analysisstartdate'] = epa_raw_data['analyzed'].apply(lambda x: "{:%Y-%m-%d}".format(x), 1)
        unitdict = {'ug/L': 'ug/l', 'NONE': 'None', 'UMHOS-CM': 'uS/cm', 'mg/L': 'mg/l'}
        epa_raw_data['resultunit'] = epa_raw_data['resultunit'].apply(lambda x: unitdict.get(x, x), 1)
        epa_raw_data['resultdetecquantlimitunit'] = epa_raw_data['resultunit']
        epa_raw_data['monitoringlocationid'] = epa_raw_data['monitoringlocationid'].apply(lambda x: str(x), 1)
        epa_raw_data['characteristicgroup'] = epa_raw_data['characteristicname'].apply(lambda x: self.chemgroups.get(x),1)
        epa_data = epa_raw_data.drop(self.epa_drop, axis=1)
        self.epa_data = epa_data

        return epa_data

    def save_data(self, user, password):

        self.start_engine(user, password)
        self.get_sde_tables()
        self.epa_data['created_user'] = self.user
        self.epa_data['last_edited_user'] = self.user
        self.epa_data['created_date'] = datetime.datetime.today()
        self.epa_data['last_edited_date'] = datetime.datetime.today()

        sdeact = self.ugs_tabs['Activities'][[['MonitoringLocationID', 'ActivityID']]]
        sdechem = self.ugs_tabs['Results'][[['MonitoringLocationID', 'ActivityID']]]

        epa_acts = self.epa_data[~self.epa_data['ActivityID'].isin(sdeact['ActivityID'])].drop_duplicates(subset=['ActivityID'])
        epa_acts[self.fieldnames['Activity']].to_csv(f"{self.save_folder:}/epa_sheet_to_sde_activity_{datetime.datetime.today():%Y%m%d%M%H%S}.csv")

        epa_results = self.epa_data[~self.epa_data['ActivityID'].isin(sdechem['ActivityID'])]
        epa_results[self.fieldnames['Result']].to_csv(f"{self.save_folder:}/epa_sheet_to_sde_result_{datetime.datetime.today():%Y%m%d%M%H%S}.csv")
        print('success!')


class StateLabtoSDE(SDEconnect):

    def __init__(self, file_path, save_path, sample_matches_file):
        SDEconnect.__init__()

        self.save_folder = save_path
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

        self.chemcols = {'Sample Number': 'activityid',
                         'Station ID': 'monitoringlocationid',
                         'Sample Date': 'activitystartdate',
                         'Sample Time': 'activitystarttime',
                         'Sample Description': 'notes',
                         'Collector': 'personnel',
                         'Method Agency': 'resultanalyticalmethodcontext',
                         'Method ID': 'resultanalyticalmethodid',
                         'Matrix Description': 'resultsamplefraction',
                         'Result Value': 'resultvalue',
                         'Lower Report Limit': 'detecquantlimitmeasure',
                         'Method Detect Limit': 'resultdetecquantlimitunit',
                         'Units': 'resultunit',
                         'Analysis Date': 'analysisstartdate'}

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
        self.get_group_names()
        self.state_lab_chem = self.run_calcs()

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
        state_lab_chem['resultid'] = state_lab_chem[['ActivityID', 'CharacteristicName']].apply(
            lambda x: x[0] + '-' + x[1],
            1)
        self.state_lab_chem = state_lab_chem
        #self.save_it(self.save_folder)
        return state_lab_chem

    def append_data(self):
        self.state_lab_chem = self.run_calcs()
        sdeact = table_to_pandas_dataframe(self.activities_table_name,
                                           field_names=['MonitoringLocationID', 'ActivityID'])
        sdechem = table_to_pandas_dataframe(self.chem_table_name, field_names=['MonitoringLocationID', 'ActivityID'])

        state_lab_chem['created_user'] = self.user
        state_lab_chem['last_edited_user'] = self.user
        state_lab_chem['created_date'] = pd.datetime.today()
        state_lab_chem['last_edited_date'] = pd.datetime.today()

        df = state_lab_chem[~state_lab_chem['ActivityID'].isin(sdeact['ActivityID'])]
        edit_table(subset, self.activities_table_name, fieldnames=fieldnames, enviro=self.enviro)
        df = state_lab_chem[~state_lab_chem['ActivityID'].isin(sdechem['ActivityID'])]
        self.edit_table(subset, self.chem_table_name, fieldnames=fieldnames, enviro=self.enviro)

    def get_sample_matches(self):
        matches = pd.read_csv(self.sample_matches_file)
        matches = matches[['Station ID', 'Sample Number']].drop_duplicates()
        matches['Station ID'] = matches['Station ID'].apply(lambda x: "{:.0f}".format(x), 1)
        matches_dict = matches[['Sample Number', 'Station ID']].set_index(['Sample Number']).to_dict()['Station ID']
        return matches_dict

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

    def get_proj_match(self):
        stations = self.pull_sde_stations()

        projectmatch = stations[['LocationID', 'QWNetworkName']].set_index('LocationID').to_dict()['QWNetworkName']

        return projectmatch


if __name__ == "__main__":
    import sys

    GetPaths(int(sys.argv[1]))
