import pandas as pd
import numpy as np
import requests
import datetime
from sqlalchemy import create_engine


class SDEconnect(object):
    def __init__(self):
        self.engine = None
        self.user = None
        self.password = None

        self.tabnames = {'Result': "ugs_ngwmn_monitoring_phy_chem_results",
                         'Activity': "ugs_ngwmn_monitoring_phy_chem_activities",
                         'Station': "ugs_ngwmn_monitoring_locations"}
        self.ugs_tabs = {}
        self.ugs_to_upload = {}
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
                                       'activitystarttime', 'notes', 'personnel', 'created_user', 'created_date',
                                       'last_edited_user',
                                       'last_edited_date']

        self.fieldnames['Result-Activity'] = ['activityid', 'activitymedia', 'activitystartdate', 'activitystarttime',
                                              'activitytimezone', 'activitytype', 'analysisstartdate',
                                              'characteristicname', 'laboratoryname', 'methodspeciation',
                                              'monitoringlocationid',
                                              'projectid',
                                              'resultanalyticalmethodcontext', 'resultanalyticalmethodid',
                                              'resultdetectioncondition',
                                              'detecquantlimitmeasure', 'resultdetecquantlimittype',
                                              'resultdetecquantlimitunit', 'resultsamplefraction', 'resultstatusid',
                                              'resultunit', 'resultvaluetype', 'sampcollectionequip',
                                              'sampcollectmethod'
                                              ]

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

    def get_result_activity_sde(self):
        sql = """SELECT
                ugs_ngwmn_monitoring_phy_chem_results.objectid,
                ugs_ngwmn_monitoring_phy_chem_results.monitoringlocationid,
                ugs_ngwmn_monitoring_phy_chem_results.activityid,
                ugs_ngwmn_monitoring_phy_chem_activities.activitystartdate,
                ugs_ngwmn_monitoring_phy_chem_activities.activitystarttime,
                ugs_ngwmn_monitoring_phy_chem_results.characteristicgroup,
                ugs_ngwmn_monitoring_phy_chem_results.characteristicname,
                ugs_ngwmn_monitoring_phy_chem_results.resultvalue,
                ugs_ngwmn_monitoring_phy_chem_results.resultunit,
                ugs_ngwmn_monitoring_phy_chem_results.resultqualifier,
                ugs_ngwmn_monitoring_phy_chem_results.detecquantlimitmeasure,
                ugs_ngwmn_monitoring_phy_chem_results.resultdetecquantlimitunit,
                ugs_ngwmn_monitoring_phy_chem_results.resultdetecquantlimittype,
                ugs_ngwmn_monitoring_phy_chem_results.resultanalyticalmethodid,
                ugs_ngwmn_monitoring_phy_chem_results.resultanalyticalmethodcontext,
                ugs_ngwmn_monitoring_phy_chem_results.resultsamplefraction,
                ugs_ngwmn_monitoring_phy_chem_results.methodspeciation,
                ugs_ngwmn_monitoring_phy_chem_results.resultdetectioncondition,
                ugs_ngwmn_monitoring_phy_chem_results.laboratoryname,
                ugs_ngwmn_monitoring_phy_chem_results.analysisstartdate,
                ugs_ngwmn_monitoring_phy_chem_results.inwqx,
                ugs_ngwmn_monitoring_phy_chem_results.created_user,
                ugs_ngwmn_monitoring_phy_chem_results.created_date,
                ugs_ngwmn_monitoring_phy_chem_results.last_edited_user,
                ugs_ngwmn_monitoring_phy_chem_results.last_edited_date,
                ugs_ngwmn_monitoring_phy_chem_results.resultid,
                ugs_ngwmn_monitoring_phy_chem_activities.projectid,
                ugs_ngwmn_monitoring_phy_chem_activities.personnel 
            FROM
                ugs_ngwmn_monitoring_phy_chem_results
            LEFT JOIN ugs_ngwmn_monitoring_phy_chem_activities 
            ON ugs_ngwmn_monitoring_phy_chem_results.activityid = ugs_ngwmn_monitoring_phy_chem_activities.activityid;"""
        df = pd.read_sql(sql, self.engine, parse_dates={'activitystartdate': '%Y-%m-%D'})

        df['activitymedia'] = 'Water'
        df['activitytimezone'] = 'MDT'
        df['sampcollectionequip'] = 'Water Bottle'
        df['sampcollectmethod'] = 'GRAB'
        df['resultvaluetype'] = 'Actual'
        df['resultstatusid'] = 'Final'
        df['activitytype'] = df['activityid'].apply(lambda x: 'Field Msr/Obs' if '-FM' in x else 'Sample-Routine', 1)

        self.ugs_tabs['Result-Activity'] = df

    def get_group_names(self):
        # "https://cdxnodengn.epa.gov/cdx-srs-rest/"
        char_domains = "http://www.epa.gov/storet/download/DW_domainvalues.xls"
        char_schema = pd.read_excel(char_domains)
        self.char_schema = char_schema[['PK_ISN', 'REGISTRY_NAME', 'CHARACTERISTIC_GROUP_TYPE', 'SRS_ID', 'CAS_NUMBER']]
        self.chemgroups = \
            self.char_schema[['REGISTRY_NAME', 'CHARACTERISTIC_GROUP_TYPE']].set_index(['REGISTRY_NAME']).to_dict()[
                'CHARACTERISTIC_GROUP_TYPE']


class SDEtoWQX(SDEconnect):
    def __init__(self, user, password, savedir):
        """
        Class to convert UGS Database data into EPA WQX format for upload;  This class uses UGS config 6441.
        :param savedir: location to save output files
        """
        # self.enviro = conn_file
        SDEconnect.__init__(self)
        self.savedir = savedir
        self.config_links = {}
        self.import_config_url = "https://cdx.epa.gov/WQXWeb/ImportConfigurationDetail.aspx?mode=import&impcfg_uid={:}"
        self.config_links['Station'] = self.import_config_url.format(6441)
        self.config_links['Result'] = self.import_config_url.format(5926)
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

        self.start_engine(user, password)
        self.get_sde_tables()
        self.get_result_activity_sde()
        self.get_wqp_tables()
        self.compare_sde_wqx()
        self.prep_station_sde()
        #self.prep_result_sde()
        self.prep_result_activity_sde()

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
        self.wqp_tabs['Result']['activityid'] = self.wqp_tabs['Result']['activityid'].apply(lambda x: x.replace('UTAHGS-',''))

        for tab in [self.wqp_tabs['Result'], self.ugs_tabs['Result'], self.ugs_tabs['Result-Activity']]:
            tab['uniqueid'] = tab[['monitoringlocationid', 'activityid', 'characteristicname']].apply(
                lambda x: "{:}-{:}-{:}".format(str(x[0]), str(x[1]), x[2]), 1)
            tab = tab.drop_duplicates(subset='uniqueid')

        self.wqp_tabs['Result-Activity'] = self.wqp_tabs['Result']

        for key, value in {'Result': 'uniqueid', 'Station': 'locationid', 'Activity': 'activityid',
                           'Result-Activity': 'uniqueid'}.items():
            self.ugs_tabs[key]['inwqx'] = self.ugs_tabs[key][value].apply(
                lambda x: 1 if x in self.wqp_tabs[key].index else 0, 1)
            self.ugs_to_upload[key] = self.ugs_tabs[key][self.ugs_tabs[key]['inwqx'] == 0]

    def prep_result_sde(self):
        self.ugs_to_upload['Result']['resultstatusid'] = 'Final'
        self.ugs_to_upload['Result']['resultvaluetype'] = 'Actual'
        self.ugs_to_upload['Result'] = self.ugs_to_upload['Result'][self.fieldnames['Result-Activity']]

    def prep_result_activity_sde(self):
        self.ugs_to_upload['Result-Activity'] = self.ugs_to_upload['Result-Activity'][self.fieldnames['Result-Activity']]

    def prep_station_sde(self):
        """

        :param sde_stat_table:
        :param save_dir:
        """
        self.ugs_to_upload['Station'] = self.ugs_to_upload['Station'][self.ugs_to_upload['Station']['send'] == 1]
        self.ugs_to_upload['Station'] = self.ugs_to_upload['Station'].reset_index()
        self.ugs_to_upload['Station']['triballandind'] = 'No'
        self.ugs_to_upload['Station']['triballandname'] = None
        self.ugs_to_upload['Station'] = self.ugs_to_upload['Station'].apply(lambda x: self.get_context(x), 1)
        self.ugs_to_upload['Station'] = self.ugs_to_upload['Station'][self.fieldnames['Station']]
        self.ugs_to_upload['Station'] = self.ugs_to_upload['Station'][
            self.ugs_to_upload['Station']['locationtype'] != 'Atmosphere']
        self.ugs_to_upload['Station'] = self.ugs_to_upload['Station'].sort_values("locationid")

    def save_file(self):
        for tab in self.tabnames.keys():
            self.ugs_to_upload[tab].to_csv(f"{self.savedir}/sde_to_wqx_{tab}_{datetime.datetime.today():%Y%m%d}.csv",
                                           index=False)

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
        epa_raw_data['characteristicgroup'] = epa_raw_data['characteristicname'].apply(lambda x: self.chemgroups.get(x),
                                                                                       1)
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

        epa_acts = self.epa_data[~self.epa_data['ActivityID'].isin(sdeact['ActivityID'])].drop_duplicates(
            subset=['ActivityID'])
        epa_acts[self.fieldnames['Activity']].to_csv(
            f"{self.save_folder:}/epa_sheet_to_sde_activity_{datetime.datetime.today():%Y%m%d%M%H%S}.csv")

        epa_results = self.epa_data[~self.epa_data['ActivityID'].isin(sdechem['ActivityID'])]
        epa_results[self.fieldnames['Result']].to_csv(
            f"{self.save_folder:}/epa_sheet_to_sde_result_{datetime.datetime.today():%Y%m%d%M%H%S}.csv")
        print('success!')


class StateLabtoSDE(SDEconnect):

    def __init__(self, file_path, save_path, sample_matches_file):
        """

        :param file_path: path to raw state lab data
        :param save_path: path of where to save output file
        :param sample_matches_file: file that contains information for matching records; should contain Sample Number,Station ID
        """
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


        self.matches = {'SPRINGVILLE': '401043111361801',
                   '(C-19-.*?4)31ada-.*?1': '390714112200401',
                   'AM.*?FK|FORK.*?WELL|wl': '402105111472601',
                   'Cedar.*?V.*?Fur': '401656112020301',
                   'FAIRFIELD': '401539112045501',
                   'clover.*?sp': '402050112330201',
                   'Pecan': '370858113220301',
                   'delle.*?sp': '403328112442201',
                   'sp.*?timpie|timpie.*?sp': '404425112384801',
                   'sp.*?simpson|simpson.*?sp': '400204112465801',
                   'sp.*?antelope|antelope.*?sp': '382238113205301',
                   'sp.*?willow.*?!piez|willow.*?sp.*?!piez|404975112573501': '404975112573501',
                   'PLACER|OPEN.*?LAND': '383709109230701',
                   'BAILEY|KEELER': '383832109243901',
                   'STUCKI': '383922109253801',
                   'Jenks|ADELE|CREEKSIDE': '383854109242901',
                   'PORCUPINE': '383453109200601',
                   'LOOP.*?ROAD': '383746109214001',
                   'GOLF.*?MOAB|MOAB.*?GOLF|383210109285801': '383210109285801',
                   'COURTHOUSE.*?WASH': '384113109391001',
                   'PRICE.*?GOLF': '393841110514801',
                   'NEPHI.*?WELL': '394313111504701',
                   'WL.*?FLOWING|FLOWING.*?WL|FLOWING.*?WELL': '3959531115425101',
                   'SP.*?GOSHEN.*?WARM|GOSHEN.*?WARM.*?SP|395717111512301': '395717111512301',
                   'WL.*?ELEMEN|ELEMEN.*?WL|ELEMEN.*?WELL': '404724111562501',
                   'GRATEFUL': '404508112522401',
                   'RHONDA|EASY.*?ST': '383040109281201',
                   'LRP1A|LOWER.*?R.*?CKY.*?PASS.*?PIEZ': '413308113492701',
                   'WLP1C|WILLOW.*?SP.*?PIEZOMETER.*?1C': '413435113475901',
                   'WLP1B|WILLOW.*?SP.*?PIEZOMETER.*?1B': '413435113475801',
                   'BRYCE|BC26W.*?RICH': '374159112121001',
                   'BRYCE.*?BC13W.*?POE': '374124112135501',
                   'BRYCE.*?BC22W.*?RUBY.*?3': '374138112103102',
                   'BRYCE.*?BC21W.*?RUBY.*?2': '374138112103101',
                   'BRYCE.*?BC15S.*?LOWER.*?BERRY': '374410112133001',
                   'BRYCE.*?BC27W.*?AIRPORT': '374236112092501',
                   'BRYCE.*?BC40W.*?ELK': '374232112100501',
                   'BRYCE.*?BC44W.*?NPS.*?1': '373755112124301',
                   'BRYCE.*?BC17S.*?UPPER.*?BERRY': '374548112142201',
                   'BRYCE.*?BC25W.*?UDOT': '374156112112401',
                   'BRYCE.*?BC51W.*?BRISTLE.*?CONE': '374646112063201',
                   'BRYCE.*?BC11S.*?NPS.*?4': '373801112124701',
                   'BRYCE.*?BC2S.*?TROPIC.*?1': '373639112152101',
                   'BRYCE.*?BC4S.*?TROPIC.*?2': '373629112151801',
                   'BRYCE.*?BC12W USFS': '374017112130801',
                   'BRYCE.*?BC53S.*?WATER': '373534112151101',
                   'BRYCE.*?BC6S.*?DRIPPING.*?VAT': '374500112023001',
                   'BRYCE.*?BC48W.*?SITLA': '374438112080601',
                   'HAMMOND': '391159111543401',
                   'NINE.*?MILE': '391020111421301',
                   'OLSEN': '392335111354601',
                   'WL.*?T34.*?190605': '394809112161101',
                   'SP.*?MUD2.*?190605': '394800112160201',
                   'WL.*?T2.*?190605': '395215112140501',
                   'SP.*?MUD1.*?190605': '395214112140301',
                   'WL.*?T53.*?190605': '395059112120501',
                   'ST.*?T15.*?190605': '395051112120101',
                   'ST.*?T8.*?190605': '395126112124501',
                   'BC35S.*?BRYCE|MOSSY.*?CAVE': '373949112065401',
                   'BC28W.*?BRYCE.*?1': '374750112043401',
                   'BC30W.*?BRYCE.*?LANDFILL.*?3': '374759112040101',
                   'BC29W.*?BRYCE.*?LANDFILL.*?2': '374801112042101',
                   'BC31S.*?BRYCE.*?TOM.*?BEST': '374855112054701',
                   'BC19W.*?BRYCE.*?KINGS.*?CG': '373647112154301',
                   'TRI.*?R.*?NCH': '395718112234301',
                   'SP.*?HUNGTINGTON|HUNTING.*?': '392025110565901',
                   'IRR.*?WELL.*?GROUSE|GROUSE.*?IRR.*?WELL': '413332113545101',
                   'IRR.*?WELL.*?13': '415657112514101',
                   'IRR.*?WELL.*?5': '415906112485001',
                   '67.*?3499|': '390714112200401',
                   'FLOWELL.*?WELL|WL.*?FLOWELL': '385822112265201',
                   'BLACK.*?SPRING': '384610112495201',
                   'ASHELY.*?GORGE': '403429109370601',
                   'WHITEROCKS.*?SP|SP.*?WHITEROCKS': '402906109572401',
                   'SG21C': '395312113244803',
                   'MUD.*?1': '395214112140301',
                   'MUD.*?2': '395214112140301',
                   'T8': '395132112124901',
                   'T15': '395051112120101',
                   'T53': '395059112120501',
                   'T34': '394809112161101',
                   'T2': '395215112140501',
                   'KGSP.*?!PIEZ|KEG.*?SPR.*?!PIEZ': '413507113472401',
                   'NORTH.*?BEDKE.*?SPR': '413810113493901',
                   'BC7W.*?RUBY.*?4': '374116112083901',
                   'BC51W.*?BRYCE|BRISTLE.*?CONE': '374646112063201',
                   'AGI3C': '385630114020202',
                   'PORCUPINE': '383453109200601'
                   }

        self.get_group_names()
        self.state_lab_chem = self.run_calcs()

    def matchids(self):
        for key, value in self.matches.items():
            fd = key + f"|{value}"
            self.state_lab_chem.loc[self.state_lab_chem['Sample Description'].str.contains(f'(?i){fd}'),
                                    'Station ID'] = value

    def chem_lookup(chem):
        url = f'https://cdxnodengn.epa.gov/cdx-srs-rest/substance/name/{chem}?qualifier=exact'
        rqob = requests.get(url).json()
        moleweight = float(rqob[0]['molecularWeight'])
        moleformula = rqob[0]['molecularFormula']
        casnumber = rqob[0]['currentCasNumber']
        epaname = rqob[0]['epaName']
        return [epaname, moleweight, moleformula, casnumber]

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
        # self.save_it(self.save_folder)
        return state_lab_chem

    def save_data(self, user, password):

        self.start_engine(user, password)
        self.get_sde_tables()
        self.state_lab_chem['created_user'] = self.user
        self.state_lab_chem['last_edited_user'] = self.user
        self.state_lab_chem['created_date'] = datetime.datetime.today()
        self.state_lab_chem['last_edited_date'] = datetime.datetime.today()

        sdeact = self.ugs_tabs['Activities'][[['MonitoringLocationID', 'ActivityID']]]
        sdechem = self.ugs_tabs['Results'][[['MonitoringLocationID', 'ActivityID']]]

        state_lab_acts = self.state_lab_chem[
            ~self.state_lab_chem['ActivityID'].isin(sdeact['ActivityID'])].drop_duplicates(subset=['ActivityID'])
        state_lab_acts[self.fieldnames['Activity']].to_csv(
            f"{self.save_folder:}/statelab_sheet_to_sde_activity_{datetime.datetime.today():%Y%m%d%M%H%S}.csv")

        state_lab_chem = self.state_lab_chem[~self.state_lab_chem['ActivityID'].isin(sdechem['ActivityID'])]
        state_lab_chem[self.fieldnames['Result']].to_csv(
            f"{self.save_folder:}/statelab_sheet_to_sde_result_{datetime.datetime.today():%Y%m%d%M%H%S}.csv")
        print('success!')

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

    sde = SDEconnect()
    sde.start_engine(sys.argv[0], sys.argv[1])
