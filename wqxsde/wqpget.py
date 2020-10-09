# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 09:50:51 2016

@author: paulinkenbrandt
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
from datetime import datetime
import numpy as np
import requests




class WQP(object):
    """Downloads Water Quality Data from thw Water Quality Portal based on parameters entered
    :param values: query parameter designating location to select site; this is the Argument for the REST parameter in
    table 1 of https://www.waterqualitydata.us/webservices_documentation/
    :param loc_type: type of query to perform; valid inputs include 'huc', 'bBox', 'countycode', 'siteid';
    this is the REST parameter of table 1 of https://www.waterqualitydata.us/webservices_documentation/
    :type loc_type: str
    :type values: str
    :param **kwargs: additional Rest Parameters

    :Example:
    >>> wq = WQP('-111.54,40.28,-111.29,40.48','bBox')
    https://www.waterqualitydata.us/Result/search?mimeType=csv&zip=no&siteType=Spring&siteType=Well&characteristicType=Inorganics%2C+Major%2C+Metals&characteristicType=Inorganics%2C+Major%2C+Non-metals&characteristicType=Nutrient&characteristicType=Physical&bBox=-111.54%2C40.28%2C-111.29%2C40.48&sorted=no&sampleMedia=Water

    """

    def __init__(self, values, loc_type, **kwargs):
        """Downloads Water Quality Data from thw Water Quality Portal based on parameters entered
        """
        self.loc_type = loc_type
        self.values = values
        self.url = 'https://www.waterqualitydata.us/'
        self.geo_criteria = ['sites', 'stateCd', 'huc', 'countyCd', 'bBox','organization']
        #self.cTgroups = ['Inorganics, Major, Metals', 'Inorganics, Major, Non-metals', 'Nutrient', 'Physical']

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
                                  'CountyCode': 'county',
                                  'AquiferName': 'aquifername',
                                  'WellDepthMeasure/MeasureValue': 'welldepth',
                                  'WellDepthMeasure/MeasureUnitCode': 'welldepthmeasureunit'}

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
                                 'DetectionQuantitationLimitMeasure/MeasureUnitCode': 'resultdetecquantlimitunit',
                                 "ActivityStartDate": "sampledate",
                                 "ActivityIdentifier": "sampleid",
                                 "ActivityStartTime/Time": "sampletime"
                                 }

        self.ParAbb = {"Alkalinity": "Alk", "Alkalinity, Carbonate as CaCO3": "Alk", "Alkalinity, total": "Alk",
                  "Arsenic": "As", "Calcium": "Ca", "Chloride": "Cl", "Carbon dioxide": "CO2", "Carbonate": "CO3",
                  "Carbonate (CO3)": "CO3", "Specific conductance": "Cond", "Conductivity": "Cond", "Copper": "Cu",
                  "Depth": "Depth", "Dissolved oxygen (DO)": "DO", "Iron": "Fe",
                  "Hardness, Ca, Mg": "Hard", "Total hardness -- SDWA NPDWR": "Hard",
                  "Bicarbonate": "HCO3", "Potassium": "K", "Magnesium": "Mg", "Kjeldahl nitrogen": "N",
                  "Nitrogen, mixed forms (NH3), (NH4), organic, (NO2) and (NO3)": "N", "Nitrogen": "N", "Sodium": "Na",
                  "Sodium plus potassium": "NaK", "Ammonia-nitrogen": "NH3_N", "Ammonia-nitrogen as N": "N",
                  "Nitrite": "NO2",
                  "Nitrate": "NO3", "Nitrate as N": "N", "pH, lab": "pH", "pH": "pH", "Phosphate-phosphorus": "PO4",
                  "Orthophosphate": "PO4", "Phosphate": "PO4", "Stream flow, instantaneous": "Q", "Flow": "Q",
                  "Flow rate, instantaneous": "Q", "Silica": "Si", "Sulfate": "SO4", "Sulfate as SO4": "SO4",
                  "Boron": "B", "Barium": "Ba", "Bromine": "Br", "Lithium": "Li", "Manganese": "Mn", "Strontium": "Sr",
                  "Total dissolved solids": "TDS", "Temperature, water": "Temp",
                  "Total Organic Carbon": "TOC", "delta Dueterium": "d2H", "delta Oxygen 18": "d18O",
                  "delta Carbon 13 from Bicarbonate": "d13CHCO3", "delta Oxygen 18 from Bicarbonate": "d18OHCO3",
                  "Total suspended solids": "TSS", "Turbidity": "Turb"}

        self.results = self.get_wqp_results('Result', **kwargs)
        self.massage_results()
        self.stations = self.get_wqp_stations('Station', **kwargs)
        self.massage_stations()
        self.activities = self.piv_chem(chems='')#self.get_wqp_activities('Activity',**kwargs)


    def get_response(self, service, **kwargs):
        """ Returns a dictionary of data requested by each function.
        :param service: options include 'Station' or 'Results'
        table 1 of https://www.waterqualitydata.us/webservices_documentation/
        """
        http_error = 'Could not connect to the API. This could be because you have no internet connection, a parameter' \
                     ' was input incorrectly, or the API is currently down. Please try again.'
        # For python 3.4
        # try:
        kwargs[self.loc_type] = self.values
        kwargs['mimeType'] = 'csv'
        kwargs['zip'] = 'yes'
        #kwargs['sorted'] = 'no'

        #if 'siteType' not in kwargs:
        #    kwargs['sampleMedia'] = 'Water'

        #if 'siteType' not in kwargs:
        #    kwargs['siteType'] = ['Spring', 'Well']
        #    print('This function is biased towards groundwater. For all sites, use')

        #if 'characteristicType' not in kwargs:
        #    kwargs['characteristicType'] = self.cTgroups

        total_url = self.url + service + '/search?'
        response_ob = requests.get(total_url, params=kwargs)

        return response_ob

    def get_wqp_stations(self, service, **kwargs):
        nwis_dict = self.get_response(service, **kwargs).url

        stations = pd.read_csv(nwis_dict, compression='zip')
        return stations

    def get_wqp_results(self, service, **kwargs):
        """Bring data from WQP site into a Pandas DataFrame for analysis"""

        # set data types
        Rdtypes = {"OrganizationIdentifier": np.str_, "OrganizationFormalName": np.str_, "ActivityIdentifier": np.str_,
                   "ActivityStartTime/Time": np.str_,
                   "ActivityTypeCode": np.str_, "ActivityMediaName": np.str_, "ActivityMediaSubdivisionName": np.str_,
                   "ActivityStartDate": np.str_, "ActivityStartTime/TimeZoneCode": np.str_,
                   "ActivityEndDate": np.str_, "ActivityEndTime/Time": np.str_, "ActivityEndTime/TimeZoneCode": np.str_,
                   "ActivityDepthHeightMeasure/MeasureValue": np.float16,
                   "ActivityDepthHeightMeasure/MeasureUnitCode": np.str_,
                   "ActivityDepthAltitudeReferencePointText": np.str_,
                   "ActivityTopDepthHeightMeasure/MeasureValue": np.float16,
                   "ActivityTopDepthHeightMeasure/MeasureUnitCode": np.str_,
                   "ActivityBottomDepthHeightMeasure/MeasureValue": np.float16,
                   "ActivityBottomDepthHeightMeasure/MeasureUnitCode": np.str_,
                   "ProjectIdentifier": np.str_, "ActivityConductingOrganizationText": np.str_,
                   "MonitoringLocationIdentifier": np.str_, "ActivityCommentText": np.str_,
                   "SampleAquifer": np.str_, "HydrologicCondition": np.str_, "HydrologicEvent": np.str_,
                   "SampleCollectionMethod/MethodIdentifier": np.str_,
                   "SampleCollectionMethod/MethodIdentifierContext": np.str_,
                   "SampleCollectionMethod/MethodName": np.str_, "SampleCollectionEquipmentName": np.str_,
                   "ResultDetectionConditionText": np.str_, "CharacteristicName": np.str_,
                   "ResultSampleFractionText": np.str_,
                   "ResultMeasureValue": np.str_, "ResultMeasure/MeasureUnitCode": np.str_,
                   "MeasureQualifierCode": np.str_,
                   "ResultStatusIdentifier": np.str_, "StatisticalBaseCode": np.str_, "ResultValueTypeName": np.str_,
                   "ResultWeightBasisText": np.str_, "ResultTimeBasisText": np.str_,
                   "ResultTemperatureBasisText": np.str_,
                   "ResultParticleSizeBasisText": np.str_, "PrecisionValue": np.str_, "ResultCommentText": np.str_,
                   "USGSPCode": np.str_, "ResultDepthHeightMeasure/MeasureValue": np.float16,
                   "ResultDepthHeightMeasure/MeasureUnitCode": np.str_,
                   "ResultDepthAltitudeReferencePointText": np.str_,
                   "SubjectTaxonomicName": np.str_, "SampleTissueAnatomyName": np.str_,
                   "ResultAnalyticalMethod/MethodIdentifier": np.str_,
                   "ResultAnalyticalMethod/MethodIdentifierContext": np.str_,
                   "ResultAnalyticalMethod/MethodName": np.str_, "MethodDescriptionText": np.str_,
                   "LaboratoryName": np.str_,
                   "AnalysisStartDate": np.str_, "ResultLaboratoryCommentText": np.str_,
                   "DetectionQuantitationLimitTypeName": np.str_,
                   "DetectionQuantitationLimitMeasure/MeasureValue": np.str_,
                   "DetectionQuantitationLimitMeasure/MeasureUnitCode": np.str_, "PreparationStartDate": np.str_,
                   "ProviderName": np.str_}

        # define date field indices
        dt = [6, 56, 61]
        csv = self.get_response(service, **kwargs).url
        print(csv)
        # read csv into DataFrame
        df = pd.read_csv(csv, dtype=Rdtypes, parse_dates=dt, compression='zip')
        #df = pd.read_csv(csv, dtype=Rdtypes, parse_dates=dt)
        return df

    def massage_results(self, df = ''):
        """Massage WQP result data for analysis

        When called, this function:
        - renames all of the results fields, abbreviating the fields and eliminating slashes and spaces.
        - parses the datetime fields, fixing errors when possible (see :func:`datetimefix`)
        - standardizes units to mg/L
        - normalizes nutrient species(See :func:`parnorm`)
        """
        if df == '':
            df = self.results

        # Map new names for columns
        #TODO Match Field names to those in the SDE
        ResFieldDict = {"AnalysisStartDate": "AnalysisDate", "ResultAnalyticalMethod/MethodIdentifier": "AnalytMeth",
                        "ResultAnalyticalMethod/MethodName": "AnalytMethId",
                        "ResultDetectionConditionText": "DetectCond",
                        "ResultLaboratoryCommentText": "LabComments", "LaboratoryName": "LabName",
                        "DetectionQuantitationLimitTypeName": "LimitType",
                        "DetectionQuantitationLimitMeasure/MeasureValue": "MDL",
                        "DetectionQuantitationLimitMeasure/MeasureUnitCode": "MDLUnit",
                        "MethodDescriptionText": "MethodDescript",
                        "OrganizationIdentifier": "OrgId", "OrganizationFormalName": "OrgName",
                        "CharacteristicName": "Param",
                        "ProjectIdentifier": "ProjectId", "MeasureQualifierCode": "QualCode",
                        "ResultCommentText": "ResultComment",
                        "ResultStatusIdentifier": "ResultStatus", "ResultMeasureValue": "ResultValue",
                        "ActivityCommentText": "SampComment", "ActivityDepthHeightMeasure/MeasureValue": "SampDepth",
                        "ActivityDepthAltitudeReferencePointText": "SampDepthRef",
                        "ActivityDepthHeightMeasure/MeasureUnitCode": "SampDepthU",
                        "SampleCollectionEquipmentName": "SampEquip",
                        "ResultSampleFractionText": "SampFrac", "ActivityStartDate": "SampleDate",
                        "ActivityIdentifier": "SampleId",
                        "ActivityStartTime/Time": "SampleTime", "ActivityMediaSubdivisionName": "SampMedia",
                        "SampleCollectionMethod/MethodIdentifier": "SampMeth",
                        "SampleCollectionMethod/MethodName": "SampMethName",
                        "ActivityTypeCode": "SampType", "MonitoringLocationIdentifier": "StationId",
                        "ResultMeasure/MeasureUnitCode": "Unit", "USGSPCode": "USGSPCode"}

        # Rename Data
        #df = self.results
        df1 = df.rename(columns=self.rename['Result'])

        # Remove unwanted and bad times
        df1["sampledate"] = df1[["sampledate", "sampletime"]].apply(lambda x: self.datetimefix(x, "%Y-%m-%d %H:%M"), 1)

        # Define unneeded fields to drop
        #TODO make sure not to remove fields that are in the UGS SDE
        resdroplist = ["ActivityBottomDepthHeightMeasure/MeasureUnitCode",
                       "ActivityBottomDepthHeightMeasure/MeasureValue",
                       "ActivityConductingOrganizationText",
                       "ActivityEndDate", "ActivityEndTime/Time",
                       "ActivityEndTime/TimeZoneCode",
                       "ActivityMediaName",
                       "ActivityStartTime/TimeZoneCode",
                       "ActivityTopDepthHeightMeasure/MeasureUnitCode",
                       "ActivityTopDepthHeightMeasure/MeasureValue",
                       "HydrologicCondition", "HydrologicEvent", "PrecisionValue", "PreparationStartDate",
                       "ProviderName",
                       "ResultDepthAltitudeReferencePointText",
                       "ResultDepthHeightMeasure/MeasureUnitCode", "ResultDepthHeightMeasure/MeasureValue",
                       "ResultParticleSizeBasisText", "ResultTemperatureBasisText",
                       "ResultTimeBasisText", "ResultValueTypeName", "ResultWeightBasisText", "SampleAquifer",
                       "SampleTissueAnatomyName",
                       "StatisticalBaseCode",
                       "SubjectTaxonomicName", "SampleTime"]

        # Drop fields
        for i in resdroplist:
            if i in df1.columns:
                df1 = df1.drop(i, axis=1)

        # convert results and mdl to float
        df1['resultvalue'] = pd.to_numeric(df1['resultvalue'], errors='coerce')
        df1['detecquantlimitmeasure'] = pd.to_numeric(df1['detecquantlimitmeasure'], errors='coerce')

        # match old and new station ids
        df1['monitoringlocationid'] = df1['monitoringlocationid'].str.replace('_WQX-', '-')

        # standardize all ug/l data to mg/l
        df1.resultunit = df1.resultunit.apply(lambda x: str(x).rstrip(), 1)
        df1.resultvalue = df1[["resultvalue", "resultunit"]].apply(
            lambda x: x[0] / 1000 if str(x[1]).lower() == "ug/l" else x[0], 1)
        df1.resultunit = df1.resultunit.apply(lambda x: self.unitfix(x), 1)

        #df1['characteristicname'], df1['resultvalue'], df1['resultunit'] = zip(
        #    *df1[['characteristicname', 'resultvalue', 'resultunit']].apply(lambda x: self.parnorm(x), 1))

        df1['characteristicname'], df1['methodspeciation'], df1['resultunit'] = zip(
            *df1[['characteristicname', 'resultunit']].apply(lambda x: self.makemethspec(x), 1))

        self.results = df1

        return df1

    def datetimefix(self, x, form):
        """This script cleans date-time errors

        :param x: date-time string
        :param form: format of date-time string

        :returns: formatted datetime type
        """
        d = str(x[0]).lstrip().rstrip()[0:10]
        t = str(x[1]).lstrip().rstrip()[0:5].zfill(5)
        try:
            int(d[0:2])
        except(ValueError, TypeError, NameError):
            return np.nan
        try:
            int(t[0:2])
            int(t[3:5])
        except(ValueError, TypeError, NameError):
            t = "00:00"

        if int(t[0:2]) > 23:
            t = "00:00"
        elif int(t[3:5]) > 59:
            t = "00:00"
        else:
            t = t[0:2].zfill(2) + ":" + t[3:5]
        return datetime.strptime(d + " " + t, form)

    def parnorm(self, x):
        """Standardizes nutrient species

        - Nitrate as N to Nitrate
        - Nitrite as N to Nitrite
        - Sulfate as s to Sulfate
        """
        p = str(x[0]).rstrip().lstrip().lower()
        u = str(x[2]).rstrip().lstrip().lower()
        if p == 'nitrate' and u == 'mg/l as n':
            return 'Nitrate', x[1] * 4.427, 'mg/l'
        elif p == 'nitrite' and u == 'mg/l as n':
            return 'Nitrite', x[1] * 3.285, 'mg/l'
        elif p == 'ammonia-nitrogen' or p == 'ammonia-nitrogen as n' or p == 'ammonia and ammonium':
            return 'Ammonium', x[1] * 1.288, 'mg/l'
        elif p == 'ammonium' and u == 'mg/l as n':
            return 'Ammonium', x[1] * 1.288, 'mg/l'
        elif p == 'sulfate as s':
            return 'Sulfate', x[1] * 2.996, 'mg/l'
        elif p in ('phosphate-phosphorus', 'phosphate-phosphorus as p', 'orthophosphate as p'):
            return 'Phosphate', x[1] * 3.066, 'mg/l'
        elif (p == 'phosphate' or p == 'orthophosphate') and u == 'mg/l as p':
            return 'Phosphate', x[1] * 3.066, 'mg/l'
        elif u == 'ug/l':
            return x[0], x[1] / 1000, 'mg/l'
        else:
            return x[0], x[1], str(x[2]).rstrip()

    def makemethspec(self, x):
        p = str(x[0]).rstrip().lstrip().lower()
        u = str(x[1]).rstrip().lstrip().lower()
        if p == 'nitrate' and u == 'mg/l as n':
            return 'Nitrate', 'as N', 'mg/l'
        elif p == 'nitrite' and u == 'mg/l as n':
            return 'Nitrite', 'as N', 'mg/l'
        elif p == 'ammonia-nitrogen' or p == 'ammonia-nitrogen as n' or p == 'ammonia and ammonium':
            return 'Ammonium', 'as N', 'mg/l'
        elif p == 'ammonium' and u == 'mg/l as n':
            return 'Ammonium', 'as N', 'mg/l'
        elif p == 'sulfate as s':
            return 'Sulfate', 'as S', 'mg/l'
        elif p in ('phosphate-phosphorus', 'phosphate-phosphorus as p', 'orthophosphate as p'):
            return 'Phosphate', 'as N', 'mg/l'
        elif (p == 'phosphate' or p == 'orthophosphate') and u == 'mg/l as p':
            return 'Phosphate', 'as P', 'mg/l'
        else:
            return x[0], None, x[1]

    def unitfix(self, x):
        """Standardizes unit labels from ug/l to mg/l

        :param x: unit label to convert
        :type x: str

        :returns: unit string as mg/l
        .. warning:: must be used with a value conversion tool
        """
        z = str(x).lower()
        if z == "ug/l":
            return "mg/l"
        elif z == "mg/l":
            return "mg/l"
        else:
            return x

    def massage_stations(self):
        """Massage WQP station data for analysis
        """
        #TODO match to fields in UGS SDE

        df = self.stations
        df.rename(columns=self.rename['Station'], inplace=True)

        statdroplist = ["ContributingDrainageAreaMeasure/MeasureUnitCode",
                        "ContributingDrainageAreaMeasure/MeasureValue",
                        "DrainageAreaMeasure/MeasureUnitCode", "DrainageAreaMeasure/MeasureValue", "CountryCode",
                        "ProviderName",
                        "SourceMapScaleNumeric"]

        df.drop(statdroplist, inplace=True, axis=1)

        TypeDict = {"River/Stream": "Stream", "Stream: Canal": "Stream",
                    "Well: Test hole not completed as a well": "Well"}

        # Make station types in the StationType field consistent for easier summary and compilation later on.
        df['locationtype'] = df['locationtype'].apply(lambda x: TypeDict.get(x, x), 1)
        df['verticalmeasure'] = df['verticalmeasure'].apply(lambda x: np.nan if x == 0.0 else round(x, 1), 1)

        # Remove preceding WQX from StationId field to remove duplicate station data created by legacy database.
        df['locationid'] = df['locationid'].str.replace('_WQX-', '-')
        df.drop_duplicates(subset=['locationid'], inplace=True)
        #self.stations = df
        return df

    def piv_chem(self, results='', chems='piper'):
        """pivots results DataFrame for input into piper class

        :param results: DataFrame of results data from WQP; default is return from call of :class:`WQP`
        :param chems: set of chemistry that must be present to retain row; default are the major ions for a piper plot
        :return: pivoted table of result values

        .. warnings:: this method drops < and > signs from values; do not use it for statistics
        """

        if results == '':
            results = self.results
        #print(results.columns)

        results['ParAbb'] = results['characteristicname'].apply(lambda x: self.ParAbb.get(x, ''), 1)
        results.dropna(subset=['sampleid'], how='any', inplace=True)
        results = results[pd.isnull(results['resultdetectioncondition'])]
        results.drop_duplicates(subset=['sampleid', 'ParAbb'], inplace=True)
        datap = results.pivot(index='sampleid', columns='ParAbb', values='resultvalue')
        if chems == '':
            pass
        elif chems == 'piper':
            datap.dropna(subset=['SO4', 'Cl', 'Ca', 'HCO3', 'pH'], how='any', inplace=True)
        else:
            datap.dropna(subset=chems, how='any', inplace=True)
        return datap

if __name__ == "__main__":
    import sys
    WQP(int(sys.argv[1]))