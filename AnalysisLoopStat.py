import stat

import h5py
import argparse
import os
import numpy as np
import pynwb
import pandas as pd
import tensortools as tt
import matplotlib.pyplot as plt
import numpy.matlib as npm
import scipy.stats as stats
import itertools
import scipy
import re
import statsmodels.stats.multicomp as mc
import statsmodels.stats.multitest as mt
from scikit_posthocs import posthoc_tukey
from scikit_posthocs import posthoc_dunn
import matplotlib
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

## Functions


def find_closest_index(original_vector, vector_to_find):
    indexes_nearest = np.ones(vector_to_find.shape)
    indexes_nearest[:] = np.nan;

    for counterItems in range(vector_to_find.shape[0]):
        currItem = vector_to_find[counterItems]
        indexes_nearest[counterItems] = np.argmin(abs(original_vector - currItem))
    indexes_nearest = indexes_nearest.astype(int);
    return indexes_nearest

def find_closest_preceding_index(original_vector, vector_to_find, indexes_shift=0):
    indexes_nearest = np.ones(vector_to_find.shape)
    indexes_nearest[:] = np.nan

    for counterItems in range(vector_to_find.shape[0]):
        currItem = vector_to_find[counterItems]
        tmpDiff = original_vector - currItem
        tmpDiff[tmpDiff > 0] = min(tmpDiff) - 1
        indexes_nearest[counterItems] = np.argmin(abs(tmpDiff)) - indexes_shift;
    indexes_nearest = indexes_nearest.astype(int)
    return indexes_nearest

def load_experiment(Path):
    db = h5py.File(Path)
    OphysData = db['processing/ophys/event_detection/data'][()]
    OphysTs = db['processing/ophys/dff/traces/timestamps'][()]
    return db, OphysData, OphysTs

def extract_events_times(db):
    AllIntervals = db['intervals']

    ListKeys = list(AllIntervals.keys());
    ListKeys.remove('trials')
    ListKeys.remove('omitted_presentations')

    AllStarts = np.empty(0)
    # AllImageNames = np.empty(0)
    AllImageID = np.empty(0)
    for counterKeys in range(len(ListKeys)):
        currKey = ListKeys[counterKeys]
        currStarts = AllIntervals[currKey]['start_time'][()]
        # AllImageNames = AllImageNames[currKey]['start_time'][()]
        AllStarts = np.concatenate((AllStarts, currStarts))
        AllImageID = np.concatenate((AllImageID, np.ones(len(currStarts)) * counterKeys))
        print(currKey)

    SortingIndices = np.argsort(AllStarts)
    AllStarts = AllStarts[SortingIndices]
    # AllImageID = AllImageID[SortingIndices]

    # IndexesNonOmissions = np.where(np.diff(AllStarts) < 1)[0] + 1
    # IndexesImagesChange = np.where(abs(np.diff(AllImageID)) > 0)[0]
    # IndexesImagesChangeNonOmit = np.intersect1d(IndexesNonOmissions, IndexesImagesChange)

    AllOmittedStarts = AllIntervals['omitted_presentations']['start_time'][()]

    IndexesImagePrecedingOmission0 = find_closest_preceding_index(AllStarts, AllOmittedStarts, indexes_shift=0)
    IndexesImagePrecedingOmission1 = find_closest_preceding_index(AllStarts, AllOmittedStarts, indexes_shift=1)
    IndexesImagePrecedingOmission2 = find_closest_preceding_index(AllStarts, AllOmittedStarts, indexes_shift=2)
    IndexesImagePrecedingOmission3 = find_closest_preceding_index(AllStarts, AllOmittedStarts, indexes_shift=3)

    Diff01 = AllStarts[IndexesImagePrecedingOmission0] - AllStarts[IndexesImagePrecedingOmission1]
    Diff12 = AllStarts[IndexesImagePrecedingOmission1] - AllStarts[IndexesImagePrecedingOmission2]
    Diff23 = AllStarts[IndexesImagePrecedingOmission2] - AllStarts[IndexesImagePrecedingOmission3]

    IndexesImagePrecedingOmissionToInclude = np.intersect1d(np.intersect1d(np.intersect1d(np.where(Diff01 < 0.8)[0], np.where(Diff12 < 0.8)[0]), np.where(Diff23 < 0.8)[0] ),np.where(IndexesImagePrecedingOmission2>0)[0])
    IndexesImagePrecedingOmission = IndexesImagePrecedingOmission1[IndexesImagePrecedingOmissionToInclude];
    AllImagesPrecedingOmissionStarts = AllStarts[IndexesImagePrecedingOmission]


    AllOmittedStarts = AllOmittedStarts[np.logical_and(np.concatenate((np.ones(1), np.diff(AllOmittedStarts) > 0.8)),
                                                       np.concatenate((np.diff(AllOmittedStarts) > 0.8, np.ones(1))))]

    # StimStartIndexImages = find_closest_index(OphysTs,AllStarts[IndexesImagesChangeNonOmit])
    #StimStartIndexImages = find_closest_index(OphysTs, AllImagesPrecedingOmissionStarts)
    #StimStartIndexOmissions = find_closest_index(OphysTs, AllOmittedStarts)

    return AllImagesPrecedingOmissionStarts, AllOmittedStarts

def extract_events_indexes(OphysTs, db, MinPrestim = 10):
    [AllImagesPrecedingOmissionStarts, AllOmittedStarts] = extract_events_times(db)

    AllOmittedStarts = AllOmittedStarts[(AllOmittedStarts-OphysTs[0])>MinPrestim]
    StimStartIndexImages = find_closest_index(OphysTs, AllImagesPrecedingOmissionStarts)
    StimStartIndexOmissions = find_closest_index(OphysTs, AllOmittedStarts)

    return StimStartIndexImages, StimStartIndexOmissions

def split_data(OphysData , StimStartIndex, WinCutPre = 10, WinCutPost = 10):
    Split = np.ones((StimStartIndex.shape[0], WinCutPre + WinCutPost, OphysData.shape[1]))
    Split[:, :] = np.nan

    for counterFrame in range(StimStartIndex.shape[0]):
        Split[counterFrame, :, :] = OphysData[(StimStartIndex[counterFrame] - WinCutPre):(StimStartIndex[counterFrame] + WinCutPost),:]

    SplitTS = np.array(range(-WinCutPre, WinCutPost))
    return Split, SplitTS

def normalize_split(Split, Normalization = 'minmax' ):
    if Normalization is 'minmax':
        Split = Split - np.min(np.min(Split, axis=1), axis=0);
        NonZeroIndexes = np.where(np.max(np.max(Split, axis=1), axis=0)>0)[0];
        Split[:,:,NonZeroIndexes] = Split[:,:,NonZeroIndexes]  / np.max(np.max(Split[:,:,NonZeroIndexes] , axis=1), axis=0);
    # else if Normalization is 'zscore':
    #    Split = Split / np.max(np.max(Split, axis=1), axis=0);
    return Split

def normalize_continuous(OphysData, Normalization = 'minmax' ):
    if Normalization is 'minmax':
        OphysData = OphysData - np.min(OphysData, axis = 0)
        OphysData = OphysData / np.max(OphysData, axis = 0)
    elif Normalization is 'zscore':
        OphysData = OphysData / OphysData.std(axis=0)[np.newaxis,:]
    return OphysData

def load_table():
    Table = pd.read_csv(r'D:\AllenInstitute\VisualBehavior\visual-behavior-ophys-1.0.1\project_metadata\ophys_experiment_table.csv')
    return Table

def filter_table(Table,Field = 'cre_line', Group = 'All'):
    Table = Table.dropna(subset=["project_code"])
    #Table = (Table[['Multiscope' in x for x in Table['project_code']]])
    Table = (Table[['VisualBehaviorMultiscope' == x for x in Table['project_code']]])
    Table = (Table[['Familiar' in x for x in Table['experience_level']]])
    # Table = (Table[['OPHYS_' in x for x in Table['session_type']]])

    Table = Table[np.logical_not(Table['passive'])]
    # Table_Block1 = (Table[['OPHYS_1' in x for x in Table['session_type']]])
    # Table_Block3 = (Table[['OPHYS_3' in x for x in Table['session_type']]])
    # Table = pd.concat((Table_Block1,Table_Block3))

    if Group == 'All':
        pass
    else:
        Table = (Table[[Group in x for x in Table[Field]]])
    return Table

def filter_table_novel(Table,Field = 'cre_line', Group = 'All'):
    Table = Table.dropna(subset=["project_code"])
    #Table = (Table[['Multiscope' in x for x in Table['project_code']]])
    Table = (Table[['VisualBehaviorMultiscope' == x for x in Table['project_code']]])
    Table = (Table[['Novel 1' in x for x in Table['experience_level']]])
    # Table = (Table[['OPHYS_' in x for x in Table['session_type']]])

    Table = Table[np.logical_not(Table['passive'])]
    # Table_Block1 = (Table[['OPHYS_1' in x for x in Table['session_type']]])
    # Table_Block3 = (Table[['OPHYS_3' in x for x in Table['session_type']]])
    # Table = pd.concat((Table_Block1,Table_Block3))

    if Group == 'All':
        pass
    else:
        Table = (Table[[Group in x for x in Table[Field]]])
    return Table

def plot_butterfly(Split, SplitTs, PathToSave = 'none'):
    fig, axes = plt.subplots(1, 1)
    plt.plot(SplitTs,np.nanmean(Split, axis=0)) # Split = trials x time x cells
    #plt.title(str(currTable['ophys_experiment_id'].iloc[currExpIndex]))
    if PathToSave != 'none':
        plt.savefig(os.path.join(PathToSave, "butterfly_plot.png"), dpi=300)
    plt.show()

def plt_TCA_reconstruction(ensemble, PathToSave = 'none'):
    fig, axes = plt.subplots(1, 2)
    tt.plot_objective(ensemble, ax=axes[0])  # plot
    tt.plot_similarity(ensemble, ax=axes[1])  # plot model similarity as a function of num components.
    fig.tight_layout()
    if PathToSave != 'none':
        plt.savefig(os.path.join(PathToSave, "tca_reconstruction.png"), dpi=300)
    plt.show()

def plot_TCA_weights(ensemble, nFactors, PathToSave = 'none', Repeat = 0):
    fig, axes = plt.subplots(1, 1)
    tt.plot_factors(ensemble.factors(nFactors)[Repeat])  # plot the low-d factors
    if PathToSave != 'none':
        plt.savefig(os.path.join(PathToSave, "tca_weights.png"), dpi=300)
    plt.show()
    return fig

def get_weights(ensemble, nFactors):
    WeightPerChannel = ensemble.factors(nFactors)[0][0];
    WeightPerTime = ensemble.factors(nFactors)[0][1];
    WeightPerTrials = ensemble.factors(nFactors)[0][2];
    return WeightPerChannel, WeightPerTime, WeightPerTrials

def plot_trial_weights_by_component(WeightPerTrials,CutIndex, Scale = 0.2, PathToSave = 'none'):
    tmp1 = WeightPerTrials[0:CutIndex, :]
    tmp2 = WeightPerTrials[CutIndex::, :]

    BigMatrixSize = np.max((tmp1.shape[0], tmp2.shape[0]))
    BigMatrix = np.zeros((BigMatrixSize, tmp1.shape[1] * 2))
    BigMatrix[:, :] = np.nan

    for counterColumn in range(tmp1.shape[1]):
        BigMatrix[0:tmp1.shape[0], counterColumn * 2] = tmp1[:, counterColumn]
        #print(counterColumn * 2)
    for counterColumn in range(tmp2.shape[1]):
        BigMatrix[0:tmp2.shape[0], counterColumn * 2 + 1] = tmp2[:, counterColumn]
        #print(counterColumn * 2 + 1)

    mask = ~np.isnan(BigMatrix)
    FilteredBigMatrix = [d[m] for d, m in zip(BigMatrix.T, mask.T)]

    fig, ax = plt.subplots(1, 1)
    plt.boxplot(FilteredBigMatrix)
    plt.ylim(-Scale, Scale)
    for counterX in range(np.round(len(FilteredBigMatrix) / 2).astype(int)):
        currX = counterX * 2 + 0.5
        plt.plot([currX, currX], [-Scale, Scale])
    if PathToSave != 'none':
        plt.savefig(os.path.join(PathToSave, "WeightsPerComponent_Image_Omissions.png"), dpi=300)
    plt.show()

def get_area_weights_single_component(WeightPerChannelCurrComponent ,ImagingLocDepth): # WeightPerChannel[currComponent]
    UniqueImagingLocDepth = np.unique(ImagingLocDepth)

    #plt.figure()
    tmp = np.empty(10)
    tmp[:] = np.nan
    for counterUniqueImagingLocDepth in range(UniqueImagingLocDepth.shape[0]):
        currUniqueImagingLocDepth = UniqueImagingLocDepth[counterUniqueImagingLocDepth]
        currIndexes = np.where([currUniqueImagingLocDepth in x for x in ImagingLocDepth])[0]
        currWeightPerChannel = WeightPerChannel[currIndexes, currComponent]

        if counterUniqueImagingLocDepth == 0:
            WeightsPerAreas = ([currWeightPerChannel])
        else:
            WeightsPerAreas = WeightsPerAreas + ([currWeightPerChannel])
        # tmp3 = np.log10(abs())
    return WeightsPerAreas, UniqueImagingLocDepth

def plot_area_weights_single_component(WeightPerChannelCurrComponent ,ImagingLocDepth, YLim = (-0.2, 0.2), PathToSave = 'none'):
    [WeightsPerAreas, AreasLabels] = get_area_weights_single_component(WeightPerChannelCurrComponent, ImagingLocDepth)
    plt.boxplot(WeightsPerAreas)
    plt.xticks(np.array(range(len(WeightsPerAreas)))+1, AreasLabels)
    plt.ylim(YLim)
    if PathToSave != 'none':
        plt.savefig(os.path.join(PathToSave, "WeightsPerChannel.png"), dpi=300)
    plt.show()

def stat_trials_weights_per_component(WeightPerTrials,CutIndex):
    tmp1 = WeightPerTrials[0:CutIndex, :]
    tmp2 = WeightPerTrials[CutIndex::, :]
    TestPValues = np.empty(0)
    MediansDiff = np.empty(0)
    MediansAbsDiff = np.empty(0)
    for counterComponent in range(tmp1.shape[1]):
        tmpP = stats.ttest_ind(tmp1[:, counterComponent], tmp2[:, counterComponent])
        tmpMedian = np.median(tmp1[:, counterComponent]) - np.median(tmp2[:, counterComponent])
        tmpMedianAbs = abs(np.median(tmp1[:, counterComponent])) - abs(np.median(tmp2[:, counterComponent]))
        TestPValues = np.concatenate((TestPValues, [tmpP[1]]))
        MediansDiff = np.concatenate((MediansDiff, [tmpMedian]))
        MediansAbsDiff = np.concatenate((MediansAbsDiff, [tmpMedianAbs]))
        #print(Test[1])
    return TestPValues, MediansDiff, MediansAbsDiff

def load_single_sess_from_table(MainFolder,currTable):
    OphysData = None
    for currExpIndex in range(currTable.shape[0]):
        Path = MainFolder + r'\behavior_ophys_experiment_' + str(currTable['ophys_experiment_id'].iloc[currExpIndex]) + r'.nwb'
        [db1, OphysData1, OphysTs1] = load_experiment(Path)
        # ImagingLocation = db1[r'general/optophysiology/imaging_plane_1/location'][()]
        currImagingLocation = currTable['targeted_structure'].iloc[currExpIndex]
        currImagingDepth = currTable['imaging_depth'].iloc[currExpIndex]
        currImagingLocDepth = currImagingLocation + str(currImagingDepth);

        if OphysData is None:
            OphysData = OphysData1
            ImagingLocation = list(itertools.repeat(currImagingLocation, OphysData1.shape[1]))
            ImagingDepth = list(itertools.repeat(currImagingDepth, OphysData1.shape[1]))
            ImagingLocDepth = list(itertools.repeat(currImagingLocDepth, OphysData1.shape[1]))
        else:
            OphysData = np.concatenate((OphysData, OphysData1), axis=1)
            ImagingLocation = ImagingLocation + list(itertools.repeat(currImagingLocation, OphysData1.shape[1]))
            ImagingDepth = ImagingDepth + list(itertools.repeat(currImagingDepth, OphysData1.shape[1]))
            ImagingLocDepth = ImagingLocDepth + list(itertools.repeat(currImagingLocDepth, OphysData1.shape[1]))
    return OphysData, ImagingLocDepth, db1, OphysTs1

def load_single_sess_exported_data(SessID , MainFolder_ExportedData = r'D:\AllenInstitute\VisualBehavior\ExportedData'):

    if np.logical_not(isinstance(SessID, str)):
        SessID = str(SessID)

    currFolder = os.path.join(MainFolder_ExportedData, SessID)
    SplitData = np.load(os.path.join(currFolder, 'SplitData.npy'))
    ImagingLocDepth = np.load(os.path.join(currFolder, 'ImagingLocDepth.npy'))
    currCutoffIndexes = np.load(os.path.join(currFolder, 'currCutoffIndexes.npy'))
    OphysTs = np.load(os.path.join(currFolder, 'OphysTs.npy'))
    WindowPrePost = np.load(os.path.join(currFolder, 'WindowPrePost.npy'))

    return SplitData, ImagingLocDepth, currCutoffIndexes, OphysTs, WindowPrePost

def stat_area_weights_single_component(WeightPerChannelCurrComponent, ImagingLocDepth, test_type = 'parametric'):
    [WeightsPerAreas, AreasLabels] = get_area_weights_single_component(WeightPerChannelCurrComponent, ImagingLocDepth)

    Medians = np.zeros(len(WeightsPerAreas))
    Medians[:] = np.nan
    Stds = np.zeros(len(WeightsPerAreas))
    Stds[:] = np.nan

    for counterColumn in range(len(WeightsPerAreas)):
        if counterColumn == 0:
            Group = list(itertools.repeat(str(counterColumn), len(WeightsPerAreas[counterColumn])))
            Values = list(WeightsPerAreas[counterColumn])
        else:
            Group = Group + list(itertools.repeat(str(counterColumn), len(WeightsPerAreas[counterColumn])))
            Values = Values + list(WeightsPerAreas[counterColumn])
        Medians[counterColumn] = np.nanmean(WeightsPerAreas[counterColumn])
        #Medians[counterColumn] = np.nanmedian(WeightsPerAreas[counterColumn])
        Stds[counterColumn] = np.nanstd(WeightsPerAreas[counterColumn])
    df = pd.DataFrame({'Values': Values, 'Group': Group})

    if test_type == 'parametric':
        stat_df = scipy.stats.f_oneway(*WeightsPerAreas)
        posthoc_df = posthoc_tukey(df, val_col="Values", group_col="Group")
    elif test_type == 'nonparametric':
        stat_df = stats.kruskal(*WeightsPerAreas)
        posthoc_df = posthoc_dunn(df, val_col="Values", group_col="Group", p_adjust="fdr_bh")

    return stat_df, posthoc_df, Medians, Stds, AreasLabels

def get_image_and_omission_components(WeightPerTrials, CutIndex, SignificanceThreshold=0.05):
    TestPValues, MediansDiff, MediansAbsDiff = stat_trials_weights_per_component(WeightPerTrials, CutIndex)

    Indexes1 = np.intersect1d(np.where(MediansAbsDiff > 0), np.where(TestPValues < SignificanceThreshold))
    Indexes2 = np.intersect1d(np.where(MediansAbsDiff < 0), np.where(TestPValues < SignificanceThreshold))

    Indexes = np.zeros(MediansDiff.shape[0])
    Indexes[Indexes1] = 1;
    Indexes[Indexes2] = 2;
    return Indexes, MediansDiff, MediansAbsDiff, TestPValues

def match_labels(AreaLabels, nLabels = 4):
    if nLabels == 8:
        DefaultLabels = ['VISl150', 'VISl225', 'VISl300', 'VISl75', 'VISp150', 'VISp225', 'VISp300', 'VISp75']
        MatchedAreas = 'none'
        MatchedIndexes = np.zeros(len(AreaLabels))
        MatchedIndexes[:] = np.nan

        DefaultLabels_Str = 'none'
        DefaultLabels_Num = np.zeros(len(DefaultLabels))
        DefaultLabels_Num[:] = np.nan
        for counterDefaultLabels in range(len(DefaultLabels)):
            currDefaultLabels = DefaultLabels[counterDefaultLabels]
            currDefaultLabels_Str = list([" ".join(re.findall("[a-zA-Z]+", currDefaultLabels))])
            DefaultLabels_Num[counterDefaultLabels] = int(" ".join(re.findall("[0-9]+", currDefaultLabels)))
            if DefaultLabels_Str == 'none':
                DefaultLabels_Str = currDefaultLabels_Str
            else:
                DefaultLabels_Str = DefaultLabels_Str + currDefaultLabels_Str

        AreaLabels_Str = 'none'
        AreaLabels_Num = np.zeros(len(AreaLabels))
        AreaLabels_Num[:] = np.nan
        for counterAreaLabels in range(len(AreaLabels)):
            currAreaLabel = AreaLabels[counterAreaLabels]
            currAreaLabel_Str = list([" ".join(re.findall("[a-zA-Z]+", currAreaLabel))])
            currAreaLabels_Num = int(" ".join(re.findall("[0-9]+", currAreaLabel)))
            AreaLabels_Num[counterAreaLabels]  = currAreaLabels_Num
            if AreaLabels_Str == 'none':
                AreaLabels_Str = currAreaLabel_Str
            else:
                AreaLabels_Str = AreaLabels_Str + currAreaLabel_Str

            tmpDiff = abs(currAreaLabels_Num - DefaultLabels_Num)
            tmpIndices = [i for i in range(len(DefaultLabels_Str)) if DefaultLabels_Str[i] != currAreaLabel_Str[0]]
            tmpDiff[tmpIndices] = max(tmpDiff)  + 1
            currIndex = np.argmin(tmpDiff)
            currMatched = DefaultLabels[currIndex]

            MatchedIndexes[counterAreaLabels] = currIndex
            if MatchedAreas == 'none':
                MatchedAreas = list([currMatched])
            else:
                MatchedAreas = MatchedAreas + list([currMatched])
    elif nLabels == 4:
        DefaultLabels = ['VISl_sup', 'VISl_deep', 'VISp_sup', 'VISp_deep']
        MatchedAreas = 'none'
        MatchedIndexes = np.zeros(len(AreaLabels))
        MatchedIndexes[:] = np.nan

        DefaultLabels_Str = 'none'
        for counterDefaultLabels in range(len(DefaultLabels)):
            currDefaultLabels = DefaultLabels[counterDefaultLabels]
            currDefaultLabels_Str = [currDefaultLabels.split('_')[0]]
            if DefaultLabels_Str == 'none':
                DefaultLabels_Str = currDefaultLabels_Str
            else:
                DefaultLabels_Str = DefaultLabels_Str + currDefaultLabels_Str

        AreaLabels_Str = 'none'
        AreaLabels_Num = np.zeros(len(AreaLabels))
        AreaLabels_Num[:] = np.nan
        for counterAreaLabels in range(len(AreaLabels)):
            currAreaLabel = AreaLabels[counterAreaLabels]
            currAreaLabel_Str = list([" ".join(re.findall("[a-zA-Z]+", currAreaLabel))])
            currAreaLabels_Num = int(" ".join(re.findall("[0-9]+", currAreaLabel)))
            AreaLabels_Num[counterAreaLabels] = currAreaLabels_Num
            if AreaLabels_Str == 'none':
                AreaLabels_Str = currAreaLabel_Str
            else:
                AreaLabels_Str = AreaLabels_Str + currAreaLabel_Str

            if currAreaLabels_Num<250:
                currMatched = currAreaLabel_Str[0] + '_sup'
            elif currAreaLabels_Num>=250:
                currMatched = currAreaLabel_Str[0] + '_deep'

            MatchedIndexes[counterAreaLabels] = DefaultLabels.index(currMatched)
            if MatchedAreas == 'none':
                MatchedAreas = list([currMatched])
            else:
                MatchedAreas = MatchedAreas + list([currMatched])

    return MatchedAreas, MatchedIndexes






ExportData = False
LoadExportedData = False
GeneratePlots = False
WindowPrePost = (3,7) #(10,10) #(3,7)
nComponents = 10  # 10 # 5
nLabels = 4
nRuns = 100
CellType = 'Slc' #Vip Slc Sst

ShuffleTrials = True #

ShuffleNeurons = False # keep false
ShuffleTimes = False # keep false

#['VISl150', 'VISl225', 'VISl300', 'VISl75', 'VISp150', 'VISp225', 'VISp300', 'VISp75']
AreasNames = ['VISl_sup', 'VISl_deep', 'VISp_sup', 'VISp_deep']

ConnectionNames = []
for counterAreas1 in range(len(AreasNames)):
    currArea1 = AreasNames[counterAreas1];
    for counterAreas2 in range(len(AreasNames)):
        currArea2 = AreasNames[counterAreas2];
        currAreas = [currArea1,currArea2]
        currAreas.sort()
        ConnectionNames.append(currAreas[0] + '-' + currAreas[1])


## Run
MainFolder = r'D:\AllenInstitute\VisualBehavior\visual-behavior-ophys-1.0.1\behavior_ophys_experiments'
MainFolderToSaveGeneral = r'D:\AllenInstitute\VisualBehavior\TCAOutputs'
save_path = r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\figures'
MainFolder_ExportedData = r'D:\AllenInstitute\VisualBehavior\ExportedData'
MainFolder_Figures = r'D:\AllenInstitute\VisualBehavior\Figures'

Table = load_table()
Table = filter_table(Table, Field='cre_line', Group=CellType) #Sst WIP #Vip #Slc
#Table = filter_table_novel(Table, Field='cre_line', Group=CellType) #Sst WIP #Vip #Slc
UniqueSessID = np.unique(Table['ophys_session_id'])




AllCorr_p_images = np.zeros((nLabels,nLabels,nRuns))
AllCorr_r_images = np.zeros((nLabels,nLabels,nRuns))
AllCorr_p_omissions = np.zeros((nLabels,nLabels,nRuns))
AllCorr_r_omissions = np.zeros((nLabels,nLabels,nRuns))
AllWeights_images = np.zeros((nLabels,nRuns))
AllWeights_omissions = np.zeros((nLabels,nRuns))
AllNComp_images = np.zeros((nRuns))
AllNComp_omissions = np.zeros((nRuns))
AllWeights_images[:,:] = np.nan
AllWeights_omissions[:,:] = np.nan
AllNComp_images[:] = np.nan
AllNComp_omissions[:] = np.nan

All_StdSingleAreas_images = np.zeros((nLabels,nRuns))
All_StdSingleAreas_omissions = np.zeros((nLabels,nRuns))
All_StdTrials_images = np.zeros((nRuns))
All_StdTrials_omissions = np.zeros((nRuns))
All_StdSingleAreas_images[:] = np.nan
All_StdSingleAreas_omissions[:] = np.nan
All_StdTrials_images[:] = np.nan
All_StdTrials_omissions[:] = np.nan

All_Ts_ComponentsOmission = np.empty((0, WindowPrePost[0]+WindowPrePost[1]));

All_Similarities = np.zeros((nRuns))
All_Objectives = np.zeros((nRuns))
All_Similarities[:] = np.nan
All_Objectives[:] = np.nan


for counterRun in range(nRuns):




    CountComponentsImage = np.zeros(UniqueSessID.shape[0])
    CountComponentsImage[:] = np.nan
    CountComponentsOmission = np.zeros(UniqueSessID.shape[0])
    CountComponentsOmission[:] = np.nan

    GrandAvg_Omission = np.empty((0, WindowPrePost[0]+WindowPrePost[1]))
    GrandAvg_Image = np.empty((0, WindowPrePost[0]+WindowPrePost[1]))

    WeightPerChannel_ComponentsUnspecific = np.zeros(UniqueSessID.shape[0])
    WeightPerChannel_ComponentsUnspecific[:] = np.nan
    WeightPerChannel_ComponentsOmission = np.zeros(UniqueSessID.shape[0])
    WeightPerChannel_ComponentsOmission[:] = np.nan
    WeightPerChannel_ComponentsImage = np.zeros(UniqueSessID.shape[0])
    WeightPerChannel_ComponentsImage[:] = np.nan

    Ts_ComponentsOmission = np.empty((0, WindowPrePost[0]+WindowPrePost[1]))
    Ts_ComponentsImage = np.empty((0, WindowPrePost[0]+WindowPrePost[1]))
    Ts_ComponentsOmission_ID = np.empty((0))
    Ts_ComponentsImage_ID = np.empty((0))

    AllStatsAreas = np.empty((nLabels, nLabels, 0))
    AllStatsAreas_image = np.empty((nLabels, nLabels, 0))
    AllStatsAreas_omissions = np.empty((nLabels, nLabels, 0))

    AllMediansAreas = np.empty((nLabels, nLabels, 0))
    AllMediansAreas_image = np.empty((nLabels, nLabels, 0))
    AllMediansAreas_omissions = np.empty((nLabels, nLabels, 0))

    AllMediansAbsAreas = np.empty((nLabels, nLabels, 0))
    AllMediansAbsAreas_image = np.empty((nLabels, nLabels, 0))
    AllMediansAbsAreas_omissions = np.empty((nLabels, nLabels, 0))

    AllMediansSingleAreas = np.empty((nLabels, 0))
    AllMediansSingleAreas_image = np.empty((nLabels, 0))
    AllMediansSingleAreas_omissions = np.empty((nLabels, 0))
    AllStdsSingleAreas = np.empty((nLabels, 0))
    AllStdsSingleAreas_image = np.empty((nLabels, 0))
    AllStdsSingleAreas_omissions = np.empty((nLabels, 0))

    PolarityTrials_ComponentsImage = np.empty((0))
    PolarityTrials_ComponentsOmission = np.empty((0))
    PolarityChannel_ComponentsImage = np.empty((0))
    PolarityChannel_ComponentsOmission = np.empty((0))

    AllCompWeight_image = np.empty((0))
    AllCompWeight_omission = np.empty((0))

    AllStdTrials_image = np.empty((0))
    AllStdTrials_omission = np.empty((0))

    currObjectives = np.empty((0))
    currSimilarities = np.empty((0))

    for currSessIndex in range(UniqueSessID.shape[0]):
        currSessID = UniqueSessID[currSessIndex]
        currTable = Table.iloc[np.where(Table['ophys_session_id'] == currSessID)[0]]
        if LoadExportedData:
            [Split, ImagingLocDepth, currCutoffIndexes, OphysTs, WindowPrePost] = load_single_sess_exported_data(currTable['ophys_session_id'].values[0])
        else:
            [OphysData, ImagingLocDepth, db, OphysTs] = load_single_sess_from_table(MainFolder, currTable)
            OphysData = scipy.ndimage.gaussian_filter1d(OphysData, 2, axis=0)

            [StimStartIndexImages, StimStartIndexOmissions] = extract_events_indexes(OphysTs, db, MinPrestim = WindowPrePost[0])
            StimStartIndex = np.concatenate((StimStartIndexImages, StimStartIndexOmissions))
            currCutoffIndexes = StimStartIndexImages.shape[0]
            Split, SplitTs = split_data(OphysData, StimStartIndex, WinCutPre=WindowPrePost[0], WinCutPost=WindowPrePost[1])
            Split = normalize_split(Split)
        #print(currCutoffIndexes)
        #print(currCutoffIndexes - Split.shape[0])

        if ExportData:
            currExportedDataFolder = os.path.join(MainFolder_ExportedData, str(currSessID))
            if np.logical_not(os.path.exists(currExportedDataFolder)):
                os.mkdir(currExportedDataFolder)
            np.save(os.path.join(currExportedDataFolder, 'SplitData'), Split)
            np.save(os.path.join(currExportedDataFolder, 'currCutoffIndexes'), currCutoffIndexes)
            np.save(os.path.join(currExportedDataFolder, 'ImagingLocDepth'), ImagingLocDepth)
            np.save(os.path.join(currExportedDataFolder, 'OphysTs'), OphysTs)
            np.save(os.path.join(currExportedDataFolder, 'WindowPrePost'), WindowPrePost)


        GrandAvg_Image = np.concatenate((GrandAvg_Image, np.nanmean(Split[1:currCutoffIndexes,:,:], axis = 0).T))
        GrandAvg_Omission = np.concatenate((GrandAvg_Omission, np.nanmean(Split[currCutoffIndexes+1::,:,:], axis = 0).T))




        #if ShuffleTimes: # double shuffling version
        #    for i1 in range(Split.shape[0]):
        #        for i2 in range(Split.shape[2]):
        #            np.random.shuffle(Split[i1, :,i2])

        #if ShuffleTrials: # double shuffling version
        #    for i1 in range(Split.shape[1]):
        #        for i2 in range(Split.shape[2]):
        #            np.random.shuffle(Split[:,i1,i2])

        #if ShuffleNeurons: # double shuffling version
        #    for i1 in range(Split.shape[0]):
        #        for i2 in range(Split.shape[1]):
        #            np.random.shuffle(Split[i1,i2,:])

        if ShuffleNeurons:
            for i1 in range(Split.shape[0]):
                tmpIndexes = np.array(list(range(Split.shape[2])))
                np.random.shuffle(tmpIndexes)
                Split[i1,:,:] = Split[i1,:,tmpIndexes].T

        if ShuffleTrials:
            for i1 in range(Split.shape[2]):
                tmpIndexes1 = np.array(list(range(currCutoffIndexes)))
                tmpIndexes2 = np.array(list(range(Split.shape[0]-currCutoffIndexes)))+currCutoffIndexes
                np.random.shuffle(tmpIndexes1)
                np.random.shuffle(tmpIndexes2)
                Split[0:currCutoffIndexes, :, i1] = Split[tmpIndexes1, :, i1]
                Split[currCutoffIndexes::, :, i1] = Split[tmpIndexes2, :, i1]



        ensemble = tt.Ensemble(nonneg=False, fit_method="cp_als") #nonneg=True
        ensemble.fit(Split.transpose(2, 1, 0), ranks=range(nComponents, nComponents+1, 2), replicates=1)
        #ensemble.fit(Split.transpose(2, 1, 0), ranks=range(nComponents, nComponents+1, 2), replicates=100) #temporary

        currObjectives = np.concatenate((currObjectives,ensemble.objectives(nComponents)))
        currSimilarities = np.concatenate((currSimilarities,ensemble.similarities(nComponents)))

        #plot_butterfly(Split, SplitTs)
        #plt_TCA_reconstruction(ensemble, PathToSave = 'none')
        #plot_TCA_weights(ensemble, 9, PathToSave = 'none', Repeat = 0)
        [WeightPerChannel, WeightPerTime, WeightPerTrials] = get_weights(ensemble, nComponents)
        WeightPerChannel = WeightPerChannel [:,1,np.newaxis] # temporary
        WeightPerTime = WeightPerTime [:,1,np.newaxis] # temporary
        WeightPerTrials = WeightPerTrials [:,1,np.newaxis] # temporary

        #plot_trial_weights_by_component(WeightPerTrials, StimStartIndexImages.shape[0])
        [Indexes, MediansDiff, MediansAbsDiff, TestPValues] = get_image_and_omission_components(WeightPerTrials, currCutoffIndexes)
        CountComponentsImage[currSessIndex] = sum(Indexes == 1)
        CountComponentsOmission[currSessIndex] = sum(Indexes == 2)

        WeightPerChannel_ComponentsUnspecific[currSessIndex] = np.nanmean(WeightPerChannel[:, Indexes == 0])
        WeightPerChannel_ComponentsImage[currSessIndex] = np.nanmean(WeightPerChannel[:, Indexes == 1])
        WeightPerChannel_ComponentsOmission[currSessIndex] = np.nanmean(WeightPerChannel[:,Indexes == 2])

        tmpPolarityTrials_ComponentsImage = np.nanmean(WeightPerTrials[:, Indexes == 1].T, axis=1)
        tmpPolarityTrials_ComponentsOmission = np.nanmean(WeightPerTrials[:, Indexes == 2].T, axis=1)
        tmpPolarityChannel_ComponentsImage = np.nanmean(WeightPerChannel[:, Indexes == 1].T, axis=1)
        tmpPolarityChannel_ComponentsOmission = np.nanmean(WeightPerChannel[:, Indexes == 2].T, axis=1)


        tmpCompSignCorrected_Image = (WeightPerTime[:, Indexes == 1] * np.sign(tmpPolarityChannel_ComponentsImage) * np.sign(tmpPolarityTrials_ComponentsImage)).T #
        tmpCompSignCorrected_Omission = (WeightPerTime[:, Indexes == 2] * np.sign(tmpPolarityChannel_ComponentsOmission) * np.sign(tmpPolarityTrials_ComponentsOmission)).T #* np.sign(tmpPolarityTrials_ComponentsOmission)
        #Ts_ComponentsImage = np.concatenate((Ts_ComponentsImage, WeightPerTime[:, Indexes == 1].T), 0)
        #Ts_ComponentsOmission = np.concatenate((Ts_ComponentsOmission, WeightPerTime[:, Indexes == 2].T), 0)
        Ts_ComponentsImage = np.concatenate((Ts_ComponentsImage, tmpCompSignCorrected_Image), axis = 0)
        Ts_ComponentsOmission = np.concatenate((Ts_ComponentsOmission, tmpCompSignCorrected_Omission), 0)
        Ts_ComponentsOmission_ID = np.concatenate((Ts_ComponentsOmission_ID, np.repeat(currSessIndex,tmpCompSignCorrected_Omission.shape[0])), 0)




        #PolarityTrials_ComponentsImage = np.concatenate((PolarityTrials_ComponentsImage,np.nanmean(WeightPerTrials[:,Indexes==1].T, axis = 0)),0)
        #PolarityTrials_ComponentsOmission = np.concatenate((PolarityTrials_ComponentsOmission,np.nanmean(WeightPerTrials[:,Indexes==2].T, axis = 0)),0)
        #PolarityChannel_ComponentsImage = np.concatenate((PolarityChannel_ComponentsImage,np.nanmean(WeightPerChannel[:,Indexes==1].T, axis = 0)),0)
        #PolarityChannel_ComponentsOmission = np.concatenate((PolarityChannel_ComponentsOmission,np.nanmean(WeightPerChannel[:,Indexes==2].T, axis = 0)),0)

        for counterComponent in range(np.where(Indexes!=0)[0].shape[0]):
                currComponent = np.where(Indexes!=0)[0][counterComponent]
                #plot_area_weights_single_component(WeightPerChannel[:,currComponent], ImagingLocDepth)


                if np.unique(ImagingLocDepth).shape[0]!=1:
                    GroupStat, PostHocStat, Medians, Stds, AreaLabels = stat_area_weights_single_component(WeightPerChannel[:,currComponent], ImagingLocDepth, test_type='parametric')
                    PostHocStat = PostHocStat.values
                else:
                    PostHocStat = np.zeros((1,1))
                    Medians = np.zeros((1,1))
                    Stds = np.zeros((1,1))
                    PostHocStat[:,:] = np.nan
                    Medians[:,:] = np.nan
                    Stds[:,:] = np.nan
                    AreaLabels = np.unique(ImagingLocDepth)
                MatchedLabels, MatchedIndexes = match_labels(AreaLabels, nLabels = nLabels)


                tmpMedianSingleArea = np.zeros(nLabels)
                tmpMedianSingleArea[:] = np.nan
                tmpStdSingleArea = np.zeros(nLabels)
                tmpStdSingleArea[:] = np.nan

                for counterArea in range(MatchedIndexes.shape[0]):
                    currIndexArea1 = MatchedIndexes[counterArea].astype(int)
                    tmpMedianSingleArea[currIndexArea1] = Medians[counterArea]
                    tmpStdSingleArea[currIndexArea1] = Stds[counterArea]


                if Indexes[currComponent] == 1:
                    AllMediansSingleAreas_image = np.concatenate((AllMediansSingleAreas_image, tmpMedianSingleArea[:, np.newaxis]), 1)
                    AllStdsSingleAreas_image = np.concatenate((AllStdsSingleAreas_image, tmpStdSingleArea[:, np.newaxis]), 1)
                    AllCompWeight_image =  np.concatenate((AllCompWeight_image, np.nanmean(np.abs(WeightPerChannel[:,currComponent]))[np.newaxis]), 0)


                    tmpWeightsPerTrials_CurrComp = WeightPerTrials[:,currComponent]
                    tmpWeightsPerTrials_CurrComp = tmpWeightsPerTrials_CurrComp[1: currCutoffIndexes]
                    AllStdTrials_image = np.concatenate((AllStdTrials_image, np.nanstd(tmpWeightsPerTrials_CurrComp)[np.newaxis]), 0)
                elif Indexes[currComponent] == 2:
                    AllMediansSingleAreas_omissions = np.concatenate((AllMediansSingleAreas_omissions, tmpMedianSingleArea[:, np.newaxis]), 1)
                    AllStdsSingleAreas_omissions = np.concatenate((AllStdsSingleAreas_omissions, tmpStdSingleArea[:, np.newaxis]), 1)
                    AllCompWeight_omission =  np.concatenate((AllCompWeight_omission, np.nanmean(np.abs(WeightPerChannel[:,currComponent]))[np.newaxis]), 0)

                    tmpWeightsPerTrials_CurrComp = WeightPerTrials[:,currComponent]
                    tmpWeightsPerTrials_CurrComp = tmpWeightsPerTrials_CurrComp[currCutoffIndexes::]
                    AllStdTrials_omission = np.concatenate((AllStdTrials_omission, np.nanstd(tmpWeightsPerTrials_CurrComp)[np.newaxis]), 0)

                    #ChannelsHighWeight = np.where(abs(WeightPerChannel[:, currComponent])>np.percentile(abs(WeightPerChannel[:, currComponent]),75))[0];
                    #ChannelsLowWeight = np.where(abs(WeightPerChannel[:, currComponent])<np.percentile(abs(WeightPerChannel[:, currComponent]),25))[0];
                    #tmpImage = Split[1:currCutoffIndexes, :, :];
                    #tmpImage_HW = np.nanmean(tmpImage[:,:,ChannelsHighWeight],0);
                    #tmpImage_LW = np.nanmean(tmpImage[:,:,ChannelsLowWeight],0);
                    #tmpOmission = Split[currCutoffIndexes+1::, :, :];
                    #tmpOmission_HW = np.nanmean(tmpOmission[:,:,ChannelsHighWeight],0);
                    #tmpOmission_LW = np.nanmean(tmpOmission[:,:,ChannelsLowWeight],0);
                    #plt.plot(tmpOmission_HW)
                    #plt.ylim([0,0.12])
                    #plt.show()
                    #plt.plot(tmpOmission_LW)
                    #plt.ylim([0,0.12])
                    #plt.show()

                else:
                    pass

                AllMediansSingleAreas = np.concatenate((AllMediansSingleAreas, tmpMedianSingleArea[:, np.newaxis]), 1)
                AllStdsSingleAreas = np.concatenate((AllStdsSingleAreas, tmpStdSingleArea[:, np.newaxis]), 1)





                MatchedPostHocStat = np.zeros((nLabels, nLabels))
                MatchedMediansDiff = np.zeros((nLabels, nLabels))
                MatchedMediansAbsDiff = np.zeros((nLabels, nLabels))
                MatchedPostHocStat[:, :] = np.nan
                MatchedMediansDiff[:, :] = np.nan
                MatchedMediansAbsDiff[:, :] = np.nan



                if GroupStat[1]<0.05:
                    for counterArea1 in range(MatchedIndexes.shape[0]):
                        currIndexArea1 = MatchedIndexes[counterArea1].astype(int)
                        Medians1 = Medians[counterArea1]
                        for counterArea2 in range(MatchedIndexes.shape[0]):
                            currIndexArea2 = MatchedIndexes[counterArea2].astype(int)
                            Medians2 = Medians[counterArea2]
                            MatchedPostHocStat[currIndexArea1, currIndexArea2] = PostHocStat[counterArea1, counterArea2]
                            MatchedMediansDiff[currIndexArea1, currIndexArea2] = Medians1 - Medians2
                            MatchedMediansAbsDiff[currIndexArea1, currIndexArea2] = abs(Medians1) - abs(Medians2)

                    if Indexes[currComponent] == 1:
                        AllStatsAreas_image = np.concatenate((AllStatsAreas_image, MatchedPostHocStat[:, :, np.newaxis]), 2)
                        AllMediansAreas_image = np.concatenate((AllMediansAreas_image, MatchedMediansDiff[:, :, np.newaxis]), 2)
                        AllMediansAbsAreas_image = np.concatenate((AllMediansAbsAreas_image, MatchedMediansAbsDiff[:, :, np.newaxis]), 2)

                    elif Indexes[currComponent] == 2:
                        AllStatsAreas_omissions = np.concatenate((AllStatsAreas_omissions, MatchedPostHocStat[:, :, np.newaxis]), 2)
                        AllMediansAreas_omissions = np.concatenate((AllMediansAreas_omissions, MatchedMediansDiff[:, :, np.newaxis]), 2)
                        AllMediansAbsAreas_omissions = np.concatenate((AllMediansAbsAreas_omissions, MatchedMediansAbsDiff[:, :, np.newaxis]), 2)

                    else:
                        pass


                AllStatsAreas = np.concatenate((AllStatsAreas, MatchedPostHocStat[:, :, np.newaxis]), 2)
                AllMediansAreas = np.concatenate((AllMediansAreas, MatchedMediansDiff[:, :, np.newaxis]), 2)
                AllMediansAbsAreas = np.concatenate((AllMediansAbsAreas, MatchedMediansAbsDiff[:, :, np.newaxis]), 2)
            print(currSessIndex)


    scipy.stats.ranksums(CountComponentsImage,CountComponentsOmission)

    scipy.stats.ranksums(abs(WeightPerChannel_ComponentsImage),abs(WeightPerChannel_ComponentsOmission))




    if GeneratePlots:
        fig, ax = plt.subplots(1,figsize=(6,6))
        plt.plot(np.nanmean(GrandAvg_Image,axis = 0))
        art = matplotlib.patches.Rectangle([WindowPrePost[0]-7, min(np.nanmean(GrandAvg_Image,axis = 0))], 2.5, max(np.nanmean(GrandAvg_Image,axis = 0))-min(np.nanmean(GrandAvg_Image,axis = 0)), alpha = 0.4, color = 'yellow')
        ax.add_patch(art)
        art = matplotlib.patches.Rectangle([WindowPrePost[0], min(np.nanmean(GrandAvg_Image,axis = 0))], 2.5, max(np.nanmean(GrandAvg_Image,axis = 0))-min(np.nanmean(GrandAvg_Image,axis = 0)), alpha = 0.4, color = 'yellow')
        ax.add_patch(art)
        art = matplotlib.patches.Rectangle([WindowPrePost[0]+7, min(np.nanmean(GrandAvg_Image,axis = 0))], 2.5, max(np.nanmean(GrandAvg_Image,axis = 0))-min(np.nanmean(GrandAvg_Image,axis = 0)), alpha = 0.4, color = 'yellow')
        ax.add_patch(art)
        plt.show()

        fig, ax = plt.subplots(1,figsize=(6,6))
        plt.plot(np.nanmean(GrandAvg_Omission,axis = 0))
        art = matplotlib.patches.Rectangle([WindowPrePost[0]-7, min(np.nanmean(GrandAvg_Omission,axis = 0))], 2.5, max(np.nanmean(GrandAvg_Omission,axis = 0))-min(np.nanmean(GrandAvg_Omission,axis = 0)), alpha = 0.4, color = 'yellow')
        ax.add_patch(art)
        art = matplotlib.patches.Rectangle([WindowPrePost[0], min(np.nanmean(GrandAvg_Omission,axis = 0))], 2.5, max(np.nanmean(GrandAvg_Omission,axis = 0))-min(np.nanmean(GrandAvg_Omission,axis = 0)), alpha = 0.4, color = 'blue')
        ax.add_patch(art)
        art = matplotlib.patches.Rectangle([WindowPrePost[0]+7, min(np.nanmean(GrandAvg_Omission,axis = 0))], 2.5, max(np.nanmean(GrandAvg_Omission,axis = 0))-min(np.nanmean(GrandAvg_Omission,axis = 0)), alpha = 0.4, color = 'yellow')
        ax.add_patch(art)
        plt.show()

        fig, ax = plt.subplots(1,figsize=(6,6))
        plt.plot((Ts_ComponentsImage.T))
        art = matplotlib.patches.Rectangle([WindowPrePost[0]-7, np.min(abs(Ts_ComponentsImage.T))], 2.5, np.max(abs(Ts_ComponentsImage.T))-np.min(abs(Ts_ComponentsImage.T)), alpha = 0.4, color = 'yellow')
        ax.add_patch(art)
        art = matplotlib.patches.Rectangle([WindowPrePost[0], np.min(abs(Ts_ComponentsImage.T))], 2.5, np.max(abs(Ts_ComponentsImage.T))-np.min(abs(Ts_ComponentsImage.T)), alpha = 0.4, color = 'yellow')
        ax.add_patch(art)
        art = matplotlib.patches.Rectangle([WindowPrePost[0]+7, np.min(abs(Ts_ComponentsImage.T))], 2.5, np.max(abs(Ts_ComponentsImage.T))-np.min(abs(Ts_ComponentsImage.T)), alpha = 0.4, color = 'yellow')
        ax.add_patch(art)
        plt.show()

        fig, ax = plt.subplots(1,figsize=(6,6))
        plt.plot((Ts_ComponentsOmission.T))
        art = matplotlib.patches.Rectangle([WindowPrePost[0]-7, np.min(abs(Ts_ComponentsOmission.T))], 2.5, np.max(abs(Ts_ComponentsOmission.T))-np.min(abs(Ts_ComponentsOmission.T)), alpha = 0.4, color = 'yellow')
        ax.add_patch(art)
        art = matplotlib.patches.Rectangle([WindowPrePost[0], np.min(abs(Ts_ComponentsOmission.T))], 2.5, np.max(abs(Ts_ComponentsOmission.T))-np.min(abs(Ts_ComponentsOmission.T)), alpha = 0.4, color = 'blue')
        ax.add_patch(art)
        art = matplotlib.patches.Rectangle([WindowPrePost[0]+7, np.min(abs(Ts_ComponentsOmission.T))], 2.5, np.max(abs(Ts_ComponentsOmission.T))-np.min(abs(Ts_ComponentsOmission.T)), alpha = 0.4, color = 'yellow')
        ax.add_patch(art)
        plt.show()

        plt.imshow(AllMediansSingleAreas_omissions)
        plt.show()

        plt.figure()
        sns.boxplot(data=abs(AllMediansSingleAreas_image).T, showfliers=False,  whis=0)
        plt.xticks([0,1,2,3],AreasNames)
        plt.show()

        plt.figure()
        sns.boxplot(data=abs(AllMediansSingleAreas_omissions).T, showfliers=False,  whis=0)
        plt.xticks([0,1,2,3],AreasNames)
        plt.show()



    tmp1 = AllStatsAreas_omissions<0.05
    tmp2 = AllStatsAreas_omissions>=0.05
    #tmp1 = AllStatsAreas_image<0.05
    #tmp2 = AllStatsAreas_image>=0.05

    tmpToPlot = np.nanmean(tmp1/(tmp1+tmp2),axis = 2);
    tmpToPlot[np.nanmean(tmp1+tmp2,axis = 2)<0.5] = np.nan
    #plt.imshow(tmpToPlot)
    #plt.colorbar()
    #plt.clim(0,0.7)
    #plt.show()





    for counterSessGroup in range(2):



        if counterSessGroup==0:
            SelectedMatrix = AllMediansSingleAreas_image;
        elif counterSessGroup==1:
            SelectedMatrix = AllMediansSingleAreas_omissions;

        #SelectedIndexes = np.where(np.nanmean(Ts_ComponentsOmission, axis  = 1)>0)[0]
        #SelectedMatrix = SelectedMatrix[:,SelectedIndexes]

        #plt.imshow(AllMediansSingleAreas_omissions)
        #plt.show()

        Correlations_p = np.zeros((SelectedMatrix.shape[0],SelectedMatrix.shape[0]))
        Correlations_r = np.zeros((SelectedMatrix.shape[0],SelectedMatrix.shape[0]))
        RatioArea = np.zeros((SelectedMatrix.shape[0],SelectedMatrix.shape[0]))
        TotArea = np.zeros((SelectedMatrix.shape[0],SelectedMatrix.shape[0]))
        SignRankTests = np.zeros((SelectedMatrix.shape[0],SelectedMatrix.shape[0]))
        Correlations_p [:,:] = np.nan;
        Correlations_r [:,:] = np.nan;
        RatioArea [:,:] = np.nan;
        TotArea [:,:] = np.nan;
        SignRankTests [:,:] = np.nan;


        for counterArea1 in range(SelectedMatrix.shape[0]):
            for counterArea2 in range(SelectedMatrix.shape[0]):

                tmp1 = SelectedMatrix[counterArea1, :]
                tmp2 = SelectedMatrix[counterArea2, :]
                NonNanIndexes = np.intersect1d(np.where(~np.isnan(tmp1))[0], np.where(~np.isnan(tmp2))[0])


                if np.sum(np.logical_and(np.logical_not(np.isnan(SelectedMatrix[counterArea1, :])),np.logical_not(np.isnan(SelectedMatrix[counterArea2, :]))))>2 :
                    #tmpCorr = scipy.stats.spearmanr(AllMediansSingleAreas_omissions_zscored[counterArea1, :], AllMediansSingleAreas_omissions_zscored[counterArea2,:], nan_policy='omit')
                    tmpCorr = scipy.stats.spearmanr(SelectedMatrix[counterArea1, :], SelectedMatrix[counterArea2, :], nan_policy='omit')
                    #tmpCorr = scipy.stats.pearsonr(tmp1[NonNanIndexes], tmp2[NonNanIndexes])
                else :
                    tmpCorr = (np.nan, np.nan)

                Correlations_r[counterArea1,counterArea2] = tmpCorr[0]
                Correlations_p[counterArea1,counterArea2] = tmpCorr[1]


                #if counterArea1!=counterArea2:
                #    tmpTest = scipy.stats.wilcoxon(tmp1[NonNanIndexes], tmp2[NonNanIndexes])
                #    tmpTest = scipy.stats.ranksums(tmp1[np.where(~np.isnan(tmp1))[0]], tmp2[np.where(~np.isnan(tmp2))[0]])
                #
                #    SignRankTests[counterArea1, counterArea2] = tmpTest[1]
                #else:
                #    pass


                Val1 = np.nansum(SelectedMatrix[counterArea1, :] > SelectedMatrix[counterArea2, :])
                Val2 = np.nansum(SelectedMatrix[counterArea1, :] < SelectedMatrix[counterArea2, :])
                RatioArea[counterArea1,counterArea2] = Val1
                TotArea[counterArea1, counterArea2] = (Val1+Val2)


        if GeneratePlots:
            CorrMatrix = (Correlations_p < 0.05) * Correlations_r;
            # CorrMatrix = (Correlations_p<0.001)*Correlations_r*(Correlations_r>0.4)*(TotArea>40);

            for counter in range(CorrMatrix.shape[0]):
                CorrMatrix[counter, counter] = np.nan;

            plt.imshow(CorrMatrix,cmap='bwr')
            plt.clim(-1,1)
            plt.show()

        if counterSessGroup==0:
            AllCorr_p_images[:,:,counterRun] = Correlations_p
            AllCorr_r_images[:,:,counterRun] = Correlations_r
        elif counterSessGroup==1:
            AllCorr_p_omissions[:,:,counterRun] = Correlations_p
            AllCorr_r_omissions[:,:,counterRun] = Correlations_r


    AllWeights_images[:,counterRun] = np.nanmean(abs(AllMediansSingleAreas_image), axis = 1)
    AllWeights_omissions[:,counterRun] = np.nanmean(abs(AllMediansSingleAreas_omissions), axis = 1)

    AllNComp_images[counterRun] = AllMediansSingleAreas_image.shape[1]
    AllNComp_omissions[counterRun] = AllMediansSingleAreas_omissions.shape[1]

    All_StdSingleAreas_images[:,counterRun] = np.nanmean(AllStdsSingleAreas_image, axis = 1)
    All_StdSingleAreas_omissions[:,counterRun] = np.nanmean(AllStdsSingleAreas_omissions, axis = 1)
    All_StdTrials_images[counterRun] = np.nanmean(AllStdTrials_image);
    All_StdTrials_omissions[counterRun] = np.nanmean(AllStdTrials_omission);

    All_Similarities[counterRun] = np.nanmean(currSimilarities)
    All_Objectives[counterRun] = np.nanmean(currObjectives)


All_Ts_ComponentsOmission = np.concatenate((All_Ts_ComponentsOmission,Ts_ComponentsOmission))



#AllCorr_p_images_NonSh = AllCorr_p_images
#AllCorr_r_images_NonSh = AllCorr_r_images

#AllCorr_p_omissions_NonSh = AllCorr_p_omissions
#AllCorr_r_omissions_NonSh = AllCorr_r_omissions

#AllWeights_images_NonSh = AllWeights_images
#AllWeights_omissions_NonSh = AllWeights_omissions

#AllNComp_images_NonSh = AllNComp_images
#AllNComp_omissions_NonSh = AllNComp_omissions


#All_StdSingleAreas_images_NonSh = All_StdSingleAreas_images
#All_StdSingleAreas_omissions_NonSh = All_StdSingleAreas_omissions
#All_StdTrials_images_NonSh = All_StdTrials_images
#All_StdTrials_omissions_NonSh = All_StdTrials_omissions





#AllCorr_p_images_ShNeurons = AllCorr_p_images
#AllCorr_r_images_ShNeurons = AllCorr_r_images

#AllCorr_p_omissions_ShNeurons = AllCorr_p_omissions
#AllCorr_r_omissions_ShNeurons = AllCorr_r_omissions

#AllWeights_images_ShNeurons = AllWeights_images
#AllWeights_omissions_ShNeurons = AllWeights_omissions

#AllNComp_images_ShNeurons = AllNComp_images
#AllNComp_omissions_ShNeurons = AllNComp_omissions


fig, ax = plt.subplots()
#plt.boxplot((All_StdTrials_omissions_NonSh,All_StdTrials_omissions))
plt.boxplot((All_StdTrials_images_NonSh,All_StdTrials_images))
plt.xticks((1,2),('NonShuffled','Shuffled'))
plt.ylabel('Variability across trials')
scipy.stats.ranksums(All_StdTrials_omissions_NonSh,All_StdTrials_omissions)
plt.show()
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\StdAcrossTrials.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\StdAcrossTrials_slc.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\StdAcrossTrials_slc_images.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\StdAcrossTrials_sst.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\StdAcrossTrials_sst_images.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\StdAcrossTrials_novel.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\StdAcrossTrials_images_novel.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\StdAcrossTrials_slc_novel.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\StdAcrossTrials_slc_images_novel.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\StdAcrossTrials_sst_novel.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\StdAcrossTrials_sst_images_novel.svg', format='svg', dpi=1200)



nArea = 3
plt.hist(AllWeights_images_NonSh[nArea,:], alpha = 0.5, color = 'red')
plt.hist(AllWeights_images[nArea,:], alpha = 0.5, color = 'gray')
#plt.xlim(0.03,0.20)
plt.show()


plt.hist(AllWeights_omissions_NonSh[nArea,:], alpha = 0.5, color = 'red')
plt.hist(AllWeights_omissions[nArea,:], alpha = 0.5, color = 'gray')
#plt.xlim(0.03,0.20)
plt.show()



nArea1 = 2
nArea2 = 3
plt.hist(AllCorr_r_images_NonSh[nArea1,nArea2,:], alpha = 0.5, color = 'red')
plt.hist(AllCorr_r_images[nArea1,nArea2,:], alpha = 0.5, color = 'grey')
scipy.stats.ranksums(AllCorr_r_images_NonSh[nArea1,nArea2,:],AllCorr_r_images[nArea1,nArea2,:])
plt.show()


plt.hist(AllCorr_r_omissions_NonSh[nArea1,nArea2,:], alpha = 0.5, color = 'red')
plt.hist(AllCorr_r_omissions[nArea1,nArea2,:], alpha = 0.5, color = 'grey')
scipy.stats.ranksums(AllCorr_r_omissions_NonSh[nArea1,nArea2,:],AllCorr_r_omissions[nArea1,nArea2,:])
plt.show()


#MainFolderToSave = MainFolderToSaveGeneral + r'\\' + CellType
#np.save(MainFolderToSave + r'\Weights_images_NonSh', AllWeights_images_NonSh)
#np.save(MainFolderToSave + r'\Weights_images_ShNeuron', AllWeights_images_ShNeurons)
#np.save(MainFolderToSave + r'\Weights_images', AllWeights_images)
#np.save(MainFolderToSave + r'\Weights_omissions_NonSh', AllWeights_omissions_NonSh)
#np.save(MainFolderToSave + r'\Weights_omissions_ShNeuron', AllWeights_omissions_ShNeurons)
#np.save(MainFolderToSave + r'\Weights_omissions', AllWeights_omissions)

#np.save(MainFolderToSave + r'\AllCorr_p_images_NonSh', AllCorr_p_images_NonSh)
#np.save(MainFolderToSave + r'\AllCorr_p_images_ShNeuron', AllCorr_p_images_ShNeurons)
#np.save(MainFolderToSave + r'\AllCorr_p_images', AllCorr_p_images)
#np.save(MainFolderToSave + r'\AllCorr_p_omissions_NonSh', AllCorr_p_omissions_NonSh)
#np.save(MainFolderToSave + r'\AllCorr_p_omissions_ShNeuron', AllCorr_p_omissions_ShNeurons)
#np.save(MainFolderToSave + r'\AllCorr_p_omissions', AllCorr_p_omissions)

#np.save(MainFolderToSave + r'\AllCorr_r_images_NonSh', AllCorr_r_images_NonSh)
#np.save(MainFolderToSave + r'\AllCorr_r_images_ShNeuron', AllCorr_r_images_ShNeurons)
#np.save(MainFolderToSave + r'\AllCorr_r_images', AllCorr_r_images)
#np.save(MainFolderToSave + r'\AllCorr_r_omissions_NonSh', AllCorr_r_omissions_NonSh)
#np.save(MainFolderToSave + r'\AllCorr_r_omissions_ShNeuron', AllCorr_r_omissions_ShNeurons)
#np.save(MainFolderToSave + r'\AllCorr_r_omissions', AllCorr_r_omissions)

#np.save(MainFolderToSave + r'\AllNComp_images_NonSh', AllNComp_images_NonSh)
#np.save(MainFolderToSave + r'\AllNComp_images_ShNeuron', AllNComp_images_ShNeurons)
#np.save(MainFolderToSave + r'\AllNComp_images', AllNComp_images)
#np.save(MainFolderToSave + r'\AllNComp_omissions_NonSh', AllNComp_omissions_NonSh)
#np.save(MainFolderToSave + r'\AllNComp_omissions_ShNeuron', AllNComp_omissions_ShNeurons)
#np.save(MainFolderToSave + r'\AllNComp_omissions', AllNComp_omissions)




NonShufflingToLoad = '_NonSh' #'_NonSh'
ShufflingToLoad = '' #'_ShNeuron' ''
CellType = 'Vip'

########## Plots Weights per area

MainFolderToLoad = MainFolderToSaveGeneral + r'\\' + CellType
AllWeights_images_NonSh = np.load(MainFolderToLoad + r'\Weights_images' + NonShufflingToLoad +'.npy')
AllWeights_images = np.load(MainFolderToLoad + r'\Weights_images' + ShufflingToLoad + '.npy')
AllWeights_omissions_NonSh = np.load(MainFolderToLoad + r'\Weights_omissions' + NonShufflingToLoad +'.npy')
AllWeights_omissions = np.load(MainFolderToLoad + r'\Weights_omissions' + ShufflingToLoad + '.npy')

AllCorr_p_images_NonSh = np.load(MainFolderToLoad + r'\AllCorr_p_images' + NonShufflingToLoad +'.npy')
AllCorr_p_images = np.load(MainFolderToLoad + r'\AllCorr_p_images' + ShufflingToLoad + '.npy')
AllCorr_p_omissions_NonSh = np.load(MainFolderToLoad + r'\AllCorr_p_omissions' + NonShufflingToLoad +'.npy')
AllCorr_p_omissions = np.load(MainFolderToLoad + r'\AllCorr_p_omissions' + ShufflingToLoad + '.npy')

AllCorr_r_images_NonSh = np.load(MainFolderToLoad + r'\AllCorr_r_images' + NonShufflingToLoad +'.npy')
AllCorr_r_images = np.load(MainFolderToLoad + r'\AllCorr_r_images' + ShufflingToLoad + '.npy')
AllCorr_r_omissions_NonSh = np.load(MainFolderToLoad + r'\AllCorr_r_omissions' + NonShufflingToLoad +'.npy')
AllCorr_r_omissions = np.load(MainFolderToLoad + r'\AllCorr_r_omissions' + ShufflingToLoad + '.npy')

AllNComp_images_NonSh = np.load(MainFolderToLoad + r'\AllNComp_images' + NonShufflingToLoad +'.npy')
AllNComp_images = np.load(MainFolderToLoad + r'\AllNComp_images' + ShufflingToLoad + '.npy')
AllNComp_omissions_NonSh = np.load(MainFolderToLoad + r'\AllNComp_omissions' + NonShufflingToLoad +'.npy')
AllNComp_omissions = np.load(MainFolderToLoad + r'\AllNComp_omissions' + ShufflingToLoad + '.npy')


#MaxData = 20
#AllWeights_images_NonSh = AllWeights_images_NonSh[:,0:MaxData]
#AllWeights_images = AllWeights_images[:,0:MaxData]
#AllWeights_omissions_NonSh = AllWeights_omissions_NonSh[:,0:MaxData]
#AllWeights_omissions = AllWeights_omissions[:,0:MaxData]
#AllNComp_images_NonSh = AllNComp_images_NonSh[0:MaxData]
#AllNComp_images = AllNComp_images[0:MaxData]
#AllNComp_omissions_NonSh = AllNComp_omissions_NonSh[0:MaxData]
#AllNComp_omissions = AllNComp_omissions[0:MaxData]


AllNComp = np.concatenate((AllNComp_images_NonSh,AllNComp_images,AllNComp_omissions_NonSh,AllNComp_omissions))
CompType = np.concatenate((np.zeros(AllNComp_images_NonSh.shape),np.zeros(AllNComp_images.shape),np.ones(AllNComp_omissions_NonSh.shape),np.ones(AllNComp_omissions.shape)))
Shuffling =  np.concatenate((np.zeros(AllNComp_images_NonSh.shape),np.ones(AllNComp_images.shape),np.zeros(AllNComp_omissions_NonSh.shape),np.ones(AllNComp_omissions.shape)))
df = pd.DataFrame({'Weights' : AllNComp,
                   'Type' : CompType,
                   'Shuffling' : Shuffling })

fig, ax = plt.subplots()
sns.boxplot(data=df, x='Type', y='Weights', hue='Shuffling')
plt.title('Number of components')
plt.xticks(range(2), ['images','omissions'])
plt.show()
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\NComponentsImagesOmissions.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\NComponentsImagesOmissions_slc.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\NComponentsImagesOmissions_sst.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\NComponentsImagesOmissions_novel.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\NComponentsImagesOmissions_slc_novel.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\NComponentsImagesOmissions_sst_novel.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\NComponentsImagesOmissions_10.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\NComponentsImagesOmissions_sst_10.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\NComponentsImagesOmissions_slc_10.svg', format='svg', dpi=1200)

dataframe = pd.DataFrame({'Shuffling': np.concatenate((np.repeat(['NonSh'], AllNComp_omissions_NonSh.shape[0]),np.repeat(['Sh'], AllNComp_omissions.shape[0]),np.repeat(['NonSh'], AllNComp_omissions_NonSh.shape[0]),np.repeat(['Sh'], AllNComp_omissions.shape[0]))),
                          'Category': np.concatenate((np.repeat(['Image'], AllNComp_omissions_NonSh.shape[0]),np.repeat(['Image'], AllNComp_omissions.shape[0]),np.repeat(['Omission'], AllNComp_omissions_NonSh.shape[0]),np.repeat(['Omission'], AllNComp_omissions.shape[0]))),
                          'nComp': np.concatenate((AllNComp_images_NonSh,AllNComp_images,AllNComp_omissions_NonSh,AllNComp_omissions))})
model = ols(
    'nComp ~ C(Shuffling) + C(Category) +\
    C(Shuffling):C(Category)', data=dataframe).fit()
sm.stats.anova_lm(model, typ=2)

#tukey = pairwise_tukeyhsd(endog=df['score'],
#                          groups=df['group'],
#                          alpha=0.05)



# N discarded components SLC: np.nanmean((21*5)-(AllNComp_images_NonSh+AllNComp_omissions_NonSh))
# N discarded components SST: np.nanmean((15*5)-(AllNComp_images_NonSh+AllNComp_omissions_NonSh))
# N discarded components VIP: np.nanmean((22*5)-(AllNComp_images_NonSh+AllNComp_omissions_NonSh))

## Boxplot Neuron weights images
DataNonSh = AllWeights_images_NonSh.copy()
DataSh = AllWeights_images.copy()

AreasNonSh = DataNonSh.copy()
AreasSh = DataSh.copy()
for counter in range(AreasSh.shape[0]):
    AreasNonSh[counter,:] = counter
    AreasSh[counter,:] = counter

ShufflingNonSh = DataNonSh.copy()
ShufflingSh = DataSh.copy()
ShufflingNonSh[:,:] = 0;
ShufflingSh[:,:] = 1;

df = pd.DataFrame({'Weights' : np.concatenate((DataNonSh,DataSh),1).flatten(),
                   'Areas' : np.concatenate((AreasNonSh,AreasSh),1).flatten(),
                   'Shuffling' : np.concatenate((ShufflingNonSh,ShufflingSh),1).flatten() })

fig, ax = plt.subplots()
sns.boxplot(data=df, x='Areas', y='Weights', hue='Shuffling')
plt.title('Neuron Weights per areas - ' + CellType + ' - images')
plt.xticks(range(4),AreasNames)
plt.show()

#model = ols( 'Weights ~ C(Areas) + C(Shuffling) + C(Areas):C(Shuffling)', data=df).fit()
#sm.stats.anova_lm(model, typ=2)

tmpDf = df.loc[df['Shuffling']==0]
f_oneway(tmpDf.loc[tmpDf['Areas']==0]['Weights'],tmpDf.loc[tmpDf['Areas']==1]['Weights'],tmpDf.loc[tmpDf['Areas']==2]['Weights'],tmpDf.loc[tmpDf['Areas']==3]['Weights'])
tukey_oneway = pairwise_tukeyhsd(tmpDf["Weights"],  tmpDf["Areas"])
print(tukey_oneway)

scipy.stats.ranksums(df.loc[np.logical_and(df['Areas'] == 0,df['Shuffling'] == 0)]['Weights'],df.loc[np.logical_and(df['Areas'] == 0,df['Shuffling'] == 1)]['Weights'])
scipy.stats.ranksums(df.loc[np.logical_and(df['Areas'] == 1,df['Shuffling'] == 0)]['Weights'],df.loc[np.logical_and(df['Areas'] == 1,df['Shuffling'] == 1)]['Weights'])
scipy.stats.ranksums(df.loc[np.logical_and(df['Areas'] == 2,df['Shuffling'] == 0)]['Weights'],df.loc[np.logical_and(df['Areas'] == 2,df['Shuffling'] == 1)]['Weights'])
scipy.stats.ranksums(df.loc[np.logical_and(df['Areas'] == 3,df['Shuffling'] == 0)]['Weights'],df.loc[np.logical_and(df['Areas'] == 3,df['Shuffling'] == 1)]['Weights'])

#scipy.stats.ttest_ind(df.loc[np.logical_and(df['Areas'] == 0,df['Shuffling'] == 0)]['Weights'],df.loc[np.logical_and(df['Areas'] == 0,df['Shuffling'] == 1)]['Weights'])
#scipy.stats.ttest_ind(df.loc[np.logical_and(df['Areas'] == 1,df['Shuffling'] == 0)]['Weights'],df.loc[np.logical_and(df['Areas'] == 1,df['Shuffling'] == 1)]['Weights'])
#scipy.stats.ttest_ind(df.loc[np.logical_and(df['Areas'] == 2,df['Shuffling'] == 0)]['Weights'],df.loc[np.logical_and(df['Areas'] == 2,df['Shuffling'] == 1)]['Weights'])
#scipy.stats.ttest_ind(df.loc[np.logical_and(df['Areas'] == 3,df['Shuffling'] == 0)]['Weights'],df.loc[np.logical_and(df['Areas'] == 3,df['Shuffling'] == 1)]['Weights'])

np.median(df.loc[np.logical_and(df['Areas'] == 0,df['Shuffling'] == 1)]['Weights'])-np.median(df.loc[np.logical_and(df['Areas'] == 0,df['Shuffling'] == 0)]['Weights'])
np.median(df.loc[np.logical_and(df['Areas'] == 1,df['Shuffling'] == 1)]['Weights'])-np.median(df.loc[np.logical_and(df['Areas'] == 1,df['Shuffling'] == 0)]['Weights'])
np.median(df.loc[np.logical_and(df['Areas'] == 2,df['Shuffling'] == 1)]['Weights'])-np.median(df.loc[np.logical_and(df['Areas'] == 2,df['Shuffling'] == 0)]['Weights'])
np.median(df.loc[np.logical_and(df['Areas'] == 3,df['Shuffling'] == 1)]['Weights'])-np.median(df.loc[np.logical_and(df['Areas'] == 3,df['Shuffling'] == 0)]['Weights'])


#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\WeightsPerArea_images.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\WeightsPerArea_images_slc.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\WeightsPerArea_images_sst.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\WeightsPerArea_images_novel.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\WeightsPerArea_slc_images_novel.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\WeightsPerArea_sst_images_novel.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\WeightsPerArea_sst_images_sst_10.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\WeightsPerArea_sst_images_slc_10.svg', format='svg', dpi=1200)



## Boxplot Neuron weights omissions
DataNonSh = AllWeights_omissions_NonSh.copy()
DataSh = AllWeights_omissions.copy()

AreasNonSh = DataNonSh.copy()
AreasSh = DataSh.copy()
for counter in range(AreasSh.shape[0]):
    AreasNonSh[counter,:] = counter
    AreasSh[counter,:] = counter

ShufflingNonSh = DataNonSh.copy()
ShufflingSh = DataSh.copy()
ShufflingNonSh[:,:] = 0;
ShufflingSh[:,:] = 1;

df = pd.DataFrame({'Weights' : np.concatenate((DataNonSh,DataSh),1).flatten(),
                   'Areas' : np.concatenate((AreasNonSh,AreasSh),1).flatten(),
                   'Shuffling' : np.concatenate((ShufflingNonSh,ShufflingSh),1).flatten() })

tmpDf = df.loc[df['Shuffling']==0]
f_oneway(tmpDf.loc[tmpDf['Areas']==0]['Weights'],tmpDf.loc[tmpDf['Areas']==1]['Weights'],tmpDf.loc[tmpDf['Areas']==2]['Weights'],tmpDf.loc[tmpDf['Areas']==3]['Weights'])
tukey_oneway = pairwise_tukeyhsd(tmpDf["Weights"],  tmpDf["Areas"])
print(tukey_oneway)

fig, ax = plt.subplots()
sns.boxplot(data=df, x='Areas', y='Weights', hue='Shuffling')
plt.title('Neuron Weights per areas - ' + CellType + ' - omissions')
plt.xticks(range(4),AreasNames)
plt.show()



scipy.stats.ranksums(df.loc[np.logical_and(df['Areas'] == 0,df['Shuffling'] == 0)]['Weights'],df.loc[np.logical_and(df['Areas'] == 0,df['Shuffling'] == 1)]['Weights'])
scipy.stats.ranksums(df.loc[np.logical_and(df['Areas'] == 1,df['Shuffling'] == 0)]['Weights'],df.loc[np.logical_and(df['Areas'] == 1,df['Shuffling'] == 1)]['Weights'])
scipy.stats.ranksums(df.loc[np.logical_and(df['Areas'] == 2,df['Shuffling'] == 0)]['Weights'],df.loc[np.logical_and(df['Areas'] == 2,df['Shuffling'] == 1)]['Weights'])
scipy.stats.ranksums(df.loc[np.logical_and(df['Areas'] == 3,df['Shuffling'] == 0)]['Weights'],df.loc[np.logical_and(df['Areas'] == 3,df['Shuffling'] == 1)]['Weights'])

np.median(df.loc[np.logical_and(df['Areas'] == 0,df['Shuffling'] == 1)]['Weights'])-np.median(df.loc[np.logical_and(df['Areas'] == 0,df['Shuffling'] == 0)]['Weights'])
np.median(df.loc[np.logical_and(df['Areas'] == 1,df['Shuffling'] == 1)]['Weights'])-np.median(df.loc[np.logical_and(df['Areas'] == 1,df['Shuffling'] == 0)]['Weights'])
np.median(df.loc[np.logical_and(df['Areas'] == 2,df['Shuffling'] == 1)]['Weights'])-np.median(df.loc[np.logical_and(df['Areas'] == 2,df['Shuffling'] == 0)]['Weights'])
np.median(df.loc[np.logical_and(df['Areas'] == 3,df['Shuffling'] == 1)]['Weights'])-np.median(df.loc[np.logical_and(df['Areas'] == 3,df['Shuffling'] == 0)]['Weights'])

#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\WeightsPerArea.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\WeightsPerArea_slc.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\WeightsPerArea_sst.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\WeightsPerArea_novel.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\WeightsPerArea_slc_novel.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\WeightsPerArea_sst_novel.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\WeightsPerArea_10.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\WeightsPerArea_sst_10.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\WeightsPerArea_slc_10.svg', format='svg', dpi=1200)







## Boxplot Correlations images without repetitions
DataCorrNonSh = AllCorr_r_images_NonSh.copy();
DataCorrSh = AllCorr_r_images.copy();

AreasNonSh = DataCorrNonSh.copy()
AreasSh = DataCorrSh.copy()
for counter0 in range(AreasSh.shape[0]):
    for counter1 in range(AreasSh.shape[1]):
        AreasNonSh[counter0,counter1,:] = counter0+counter1*4
        AreasSh[counter0,counter1,:] = counter0+counter1*4

ShufflingNonSh = DataCorrNonSh.copy()
ShufflingSh = DataCorrSh.copy()
ShufflingNonSh[:,:,:] = 0;
ShufflingSh[:,:,:] = 1;

currConnectionNames = ConnectionNames.copy()

df = pd.DataFrame({'Weights' : np.concatenate((DataCorrNonSh,DataCorrSh),1).flatten(),
                   'Areas' : np.concatenate((AreasNonSh,AreasSh),1).flatten(),
                   'Shuffling' : np.concatenate((ShufflingNonSh,ShufflingSh),1).flatten() })



UniqueCurrConnectionNames = np.unique(currConnectionNames, return_index=True)
tmp = abs(df.Areas.values[:,np.newaxis]-UniqueCurrConnectionNames[1][np.newaxis,:])
SelectedIndexes = np.where(np.min(tmp, axis = 1)==0)[0]
currConnectionNames = np.array(currConnectionNames)[UniqueCurrConnectionNames[1]]
df = df.iloc[SelectedIndexes]

sns.boxplot(data=df, x='Areas', y='Weights', hue='Shuffling')
plt.title('Connectivity between areas - ' + CellType + ' - images')
plt.xticks(range(len(currConnectionNames)),currConnectionNames, rotation = 45, ha="right", rotation_mode="anchor")
plt.subplots_adjust(bottom=0.3)
plt.show()






## Boxplot Correlations omissions with repetitions
DataCorrNonSh = AllCorr_r_omissions_NonSh.copy();
DataCorrSh = AllCorr_r_omissions.copy();

AreasNonSh = DataCorrNonSh.copy()
AreasSh = DataCorrSh.copy()
for counter0 in range(AreasSh.shape[0]):
    for counter1 in range(AreasSh.shape[1]):
        AreasNonSh[counter0,counter1,:] = counter0+counter1*4
        AreasSh[counter0,counter1,:] = counter0+counter1*4

ShufflingNonSh = DataCorrNonSh.copy()
ShufflingSh = DataCorrSh.copy()
ShufflingNonSh[:,:,:] = 0;
ShufflingSh[:,:,:] = 1;

currConnectionNames = ConnectionNames.copy()

df = pd.DataFrame({'Weights' : np.concatenate((DataCorrNonSh,DataCorrSh),1).flatten(),
                   'Areas' : np.concatenate((AreasNonSh,AreasSh),1).flatten(),
                   'Shuffling' : np.concatenate((ShufflingNonSh,ShufflingSh),1).flatten() })

UniqueCurrConnectionNames = np.unique(currConnectionNames, return_index=True)
tmp = abs(df.Areas.values[:,np.newaxis]-UniqueCurrConnectionNames[1][np.newaxis,:])
SelectedIndexes = np.where(np.min(tmp, axis = 1)==0)[0]
currConnectionNames = np.array(currConnectionNames)[UniqueCurrConnectionNames[1]]
df = df.iloc[SelectedIndexes]

fig, ax = plt.subplots()
sns.boxplot(data=df, x='Areas', y='Weights', hue='Shuffling')
plt.title('Connectivity between areas - ' + CellType + ' - omissions')
plt.xticks(range(len(currConnectionNames)),currConnectionNames, rotation = 45, ha="right", rotation_mode="anchor")
plt.subplots_adjust(bottom=0.3)
plt.show()
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\CorrelationBetweenAreas_Shuff_NonShuff.svg', format='svg', dpi=1200)





















## Boxplot Correlations images with repetitions
#DataCorrNonSh = AllCorr_r_omissions_NonSh.copy();
#DataCorrSh = AllCorr_r_omissions.copy();
#
#AreasNonSh = DataCorrNonSh.copy()
#AreasSh = DataCorrSh.copy()
#for counter0 in range(AreasSh.shape[0]):
#    for counter1 in range(AreasSh.shape[1]):
#        AreasNonSh[counter0,counter1,:] = counter0+counter1*4
#        AreasSh[counter0,counter1,:] = counter0+counter1*4
#
#ShufflingNonSh = DataCorrNonSh.copy()
#ShufflingSh = DataCorrSh.copy()
#ShufflingNonSh[:,:,:] = 0;
#ShufflingSh[:,:,:] = 1;
#
#df = pd.DataFrame({'Weights' : np.concatenate((DataCorrNonSh,DataCorrSh),1).flatten(),
#                   'Areas' : np.concatenate((AreasNonSh,AreasSh),1).flatten(),
#                   'Shuffling' : np.concatenate((ShufflingNonSh,ShufflingSh),1).flatten() })
#
#sns.boxplot(data=df, x='Areas', y='Weights', hue='Shuffling')
#plt.title('Connectivity between areas - omissions')
#plt.xticks(range(len(ConnectionNames)),ConnectionNames, rotation = 45, ha="right", rotation_mode="anchor")
#plt.subplots_adjust(bottom=0.3)
#plt.show()




## Boxplot Correlations omissions with repetitions
#DataCorrNonSh = AllCorr_r_omissions_NonSh.copy();
#DataCorrSh = AllCorr_r_omissions.copy();
#
#AreasNonSh = DataCorrNonSh.copy()
#AreasSh = DataCorrSh.copy()
#for counter0 in range(AreasSh.shape[0]):
#    for counter1 in range(AreasSh.shape[1]):
#        AreasNonSh[counter0,counter1,:] = counter0+counter1*4
#        AreasSh[counter0,counter1,:] = counter0+counter1*4
#
#ShufflingNonSh = DataCorrNonSh.copy()
#ShufflingSh = DataCorrSh.copy()
#ShufflingNonSh[:,:,:] = 0;
#ShufflingSh[:,:,:] = 1;
#
#df = pd.DataFrame({'Weights' : np.concatenate((DataCorrNonSh,DataCorrSh),1).flatten(),
#                   'Areas' : np.concatenate((AreasNonSh,AreasSh),1).flatten(),
#                   'Shuffling' : np.concatenate((ShufflingNonSh,ShufflingSh),1).flatten() })
#
#sns.boxplot(data=df, x='Areas', y='Weights', hue='Shuffling')
#plt.title('Connectivity between areas - images')
#plt.xticks(range(len(ConnectionNames)),ConnectionNames, rotation = 45, ha="right", rotation_mode="anchor")
#plt.subplots_adjust(bottom=0.3)
#plt.show()









colors = [plt.cm.hsv(i) for i in np.linspace(0, 1, 600)]

ColorsIndexes = ((Ts_ComponentsOmission_ID/max(Ts_ComponentsOmission_ID)*(600-1)).astype(int));
fig, ax = plt.subplots()
for counter in range(Ts_ComponentsOmission.shape[0]):
    plt.plot(All_Ts_ComponentsOmission[counter],color = colors[ColorsIndexes[counter]])

plt.ylabel('Weights')
plt.xlabel('Time')
plt.show()
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\TimeWeights.svg', format='svg', dpi=1200)
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\TimeWeights_NonSh.svg', format='svg', dpi=1200)






























# Indexes==2

#WeightPerChannelOrig = WeightPerChannel[:,2];
#WeightPerTrialsOrig = WeightPerTrials[:,2];
#WeightPerTimeOrig = WeightPerTime[:,2];

WeightPerChannelShuff = WeightPerChannel[:,3];
WeightPerTrialsShuff = WeightPerTrials[:,3];
WeightPerTimeShuff = WeightPerTime[:,3];

fig, ax = plt.subplots()
plt.plot(WeightPerChannelOrig,'r')
plt.plot(-WeightPerChannelShuff,'b')
plt.ylabel('Weights')
plt.xlabel('Neurons [#]')
plt.show()
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\RepresentativeWeights_Neurons.svg', format='svg', dpi=1200)

fig, ax = plt.subplots()
plt.plot(WeightPerTrialsOrig,'r')
plt.plot(-WeightPerTrialsShuff,'b')
plt.ylabel('Weights')
plt.xlabel('Trials [#] - images until 159')
plt.show()
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\RepresentativeWeights_Trials.svg', format='svg', dpi=1200)

fig, ax = plt.subplots()
plt.plot(WeightPerTimeOrig,'r')
plt.plot(WeightPerTimeShuff,'b')
plt.ylabel('Weights')
plt.xlabel('Time')
plt.show()
#fig.savefig(r'C:\Users\simon\OneDrive\Desktop\VisualBehavior\SVGFigures\RepresentativeWeights_Time.svg', format='svg', dpi=1200)



