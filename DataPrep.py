import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
from itertools import repeat


def twos(l):
    ''''
    This function will return all the combinations of all the chars in a length of two given a string
    '''
    yield from itertools.product(*([l] * 2))

def threes(l):
    ''''
    This function will return all the combinations of all the chars in a length of three given a string
    '''
    yield from itertools.product(*([l] * 3))

def sixs(l):
    ''''
    This function will return all the combinations of all the chars in a length of six given a string
    '''
    yield from itertools.product(*([l] * 6))

def clean_db(X):

    '''
    This function will clean the data in the DataFrame and return it ready for the train phase
    '''

    # drop 0 variance columns:
    del_col = []
    for i in X.columns[:-1]:
        if np.std(X.loc[:, i]) < 0.01 * np.mean(X.loc[:, i]):
            del_col.append(i)
    print(del_col, 'have 1% variance, will be dropped from db')
    X = X.drop(del_col, axis=1)

    # drop 90% correlated columns
    corr_db = pd.DataFrame(np.corrcoef(X.transpose().astype(float)),
                           index=X.columns,
                           columns=X.columns)
    del_col = []
    for c_index, c in enumerate(corr_db.columns):
        for b in corr_db.index[c_index + 1:]:
            if corr_db.loc[c, b] >= 0.8 and b != c:
                # print(c, ' and ', b, ' are strongly associated: ', corr_db.loc[c, b])
                if b not in del_col:
                    del_col.append(b)
    print("deleting column ", del_col)
    print("Total deleted columns = ", len(del_col))
    X = X.drop(del_col, axis=1)
    return X

def FeatureEnrichment(data):

    '''
    This function will enrich the data with the calculations of single nucleotides, paired nucleotides and entropies
    '''

    relaventColumns = data.columns[12:] #Only Trios of codons

    #Single codon features

    dataA = data.copy()
    dataC = data.copy()
    dataG = data.copy()
    dataT = data.copy()

    for col in relaventColumns:
        if col.count('A') == 2:
            dataA[col+"2"] = data[col]
        if col.count('C') == 2:
            dataC[col+"2"] = data[col]
        if col.count('T') == 2:
            dataT[col+"2"] = data[col]
        if col.count('G') == 2:
            dataG[col+"2"] = data[col]

        if col.count('A') == 3:
            dataA[col+"2"] = data[col]
            dataA[col + "3"] = data[col]
        if col.count('C') == 3:
            dataC[col+"2"] = data[col]
            dataC[col + "3"] = data[col]
        if col.count('G') == 3:
            dataG[col+"2"] = data[col]
            dataG[col + "3"] = data[col]
        if col.count('T') == 3:
            dataT[col+"2"] = data[col]
            dataT[col + "3"] = data[col]

    relaventColumnsA = dataA.columns[12:]
    relaventColumnsC = dataC.columns[12:]
    relaventColumnsG = dataG.columns[12:]
    relaventColumnsT = dataT.columns[12:]

    data['A'] = dataA[[str for str in relaventColumnsA if any(sub in str for sub in ['A'])]].sum(axis=1)
    data['G'] = dataG[[str for str in relaventColumnsG if any(sub in str for sub in ['G'])]].sum(axis=1)
    data['C'] = dataC[[str for str in relaventColumnsC if any(sub in str for sub in ['C'])]].sum(axis=1)
    data['T'] = dataT[[str for str in relaventColumnsT if any(sub in str for sub in ['T'])]].sum(axis=1)
    data['%A'] = data['A'] / 3 / data['# Codons']
    data['%G'] = data['G'] / 3 / data['# Codons']
    data['%C'] = data['C'] / 3 / data['# Codons']
    data['%T'] = data['T'] / 3 / data['# Codons']

    #Paired codon data

    relaventColumns = data.columns[12:]

    data1 = data.copy()
    data1['TTT2'] = data['TTT']
    data1['GGG2'] = data['GGG']
    data1['AAA2'] = data['AAA']
    data1['CCC2'] = data['CCC']

    relaventColumns2 = data1.columns[12:]

    for x in twos('TGCA'):
        data[x[0]+x[1]] = data1[[str for str in relaventColumns2 if any(sub in str for sub in [x[0]+x[1]])]].sum(axis=1)
        data['%' + x[0]+x[1]] = data[x[0]+x[1]] / 2 / data['# Codons']


    #precentage of codons in trios

    for x in threes('TGCA'):
        #if(x[0]+x[1]+x[2] != 'GGG'): #No GGG
        data['%' + x[0]+x[1]+x[2]] = data[x[0]+x[1]+x[2]] / data['# Codons']

    cols2 = []
    for x in twos('TGCA'):
        cols2.append('%' + x[0] + x[1])
        data['%of2'] = data[cols2].sum(axis=1)

    cols3 = []
    for x in threes('TGCA'):
        #if (x[0] + x[1] + x[2] != 'GGG'):
        cols3.append('%' + x[0] + x[1] + x[2])
        data['%of3'] = data[cols3].sum(axis=1)

    data['%of1'] = data['%A'] + data['%C'] + data['%T'] + data['%G']

    print("average of 1: " + str(data['%of1'].mean()))
    print("average of 2: " + str(data['%of2'].mean()))
    print("average of 3: " + str(data['%of3'].mean()))

    #Entropy

    c1 = ['%A', '%G' ,'%C', '%T']
    data['entropy1'] = -1 * (data[c1] * np.log2(data[c1])).sum(axis = 1)

    c2 = ['%'+x[0]+x[1] for x in twos('TGCA')]
    data['entropy2'] = -1 * (data[c2] * np.log2(data[c2])).sum(axis=1)

    c3 = ['%'+x[0]+x[1]+x[2] for x in threes('TGCA') ]#if x != ('G','G','G')]
    data['entropy3'] = -1 * (data[c3] * np.log2(data[c3])).sum(axis=1)

    return data

def FeatureEnrichmentBi(data):
    '''
    This function will enrich the data with the calculations of codon-pairs and corresponding entropies
    '''

    relaventColumns = data.columns[9:]

    for x in sixs('tcga'):
        data['%' + x[0]+x[1]+x[2]+x[3]+x[4]+x[5]] = data[x[0]+x[1]+x[2]+x[3]+x[4]+x[5]] / data['# Codon Pairs']

    c6 = ['%' + x[0]+x[1]+x[2]+x[3]+x[4]+x[5] for x in sixs('tcga')]
    data['entropy2'] = -1 * (data[c6] * np.log2(data[c6])).sum(axis=1)
    return data

def Preprocess(data):
    data = data[data['Organelle'] == 'genomic']
    data.pop('Division')
    data.pop('Organelle')
    data.pop('Taxid')
    data.pop('Species')
    data.pop('Assembly')
    data.pop('Translation Table')
    cols = ['%' + x[0]+x[1] for x in twos('TGCA')] + ['%' + x[0]+x[1]+x[2] for x in threes('TGCA')] + ['%A', '%T', '%C', '%G'] + ['entropy1', 'entropy2', 'entropy3']
    data = data[cols]
    #data = data.drop(columns=['Unnamed: 10', 'Unnamed: 0.1', 'Unnamed: 0', 'Division', 'Assembly'])
    ranks = [f'rank {i}' for i in range(2,10)]
    data['species'] = data['species'].str.replace('\t', '')
    for i in ranks:
        data[i] = data[i].str.replace('\t', '')

    return data

def Visualize(data):

    '''
    Helpful data visualizations as part of the initial EDA
    '''


    sns.violinplot(data=data, x="Organelle", y="entropy1")
    plt.title('Entropy 1 for each Organelle')
    plt.savefig("Entropy1.jpg")
    plt.show()

    sns.violinplot(data=data, x="Organelle", y="entropy2")
    plt.title('Entropy 2 for each Organelle')
    plt.savefig("Entropy2.jpg")
    plt.show()

    sns.violinplot(data=data, x="Organelle", y="entropy3")
    plt.title('Entropy 3 for each Organelle')
    plt.savefig("Entropy3.jpg")
    plt.show()

    #% of 1 codon visualization

    sns.violinplot(data=data, x="Organelle", y="%A")
    plt.title('%A for each Organelle')
    plt.savefig("A.jpg")
    plt.show()

    sns.violinplot(data=data, x="Organelle", y="%C")
    plt.title('%C for each Organelle')
    plt.savefig("C.jpg")
    plt.show()

    sns.violinplot(data=data, x="Organelle", y="%G")
    plt.title('%G for each Organelle')
    plt.savefig("G.jpg")
    plt.show()

    sns.violinplot(data=data, x="Organelle", y="%T")
    plt.title('%T for each Organelle')
    plt.savefig("T.jpg")
    plt.show()

    #% of single codonds violine plot

    #% codong plots

    data_to_plot = [data['%A'], data['%C'], data['%T'], data['%G']]
    red_patch = mpatches.Patch(color='blue')
    pos   = [1, 2, 3, 4]
    label = ['%A','%C','%T','%G']

    fake_handles = repeat(red_patch, len(pos))

    plt.figure()
    ax = plt.subplot(111)
    plt.violinplot(data_to_plot, pos, vert=False)
    ax.legend(fake_handles, label)
    plt.title("Codon % for genomic organelle")
    plt.show()

    #entropy plots

    data_to_plot = [data['entropy1'], data['entropy2'], data['entropy3']]
    red_patch = mpatches.Patch(color='blue')
    pos   = [1, 2, 3]
    label = ['entropy1','entropy2','entropy3']

    fake_handles = repeat(red_patch, len(pos))

    plt.figure()
    ax = plt.subplot(111)
    plt.violinplot(data_to_plot, pos, vert=False)
    ax.legend(fake_handles, label)
    plt.title("Entropy for genomic organelle")
    plt.show()


def makeEntropyBins(data):

    '''
    Entropy distributions visualizations
    '''

    x1 = data['entropy1']
    x2 = data['entropy2']
    x3 = data['entropy3']

    hist1, bin_edges = np.histogram(x1, bins=10)
    hist2, bin_edges = np.histogram(x2, bins=10)
    hist3, bin_edges = np.histogram(x3, bins=10)

    print(hist1)
    print("-----")
    print(hist2)
    print("-----")
    print(hist3)
    print("-----")


    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=10)

    print(str(x1.mean()) + 'std' + str(np.std(x1)))
    print(str(x2.mean()) + 'std' + str(np.std(x2)))
    print(str(x3.mean()) + 'std' + str(np.std(x3)))

    plt.hist(x1, **kwargs, label= 'Entropy 1')
    plt.hist(x2, **kwargs, label= 'Entropy 2')
    plt.hist(x3, **kwargs, label= 'Entropy 3')
    plt.legend()
    plt.title('Entropy histogram - 10 bins')
    plt.xlabel('Entropy')
    plt.ylabel('Count')
    #plt.savefig("entropy - histograms.jpg")
    plt.show()

def makeCorrelationMatrix(data):

    '''
    Correlation Heatmaps
    '''

    corr = data.corr()
    components = list()
    visited = set()
    for col in data.columns:
        if col in visited:
            continue

        component = set([col, ])
        just_visited = [col, ]
        visited.add(col)
        while just_visited:
            c = just_visited.pop(0)
            for idx, val in corr[c].items():
                if abs(val) > 0.0 and idx not in visited:
                    just_visited.append(idx)
                    visited.add(idx)
                    component.add(idx)
        components.append(component)

    for component in components:
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr.loc[component, component], cmap="Reds")

    if len(data.columns[0]) == 1:
        plt.title("Correlation for single codons")
    if len(data.columns[0]) == 2:
        plt.title("Correlation for paired codons")
    else:
        plt.title("Codon correlation")
    plt.show()




# Start

dataset_selection = "codon" #codon/ bicodon/ both
mission = "clustering" #preprocess/ EDA /

if dataset_selection == 'codon':

    #Creating filltered & feature enriched csv
    if mission == 'preprocess':
        data = pd.read_table('o537-genbank_species.tsv')
        data = data.shift(periods = 1, axis = 1)
        data = FeatureEnrichment(data)

        d = pd.read_csv('rankedlineage122.csv', sep='|')  # Taxnomy
        d = d.rename({d.columns[0]: 'Taxid', d.columns[1]: 'species', d.columns[2]: 'rank 2',
                      d.columns[3]: 'rank 3', d.columns[4]: 'rank 4',
                      d.columns[5]: 'rank 5', d.columns[6]: 'rank 6',
                      d.columns[7]: 'rank 7', d.columns[8]: 'rank 8', d.columns[9]: 'rank 9'}, axis=1)

        merged = d.merge(data, on='Taxid')
        merged = Preprocess(data)
        merged.to_csv("ready_to_run_codon.csv")


    #Data EDA
    elif mission == 'EDA':
        data = pd.read_csv('clustering_codon.csv')
        Visualize(data)
        makeEntropyBins(data)
        makeCorrelationMatrix(data[['A','C','T','G']])
        makeCorrelationMatrix(data[[x[0]+x[1] for x in twos('TGCA')]])
        makeCorrelationMatrix(data[[x[0]+x[1]+x[2] for x in threes('TGCA')]])


elif dataset_selection == 'bicodon':

    if mission == 'preprocess':
        data = pd.read_csv('o537-genbank_Bicod.tsv', sep='|')
        data = FeatureEnrichmentBi(data)

        d = pd.read_csv('rankedlineage122.csv', sep='|')  # Taxnomy
        d = d.rename({d.columns[0]: 'Taxid', d.columns[1]: 'species', d.columns[2]: 'rank 2',
                      d.columns[3]: 'rank 3', d.columns[4]: 'rank 4',
                      d.columns[5]: 'rank 5', d.columns[6]: 'rank 6',
                      d.columns[7]: 'rank 7', d.columns[8]: 'rank 8', d.columns[9]: 'rank 9'}, axis=1)

        merged = d.merge(data, on='Taxid')
        merged = Preprocess(data)
        merged.to_csv("ready_to_run_bicodon.csv")



