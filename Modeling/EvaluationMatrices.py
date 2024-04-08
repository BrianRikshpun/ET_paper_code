import numpy as np


class EvaluationMatrices():

    def calcTreeTaxonomy(self, df):

        #TC SCORE CALCULATION

        ranks = ['rank 2', 'rank 3', 'rank 4', 'rank 5', 'rank 6', 'rank 7', 'rank 8']

        ranks_entropies = {}

        for s in df['rank 2'].unique():

            if s != '':
                df2 = df.copy()
                df2 = df2[df2['rank 2'] == s]

                probs = []
                for c in df2['clusters'].unique():
                    probs.append(len(df[(df['rank 2'] == s) & (df['clusters'] == c)]) / len(df[df['rank 2'] == s]))

                ranks_entropies[s] = -1 * sum([i * np.log2(i) for i in probs])

        for i, rank in enumerate(ranks[1:]):
            i = i + 1

            for node in df[rank].unique():

                if str(node) != 'nan':
                    d = df.copy()
                    d = d[d[rank] == node]
                    ranks_entropies[node] = 0

                    for w in d[ranks[i - 1]].unique():
                        if str(w) != 'nan':
                            ranks_entropies[node] += (len(d[d[ranks[i - 1]] == w]) / len(d)) * ranks_entropies[w]

        summ = 0
        for node in df['rank 8'].unique():
            if str(node) != 'nan':
                print(f'{node} entropy is {ranks_entropies[node]} ')
                summ += ranks_entropies[node]

        print(f'the sum is {summ}')

        return summ



