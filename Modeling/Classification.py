import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def make_targets(df):
    ranks = [9,8,7,6,5,4,3,2]

    dfs = []
    labels = {9:{},8:{},7:{},6:{},5:{},4:{},3:{},2:{}}


    for r in ranks:
        d1 = df.copy()
        d1 = d1[d1[f'rank {r}'] != '']
        label_encoder = LabelEncoder()
        d1['target_names'] = label_encoder.fit_transform(d1[f'rank {r}'])

        dfs.append(d1)
        print(f'appended {r}')

        for original_label, encoded_label in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
            labels[r][encoded_label] = original_label

    return dfs,labels

def classify(dfs, labels):

    for data in dfs:

        data = data.groupby('target_names').filter(lambda x: len(x) > 1)
        counter = 3
        X = data[data.columns[16:-4]]
        y = data['target_names']

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Define the models to be evaluated with one set of parameters each
        models = [
            {
                'name': 'Decision Tree',
                'model': DecisionTreeClassifier(criterion='gini', max_depth=5)
            },
            {
                'name': 'SVM',
                'model': SVC(C=1, kernel='rbf')
            },
            {
                'name': 'Random Forest',
                'model': RandomForestClassifier(n_estimators=100, max_depth=5)
            },
            {
                'name': 'Logistic Regression',
                'model': LogisticRegression(C=1)
            }

        ]

        # Lists to store the best scores
        results = []

        # Evaluate models
        for model in models:

            print(f"Evaluating {model['name']}")
            clf = model['model']
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Test accuracy: {accuracy}")
            print("")

            report = classification_report(y_test, y_pred, output_dict=True)
            accuracies = {f"accuracy_class_{labels[counter][int(i)]}": report[str(i)]['precision'] for i in list(report.keys())[:-3]}
            accuracies['total_accuracy'] = accuracy_score(y_test, y_pred)
            accuracies['model'] = model['name']

            results.append(accuracies)

        # Convert results to DataFrame
        df = pd.DataFrame(results)
        df.to_csv(f'scores_for_rank_{counter}.csv')


if __name__ == '__main__':

    data = pd.read_csv('ready_to_run_codon.csv')
    dfs,labels = make_targets(data)
    classify(dfs, labels)