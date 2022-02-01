import warnings
warnings.filterwarnings('ignore')
import pandas as pd

# Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#mathplotlib Librarries
import matplotlib.pyplot as plt
from sklearn import metrics
# Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def Algorithm(filename,Algorithm):
    from sklearn import metrics
    ### loadDataset(fileName):
    dataset = pd.read_csv(filename)
    dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    standardScaler = StandardScaler()
    columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
    y = dataset['target']
    X = dataset.drop(['target'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


    if Algorithm=='KNN':
        knn_scores = []
        for k in range(1,21):
            knn_classifier = KNeighborsClassifier(n_neighbors = k)
            knn_classifier.fit(X_train, y_train)
            knn_scores.append(knn_classifier.score(X_test, y_test))

        y_pred = knn_classifier.predict(X_test)
        # print('Áccuracy ',metrics.accuracy_score(y_test, y_pred))

        plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
        for i in range(1,21):
            plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
        plt.xticks([i for i in range(1, 21)])
        plt.xlabel('Number of Neighbors (K)')
        plt.ylabel('Scores')
        plt.title('K Neighbors Classifier scores for different K values')
        plt.show()


    elif Algorithm=='Decision Tree':
        dt_scores = []
        for i in range(1, 21):
            dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
            dt_classifier.fit(X_train, y_train)
            dt_scores.append(dt_classifier.score(X_test, y_test))

        y_pred = dt_classifier.predict(X_test)
        # print('Áccuracy',metrics.accuracy_score(y_test, y_pred))

        plt.plot([i for i in range(1, 21)], dt_scores, color = 'green')
        for i in range(1, 21):
            plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))
        plt.xticks([i for i in range(1, 21)])
        plt.xlabel('Max features')
        plt.ylabel('Scores')
        plt.title('Decision Tree Classifier scores for different number of maximum features')
        plt.show()


    elif Algorithm=='Random Forest':
        rf_scores = []
        for i in range(1, 21):
            rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)
            rf_classifier.fit(X_train, y_train)
            rf_scores.append(rf_classifier.score(X_test, y_test))

        y_pred = rf_classifier.predict(X_test)
        # print('Áccuracy ',metrics.accuracy_score(y_test, y_pred))

        plt.plot([i for i in range(1, 21)], rf_scores, color = 'black')
        for i in range(1, 21):
            plt.text(i, rf_scores[i-1], (i, rf_scores[i-1]))
        plt.xticks([i for i in range(1, 21)])
        plt.xlabel('Max features')
        plt.ylabel('Scores')
        plt.title('Random Forest Classifier scores for different number of maximum features')
        plt.show()
