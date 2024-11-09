import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import csv
import pandas as pd

data=np.load("../report_src/data.npz")

# Question 2

class ClassificationStump():
    def __init__(self):
        return

    def fit(self, X_trn, y_trn):
        
        # do stuff here
        D = len(X_trn[0])
        N = len(X_trn)
        dim = -1
        thresh = -1
        min_error = np.inf
        c_left = 0
        c_right = 0

        for i in range(D):

            sorted_indices = np.argsort(X_trn[:, i])
            Z = X_trn[sorted_indices]
            y_sorted = y_trn[sorted_indices]


            for n in range(N-1):
                t = (Z[n][i] + Z[n+1][i])/2

                R1 = y_sorted[:n+1] # x <= t
                R2 = y_sorted[n+1:] # x > t

                R1_count = np.bincount(R1, minlength=np.max(y_sorted) + 1)
                R2_count = np.bincount(R2, minlength=np.max(y_sorted) + 1)
                
                c1 = np.argmax(R1_count)
                c2 = np.argmax(R2_count)

                p1 = R1_count / len(R1) # probability of each class in R1
                p2 = R2_count / len(R2) # probability of each class in R2

                gini_left = np.sum(p1 * (1-p1)) # sum pi(1-pi) for each class in R1
                gini_right = np.sum(p2 * (1-p2))
                
                gini_total = (len(R1) / N) * gini_left + (len(R2) / N) * gini_right # weighted average of gini impurity
                
                error = gini_total
                if error < min_error:
                    min_error = error
                    dim = i
                    c_left = c1
                    c_right = c2
                    thresh = t

        self.model = (dim, thresh, c_left, c_right)
        self.model = {'dim': dim, 'thresh': thresh, 'c_left': c_left, 'c_right': c_right}
        return 

    def predict(self, X_val, y_val=None):
        assert hasattr(self, "model"), "No fitted model!"
        # do stuff here (use self.model for prediction)
        y_pred = []
        for x_val in X_val:
            y_pred.append(self.model['c_left'] if x_val[self.model['dim']] <= self.model['thresh'] else self.model['c_right'])
        
        return y_pred


X_trn = data['X_trn']
y_trn = data['y_trn']
X_tst = data['X_tst']

clf = ClassificationStump()
X_trn = X_trn.reshape((6000, 3 * 29 * 29))

clf.fit(X_trn, y_trn)
y_pred = clf.predict(X_trn)
correct_pred = sum(y_pred == y_trn)
e = 1 - correct_pred / len(y_pred)
print("Training classification error: ", e)


# Question 3

# train classifcation trees

X_trn = data['X_trn']
y_trn = data['y_trn']

X_trn = X_trn.reshape((len(X_trn), 3 * 29 * 29))

max_depths = [1,3,6,9,12,14]
# return a 6x1 table of classification errors
print(" DEPTH | TRAINING ERROR")
for max_depth in max_depths:
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_trn, y_trn)
    y_pred = clf.predict(X_trn)
    correct_pred = sum(y_pred == y_trn)
    acc = correct_pred / len(y_trn)
    e = 1 - acc 
    print(f"  {max_depth}  | {e}")

# Question 6

# linear classifier model with logistic loss and ridge regularization only using sklearn.linear and decision_function() method



X_trn = data['X_trn']
y_trn = data['y_trn']
X_tst = data['X_tst']

X_trn = X_trn.reshape((6000, 3*29*29))  


lambda_vals = [0.1, 1, 10, 100, 1000]

# report training classification error and logistic loss for each lambda value
print(" LAMBDA | TRAINING ERROR | LOGISTIC LOSS")
for lambda_val in lambda_vals:
    clf = LogisticRegression(penalty='l2', C=1/lambda_val, max_iter=10000, tol=.001)
    clf.fit(X_trn, y_trn)
    
    confidence_scores = clf.decision_function(X_trn) # get confidence score per class per sample 
   
    exp_scores = np.exp(confidence_scores) # prepare for softmax
    sum_exp_scores_per_sample = [sum(scores) for scores in exp_scores]

    probs = [exp_scores[i] / sum_exp_scores_per_sample[i] for i in range(len(exp_scores))] # softmax to get probs
    y_pred = [np.argmax(prob) for prob in probs] # get prediction from highest prob
    logistic_loss = log_loss(y_trn, probs) # calculate logistic loss
   
    correct_preds = np.sum(y_pred == y_trn)
    e = 1 - (correct_preds / len(y_trn)) # calculate classification error
    print(f"  {lambda_val}  | {e} | {logistic_loss}")


# Question 7

# K nearest neighbors classifier

X_trn = data['X_trn']
y_trn = data['y_trn']
X_tst = data['X_tst']

X_trn = X_trn.reshape((len(X_trn), 3 * 29 * 29))

k_neighbors = [1, 3, 5, 7, 9, 11]

print(" K | TRAINING ERROR")
for k in k_neighbors:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_trn, y_trn)
    y_pred = clf.predict(X_trn)
    correct_pred = sum(y_pred == y_trn)
    acc = correct_pred / len(y_trn)
    e = 1 - acc
    print(f" {k} | {e}")


# Question 9


class Classifier():
    def __init__(self, model):
        self.model = model
        return

    def fit(self, X_trn, y_trn):
        self.model.fit(X_trn, y_trn)
        # self.model is stored
        return

    def predict(self, X_val, y_val=None):
        # self.model is used
        y_pred = self.model.predict(X_val)
        return y_pred


def cross_validation(classifier, X_trn, y_trn, n_folds=5):
    # do stuff here

    kf = KFold(n_splits=n_folds)
    outputs = []

    for train_index, test_index in kf.split(X_trn):
        
        x_train, x_test = X_trn[train_index], X_trn[test_index]
        y_train, y_test = y_trn[train_index], y_trn[test_index]
        
        
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        correct_preds = np.sum(y_pred == y_test)
        e = 1 - (correct_preds / len(y_test))
        
        outputs.append((classifier, e))
    
    # Return the paired (model and error) for all folds
    return outputs # [(model1, error1), (model2, error2), ..., (modelK, errorK)]
    
# Question 10

X_trn = data['X_trn']
y_trn = data['y_trn']

X_trn = X_trn.reshape((6000, 3 * 29 * 29))

results = []

# Perform cross-validation for different max_depth values
for max_depth in [1, 3, 6, 9, 12, 14]:
    classifier = Classifier(DecisionTreeClassifier(max_depth=max_depth))
    N = 5
    outputs = cross_validation(classifier, X_trn, y_trn, n_folds=N)
    
    # Collect the errors for each fold
    fold_errors = [out_[1] for out_ in outputs]
    avg_error = np.mean(fold_errors)
    
    # Append the results to the list
    results.append([max_depth] + fold_errors + [avg_error])

# Create a DataFrame from the results
columns = ["MAX_DEPTH", "FOLD_1_ERROR", "FOLD_2_ERROR", "FOLD_3_ERROR", "FOLD_4_ERROR", "FOLD_5_ERROR", "AVG_ERROR"]
df = pd.DataFrame(results, columns=columns)

# Display the DataFrame as a table
print(df.to_string(index=False))

# Question 11


X_trn = data['X_trn']
y_trn = data['y_trn']

X_trn = X_trn.reshape((6000, 3 * 29 * 29))

results = []

# Perform cross-validation for different lambda values
for lambda_val in [0.1, 1, 10, 100, 1000]:
    classifier = Classifier(LogisticRegression(penalty='l2', C=1/lambda_val, max_iter=10000, tol=.001))
    N = 5
    outputs = cross_validation(classifier, X_trn, y_trn, n_folds=N)
    
    # Collect the errors for each fold
    fold_errors = [out_[1] for out_ in outputs]
    avg_error = np.mean(fold_errors)
    
    # Append the results to the list
    results.append([lambda_val] + fold_errors + [avg_error])

# Create a DataFrame from the results
columns = ["LAMBDA", "FOLD_1_ERROR", "FOLD_2_ERROR", "FOLD_3_ERROR", "FOLD_4_ERROR", "FOLD_5_ERROR", "AVG_ERROR"]
df = pd.DataFrame(results, columns=columns)

# Display the DataFrame as a table
print(df.to_string(index=False))

# Question 12

class KNNClassifier():
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        return

    def fit(self, X_trn, y_trn):
        # Just store X_trn, y_trn
        self.model = (X_trn, y_trn)

    def predict(self, X_val, y_val=None):
        assert hasattr(self, "model"), "No fitted model!"
        # do stuff here (use self.model for prediction)
        
        def KNN_predict_(X_trn, y_trn, x, K): 
            # Dictionary to store n: distance pairs
            distances = {}
            for n in range(len(X_trn)):
                # Calculate distance between x and x[n]
                distance = np.sqrt(np.sum((x - X_trn[n])**2))
                distances[n] = distance
            
            # Sort distances in ascending order
            sorted_distances = sorted(distances.items(), key=lambda x: x[1])

            # Get the K nearest neighbors
            y_neighbor = [] 
            for n in range(K):
                n_nearest = sorted_distances[n][0]
                y_neighbor.append(y_trn[n_nearest])
            
            # get majority class
            c_pred = np.argmax(np.bincount(y_neighbor))
            return c_pred
        
        X_trn, y_trn = self.model
        y_pred = []
        for x in X_val:
            y_pred.append(KNN_predict_(X_trn, y_trn, x, self.n_neighbors))

        return y_pred

k_neighbors = [1, 3, 5, 7, 9, 11]

# Initialize a list to store the results
results = []

# Perform cross-validation for different k values
for k in k_neighbors:
    knn = KNNClassifier(k)
    knn.fit(X_trn, y_trn)
    models = cross_validation(knn, X_trn, y_trn, n_folds=5)
    
    # Collect the errors for each fold
    fold_errors = [model[1] for model in models]
    avg_error = np.mean(fold_errors)
    
    # Append the results to the list
    results.append([k] + fold_errors + [avg_error])

# Create a DataFrame from the results
columns = ["K", "FOLD_1_ERROR", "FOLD_2_ERROR", "FOLD_3_ERROR", "FOLD_4_ERROR", "FOLD_5_ERROR", "AVG_ERROR"]
df = pd.DataFrame(results, columns=columns)

# Display the DataFrame as a table
print(df.to_string(index=False))

# Question 13

def write_csv(y_pred, filename):
    """Write a 1d numpy array to a Kaggle-compatible .csv file"""
    with open(filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Id', 'Category'])
        for idx, y in enumerate(y_pred):
            csv_writer.writerow([idx, y])

# Train best tree model on test data and save predictions

X_trn = data['X_trn']
y_trn = data['y_trn']
X_tst = data['X_tst']

X_trn = X_trn.reshape((len(X_trn), 3 * 29 * 29))
X_tst = X_tst.reshape((len(X_tst), 3 * 29 * 29))

clf = DecisionTreeClassifier(max_depth = 9)
clf.fit(X_trn, y_trn)
y_pred = clf.predict(X_tst)
write_csv(y_pred, 'tree_predictions_9.csv')

# Train best logistic regression model on test data and save predictions

X_trn = data['X_trn']
y_trn = data['y_trn']
X_tst = data['X_tst']

X_trn = X_trn.reshape((len(X_trn), 3 * 29 * 29))
X_tst = X_tst.reshape((len(X_tst), 3 * 29 * 29))

lambda_val = 100
clf = LogisticRegression(penalty='l2', C=1/lambda_val, max_iter=10000, tol=.001)
clf.fit(X_trn, y_trn)
y_pred = clf.predict(X_tst)
write_csv(y_pred, 'logistic_predictions_100.csv')

# Train best k nearest neighbor on test data and save predictions

X_trn = data['X_trn']
y_trn = data['y_trn']
X_tst = data['X_tst']

X_trn = X_trn.reshape((len(X_trn), 3 * 29 * 29))
X_tst = X_tst.reshape((len(X_tst), 3 * 29 * 29))

k = 9
clf = KNNClassifier(k)
clf.fit(X_trn, y_trn)
y_pred = clf.predict(X_tst)
write_csv(y_pred, 'knn_predictions_9.csv')