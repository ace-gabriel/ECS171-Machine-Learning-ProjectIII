"""

ECS171 HW3
Gabriel Yin #999885129

5. Attribute Information:
	1. ID: profile ID (integer)
	2. Strain: E. coli strain name (string)
		(1) BW25113
		(2) CG2
		(3) DH5alpha
		(4) MG1655
		(5) P2
		(6) P4X
		(7) W3110
		(8) rpoA14
		(9) rpoA27
		(10) rpoD3
	3. Medium: supplied medium indexed as MDXXX (string)
		(1) MD001 : M9 + Glucose (5mL of 40% solution of Glucose)
		(2) MD002 : M9 + Glycerol (5mL of 40% solution of Glycerol)
		(3) MD003 : M9 + Lactose (5mL of 40% solution of Lactose)
		(4) MD004 : LB (undefined medium)
		(5) MD005 : minimal medium + low-Glucose (0.1g/L)
		(6) MD006 : LB + KCl
		(7) MD007 : minimal medium + Glucose
		(8) MD008 : minimal medium + Glucose + Arginine
		(9) MD009 : mineral medium + low-Glucose (0.1g/L)
		(10) MD010 : GGM
		(11) MD011 : defined medium + Glycerol
		(12) MD012 : M9 + Glucose + Galactose + Glycerol + Lactose + Maltose
		(13) MD013 : defined mineral medium + Glucose
		(14) MD014 : modified MOPS + Glycerol
		(15) MD015 : MOPS + Glucose + NH4Cl
		(16) MD016 : TB (undefined medium)
		(17) MD017 : MOPS + Succinate
		(18) MD018 : Evans
	4. Environmental_perturbation: stresses present in the environment (string)
		(1) Indole : inhibitor of quorum-sensing signal
		(2) O2-starvation : oxygen starvation
		(3) RP-overexpress : overexpression of a recombinant protein
		(4) Antibacterial : presence of antibiotics (no resistant genes in the strain)
		(5) Carbon-limitation : limited carbon source
		(6) Dna-damage : DNA damage agents
		(7) Zinc-limitation : limited Zinc source
		(8) None : no stress
	5. Gene_Perturbation: knock-out or overexpression of a specific gene (string)
		(1) appY_KO : Knock-out of the appY gene
		(2) arcA_KO : Knock-out of arcA gene
		(3) argR_KO : Knock-out of argR gene
		(4) cya_KO : Knock-out of cya gene
		(5) fis_OE : Over-expressino of fis gene
		(6) fnr_KO : Knock-out of fnr gene
		(7) frdC_KO : Knock-out of frdC gene
		(8) na_WT : wild-type
		(9) oxyR_KO : Knock-out of oxyR
		(10) rpoS_KO : Knock-out of rpoS
		(11) soxS_KO : Knock-out of soxS
		(12) tnaA_KO : Knock-out of tnaA
	6. Growth_Rate (floating point)
	7-4502. Gene expression for all 4496 genes. GeneID is given by bnumber and gene names can be found in the
		gene_name.txt. For more information about each gene, visit www.ecocyc.org
"""

from sklearn import linear_model, model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.decomposition import PCA

# Problem 1
dataset = pd.read_csv("ecs171.dataset.txt", sep='\s+', header = None).values
dataset_x = dataset[1:,7:4502].astype(float)
dataset_y = dataset[1:,5].astype(float)
# Find optimal alpha using bic criteria
bic_model = linear_model.LassoLarsIC(criterion = 'bic')
bic_model.fit(dataset_x, dataset_y)
optimal_alpha = bic_model.alpha_
clf = linear_model.Lasso(alpha = optimal_alpha)
clf.fit(dataset_x, dataset_y)
#count the number of non-zero features
count = 0
for feature in clf.coef_:
    if feature != 0:
        count += 1
print("The number of non-zero features is", count)
# Cross validation
cv_model = linear_model.LassoLarsCV(cv = 10).fit(dataset_x, dataset_y)
print("Ten fold cross-validation error is", cv_model.cv_mse_path_.mean())

# Problem 2

iterations = 1000 # Number of iterations
size_n = int(len(dataset_x) * 0.5) # determine the size of n

data = [] # global array to hold scores statistics
for i in range(iterations):
	rand_index = np.random.choice(range(0,len(dataset_y)), len(dataset_y))
	x_set = dataset_x[rand_index]
	y_set = dataset_y[rand_index]
	# using ridge model
	csv_model = linear_model.Ridge(fit_intercept = True, alpha = optimal_alpha)
	csv_model.fit(x_set, y_set)
	# Compute stats
	test_index = np.array([item for item in range(0, len(dataset_y)) if item not in rand_index])
	x_test_set = dataset_x[test_index]
	y_test_set = dataset_y[test_index]

	stats = csv_model.predict(x_test_set)
	score = csv_model.score(x_test_set, y_test_set)
	data.append(score)
# Get Confidence Interval 90
alpha_val = 0.9
p = 100 * ((1 - alpha_val)/2.0)
lower1 = max(0.0, np.percentile(data, int(p)))
p = 100 * (alpha_val + ((1.0 - alpha_val)/2.0))
upper1 = min(1.0, np.percentile(data, int(p)))
print("90 interval: lower:", lower1, "upper:", upper1)
# Get Confidence Interval 95
alpha_val2 = 0.95
p = 100 * ((1 - alpha_val2)/2.0)
lower2 = max(0.0, np.percentile(data, int(p)))
p = 100 * (alpha_val + ((1.0 - alpha_val)/2.0))
upper2 = min(1.0, np.percentile(data, int(p)))
print("95 interval: lower:", lower2, "upper:", upper2)

# Problem 3
mean_value = []
for i in range(0, (np.shape(dataset_x)[1])):
	mean_value.append((dataset_x[:,i].mean()))
mean_value = np.array(mean_value).reshape(1,-1)
result = clf.predict(mean_value)[0]
print("Mean expression gene growth value:", result)

# Problem 4
"""
Strain type. Medium type. Environmental pertub. Gene pertub.
"""
# transform data
Encoder = LabelEncoder()
strain, medium, env_port, gene_port = dataset[1:,1], dataset[1:,2], dataset[1:,3], dataset[1:,4]
strain = Encoder.fit_transform(strain)
medium = Encoder.fit_transform(medium)
gene_port = Encoder.fit_transform(gene_port)
env_port = Encoder.fit_transform(env_port)
# create SVM classifier and do a feature selection
dataset_x_medium, dataset_x_strain, dataset_x_env, dataset_x_gene = dataset_x, dataset_x, dataset_x, dataset_x
# strain
strain_svc = LinearSVC(C = 0.05, penalty = "l1", dual = False).fit(dataset_x_strain, strain)
strain_model = SelectFromModel(strain_svc, prefit = True)
dataset_x_strain = strain_model.transform(dataset_x_strain)
# medium
medium_svc = LinearSVC(C = 0.05, penalty = "l1", dual = False).fit(dataset_x_medium, medium)
medium_model = SelectFromModel(medium_svc, prefit = True)
dataset_x_medium = medium_model.transform(dataset_x_medium)
# env
env_svc = LinearSVC(C = 0.05, penalty = "l1", dual = False).fit(dataset_x_env, env_port)
env_model = SelectFromModel(env_svc, prefit = True)
dataset_x_env = env_model.transform(dataset_x_env)
# gene
gene_svc = LinearSVC(C = 0.05, penalty = "l1", dual = False).fit(dataset_x_gene, gene_port)
gene_model = SelectFromModel(gene_svc, prefit = True)
dataset_x_gene = gene_model.transform(dataset_x_gene)



""" Plot ROC curve with AUC value"""
# strain
names = [1,2,3,4,5,6,7,8,9,10]
random_state = np.random.RandomState(0)
n_samples, n_features = dataset_x_strain.shape
y = label_binarize(strain, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(dataset_x_strain, y, test_size=.8,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0, n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = ["red", "green", "blue", "yellow", "darkorange", "cyan", "purple", "pink", "black", "orange"]
plt.figure(figsize=(10,8))
lw = 2
for i in range(0, n_classes):
	plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw, label='ROC curve for class %d (AUC = %0.2f)' % (names[i] , roc_auc[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC Curve for Strain')
plt.legend(loc="lower right")
plt.show()

# medium
names = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
random_state = np.random.RandomState(0)
n_samples, n_features = dataset_x_medium.shape
y = label_binarize(medium, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(dataset_x_medium, y, test_size=.8,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0, n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = ["red", "green", "blue", "yellow", "darkorange", "cyan", "purple", "pink", "black", "orange",
          "navy", "lightblue", "darkgreen", "lightgreen", "lightpink", "darkblue", "darkcyan", "aqua"]
plt.figure(figsize=(10,8))

lw = 2
for i in range(0, n_classes):
	plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw, label='ROC curve for class %d (AUC = %0.2f)' % (names[i] , roc_auc[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC Curve for Medium')
plt.legend(loc="lower right")
plt.show()

# env
names = [1,2,3,4,5,6,7,8]
random_state = np.random.RandomState(0)
n_samples, n_features = dataset_x_env.shape
y = label_binarize(env_port, classes=[0,1,2,3,4,5,6,7])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(dataset_x_env, y, test_size=.8,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0, n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = ["red", "green", "blue", "yellow", "darkorange", "cyan", "purple", "pink"]
plt.figure(figsize=(10,8))

lw = 2
for i in range(0, n_classes):
	plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw, label='ROC curve for class %d (AUC = %0.2f)' % (names[i] , roc_auc[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC Curve for Environmental_perturbation')
plt.legend(loc="lower right")
plt.show()

# gene
names = [1,2,3,4,5,6,7,8,9,10,11,12]
random_state = np.random.RandomState(0)
n_samples, n_features = dataset_x_gene.shape
y = label_binarize(medium, classes=[0,1,2,3,4,5,6,7,8,9,10,11])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(dataset_x_gene, y, test_size=.8,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0, n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = ["red", "green", "blue", "yellow", "darkorange", "cyan", "purple", "pink", "black", "orange",
          "navy", "lightblue"]
plt.figure(figsize=(10,8))

lw = 2
for i in range(0, n_classes):
	plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw, label='ROC curve for class %d (AUC = %0.2f)' % (names[i] , roc_auc[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC Curve for Gene_Perturbation')
plt.legend(loc="lower right")
plt.show()

""" Plot PR curve with area value"""
# strain
strain = label_binarize(strain, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
n_classes = strain.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(dataset_x_strain, strain, test_size=.8,
                                                    random_state=random_state)
classifier = OneVsRestClassifier(svm.LinearSVC(random_state = 0))
classifier.fit(X_train, Y_train)
y_score = classifier.decision_function(X_test)
colors = ["red", "green", "blue", "yellow", "darkorange", "cyan", "purple", "pink", "black", "orange"]
precision = dict()
recall = dict()
average_precision = dict()

for i in range(0, n_classes):
	precision[i], recall[i], _ = precision_recall_curve(Y_test[:,i], y_score[:,i])
	average_precision[i] = average_precision_score(Y_test[:,i], y_score[:,i])

plt.figure(figsize=(10,8))
lines = []
labels = []
for i in range(0, n_classes):
	l, = plt.plot(recall[i], precision[i], color = colors[i], lw=2)
	lines.append(l)
	labels.append('Precision-recall for class {0} (area = {1:0.2f})'''.format(i+1, average_precision[i]))
fig = plt.gcf()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve for Strain class')
plt.legend(lines, labels, loc='lower right')
plt.show()

# medium
medium = label_binarize(medium, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
n_classes = medium.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(dataset_x_medium, medium, test_size=.8,
                                                    random_state=random_state)
classifier = OneVsRestClassifier(svm.LinearSVC(random_state = 0))
classifier.fit(X_train, Y_train)
y_score = classifier.decision_function(X_test)
colors = ["red", "green", "blue", "yellow", "darkorange", "cyan", "purple", "pink", "black", "orange",
          "navy", "lightblue", "darkgreen", "lightgreen", "lightpink", "darkblue", "darkcyan", "aqua"]
precision = dict()
recall = dict()
average_precision = dict()

for i in range(0, n_classes):
	precision[i], recall[i], _ = precision_recall_curve(Y_test[:,i], y_score[:,i])
	average_precision[i] = average_precision_score(Y_test[:,i], y_score[:,i])

plt.figure(figsize=(10,8))
lines = []
labels = []
for i in range(0, n_classes):
	l, = plt.plot(recall[i], precision[i], color = colors[i], lw=2)
	lines.append(l)
	labels.append('Precision-recall for class {0} (area = {1:0.2f})'''.format(i+1, average_precision[i]))
fig = plt.gcf()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve for Medium class')
plt.legend(lines, labels, loc='lower right')
plt.show()

# env
env_port = label_binarize(env_port, classes=[0,1,2,3,4,5,6,7])
n_classes = env_port.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(dataset_x_env, env_port, test_size=.8,
                                                    random_state=random_state)
classifier = OneVsRestClassifier(svm.LinearSVC(random_state = 0))
classifier.fit(X_train, Y_train)
y_score = classifier.decision_function(X_test)
colors = ["red", "green", "blue", "yellow", "darkorange", "cyan", "purple", "pink", "black"]
precision = dict()
recall = dict()
average_precision = dict()

for i in range(0, n_classes):
	precision[i], recall[i], _ = precision_recall_curve(Y_test[:,i], y_score[:,i])
	average_precision[i] = average_precision_score(Y_test[:,i], y_score[:,i])

plt.figure(figsize=(10,8))
lines = []
labels = []
for i in range(0, n_classes):
	l, = plt.plot(recall[i], precision[i], color = colors[i], lw=2)
	lines.append(l)
	labels.append('Precision-recall for class {0} (area = {1:0.2f})'''.format(i+1, average_precision[i]))
fig = plt.gcf()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve for Environmental_perturbation class')
plt.legend(lines, labels, loc='lower right')
plt.show()

# gene
gene_port = label_binarize(gene_port, classes=[0,1,2,3,4,5,6,7,8,9,10,11])
n_classes = gene_port.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(dataset_x_gene, gene_port, test_size=.8,
                                                    random_state=random_state)
classifier = OneVsRestClassifier(svm.LinearSVC(random_state = 0))
classifier.fit(X_train, Y_train)
y_score = classifier.decision_function(X_test)
colors = ["red", "green", "blue", "yellow", "darkorange", "cyan", "purple", "pink", "black", "orange",
          "navy", "lightblue"]
precision = dict()
recall = dict()
average_precision = dict()

for i in range(0, n_classes):
	precision[i], recall[i], _ = precision_recall_curve(Y_test[:,i], y_score[:,i])
	average_precision[i] = average_precision_score(Y_test[:,i], y_score[:,i])

plt.figure(figsize=(10,8))
lines = []
labels = []
for i in range(0, n_classes):
	l, = plt.plot(recall[i], precision[i], color = colors[i], lw=2)
	lines.append(l)
	labels.append('Precision-recall for class {0} (area = {1:0.2f})'''.format(i+1, average_precision[i]))
fig = plt.gcf()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve for Gene_perturbation class')
plt.legend(lines, labels, loc='lower right')
plt.show()

# Problem 5
# Combine them together
medium_and_stress = dataset[1:,2] + dataset[1:,3]
m_and_s = Encoder.fit_transform(medium_and_stress)
dataset_x_medium_and_stress = dataset_x

m_and_s_svc = LinearSVC(C = 0.0064, penalty = "l1", dual = False).fit(dataset_x_medium_and_stress, medium_and_stress)
strain_model = SelectFromModel(m_and_s_svc, prefit = True)
dataset_x_medium_and_stress = strain_model.transform(dataset_x_medium_and_stress)

""" Plot PRC curve """
clsp = list(i for i in range(0,18))
medium_and_stress = label_binarize(m_and_s, classes=clsp)
n_classes = medium_and_stress.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(dataset_x_medium_and_stress, medium_and_stress, test_size=.8,
                                                    random_state=random_state)
classifier = OneVsRestClassifier(svm.LinearSVC(random_state = 0))
classifier.fit(X_train, Y_train)
y_score = classifier.decision_function(X_test)

precision = dict()
recall = dict()
average_precision = dict()

colors = ["red", "green", "blue", "yellow", "darkorange", "cyan", "purple", "pink", "black", "orange",
          "navy", "lightblue", "darkgreen", "lightgreen", "lightpink", "darkblue", "darkcyan", "aqua"]

for i in range(0, n_classes):
	precision[i], recall[i], _ = precision_recall_curve(Y_test[:,i], y_score[:,i])
	average_precision[i] = average_precision_score(Y_test[:,i], y_score[:,i])

plt.figure(figsize=(10,8))
lines = []
labels = []
for i in range(0, n_classes):
	l, = plt.plot(recall[i], precision[i], color = colors[i], lw=2)
	lines.append(l)
	labels.append('Precision-recall for class {0} (area = {1:0.2f})'''.format(i+1, average_precision[i]))
fig = plt.gcf()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve for Medium and Stress class')
plt.legend(lines, labels, loc='lower right')
plt.show()

""" Plot ROC curve """

# medium

random_state = np.random.RandomState(0)
n_samples, n_features = dataset_x_medium_and_stress.shape
y = label_binarize(medium_and_stress, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(dataset_x_medium_and_stress, y, test_size=.8,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0, n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = ["red", "green", "blue", "yellow", "darkorange", "cyan", "purple", "pink", "black", "orange",
          "navy", "lightblue", "darkgreen", "lightgreen", "lightpink", "darkblue", "darkcyan", "aqua"]
plt.figure(figsize=(10,8))

lw = 2
for i in range(0, n_classes):
	plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw, label='ROC curve for Medium and Stress feature %d (AUC = %0.2f)' % (i + 1, roc_auc[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC Curve for Medium and Stress')
plt.legend(loc="lower right")
plt.show()

# Problem 6
"""PR Curve"""
# dimensionality reduction
clf_pca = PCA(n_components = 3)
clf_pca.fit(dataset_x)
strain_new_x = clf_pca.transform(dataset_x)
medium_new_x = clf_pca.transform(dataset_x)
env_new_x = clf_pca.transform(dataset_x)
gene_new_x = clf_pca.transform(dataset_x)
# strain
strain = label_binarize(strain, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
n_classes = strain.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(strain_new_x, strain, test_size=.8,
                                                    random_state=random_state)
classifier = OneVsRestClassifier(svm.LinearSVC(random_state = 0))
classifier.fit(X_train, Y_train)
y_score = classifier.decision_function(X_test)
colors = ["red", "green", "blue", "yellow", "darkorange", "cyan", "purple", "pink", "black", "orange"]
precision = dict()
recall = dict()
average_precision = dict()

for i in range(0, n_classes):
	precision[i], recall[i], _ = precision_recall_curve(Y_test[:,i], y_score[:,i])
	average_precision[i] = average_precision_score(Y_test[:,i], y_score[:,i])

plt.figure(figsize=(10,8))
lines = []
labels = []
for i in range(0, n_classes):
	l, = plt.plot(recall[i], precision[i], color = colors[i], lw=2)
	lines.append(l)
	labels.append('Precision-recall for class {0} (area = {1:0.2f})'''.format(i+1, average_precision[i]))
fig = plt.gcf()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve for Strain class with PCA = 3')
plt.legend(lines, labels, loc='lower right')
plt.show()
# medium
medium = label_binarize(medium, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
n_classes = medium.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(medium_new_x, medium, test_size=.8,
                                                    random_state=random_state)
classifier = OneVsRestClassifier(svm.LinearSVC(random_state = 0))
classifier.fit(X_train, Y_train)
y_score = classifier.decision_function(X_test)
colors = ["red", "green", "blue", "yellow", "darkorange", "cyan", "purple", "pink", "black", "orange",
          "navy", "lightblue", "darkgreen", "lightgreen", "lightpink", "darkblue", "darkcyan", "aqua"]
precision = dict()
recall = dict()
average_precision = dict()

for i in range(0, n_classes):
	precision[i], recall[i], _ = precision_recall_curve(Y_test[:,i], y_score[:,i])
	average_precision[i] = average_precision_score(Y_test[:,i], y_score[:,i])

plt.figure(figsize=(10,8))
lines = []
labels = []
for i in range(0, n_classes):
	l, = plt.plot(recall[i], precision[i], color = colors[i], lw=2)
	lines.append(l)
	labels.append('Precision-recall for class {0} (area = {1:0.2f})'''.format(i+1, average_precision[i]))
fig = plt.gcf()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve for Medium class with PCA = 3')
plt.legend(lines, labels, loc='lower right')
plt.show()

# env
env_port = label_binarize(env_port, classes=[0,1,2,3,4,5,6,7])
n_classes = env_port.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(env_new_x, env_port, test_size=.8,
                                                    random_state=random_state)
classifier = OneVsRestClassifier(svm.LinearSVC(random_state = 0))
classifier.fit(X_train, Y_train)
y_score = classifier.decision_function(X_test)
colors = ["red", "green", "blue", "yellow", "darkorange", "cyan", "purple", "pink", "black"]
precision = dict()
recall = dict()
average_precision = dict()

for i in range(0, n_classes):
	precision[i], recall[i], _ = precision_recall_curve(Y_test[:,i], y_score[:,i])
	average_precision[i] = average_precision_score(Y_test[:,i], y_score[:,i])

plt.figure(figsize=(10,8))
lines = []
labels = []
for i in range(0, n_classes):
	l, = plt.plot(recall[i], precision[i], color = colors[i], lw=2)
	lines.append(l)
	labels.append('Precision-recall for class {0} (area = {1:0.2f})'''.format(i+1, average_precision[i]))
fig = plt.gcf()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve for Environmental_perturbation class with PCA = 3')
plt.legend(lines, labels, loc='lower right')
plt.show()

# gene
gene_port = label_binarize(gene_port, classes=[0,1,2,3,4,5,6,7,8,9,10,11])
n_classes = gene_port.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(gene_new_x, gene_port, test_size=.8,
                                                    random_state=random_state)
classifier = OneVsRestClassifier(svm.LinearSVC(random_state = 0))
classifier.fit(X_train, Y_train)
y_score = classifier.decision_function(X_test)
colors = ["red", "green", "blue", "yellow", "darkorange", "cyan", "purple", "pink", "black", "orange",
          "navy", "lightblue"]
precision = dict()
recall = dict()
average_precision = dict()

for i in range(0, n_classes):
	precision[i], recall[i], _ = precision_recall_curve(Y_test[:,i], y_score[:,i])
	average_precision[i] = average_precision_score(Y_test[:,i], y_score[:,i])

plt.figure(figsize=(10,8))
lines = []
labels = []
for i in range(0, n_classes):
	l, = plt.plot(recall[i], precision[i], color = colors[i], lw=2)
	lines.append(l)
	labels.append('Precision-recall for class {0} (area = {1:0.2f})'''.format(i+1, average_precision[i]))
fig = plt.gcf()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve for Gene_perturbation class with PCA = 3')
plt.legend(lines, labels, loc='lower right')
plt.show()

""" ROC curve """

# strain
names = [1,2,3,4,5,6,7,8,9,10]
random_state = np.random.RandomState(0)
n_samples, n_features = dataset_x_strain.shape
y = label_binarize(strain, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(strain_new_x, y, test_size=.8,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0, n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = ["red", "green", "blue", "yellow", "darkorange", "cyan", "purple", "pink", "black", "orange"]
plt.figure(figsize=(10,8))
lw = 2
for i in range(0, n_classes):
	plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw, label='ROC curve for class %d (AUC = %0.2f)' % (names[i] , roc_auc[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC Curve for Strain with PCA = 3')
plt.legend(loc="lower right")
plt.show()

# medium
names = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
random_state = np.random.RandomState(0)
n_samples, n_features = dataset_x_medium.shape
y = label_binarize(medium, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(medium_new_x, y, test_size=.8,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0, n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = ["red", "green", "blue", "yellow", "darkorange", "cyan", "purple", "pink", "black", "orange",
          "navy", "lightblue", "darkgreen", "lightgreen", "lightpink", "darkblue", "darkcyan", "aqua"]
plt.figure(figsize=(10,8))

lw = 2
for i in range(0, n_classes):
	plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw, label='ROC curve for class %d (AUC = %0.2f)' % (names[i] , roc_auc[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC Curve for Medium with PCA = 3')
plt.legend(loc="lower right")
plt.show()

# env
names = [1,2,3,4,5,6,7,8]
random_state = np.random.RandomState(0)
n_samples, n_features = dataset_x_env.shape
y = label_binarize(env_port, classes=[0,1,2,3,4,5,6,7])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(env_new_x, y, test_size=.8,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0, n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = ["red", "green", "blue", "yellow", "darkorange", "cyan", "purple", "pink"]
plt.figure(figsize=(10,8))

lw = 2
for i in range(0, n_classes):
	plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw, label='ROC curve for class %d (AUC = %0.2f)' % (names[i] , roc_auc[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC Curve for Environmental_perturbation with PCA = 3')
plt.legend(loc="lower right")
plt.show()

# gene
names = [1,2,3,4,5,6,7,8,9,10,11,12]
random_state = np.random.RandomState(0)
n_samples, n_features = dataset_x_gene.shape
y = label_binarize(gene_port, classes=[0,1,2,3,4,5,6,7,8,9,10,11])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(gene_new_x, y, test_size=.8,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0, n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = ["red", "green", "blue", "yellow", "darkorange", "cyan", "purple", "pink", "black", "orange",
          "navy", "lightblue"]
plt.figure(figsize=(10,8))

lw = 2
for i in range(0, n_classes):
	plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw, label='ROC curve for class %d (AUC = %0.2f)' % (names[i] , roc_auc[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC Curve for Gene_Perturbation with PCA = 3')
plt.legend(loc="lower right")
plt.show()
