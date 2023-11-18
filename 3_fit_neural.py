# -*- coding: utf-8 -*-

# from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, merge, Lambda
from keras.models import Model
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils import np_utils
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from keras.regularizers import l2, activity_l2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import confusion_matrix as cm
import pandas as pd
import roc
from collections import Counter
print('Loading normalized data from HDF5...')
import h5py
h5f = h5py.File('data.h5', 'r')

normalized_X = h5f['normalized_X'][()]
labeled_Y = h5f['labeled_Y'][()]
Y = h5f['Y'][()]
# Y_A = h5f['Y_A'][()]
h5f.close()
# print(normalized_X)
Y = pd.DataFrame(Y)

Y = Y[:10000][:]
normalized_X = normalized_X[:10000]
labeled_Y = labeled_Y[:10000]

#根据相关性，选择特征
multi_data = pd.DataFrame(normalized_X)
# label encoding (0,1,2,3,4,5,6,7,8) multi-class labels
le2 = preprocessing.LabelEncoder()
enc_label = Y.apply(le2.fit_transform)
multi_data['label'] = enc_label
# YY = multi_data['label']
# print(multi_data.info())

# Correlation Matrix for Multi-class Labels
num_col = list(multi_data.select_dtypes(include='number').columns)
# Correlation Matrix for Multi-class Labels
plt.figure(figsize=(20, 8))
# 计算相关性系数
corr_multi = multi_data[num_col].corr()
# print(corr_multi['label'])
sns.heatmap(corr_multi,vmax=1.0,annot=False)
plt.title('Correlation Matrix for Multi Labels',fontsize=16)
plt.savefig('./res/correlation_matrix_multi.png')
# plt.show()
# finding the attributes which have more than 0.3 correlation with encoded attack label attribute

# print(corr_multi['label'])
corr_ymulti = abs(corr_multi['label'])
print("特征相关系数排行:", len(corr_ymulti), corr_ymulti.sort_values(ascending=True))
del corr_ymulti['label']
del corr_ymulti[102]
# corr_ymulti.drop(labels=102, axis=1, inplace=True)
print(len(corr_ymulti), corr_ymulti)
cc = corr_ymulti.sort_values(ascending=True)
# highest = cc[-30:]  #取后20个
highest = cc[:30]  #取后20个

# highest_corr_multi = corr_ymulti[corr_ymulti > 0.04]
highest_corr_multi = highest
print(highest_corr_multi.sort_values(ascending=True))
# selecting attributes found by using pearson correlation coefficient
multi_cols = highest_corr_multi.index
print(multi_cols)

# Multi-class labelled Dataset
multi_data = multi_data[multi_cols].copy()
# print(multi_data)
multi_data.to_csv('./datasets/res/multi_data.csv')

#  MULTI-CLASS CLASSIFICATION
  # Data Splitting
# X = multi_data.drop(columns=['label'], axis=1)
X = np.array(multi_data)
print(X.shape)

# YY = labeled_Y

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=100)
print(X_train.shape)
print(X_test.shape)
num_of_features = X_train.shape[1]
nb_classes = Y_train.shape[1]


h5f = h5py.File('datasets.h5', 'r')
X_train = h5f['X_nn_train'][()]
Y_train = h5f['Y_nn_train'][()].astype(np.float32)
X_test = h5f['X_nn_test'][()]
Y_test = h5f['Y_nn_test'][()].astype(np.float32)

X_train2 = h5f['X_rf_train'][()]
Y_train2 = h5f['Y_rf_train'][()].astype(np.float32)
X_test2 = h5f['X_rf_test'][()]
Y_test2 = h5f['Y_rf_test'][()].astype(np.float32)
h5f.close()

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from sklearn.ensemble import ExtraTreesClassifier

print('Training ExtraTreesClassifier for "attack or not" labels...')
model2 = ExtraTreesClassifier(n_estimators=31, criterion='entropy')
model2 = model2.fit(X_train2, Y_train2)

Y_pred2 = model2.predict_proba(X_test2)[:,1]

print('Testing accuracy...')
score2 = accuracy_score(Y_test2, np.around(Y_pred2))
print(score2)
print(classification_report(Y_test2, np.around(Y_pred2)))

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
    for i in range(len(y_hat)): 
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==0:
           TN += 1
    for i in range(len(y_hat)): 
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

TP, FP, TN, FN = perf_measure(np.around(Y_pred2), Y_test2)

fp_rate = FP/(TN+FP)
tn_rate = TN/(TN+FP)

accuracy = (TN+TP)/(TN+FP+TP+TN)
precision = TP/(TN+FP)
hitrate = TP/(TN+FN)

print('TP:', TP, 'FP:', FP, 'TN:', TN, 'FN:', FN)
print('Accuracy:', accuracy)
print('False Positive rate:', fp_rate, 'True Negative Rate', tn_rate)

def to_cat(y):
    y_tmp = np.ndarray(shape=(y.shape[0], 2), dtype=np.float32)
    for i in range(y.shape[0]):
        y_tmp[i, :] = np.array([1-y[i], y[i]])   # np.array([0,1]) if y[i] else np.array([1,0])
    return y_tmp

cm.plot_confusion_matrix(Y_test2, np.round(Y_pred2), classes=list(range(2)),
                          normalize=True,
                          title='"Attack or not" confusion matrix')
roc.plot_roc_curve(to_cat(Y_test2), to_cat(Y_pred2), 2, 0, "roc2", title='Receiver operating characteristic (attack_or_not = 0)')
roc.plot_roc_curve(to_cat(Y_test2), to_cat(Y_pred2), 2, 1, "roc1", title='Receiver operating characteristic (attack_or_not = 1)')


print('Combining predicted "attack or not" labels to neural network testing data...')
X_test = np.concatenate((Y_pred2[:, np.newaxis], X_test), axis=1)


print('Creating neural network...')
num_of_features = X_train.shape[1]
nb_classes = Y_train.shape[1]

def baseline_model():
    def branch2(x):
        # x = Dense(int(np.floor(num_of_features*50)), activation='sigmoid')(x)
        # x = Dropout(0.75)(x)
        #
        # x = Dense(int(np.floor(num_of_features*20)), activation='sigmoid')(x)
        # x = Dropout(0.5)(x)
        #
        # x = Dense(int(np.floor(num_of_features)), activation='sigmoid')(x)
        # x = Dropout(0.1)(x)
        x = Dense(1024, input_dim=41, activation='relu')(x)
        x = Dropout(0.1)(x)

        x = Dense(768, activation='sigmoid')(x)
        x = Dropout(0.01)(x)

        x = Dense(512, activation='sigmoid')(x)
        x = Dropout(0.01)(x)

        x = Dense(256, activation='sigmoid')(x)
        x = Dropout(0.01)(x)

        x = Dense(128, activation='sigmoid')(x)
        x = Dropout(0.01)(x)

        return x
    
    main_input = Input(shape=(num_of_features,), name='main_input')
    x = main_input
    x = branch2(x)
    main_output = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_crossentropy'])
    return model

model = baseline_model()

print('Training neural network...')
history = model.fit(X_train, Y_train,
                    epochs=100,
                    batch_size=128
                    )

print('Plotting training history data...')
print(history.history.keys())

from  epoch_history_plot import  plot_hist

plot_hist(history, ['loss', 'accuracy'])

# summarize history for accuracy
import matplotlib.pyplot as plt
plt.figure()
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig('./res/acc.png')

plt.figure()
# summarize history for loss
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig('./res/loss.png')

print('Testing neural network...')
Y_predicted = model.predict(X_test)

max_probs = np.argmax(Y_predicted, axis=1)
Y_pred = np.zeros(Y_predicted.shape)
for row, col in enumerate(max_probs):
    Y_pred[row,col] = 1

# score = accuracy_score(Y_test, Y_pred)
# print(score)

nn_a = accuracy_score(Y_test, Y_pred)
print("nn Accuracy:", nn_a)
nn_f = f1_score(Y_test, Y_pred, average='weighted')
print("nn F1 Score:", nn_f)
nn_p = precision_score(Y_test,Y_pred, average='weighted')
print("nn Precision: ", nn_p)
nn_r = recall_score(Y_test,Y_pred,average = 'weighted')
print("nn Recall: ", nn_r)

print(classification_report(Y_test.argmax(axis=-1), Y_pred.argmax(axis=-1)))


cm.plot_confusion_matrix(Y_test.argmax(axis=-1), Y_pred.argmax(axis=-1), classes=list(range(10)),
                          normalize=True,
                          title='Confusion matrix')

print('Saving neural network model...')
json_string = model.to_json()
with open('neural_model1.json', 'w') as f:
    f.write(json_string)
model.save_weights('neural_model_weights1.h5')

model.save('neural_model1.h5')

roc.plot_roc_curve(Y_test, Y_predicted, nb_classes, 6, "nb6", title='Receiver operating characteristic (class 6)')
roc.plot_roc_curve(Y_test, Y_predicted, nb_classes, 4, "nb4", title='Receiver operating characteristic (class 4)')
roc.plot_roc_curve(Y_test, Y_predicted, nb_classes, 2, "nb2", title='Receiver operating characteristic (class 2)')
roc.plot_roc_curve(Y_test, Y_predicted, nb_classes, 0, "nb1", title='Receiver operating characteristic (class 0)')


model3 = ExtraTreesClassifier(n_estimators=5, criterion='entropy')
print('Fitting...')
model3 = model2.fit(X_train, Y_train.argmax(axis=-1))
print('Predicting...')
Y_predicted3 = model3.predict(X_test)

print('Testing accuracy...')
score3 = accuracy_score(Y_test.argmax(axis=-1), Y_predicted3)
print(score3)
print(classification_report(Y_test.argmax(axis=-1), Y_predicted3))

cm.plot_confusion_matrix(Y_test.argmax(axis=-1), Y_predicted3, classes=list(range(10)),
                          normalize=True,
                          title='Extratrees Confusion matrix')



print('Saving X and Y to HDF5')

h5f = h5py.File('results.h5', 'w')
h5f.create_dataset('Y_predicted', data=Y_pred)
h5f.create_dataset('Y_expected', data=Y_test)
h5f.close()


#-------------------
y_train = Y_train
y_test = Y_test

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
dic = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(dic)

label_encoder = preprocessing.LabelEncoder()
y_test = label_encoder.fit_transform(y_test)
dic = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(dic)
print(y_test)

x_train = X_train
x_test = X_test

print(Counter(y_train))
print(Counter(y_test))

from sklearn import tree
model = tree.DecisionTreeClassifier(criterion="entropy")
model.fit(x_train, y_train)

from sklearn.metrics import confusion_matrix
y_pred=model.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(13,7))
sns.heatmap(cm, annot =True,fmt='g')
# plt.show()
plt.savefig('./res/dt.png')

print("Classification Metrices : ")
dt_a = accuracy_score(y_test, y_pred)
print("DT Accuracy:", dt_a)
dt_f = f1_score(y_test, y_pred, average='weighted')
print("DT F1 Score:", dt_f)
dt_p = precision_score(y_test,y_pred, average='weighted')
print("DT Precision: ", dt_p)
dt_r = recall_score(y_test,y_pred,average = 'weighted')
print("DT Recall: ", dt_r)

from sklearn.naive_bayes import MultinomialNB
nbmodel = MultinomialNB()
nbmodel.fit(x_train,y_train)
from sklearn.metrics import confusion_matrix
y_pred=nbmodel.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(13,7))
sns.heatmap(cm, annot =True,fmt='g')
plt.savefig('./res/mnb.png')

print("Classification Metrices : ")
nb_a = accuracy_score(y_test, y_pred)
print("MNB Accuracy:", nb_a)
nb_f = f1_score(y_test, y_pred, average='weighted')
print("MNB F1 Score:", nb_f)
nb_p = precision_score(y_test,y_pred, average='weighted')
print("MNB Precision: ", nb_p)
nb_r = recall_score(y_test,y_pred,average = 'weighted')
print("MNB Recall: ", nb_r)


from sklearn.ensemble import RandomForestClassifier
classifer_rf=RandomForestClassifier(random_state=42,n_jobs=-1,max_depth=5,n_estimators=100,oob_score=True)
classifer_rf.fit(x_train,y_train)
classifer_rf.score(x_test,y_test)
from sklearn.metrics import confusion_matrix
y_pred=classifer_rf.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(13,7))
sns.heatmap(cm, annot =True,fmt='g')
plt.savefig('./res/rf.png')

print("Classification Metrices : ")
rf_a = accuracy_score(y_test, y_pred)
print("RF Accuracy:", rf_a)
rf_f = f1_score(y_test, y_pred, average='weighted')
print("RF F1 Score:", rf_f)
rf_p = precision_score(y_test,y_pred, average='weighted')
print("RF Precision: ", rf_p)
rf_r = recall_score(y_test,y_pred,average = 'weighted')
print("RF Recall: ", rf_r)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2, n_jobs=10)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(13,7))
sns.heatmap(cm, annot =True,fmt='g')
plt.savefig('./res/knn.png')

print("Classification Metrices : ")
knn_a = accuracy_score(y_test, y_pred)
print("KNN Accuracy:", knn_a)
knn_f = f1_score(y_test, y_pred, average='weighted')
print("KNN F1 Score:", knn_f)
knn_p = precision_score(y_test,y_pred, average='weighted')
print("KNN Precision: ", knn_p)
knn_r = recall_score(y_test,y_pred,average = 'weighted')
print("KNN Recall: ", knn_r)

import xgboost as xgb
params={'base_score': 0.5,
  'booster': 'gbtree',
  'tree_method': 'auto',
  'colsample_bylevel': 1,
  'colsample_bynode': 1,
  'colsample_bytree': 1,
  'gamma': 0,
  'learning_rate': 0.1,
  'max_delta_step': 0,
  'max_depth': 2,
  'min_child_weight': 1,
  'missing': None,
  'n_estimators': 100,
  'n_jobs': 1,
  'nthread': 2,
  'objective': 'multi:softprob',
  'random_state': 0,
  'reg_alpha': 0,
  'reg_lambda': 1,
  'scale_pos_weight': 1,
  'seed': 0,
  'silent': 1,
  'subsample': 1,
  'verbosity': 1,
  'eta': 0.1,
  'num_class': nb_classes}

xgb = xgb.XGBClassifier(params=params)


xgb.fit(x_train, y_train, eval_metric="mlogloss")
xgb.score(x_test, y_test)
y_pred=xgb.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(13,7))
sns.heatmap(cm, annot=True, fmt='g')
plt.savefig('./res/xgb.png')

print("Classification Metrices : ")
xgb_a = accuracy_score(y_test, y_pred)
print("xgb Accuracy:", xgb_a)
xgb_f = f1_score(y_test, y_pred, average='weighted')
print("xgb F1 Score:", xgb_f)
xgb_p = precision_score(y_test, y_pred, average='weighted')
print("xgb Precision: ", xgb_p)
xgb_r = recall_score(y_test, y_pred,average='weighted')
print("xgb Recall: ", xgb_r)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=42)
lr.fit(x_train, y_train)
lr.score(x_test, y_test)
y_pred=lr.predict(x_test)
cm=confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(13,7))
sns.heatmap(cm, annot=True, fmt='g')
plt.savefig('./res/lr.png')

print("Classification Metrices : ")
lr_a = accuracy_score(y_test, y_pred)
print("lr Accuracy:", lr_a)
lr_f = f1_score(y_test, y_pred, average='weighted')
print("lr F1 Score:", lr_f)
lr_p = precision_score(y_test, y_pred, average='weighted')
print("lr Precision: ", lr_p)
lr_r = recall_score(y_test, y_pred,average='weighted')
print("lr Recall: ", lr_r)

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.0001, max_iter=500, random_state=42, verbose=1,
                    n_jobs=256)
sgd.fit(x_train, y_train)
sgd.score(x_test, y_test)
y_pred = sgd.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(13, 7))
sns.heatmap(cm, annot=True, fmt='g')
plt.savefig('./res/sgd.png')

print("Classification Metrices : ")
sgd_a = accuracy_score(y_test, y_pred)
print("sgd Accuracy:", sgd_a)
sgd_f = f1_score(y_test, y_pred, average='weighted')
print("sgd F1 Score:", sgd_f)
sgd_p = precision_score(y_test, y_pred, average='weighted')
print("sgd Precision: ", sgd_p)
sgd_r = recall_score(y_test, y_pred,average='weighted')
print("sgd Recall: ", sgd_r)
''''''

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=200)
gbc.fit(x_train, y_train)
gbc.score(x_test, y_test)
y_pred = gbc.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(13, 7))
sns.heatmap(cm, annot=True, fmt='g')
plt.savefig('./res/gbc.png')

print("Classification Metrices : ")
gbc_a = accuracy_score(y_test, y_pred)
print("gbc Accuracy:", gbc_a)
gbc_f = f1_score(y_test, y_pred, average='weighted')
print("gbc F1 Score:", gbc_f)
gbc_p = precision_score(y_test, y_pred, average='weighted')
print("gbc Precision: ", gbc_p)
gbc_r = recall_score(y_test, y_pred,average='weighted')
print("gbc Recall: ", gbc_r)

from sklearn.ensemble import AdaBoostClassifier
ac = AdaBoostClassifier()
ac.fit(x_train, y_train)
ac.score(x_test, y_test)
y_pred = ac.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(13, 7))
sns.heatmap(cm, annot=True, fmt='g')
plt.savefig('./res/ac.png')

print("Classification Metrices : ")
ac_a = accuracy_score(y_test, y_pred)
print("ac Accuracy:", ac_a)
ac_f = f1_score(y_test, y_pred, average='weighted')
print("ac F1 Score:", ac_f)
ac_p = precision_score(y_test, y_pred, average='weighted')
print("ac Precision: ", ac_p)
ac_r = recall_score(y_test, y_pred,average='weighted')
print("ac Recall: ", ac_r)

from sklearn.svm import SVC
svm = SVC(kernel='linear')
svm.fit(x_train, y_train)
svm.score(x_test, y_test)
y_pred=svm.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(13,7))
sns.heatmap(cm, annot =True,fmt='g')
plt.savefig('./res/svm.png')

print("Classification Metrices : ")
svm_a = accuracy_score(y_test, y_pred)
print("svm Accuracy:", svm_a)
svm_f = f1_score(y_test, y_pred, average='weighted')
print("svm F1 Score:", svm_f)
svm_p = precision_score(y_test,y_pred, average='weighted')
print("svm Precision: ", svm_p)
svm_r = recall_score(y_test,y_pred,average = 'weighted')
print("svm Recall: ", svm_r)

