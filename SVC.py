from sklearn.svm import SVC
from sklearn.metrics import f1_score


# train in SVM with RBF kernel
clf = SVC()
clf.fit(train_fc_features, train_labels)

# check results
predict_train = clf.predict(train_fc_features)
print ("This is the train data fit accuracy: ", f1_score(train_labels, predict_train, average='macro'))
devel = snore_data_extractor("./snore_segments/", one_hot=False, data_mode="devel", resize=(224, 224))
devel_features, devel_labels = devel.full_data()
devel_fc_features = model.predict(np.array(devel_features))
predict_devel = clf.predict(devel_fc_features)
print ("This is the devel data fit accuracy: ", f1_score(devel_labels, predict_devel, average='macro'))
