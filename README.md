# ODC-ML-DGS


model_cl.set_params(alpha=alpha_cl_ERREUR)

model_cl.fit(X_train_cl,y_train_cl)

cl_score = model_cl.score(X_train_cl,y_train_cl)
tmp_test_predict = model_cl.predict(X_train_cl)
cl_accuracy = metrics.accuracy_score(y_train_cl,tmp_test_predict)
cl_log_loss = metrics.log_loss(y_train_cl,tmp_test_predict)
cl_hinge_loss = metrics.hinge_loss(y_train_cl,tmp_test_predict) #Renvoie la perte de charnière moyenne.
cl_hamming_loss = metrics.hamming_loss(y_train_cl,tmp_test_predict) #Renvoie la perte de Hamming moyenne
cl_confusion_matrix = metrics.confusion_matrix(y_train_cl,tmp_test_predict) #Calculer la matrice de confusion pour évaluer la précision d'un classifieur.
cl_f1_score = metrics.f1_score(y_train_cl,tmp_test_predict) #Calcul le score F1 (moyenne harmonique de la précision et du rappel)
cl_recall_score = metrics.recall_score(y_train_cl,tmp_test_predict)

cl_score, cl_accuracy, cl_log_loss, cl_hinge_loss, cl_hamming_loss, cl_confusion_matrix, cl_f1_score, cl_recall_score
# cl_confusion_matrix
