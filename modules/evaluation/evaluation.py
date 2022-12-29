from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as ssn

class model_evaluation:

    def evaluation(model,train_set,test_set):
        eva1 = model.evaluate(test_set)
        print('Test Accuracy of model: {}'.format(eva1[1]))

        pred = model.predict(test_set)
        thresh = 0.5
        classes_x = [1 if p >= thresh else 0 for p in pred]

        #train_set.labels

        cm = confusion_matrix(test_set.labels, classes_x, normalize='pred')
        plt.figure(figsize = (10, 10))
        ssn.heatmap(cm, annot = True)

