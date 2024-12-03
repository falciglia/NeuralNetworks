import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def compute_accuracy(predictions, list_predictions, list_labels):
    cnt = 0
    for i in range(len(predictions)):
        if np.argmax(list_predictions[i]) == list_labels[i]:
            cnt = cnt + 1
            
    accuracy = cnt/len(predictions)
    return accuracy


def compute_confusion_matrix(list_predictions, list_labels, classes):
    # Confusion matrix
    all_predictions = []
    for i in range(len(list_predictions)):
        all_predictions.append(np.argmax(list_predictions[i]))
    all_predictions = np.array(all_predictions)
    all_labels = np.array(list_labels)
    
    cf_matrix = confusion_matrix(all_labels, all_predictions)
    df_cm = cmn = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    #df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*10, index=[i for i in classes], columns=[i for i in classes])
    #df_cm.drop(["O"], axis = 1, inplace = True)
    #df_cm.drop(["O"], axis = 0, inplace = True)
    plt.figure(figsize=(12,7))
    sns.heatmap(df_cm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes)
    plt.savefig('/home/s.falciglia/copia-da-prod/salvatore-backup-advanced/RieManiSpectraNet_going_to_github/saved_pics/confusion_matrix.png')# format="pdf", dpi=600,)
    plt.show()
    plt.close()