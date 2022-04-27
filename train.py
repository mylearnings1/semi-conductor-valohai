import uuid

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import valohai

def main():
  
  valohai.prepare(
        step='train-model',
        image='valohai/sklearn:1.0',
        default_inputs={
          'dataset1': 'datum://018069f2-73d9-608d-6ea6-50449d188804',
          'dataset2': 'datum://018069f2-771f-7265-9e83-479a47f9a893',
          'dataset3': 'datum://018069f2-7561-9ef9-50b9-f4fefd2476a2',
          'dataset4': 'datum://018069f2-78bc-5568-d2ab-ca99305ed284',
        },
    )
  
  x_train = pd.read_csv(valohai.inputs('dataset1').path())
  y_train = pd.read_csv(valohai.inputs('dataset2').path())
  x_test = pd.read_csv(valohai.inputs('dataset3').path())
  y_test = pd.read_csv(valohai.inputs('dataset4').path())
  
  rf = RandomForestClassifier(n_estimators=100, random_state=1,verbose=0)
  rf.fit(x_train, y_train)
  y_pred2 = rf.predict(x_test)
  test_accuracy_rf = rf.score(y_test,y_pred2)*100
  
  with valohai.logger() as logger:
      logger.log('test_accuracy_rf', test_accuracy_rf)
      
  # printing the confusion matrix
  #cm = confusion_matrix(y_test_os, y_pred)
  #sns.heatmap(cm, annot = True, cmap = 'rainbow')
  #print("Accuracy: ", lr.score(x_test_os,y_test_os)*100)
  #cm = confusion_matrix(y_test_os, y_pred)
  #sns.heatmap(cm, annot = True, cmap = 'rainbow')
  suffix = uuid.uuid4()
  output_path = valohai.outputs().path('model_rf.h5')
  rf.save(output_path)

if __name__ == '__main__':
    main()
