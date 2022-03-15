#import relevant data and split data
import pandas as pd
from sklearn.model_selection import train_test_split

df_modified=pd.read_csv(r'df_modified.csv', engine='python').sample(frac=1).reset_index(drop=True) #cleaned dataset, with some rows (with outliers) removed
print(df_modified.shape)
data_best = [df_modified["Dexa_During_Rx_Y"] , df_modified["Comorb_Encounter_For_Screening_For_Malignant_Neoplasms_Y"], df_modified["Comorb_Encounter_For_Immunization_Y"] , df_modified["Comorb_Encntr_For_General_Exam_W_O_Complaint,_Susp_Or_Reprtd_Dx_Y"] , df_modified["Comorb_Long_Term_Current_Drug_Therapy_Y"] , df_modified["Concom_Viral_Vaccines_Y"], df_modified["Persistency_Flag_Persistent"]]
headers = ["Dexa_During_Rx_Y" , "Comorb_Encounter_For_Screening_For_Malignant_Neoplasms_Y", "Comorb_Encounter_For_Immunization_Y" ,"Comorb_Encntr_For_General_Exam_W_O_Complaint,_Susp_Or_Reprtd_Dx_Y" , "Comorb_Long_Term_Current_Drug_Therapy_Y" , "Concom_Viral_Vaccines_Y", "Persistency_Flag_Persistent"]
df_best=pd.concat(data_best, axis=1, keys=headers)
print(df_best.shape)

X=df_best.loc[:,df_best.columns!="Persistency_Flag_Persistent"]
y=df_best.loc[:,df_best.columns=="Persistency_Flag_Persistent"].values.ravel()
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.30,random_state=123) 


#train neural network with the optimal paramters we found

from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import load_model
def load_model():
    # loading model
    neural_network_best_model = model_from_json(open('neural_network_best_model.json').read())
    neural_network_best_model.load_weights('neural_network_best_model.h5')
    neural_network_best_model.compile(loss='categorical_crossentropy', optimizer='adam')
    return neural_network_best_model

def create_nn(optimizer='uniform', init='adam'):
    nn = Sequential()
    nn.add(Dense(5, input_dim=6, 
                 activation='relu')) #let's use 2/3 size of input layer + size of output layer for number of nodes
    nn.add(Dense(5, activation='relu'))
    nn.add(Dense(1, activation='relu'))
    nn.compile(loss='binary_crossentropy'#, optimizer='adam', metrics='accuracy'
              )
    return nn

neural_network_best_model=KerasClassifier(build_fn=create_nn, verbose=0,batch_size=5, epochs=150, #init='uniform', optimizer='adam'
                     )
neural_network_best_model._estimator_type="classifier"
neural_network_best_model.fit(X_train,y_train)


import numpy as np
from flask import Flask, request, render_template
# from joblib import load

app=Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    features=[float(x) for x in request.form.values()]
    final_features=np.array(features).reshape(1,6)
    prediction=neural_network_best_model.predict(final_features)
    pred_round=np.round(prediction)[0][0]
    
    output=""
    if pred_round>=1:
          output+="persistent"
        
    else:
          output+="non-persistent"
       

    return(render_template('index.html', prediction_text='This patient is {}'.format(output)                      
                              ))
if __name__=="__main__":
    app.run(port=5000, debug=True, use_reloader=False)
#print(final_features.shape)


