
import numpy as np
from flask import Flask,request
from flask_cors import CORS
from sklearn.externals import joblib

app = Flask(__name__)

cors = CORS(app)


@app.route('/pred',methods=['Post'])
def predict():
    # ml = joblib.load('Knn_Down.pkl')
    # ml1 = joblib.load('rf_Down.pkl')
    # ml2 = joblib.load('dTree_Down.pkl')
    # ml3 = joblib.load('bernoli_Down.pkl')

    data = request.get_json()
    # print(data['Employer_Name'])
    x = data['model_choice']
    if x == 1:
        ml = joblib.load('bernoli_Down.pkl')
    elif x == 2:
        ml = joblib.load('Knn_Down.pkl')
    elif x == 3:
        ml = joblib.load('dTree_Down.pkl')
    else:
        ml = joblib.load('rf_Down.pkl')




    en = data['Employer_Name']

    es = data['Employer_State']
    sc = data['SOC_Code']
    nc = data['NAICS_code']
    tow = data['Total_workers']
    fp = data['FullTime_Position']
    hd = data['H1b_dependent']
    wv = data['Willful_Violator']
    twa = data['Total_wage']
    # print(en,es,sc,nc,tow,fp,hd,wv,twa)


    #[58802,53,84,1505,1,1,0,0,0,6654] --input
    #[0] --op
    # a = ml.predict(np.array([[58802,53,84,1505,1,1,0,0,6654]]).reshape(1,9))
    a = ml.predict(np.array([[en,es,sc , nc, tow ,fp , hd, wv, twa]]).reshape(1, 9))
    # b = str(a.item(0))
    print(a)
    if a.item(0) == 0:
        response = 'Approved'
    else:
        response = 'Denied'
    return response, 200


if __name__=='__main__':
    app.run(host="0.0.0.0",port=90)


