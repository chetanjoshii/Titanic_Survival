from flask import Flask, jsonify, render_template, request
from project_app.utils import TitanicSurvival
import config

app = Flask(__name__)

########################################################################################
################################## Model API ############################################
########################################################################################

@app.route('/predict_survival', methods = ['POST','GET'])
def get_survival_titanic():
    if request.method == 'POST':
        print('We are in POST Method')
        data = request.form  
        Pclass = eval(data['Pclass'])
        Gender = data['Gender']
        Age = eval(data['Age'])
        SibSp = eval(data['SibSp'])
        Parch = eval(data['Parch'])
        Fare = eval(data['Fare'])
        Embarked = data['Embarked']   
        
        titanic_survival = TitanicSurvival(Pclass, Gender, Age, SibSp, Parch, Fare,Embarked)
        survived = titanic_survival.get_survival_prediction()
        return jsonify({'Result': f'Prediction for Survival is {survived}'})

    else:
        pass

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = config.PORT_NUMBER, debug =True)