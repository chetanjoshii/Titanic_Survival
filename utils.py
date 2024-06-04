import pickle
import json
import numpy as np
import config

class TitanicSurvival(): 

    def __init__(self, Pclass, Gender, Age, SibSp, Parch, Fare,Embarked):
        self.Pclass = Pclass
        self.Gender = Gender
        self.Age = Age
        self.SibSp = SibSp
        self.Parch = Parch
        self.Fare = Fare
        self.Embarked = Embarked

    def load_model(self):
        with open(config.MODEL_FILE_PATH, 'rb') as f:
            self.model = pickle.load(f)

        with open(config.JSON_FILE_PATH, 'r') as f:
            self.project_data = json.load(f)

    def get_survival_prediction(self):
        self.load_model()

        test_array = np.zeros(len(self.project_data['columns']))
        test_array[0] = self.Pclass
        test_array[1] = self.project_data['Gender'][self.Gender]
        test_array[2] = self.Age
        test_array[3] = self.SibSp
        test_array[4] = self.Parch
        test_array[5] = self.Fare
        test_array[6] = self.project_data['Embarked'][self.Embarked]
        print('Test Array :', test_array)

        predicted_survival = self.model.predict([test_array])[0]
        print(f'Prediction for Survival is:',predicted_survival)
        return predicted_survival

        

if __name__ == '__main__':
    Pclass = 3
    Gender = 'male'
    Age = 29
    SibSp = 1
    Parch = 1
    Fare = 17.20
    Embarked = 'Q'
    titanic_survival = TitanicSurvival(Pclass, Gender, Age, SibSp, Parch, Fare,Embarked)
    titanic_survival.get_survival_prediction()