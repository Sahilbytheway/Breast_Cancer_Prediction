from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Label
from kivy.properties import ObjectProperty
from kivy.uix.listview import ListItemButton

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
class StudentListButton(ListItemButton):
    pass


class BreastCancer(BoxLayout):
    radius_mean_text_input = ObjectProperty()
    texture_mean_text_input = ObjectProperty()
    perimeter_mean_text_input = ObjectProperty()
    area_mean_text_input = ObjectProperty()
    smoothness_mean_text_input = ObjectProperty()
    compactness_mean_text_input = ObjectProperty()
    concavity_mean_text_input = ObjectProperty()
    concave_points_mean_text_input = ObjectProperty()
    symmetry_mean_text_input = ObjectProperty()
    fractal_dimension_mean_text_input = ObjectProperty()

    radius_standard_error_text_input = ObjectProperty()
    texture_standard_error_text_input = ObjectProperty()
    perimeter_standard_error_text_input = ObjectProperty()
    area_standard_error_text_input = ObjectProperty()
    smoothness_standard_error_text_input = ObjectProperty()
    compactness_standard_error_text_input = ObjectProperty()
    concavity_standard_error_text_input = ObjectProperty()
    concave_points_standard_error_text_input = ObjectProperty()
    symmetry_standard_error_text_input = ObjectProperty()
    fractal_dimension_standard_error_text_input = ObjectProperty()

    radius_worst_text_input = ObjectProperty()
    texture_worst_text_input = ObjectProperty()
    perimeter_worst_text_input = ObjectProperty()
    area_worst_text_input = ObjectProperty()
    smoothness_worst_text_input = ObjectProperty()
    compactness_worst_text_input = ObjectProperty()
    concavity_worst_text_input = ObjectProperty()
    concave_points_worst_text_input = ObjectProperty()
    symmetry_worst_text_input = ObjectProperty()
    fractal_dimension_worst_text_input = ObjectProperty()

    student_list = ObjectProperty()

    def submit_report(self):
        student_name = '242423'+" "+'B'+" "+self.radius_mean_text_input.text+" "+self.texture_mean_text_input.text+" "+self.perimeter_mean_text_input.text+" "+self.area_mean_text_input.text+" "+self.smoothness_mean_text_input.text+" "+self.compactness_mean_text_input.text+" "+self.concavity_mean_text_input.text+" "+self.concave_points_mean_text_input.text+" "+self.symmetry_mean_text_input.text+" "+self.fractal_dimension_mean_text_input.text+" "+self.radius_standard_error_text_input.text+" "+self.texture_standard_error_text_input.text+" "+self.perimeter_standard_error_text_input.text+" "+self.area_standard_error_text_input.text+" "+self.smoothness_standard_error_text_input.text+" "+self.compactness_standard_error_text_input.text+" "+self.concavity_standard_error_text_input.text+" "+self.concave_points_standard_error_text_input.text+" "+self.symmetry_standard_error_text_input.text+" "+self.fractal_dimension_standard_error_text_input.text+" "+self.radius_worst_text_input.text+" "+self.texture_worst_text_input.text+" "+self.perimeter_worst_text_input.text+" "+self.area_worst_text_input.text+" "+self.smoothness_worst_text_input.text+" "+self.compactness_worst_text_input.text+" "+self.concavity_worst_text_input.text+" "+self.concave_points_worst_text_input.text+" "+self.symmetry_worst_text_input.text+" "+self.fractal_dimension_worst_text_input.text
        self.student_list.adapter.data.extend([student_name])
        self.student_list._trigger_reset_populate()

        student_name = list(student_name.split(" "))
        data = pd.read_csv("/Users/sahil/Downloads/data.csv", error_bad_lines= False)

        with open("/Users/sahil/Downloads/data.csv", 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow("\n")
            writer.writerow(student_name)

        csvFile.close()
        data.head()
        data = data.set_index('id')
        data.diagnosis.replace(to_replace=dict(B=0, M=1), inplace=True)
        target = np.array(data['diagnosis'])
        cancer = data.drop('diagnosis', axis=1)
        feature_name = np.array(list(cancer))
        cancer = np.array(cancer)
        #cancer = np.append(cancer, [student_name], axis=0)

        target_names = np.array(['malignant', 'benign'])
        x_train, x_test, y_train, y_test = train_test_split(cancer, target, test_size=0.20, random_state=0)
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaledX = scaler.fit_transform(x_train)
        X_test_scaled = scaler.fit_transform(x_test)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(x_train)
        rescaledX = scaler.transform(x_train)
        X_test_scaled = scaler.transform(x_test)
        svm2 = SVC(C=2.0)
        svm2.probability = True
        svm2.fit(rescaledX, y_train)
        s = svm2.predict_proba(X_test_scaled[-1:-2:-1])
        v = s[0]
        print(v)
        if v[0] > v[1] * 10:
            print("B")
            return Label(text="B")
        else:
            print("M")
            return Label(text='M')


# if you want to delete
    def delete_report(self):
        pass


class BreastCancerApp(App):
    def build(self):

        #return Label(s)
        return BreastCancer()


bc_App = BreastCancerApp()
bc_App.run()


#self.radius_mean_text_input.text+self.texture_mean_text_input.text+self.perimeter_mean_text_input.text+self.area_mean_text_input.text+self.smoothness_mean_text_input.text+self.compactness_mean_text_input.text+self.concavity_mean_text_input.text+self.concave_points_mean_text_input.text+self.symmetry_mean_text_input.text+self.fractal_dimension_mean_text_input.text+self.radius_standard_error_text_input.text+self.texture_standard_error_text_input.text+self.perimeter_standard_error_text_input.text+self.area_standard_error_text_input.text+self.smoothness_standard_error_text_input.text+self.compactness_standard_error_text_input.text+self.concavity_standard_error_text_input.text+self.concave_points_standard_error_text_input.text+self.symmetry_standard_error_text_input.text+self.fractal_dimension_standard_error_text_input.text+self.radius_worst_text_input.text+self.texture_worst_text_input.text+self.perimeter_worst_text_input.text+self.area_worst_text_input.text+self.smoothness_worst_text_input.text+self.worst_mean_text_input.text+self.concavity_worst_text_input.text+self.concave_points_worst_text_input.text+self.symmetry_worst_text_input.text+self.fractal_dimension_worst_text_input.text
