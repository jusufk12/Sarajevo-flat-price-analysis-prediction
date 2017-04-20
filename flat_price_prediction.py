import pandas as pd                 # pandas is a dataframe library
import matplotlib.pyplot as plt     # matplotlib.pyplot plots data
import numpy as np
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn import tree
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import data_helper as dh

def preprocessing1():
	alipasino1, alipasino2, alipasino3, bolnica, buca_potok, cvila, dobrinja, \
			drvenija, grbavica, hadzici, hrasnica, hrasno, ilidza, ilijas, jezero,\
			 marin_dvor, marin_dvor2, mojmilo, mom_uzeir, nahorevo, obn, otoka2, otoka, \
			 pofalici, sip, skenderija, stup, titova, vogosca, vraca, vratnik = dh.assign_cluster_numbers()


	data = [alipasino1, alipasino2, alipasino3, bolnica, buca_potok, cvila, dobrinja, 
			drvenija, grbavica, hadzici, hrasnica, hrasno, ilidza, ilijas, jezero,
			 marin_dvor, marin_dvor2, mojmilo, mom_uzeir, nahorevo, obn, otoka2, otoka, 
			 pofalici, sip, skenderija, stup, titova, vogosca, vraca, vratnik]

	data = pd.concat(data)

	print(data.shape)


	data.to_csv('final_data.csv', header=True)



def preprocessing2():
	data = pd.read_csv('final_data.csv')
	data = dh.choose_columns(data)
	print(data.shape)


	print("Task 3) Dropping NaN values from 'Cijena' and 'Kvadrata'\n")
	data = data.dropna(subset=['Cijena'])
	data = data.dropna(subset=['Kvadrata'])
	data = data.dropna(subset=['Latitude'])
	data = data.dropna(subset=['Longitude'])

	print("Task 4) Dropping 'Po dogovoru' prices and making 'Cijena' usable\n")
	data = data[data.Cijena != 'Po dogovoru']
	data.Cijena = data.Cijena.str.replace("KM", "").str.replace(",", "").str.replace(".","").astype(float)

	print("Task 5) Cleaning 'Kvadrata'\n")
	data = data[data.Kvadrata.str.contains("-") == False]
	data = data[data.Kvadrata.str.contains("/") == False]
	data = data[data.Kvadrata.str.contains("od ") == False]
	data.Kvadrata = data.Kvadrata.str.extract('(\d+)', expand=False).astype(float)

	print("Task 6) Numerating 'Broj soba'\n")
	broj_soba_map = {'Garsonjera': 0.5, 'Jednosoban (1)': 1, 'Jednoiposoban (1.5)': 1.5,'Dvosoban (2)': 2,
					 'Trosoban (3)': 3, '╚etverosoban (4)': 4, 'Petosoban i viџe': 5}
	data['Broj soba'] = data['Broj soba'].map(broj_soba_map)


	print("Task 7) Numeratin 'Grijanje'\n")
	grijanje_map = {'Centralno (Plin)': 1, 'Centralno (Kotlovnica)': 2, 'Plin': 3, 'Centralno (gradsko)': 4,
					 'Struja': 5, 'Ostalo': 6}
	data ['Vrsta grijanja'] = data['Vrsta grijanja'].map(grijanje_map)
	
	print("Task 9) Cleaning 'Sprat'\n")
	data['Sprat']= data['Sprat'].replace(['Visoko prizemlje', 'Prizemlje', 'Suteren'], [0.9, 0.5, 0])
	data['Sprat'] = data['Sprat'].str.extract('(\d+)', expand=False)

	print("Task 12) Converting 'Da', 'Ne' and 'marked' into numbers from 'Balkon', 'Lift', 'Novogradnja', 'Plin' and 'Podrum/Tavan'\n")
	data['Balkon']= data['Balkon'].replace(['Da', 'Ne', 'marked'], [1, 0, 1])
	data['Lift']= data['Lift'].replace(['Da', 'Ne', 'marked'], [1, 0, 1])
	data['Novogradnja']= data['Novogradnja'].replace(['Da', 'Ne', 'marked'], [1, 0, 1])
	data['Plin']= data['Plin'].replace(['Da', 'Ne', 'marked'], [1, 0, 1])
	data['Podrum/Tavan']= data['Podrum/Tavan'].replace(['Da', 'Ne', 'marked'], [1, 0, 1])

	print("Task 10) Assuming that user did not fill in some value if it is 'NaN', than \
		i will just replace it with zeros (If there is that value it is 1)\n")
	data = data.fillna(value=0)

	data['Cijena'] = data.apply(lambda row: (row['Cijena']*row['Kvadrata']
                                               if row['Cijena'] < 30000
                                               else row['Cijena']* 1 ), axis=1)

	data['Cijena'] = data['Cijena'].drop(data['Cijena'][data.Cijena < 30000].index)
	data['Cijena'] = data['Cijena'].drop(data['Cijena'][data.Cijena > 500000].index)
	data = data.dropna(subset=['Cijena'])


	print("Task 14) Saving into csv file call 'final.csv'\n")
	data.to_csv('final.csv', header=True)
	print(data.shape)

def preprocessing3():
	data = pd.read_csv('final.csv')
	data = data.drop('Unnamed: 0', axis=1)
	#dh.plot_corr(data)
	corr = data.corr()
	


	X = data.drop('Cijena', axis=1).as_matrix()
	Y = data['Cijena'].as_matrix()


	min_max_scaler =  MinMaxScaler(feature_range=(0, 1))
	X = min_max_scaler.fit_transform(X)
	Y = min_max_scaler.fit_transform(Y)
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,Y,test_size=0.3)
	#print(data.corr())


	DTree = tree.DecisionTreeRegressor()
	DTree = DTree.fit(X_train, y_train)
	accuracy = DTree.score(X_train, y_train)
	accuracy1 = DTree.score(X_test, y_test)
	print("Decision Tree Regressor training: ", accuracy)
	print("Decision Tree Regressor testing: ", accuracy1)


	Lregression = LinearRegression()
	Lregression.fit(X_train,y_train)
	accuracy2 = Lregression.score(X_train, y_train)
	accuracy3 = Lregression.score(X_test, y_test)
	print("Linear regression training: ", accuracy2)
	print("Linear regression testing: ", accuracy3)

	SVR_Regressor = svm.SVR()
	SVR_Regressor.fit(X_train,y_train)
	accuracy2 = SVR_Regressor.score(X_train, y_train)
	accuracy3 = SVR_Regressor.score(X_test, y_test)
	print("SVR_Regressor training: ", accuracy2)
	print("SVR_Regressor testing: ", accuracy3)


	preds = SVR_Regressor.predict(X = X_test)


	data1 = {'predicted': [preds], 'true': [y_test]}
	df = pd.DataFrame(data1, columns = ['predicted', 'true'])
	df.to_csv('example.csv',  sep=',', header=True)

	plt.figure(figsize = (14,8))
	plt.plot(preds[:50])
	plt.plot(y_test[:50])


preprocessing1()
preprocessing2()
preprocessing3()


