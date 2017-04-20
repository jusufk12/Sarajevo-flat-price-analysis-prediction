import pandas as pd                 # pandas is a dataframe library
import matplotlib.pyplot as plt     # matplotlib.pyplot plots data
import numpy as np
from sklearn import preprocessing



def load_data():
	alipasino1 = pd.read_csv('data/alipasino1.csv', encoding ='cp855')
	alipasino2 = pd.read_csv('data/alipasino2.csv', encoding ='cp855')
	alipasino3 = pd.read_csv('data/alipasino3.csv', encoding ='cp855')
	bolnica = pd.read_csv('data/bolnica.csv', encoding ='cp855')
	buca_potok = pd.read_csv('data/buca_potok.csv', encoding ='cp855')
	cvila = pd.read_csv('data/cvila.csv', encoding ='cp855')
	dobrinja = pd.read_csv('data/dobrinja.csv', encoding ='cp855')


	drvenija = pd.read_csv('data/drvenija.csv', encoding ='cp855')
	grbavica = pd.read_csv('data/grbavica.csv', encoding ='cp855')
	hadzici = pd.read_csv('data/hadzici.csv', encoding ='cp855')
	hrasnica = pd.read_csv('data/hrasnica.csv', encoding ='cp855')
	hrasno = pd.read_csv('data/hrasno.csv', encoding ='cp855')	
	ilidza = pd.read_csv('data/ilidza.csv', encoding ='cp855')
	ilijas = pd.read_csv('data/ilijas.csv', encoding ='cp855') 
	jezero = pd.read_csv('data/jezero.csv', encoding ='cp855')
	marin_dvor = pd.read_csv('data/marin_dvor.csv', encoding ='cp855')
	marin_dvor2 = pd.read_csv('data/marin_dvor2.csv', encoding ='cp855')

	mojmilo = pd.read_csv('data/mojmilo.csv', encoding ='cp855')
	mom_uzeir = pd.read_csv('data/mom_uzeir.csv', encoding ='cp855')
	nahorevo = pd.read_csv('data/nahorevo.csv', encoding ='cp855')
	obn = pd.read_csv('data/obn.csv', encoding ='cp855')
	otoka = pd.read_csv('data/otoka.csv', encoding ='cp855')
	otoka2 = pd.read_csv('data/otoka2.csv', encoding ='iso8859_5')
	pofalici = pd.read_csv('data/pofalici.csv', encoding ='cp855')
	sip = pd.read_csv('data/sip.csv', encoding ='cp855')
	skenderija = pd.read_csv('data/skenderija.csv', encoding ='cp855')
	stup = pd.read_csv('data/stup.csv', encoding ='cp855')
	titova = pd.read_csv('data/titova.csv', encoding ='cp855')
	vogosca = pd.read_csv('data/velesici.csv', encoding ='cp855')
	vraca = pd.read_csv('data/vraca.csv', encoding ='cp855')
	vratnik = pd.read_csv('data/vratnik.csv', encoding ='cp855')

	return [alipasino1, alipasino2, alipasino3, bolnica, buca_potok, cvila, dobrinja, drvenija, grbavica, 
			hadzici, hrasnica, hrasno, ilidza, ilijas, jezero, marin_dvor, marin_dvor2, mojmilo, mom_uzeir,
			nahorevo, obn, otoka2, otoka, pofalici, sip, skenderija, stup, titova, vogosca, vraca, vratnik]



def assign_cluster_numbers():
	alipasino1, alipasino2, alipasino3, bolnica, buca_potok, cvila, dobrinja, \
		drvenija, grbavica, hadzici, hrasnica, hrasno, ilidza, ilijas, jezero,\
		 marin_dvor, marin_dvor2, mojmilo, mom_uzeir, nahorevo, obn, otoka2, otoka, \
		 pofalici, sip, skenderija, stup, titova, vogosca, vraca, vratnik = load_data()
		
	for col in alipasino1:
		alipasino1['cluster']=1

	for col in alipasino2:
		alipasino2['cluster']=2

	for col in alipasino3:
		alipasino3['cluster']=3

	for col in bolnica:
		bolnica['cluster']=4

	for col in buca_potok:
		buca_potok['cluster']=5

	for col in cvila:
		cvila['cluster']=6

	for col in drvenija:
		drvenija['cluster']=7

	for col in dobrinja:
		dobrinja['cluster']=8

	for col in grbavica:
		grbavica['cluster']=9	

	for col in hadzici:
		hadzici["cluster"]=10

	for col in hrasnica:
		hrasnica['cluster']=11

	for col in hrasno:
		hrasno['cluster']=12

	for col in ilidza:
		ilidza['cluster']=13

	for col in ilijas:
		ilijas['cluster']=14

	for col in jezero:
		jezero['cluster']=15

	for col in marin_dvor:
		marin_dvor['cluster']=16

	for col in marin_dvor2:
		marin_dvor2['cluster']=17

	for col in mojmilo:
		mojmilo['cluster']=18

	for col in mom_uzeir:
		mom_uzeir['cluster']=19

	for col in nahorevo:
		nahorevo['cluster']=20

	for col in obn:
		obn['cluster']=21

	for col in otoka2:
		otoka2['cluster']=22

	for col in pofalici:
		pofalici['cluster']=23

	for col in sip:
		sip['cluster']=24

	for col in skenderija:
		skenderija['cluster']=25

	for col in stup:
		stup['cluster']=26

	for col in titova:
		titova['cluster']=27

	for col in vogosca:
		vogosca['cluster']=28

	for col in vraca:
		vraca['cluster']=29

	for col in vratnik:
		vratnik['cluster']=30

	for col in otoka:
		otoka['cluster']=31

	return [alipasino1, alipasino2, alipasino3, bolnica, buca_potok, cvila, dobrinja, drvenija, grbavica, 
		hadzici, hrasnica, hrasno, ilidza, ilijas, jezero, marin_dvor, marin_dvor2, mojmilo, mom_uzeir,
		nahorevo, obn, otoka2, otoka, pofalici, sip, skenderija, stup, titova, vogosca, vraca, vratnik]


def choose_columns(data):
	print("Task 2) Firstly dropping renting rows and then dropping 'Vrste oglasa '\n")
	data = data[data['Vrsta oglasa'] == "Prodaja"]
	cols = data.columns.tolist()
	data = data[['Kvadrata','Broj soba','Latitude','Longitude', 'Sprat','Balkon',
					 'Lift', 'Novogradnja', 'Plin', 'Podrum/Tavan', 'Vrsta grijanja', 'cluster', 'Cijena']]
	return data

def plot_corr(df, size=15):
    corr = df.corr()    # data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)   # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks
    plt.show()
