from Lilygo.Recording import Recording
from Lilygo.Dataset import Dataset
import numpy as np
import pickle as pkl
import sklearn
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import scipy
from scipy.fft import fft, fftfreq
from scipy import signal
import sys

DATA_POINTS = 179
filepath = './data/'


def filtered_trace(dataset):
	"""
	Filter unvalid traces of altitude signal.
	"""

	unique_values = len(np.unique(dataset.values))
	length_minutes = dataset.total_time/60
	if (unique_values/length_minutes)<50: return True
	timestamps = np.array([d[1] for d in dataset.raw_timestamps])/1000
	if np.diff(timestamps).max()>10: return True


def get_features_slope(time_series,	window_num=20):
	"""
	For altitude and pressure signals generate slopes features on sequential windows.
	"""

	time_window=len(time_series)//window_num
	slopes = []
	features = [time_series[0] - time_series[-1]]
	for i in range (window_num):
		low = i*time_window
		high = (i+1)*time_window
		slopes.append(np.diff(time_series[low:high]).mean())
	features.extend(slopes)
	return features


def create_energy_signal(rec, input_signal):
	"""
	Generates energy signal of 3 dimensions of given input signal.
	"""

	y_time=[]
	for direct in ['x', 'y', 'z']:
		sg=np.array(rec.data[input_signal+direct].values)
		N=len(sg)
		if direct=='x':
			y_time = (sg**2)
		else:
			y_time+=(sg**2)
	return y_time - np.mean(y_time)


def get_features_energy(series, window_num=25):
	"""
	For signal in frequency domain calculate mean energy on moving window.

	"""

	time_window=len(series)//window_num
	features = []
	for i in range (window_num):
		low = i*time_window
		high = (i+1)*time_window
		features.append(np.mean(series[low:high]))
	return features


def freq_features (time_series, freq_range=5):
	"""
	Convert signal to frequency domain and calculate features.
	"""

	N = len(time_series)
	xf = fftfreq(N, 1 / 200)
	yf = np.abs(fft(time_series))
	indices = np.where(np.array(xf>=0) & np.array(xf<=freq_range))
	xf = xf[indices]
	yf = yf[indices]
	yf[yf<np.quantile(yf, 0.95, axis=0)]=0
	features=[]
	features = get_features_energy(yf)
	return features


def generate_data_path ():
	"""
	Generate two sets of features from altitude and pressure signal.
	"""
	
	data_pressure = []
	data_altitude=[]
	labels_pressure = []
	labels_altitude = []
	for i in range(DATA_POINTS):
		print (i)
		file_num = str(i)
		if i<100: file_num = '0' + file_num
		if i<10: file_num = '0' + file_num
		filename =filepath + f'trace_{file_num}.json'
		if i==9 or i==175 or i==151: 
			print('filtered trace:' + file_num)
			continue
		rec = Recording(filename, no_labels=True, mute=True)
		filter_trace = filtered_trace(rec.data['altitude']) 
		if filter_trace: print (f'filtered trace:{i}')
		new_input = []
		if 'phone_pressure' in rec.data: 
			time_series = np.array(rec.data['phone_pressure'].values)
			new_input.extend(get_features_slope(time_series))
			data_pressure.append(new_input)
			labels_pressure.append(rec.labels['path_idx'])
		new_input = []
		if not filter_trace and i not in [2, 5, 6, 11, 42, 154]: 
			time_series = np.array(rec.data['altitude'].values)
			new_input.extend(get_features_slope(time_series))
			data_altitude.append(new_input)
			labels_altitude.append(rec.labels['path_idx'])
	return data_pressure, data_altitude, labels_pressure, labels_altitude


def generate_data_path_trace (rec):
	"""
	Generate data for pressure and altitude path models.
	"""
	
	data_pressure = []
	data_altitude=[]
	new_input = []
	if 'phone_pressure' in rec.data: 
		time_series = np.array(rec.data['phone_pressure'].values)
		new_input.extend(get_features_slope(time_series))
		data_pressure.append(new_input)
	new_input = []
	time_series = np.array(rec.data['altitude'].values)
	new_input.extend(get_features_slope(time_series))
	data_altitude.append(new_input)
	return data_pressure, data_altitude


def generate_data_activities_and_location (activity=None, location=None):
	"""
	Generate data where labels are given either by activity or locaiton.
	"""

	data = []
	labels = []
	for i in range(DATA_POINTS):
		print (i)
		file_num = str(i)
		if i<100: file_num = '0' + file_num
		if i<10: file_num = '0' + file_num
		filename =filepath + f'trace_{file_num}.json'
		rec = Recording(filename, no_labels=True, mute=True)
		new_input = []
		time_series = create_energy_signal(rec, 'a')
		new_input.extend(freq_features(time_series))
		data.append(new_input)
		if activity!=None:
			labels.append(activity in rec.labels['activities'])		
		if location!=None:
			labels.append(int(rec.labels['board_loc']==location))	
	return data, labels


def generate_data_activities_and_location_trace (rec):
	"""
	Generate data for given trace for activity and location models.
	"""

	data = []
	new_input = []
	time_series = create_energy_signal(rec, 'a')
	new_input.extend(freq_features(time_series))
	data.append(new_input)	
	return data


def get_balanced_subsample(X, y):
	"""
	For binary labels y create subsample of balanced data. 
	"""

	indices_pos = np.where(y==1)[0]
	indices_neg = np.where(y==0)[0]
	if len(indices_pos)>len(indices_neg):
		indices_pos = sklearn.utils.resample(indices_pos, n_samples=len(indices_neg))
	else:
		indices_neg = sklearn.utils.resample(indices_neg,n_samples=len(indices_pos))
	X = np.concatenate(((X[indices_pos], X[indices_neg])))
	y = np.concatenate(((y[indices_pos], y[indices_neg])))
	return X, y 


def train_model_KFOLD(X, y, plot_name='tree', task='path'):
	"""
	Train and test models by spliting the data into training and validation sets.
	"""

	results = []
	results_val=[]
	num_folds = 5
	kfold = sklearn.model_selection.KFold(n_splits=num_folds, shuffle=True)
	for train_index, test_index in kfold.split(X):
		model = RandomForestClassifier()
		if task!='path': 
			X_train, y_train = get_balanced_subsample(X[train_index], y[train_index])
			X_test, y_test = get_balanced_subsample(X[test_index], y[test_index])
		model.fit(X[train_index], y[train_index])	
		# plt.figure()
		# fig = plt.figure(figsize=(25,20))
		# _ = sklearn.tree.plot_tree(model.estimators_[0], 
		# 			   class_names=True,
		# 			   filled=True)
		# fig.savefig(plot_name + ".png")
		results.append(model.score(X[train_index], y[train_index]))
		results_val.append(model.score(X[test_index], y[test_index]))
	print(f'Average train accuracy:{np.mean(results)},average validation accuracy:{np.mean(results_val)}')


def train_final_model(X, y, model_name='', task='path'):
	"""
	Train final model with full input data.
	"""

	results = []
	model = RandomForestClassifier()
	if task!='path': 
		X, y = get_balanced_subsample(X, y)
		print (f'model:{model_name}, shape: {np.shape(X)}' )
	model.fit(X, y)
	#plt.figure()
	# fig = plt.figure(figsize=(25,20))
	# _ = sklearn.tree.plot_tree(model.estimators_[0], 
	# 			   class_names=True,
	# 			   filled=True)
	# fig.savefig(f'tree_{model_name}.png')
	results.append(model.score(X, y))
	print(f'Model: {model_name}, Average train accuracy of final model:{np.mean(results)}')
	pkl.dump(model, open(f'./pickled_models/model_{model_name}.pkl', 'wb')) 


# Example of testing model of walking activity=1 with KFOLD
# activity=1
#location=None
# X, y = generate_data_activities_and_location(activity, location)
# X, y = np.array(X), np.array(y)
# pkl.dump((X, y), open(f'./pickled_datasets/data_location{location}.pkl', 'wb'))
# train_model_KFOLD (X, y)

# Generete final models and store them in .pkl files
# Path
# X_press, X_alt, y_press, y_alt = pkl.load(open('./pickled_datasets/data_path.pkl', 'rb'))
# train_final_model (X_press[np.logical_or(y_press==3, y_press==4)], y_press[np.logical_or(y_press==3, y_press==4)], model_name='path_3_4_press')
# train_final_model (X_alt[np.logical_or(y_alt==3, y_alt==4)], y_alt[np.logical_or(y_alt==3, y_alt==4)], model_name='path_3_4_alt')
# model = pkl.load(open('./pickled_models/model_path_3_4_alt.pkl', 'rb'))
# print(model.score(X_alt[np.logical_or(y_alt==3, y_alt==4)], y_alt[np.logical_or(y_alt==3, y_alt==4)])) 
# X_up = X_press[np.logical_or(np.logical_or(y_press==0, y_press==1), y_press==2)]
# y_up = y_press[np.logical_or(np.logical_or(y_press==0, y_press==1), y_press==2)]
# train_final_model (X_up, y_up, model_name='path_0_1_2_press')
# X_up = X_alt[np.logical_or(np.logical_or(y_alt==0, y_alt==1), y_alt==2)]
# y_up = y_alt[np.logical_or(np.logical_or(y_alt==0, y_alt==1), y_alt==2)]
# train_final_model (X_up, y_up, model_name='path_0_1_2_alt')
# # Activity
# X, y = pkl.load(open('./pickled_datasets/data_activity0.pkl', 'rb'))
# train_final_model (X, y, model_name='activity_0', task='activity')
# X, y = pkl.load(open('./pickled_datasets/data_activity1.pkl', 'rb'))
# train_final_model (X, y, model_name='activity_1', task='activity')
# X, y = pkl.load(open('./pickled_datasets/data_activity2.pkl', 'rb'))
# train_final_model (X, y, model_name='activity_2', task='activity')
# X, y = pkl.load(open('./pickled_datasets/data_activity3.pkl', 'rb'))
# train_final_model (X, y, model_name='activity_3', task='activity')
# # Location
# X, y = pkl.load(open('./pickled_datasets/data_location0.pkl', 'rb'))
# train_final_model (X, y, model_name='location_0', task='location')
# X, y = pkl.load(open('./pickled_datasets/data_location1.pkl', 'rb'))
# train_final_model (X, y, model_name='location_1', task='location')
# X, y = pkl.load(open('./pickled_datasets/data_location2.pkl', 'rb'))
# train_final_model (X, y, model_name='location_2', task='location')


#filename = sys.argv[1] # e.g. 'someDataTrace.json'
def get_activities_confidences(rec):
	"""
	Returns activity confidences for given filename.
	"""

	confidences = []
	X = generate_data_activities_and_location_trace(rec)
	for i in range(4):
		model = pkl.load(open(f'./pickled_models/model_activity_{i}.pkl', 'rb'))
		confidences.append(model.predict_proba(X)[0][1])
	return confidences
#print(get_activities_confidences(filename))

def get_locations_confidences(rec):
	"""
	Returns locations confidences for given filename.
	"""

	confidences = []
	X = generate_data_activities_and_location_trace(rec)
	for i in range(3):
		model = pkl.load(open(f'./pickled_models/model_location_{i}.pkl', 'rb'))
		confidences.append(model.predict_proba(X)[0][1])
	return confidences
#print(get_locations_confidences(filename))


def get_path(rec):
	"""
	Returns path index for given recording.
	"""

	confidences = []
	X_press, X_alt = generate_data_path_trace(rec)
	if len(X_press)>0:
		model_downhill = pkl.load(open(f'./pickled_models/model_path_3_4_press.pkl', 'rb'))
		model_uphill = pkl.load(open(f'./pickled_models/model_path_0_1_2_press.pkl', 'rb'))
		X=X_press
		uphill_flag = X[0][0]>0
	else:
		model_downhill = pkl.load(open(f'./pickled_models/model_path_3_4_alt.pkl', 'rb'))
		model_uphill = pkl.load(open(f'./pickled_models/model_path_0_1_2_alt.pkl', 'rb'))
		X=X_alt
		uphill_flag = X[0][0]<0
	if uphill_flag:
		path = np.argmax(model_uphill.predict_proba(X)[0])
	else:
		path = 3 + np.argmax(model_downhill.predict_proba(X)[0])
	return path
#print(f'Predicted path is: {get_path(filename)}')












