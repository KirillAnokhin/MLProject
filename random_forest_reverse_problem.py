import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def read_file(file_name):
    file = open(file_name, "r")
    list_string = []
    i = 0
    while True:
        line = file.readline()
        list_string.append(line)
        if not line:
            break
        file.close
    return list_string

file_name = "Sim4MLX.dat"
main_data = []
list_string = read_file(file_name)
# last string is empty
list_string.pop()

# Parse data
param_data = []
signal_data = []
for string in list_string:
    data = np.fromstring(string, dtype=float, sep=' ')
    # Removing the row index at the beggining as it's redundant
    data = np.delete(data, 0)
    
    resistance_depth = data[:7]
    param_data.append(resistance_depth)

    # Up to 56 index, because value for 57 index is zero
    signal_values = np.log(data[7:56])
    signal_data.append(signal_values)

print("Parsing data complete")

from sklearn.ensemble import RandomForestRegressor

# Inverse problem: by signal predict parameters

# Scaling signal data
scaler = StandardScaler().fit(signal_data)
signal_data_scaled = scaler.transform(signal_data)

# Split data to train and test
signal_train, signal_test, params_train, params_test = train_test_split(signal_data_scaled, param_data, 
                                                    train_size=0.45, 
                                                    random_state=42)

print("Preprocessing complete")

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor.fit(signal_train, params_train)

print("Fitting model complete")

from matplotlib.backends.backend_pdf import PdfPages

p = PdfPages('random_forest_inverse_results.pdf')

params_pred = regressor.predict(signal_test)

print("Mean squared error: %.25f" % mean_squared_error(params_test, params_pred))
print("Coefficient of determination: %.2f" % r2_score(params_test, params_pred))
print("Model score:  %.25f" %regressor.score(signal_test, params_test))
# Visualize 10 first pred and test params

for i in range (10):
    plt.plot(params_pred[i], label='pred_params')
    plt.plot(params_test[i], label='test_params')
    plt.legend()
    plt.ylabel('params value')
    fig = plt.figure()

fig_nums = plt.get_fignums()  
figs = [plt.figure(n) for n in fig_nums]

# iterating over the numbers in list
for fig in figs:     
    # and saving the files
    fig.savefig(p, format='pdf') 

# close the object
p.close() 

fp = open('random_forest_inverse_results.txt', 'w')
print("Mean squared error: %.25f" % mean_squared_error(params_test, params_pred), file=fp)
print("Coefficient of determination: %.2f" % r2_score(params_test, params_pred), file=fp)
print("Model score:  %.25f" %regressor.score(signal_test, params_test), file=fp)

fp.close()