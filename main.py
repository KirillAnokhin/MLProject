import sys
import numpy as np

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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError ("Incorrect arguements\nUsage: python3 main.py file.dat")    
    file_name = sys.argv[1]
    main_data = []
    
    list_string = read_file(file_name)
    for string in list_string:
        data = np.fromstring(string, dtype=float, sep=' ')
        # Removing the row index at the beggining as it's redundant
        data = np.delete(data, 0)
        resistance = data[:4]
        depth = data[4:7]
        signal_values = data[7:57]
        data = np.array([resistance, depth, signal_values], dtype=object)
        main_data.append(data) 