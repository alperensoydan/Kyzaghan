import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, vstack, csc_matrix
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from scipy.sparse import coo_matrix


data = pd.read_csv("IP-Monday-WorkingHours.pcap_ISCX.csv", sep=",", dtype=object, names=["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp", "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
"Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
"Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
"Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
"Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
"Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
"Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
"FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length",
"Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
"Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label"])
data = data.iloc[1:-1]

# #Features that we use -- Source IP, Destination IP and Timestamp will be added later when they fixed
data = data[['Source Port', 'Destination Port', 'Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Max', 'Active Min', 'Idle Max', 'Idle Min', 'Label']]
data_non_numeric = data[['Source Port', 'Destination Port', 'Protocol', 'Label']]
data_numeric = data.drop(data_non_numeric, axis=1).astype(np.float64) #model can not work on numeric datas when they are object

data_numeric = data_numeric.replace([np.inf, -np.inf], np.nan) #there is inf or float64 extended values so we have to find and fill them with biggest number in columns.
data_numeric.update(data_numeric[['Flow Bytes/s']].select_dtypes(include=[np.number]).fillna(2.071000e+09))
data_numeric.update(data_numeric[['Flow Packets/s']].select_dtypes(include=[np.number]).fillna(3000000.0))

standardization_data_numeric = StandardScaler().fit_transform(data_numeric) #we are done with numeric datas


le_SP = LabelEncoder().fit_transform(data_non_numeric[['Source Port']].values.ravel()).reshape(-1, 1) #le_SP can be data_non_numeric["Source Port"]
ohe_SP = OneHotEncoder(sparse=False).fit_transform(le_SP)

le_DP = LabelEncoder().fit_transform(data_non_numeric[['Destination Port']].values.ravel()).reshape(-1, 1)
ohe_DP = OneHotEncoder(sparse=False).fit_transform(le_DP)

le_Proto = LabelEncoder().fit_transform(data_non_numeric[['Protocol']].values.ravel()).reshape(-1, 1)
ohe_Proto = OneHotEncoder(sparse=False).fit_transform(le_Proto)

concated_non_numeric = np.column_stack((ohe_SP, ohe_DP, ohe_Proto))
concated_data_train = np.column_stack((standardization_data_numeric, concated_non_numeric))

pickle_data = open("/home/alperen/Desktop/Thesis_Test/data_train", 'wb')
pickle.dump(concated_data_train, pickle_data)
pickle_data.close()

########################################################################################################################

data_test = pd.read_csv("Tuesday-WorkingHours.pcap_ISCX.csv", sep=",", dtype=object, names=["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp", "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
"Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
"Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
"Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
"Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
"Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
"Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
"FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length",
"Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
"Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label"])
data_test = data_test.iloc[1:-1]
data_test = data_test.iloc[:431930]

data_test = data_test[['Source Port', 'Destination Port', 'Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Max', 'Active Min', 'Idle Max', 'Idle Min', 'Label']]
data_test_non_numeric = data_test[['Source Port', 'Destination Port', 'Protocol', 'Label']]
data_test_numeric = data_test.drop(data_test_non_numeric, axis=1).astype(np.float64) #model can not work on numeric datas when they are object

data_test_numeric = data_test_numeric.replace([np.inf, -np.inf], np.nan) #there is inf or float64 extended values so we have to find and fill them with biggest number in columns.
data_test_numeric.update(data_test_numeric[['Flow Bytes/s']].select_dtypes(include=[np.number]).fillna(2.070000e+09))
data_test_numeric.update(data_test_numeric[['Flow Packets/s']].select_dtypes(include=[np.number]).fillna(3000000.0))

standardization_data_test_numeric = StandardScaler().fit_transform(data_test_numeric) #we are done with numeric datas


le_test_SP = LabelEncoder().fit_transform(data_test_non_numeric[['Source Port']].values.ravel()).reshape(-1, 1) #le_SP can be data_non_numeric["Source Port"]
ohe_test_SP = OneHotEncoder(sparse=False).fit_transform(le_test_SP)

le_test_DP = LabelEncoder().fit_transform(data_test_non_numeric[['Destination Port']].values.ravel()).reshape(-1, 1)
ohe_test_DP = OneHotEncoder(sparse=False).fit_transform(le_test_DP)

le_test_Proto = LabelEncoder().fit_transform(data_test_non_numeric[['Protocol']].values.ravel()).reshape(-1, 1)
ohe_test_Proto = OneHotEncoder(sparse=False).fit_transform(le_test_Proto)

data_test_non_numeric['Enc_Label'] = data_test_non_numeric['Label'].apply(lambda x: 0 if x == "BENIGN" else 1)
data_test_non_numeric = data_test_non_numeric.drop(columns=('Label'))
test_label = data_test_non_numeric['Enc_Label']

concated_non_numeric_test = np.column_stack((ohe_test_SP, ohe_test_DP, ohe_test_Proto))
concated_data_test = np.column_stack((concated_non_numeric_test, standardization_data_test_numeric))

#pycharm throws memory error so whe use pickle to flush memory
pickle_test_label = open("/home/alperen/Desktop/Thesis_Test/test_label", 'wb')
pickle.dump(test_label, pickle_test_label)
pickle_test_label.close()

pickle_data = open("/home/alperen/Desktop/Thesis_Test/data_test", 'wb')
pickle.dump(concated_data_test, pickle_data)
pickle_data.close()

print("It's okay!")
#

########################################################################################################################

#one-class svm algorithm



concating_data_train = open("/home/alperen/Desktop/Thesis_Test/data_train", 'rb')
data_train = pickle.load(concating_data_train)
concating_data_train.close()

concating_data_test = open("/home/alperen/Desktop/Thesis_Test/data_test", 'rb')
data_test = pickle.load(concating_data_test)
concating_data_test.close()

pickle_test_label = open("/home/alperen/Desktop/Thesis_Test/test_label", 'rb')
test_label = pickle.load(pickle_test_label)
pickle_test_label.close()

print(data_train.shape)
print(data_test.shape)

o_svm = OneClassSVM()
o_svm.fit(data_train)
anomaly_detect = o_svm.predict(data_test)

pickle_predicted_data = open("/home/alperen/Desktop/Thesis_Test/predicted_data", 'wb')
predicted_data = pickle.dump(anomaly_detect, pickle_predicted_data)
pickle_predicted_data.close()

print("Model is trained successfully! :)")

unique, counts = np.unique(anomaly_detect, return_counts=True)
print(np.asarray((unique, counts)).T)

test_label = test_label.as_matrix()

print(anomaly_detect)

TP = FN = FP = TN = 0
for i in range(0, 25920):
    if test_label[i] == 0 and anomaly_detect[i] == 1:
        TP = TP + 1
    elif test_label[i] == 0 and anomaly_detect[i] == -1:
        FN = FN + 1
    elif test_label[i] == 1 and anomaly_detect[i] == 1:
        FP = FP + 1
    else:
        TN = TN + 1
print(TP,  FN,  FP,  TN)

# Performance Matrix

accuracy = (TP+TN)/(TP+FN+FP+TN)
print("accuracy: " + str(accuracy))


