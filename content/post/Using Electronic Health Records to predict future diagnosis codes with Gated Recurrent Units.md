

---
date: 2019-03-11T10:58:08-04:00
description: "Demystifying the math behind GRUs"
author: "Sparkle Russell-Puleri and Dorian Puleri"
featured_image: "/images/gru2.png"
tags: ["Healthcare", "Electronic Health Records","GRUs", "deep Learning", "RNNs", "deep learning", "machine learning"]
title: "Using Electronic Health Records to predict future diagnosis codes with Gated Recurrent Units"
---

<strong>Background: Detailed review of Doctor AI: Predicting Clinical Events via Recurrent Neural Nets (Choi et.al 2016)</strong>


by: [Sparkle Russell-Puleri](https://www.linkedin.com/in/sparkle-russell-puleri-ph-d-a6b52643) and [Dorian Puleri](https://www.linkedin.com/in/dorian-puleri-ph-d-25114511)

Electronic medical records( EMRs), which is sometimes interchangeably called Electronic health records(EHRs) are primarily used to electronically-store patient health data digitally. While the use of these systems seem commonplace today, most notably due to the passing of the Health Information Technology for Economic and Clinical Health Act in 2014. There adaption implementation into healthcare facilities across the US was very slow. Nonetheless, EHR/EMR systems now posses a wealth of longitudinal patient data which can move us closer to developing patient-centered personalized healthcare solutions. That being said, EHR data can be very messy and sparse. Despite these challenges, if harnessed properly EHR data can provide real world data insights on patient journeys, treatment patterns, predict a patients next diagnosis code or risk of readmission and mortality etc.

As of 2018, it has been reported that healthcare data alone accounts for about 30% of the world’s data production. As a result, there is no secret that many companies and investors are looking to healthcare as the next big investment. However, in order to deliver real solutions on the patient level, we need to understand how to harness and process the wealth of data that we have at hand. To this end, this tutorial will focus on breaking down how to process EHR data for use in AI algorithms. The hope is that with this insight using simulated data, we can get more data scientist and enthusiast on board to democratizing healthcare data and inch closer to making EHR data actionable on the patient level.

For the purpose of this three part tutorial, we generated some artificial EHR data to demonstrate how EHR data should be processed for use in sequence models. Please note that this data has no clinical relevance and was just created for training purposes only.

<strong>This tutorial is broken down into the following sections:</strong>

**Part 1:** Generating artificial EHR data

**Part 2:** Pre-processing artificially generated EHR data

**Part 3:** Doctor AI Pytorch minimal implementation

If you need a quick review on the inner workings of GRUs , see the [**Gated Recurrent Units Review**](https://medium.com/@sparklerussell/gated-recurrent-units-explained-with-matrices-part-2-training-and-loss-function-7e7147b7f2ae).

Github code: <https://github.com/sparalic/Electronic-Health-Records-GRUs>

------

<strong>Part 1: Generating artificial Electronic Health Records(EHR) data</strong>

<strong>Patient Admission Table</strong>

This table contains information on the patient admission history and times. The features generated were:

1. `PatientID`- Unique identifier that stay with the patient permanently
2. `Admission ID` - Specific to each visit
3. `AdmissionStartDate` - Date and time of admission
4. `AdmissionEndDate` - Date and time of discharge after care for a specific admission ID

```
import pandas as pd
import numpy as np

admission_table = {'Patient 1': {'PatientID':'A1234-B456', 
                          'Admission ID':[12,34,15], 
                          'AdmissionStartDate':['2019-01-03 9:34:55','2019-02-03 10:50:55','2019-04-03 12:34:55'],
                          'AdmissionEndDate':['2019-01-07 8:45:43','2019-03-04 1:50:32','2019-04-03 5:38:18']},
                   'Patient 2': {'PatientID':'B1234-C456', 
                          'Admission ID':[13,34], 
                          'AdmissionStartDate':['2018-01-03 9:34:55','2018-02-03 10:50:55'],
                          'AdmissionEndDate':['2018-01-07 8:45:43','2018-03-04 1:50:32']}}
admission_table = (pd.concat({k: pd.DataFrame(v) for k, v in admission_table.items()}).reset_index(level=1, drop=True))
admission_table = admission_table.reset_index(drop=True)
```

<strong>Patient Diagnosis Table</strong>

The diagnosis table is quite unique, as it can contain several diagnosis codes for the same visit. For example, Patient 1 was diagnosed with diabetes (`PrimaryDiagnosisCode`:E11.64) during his/her first visit (`Admission ID`:12). However, this code also shows up on subsequent visits (`Admission ID`:34, 15), why is that? Well if a patient is diagnosed with an uncurable condition he/she that code will always be associated all subsequent visits. On the other hand, codes associated with acute care, will come and go as seen with `PrimaryDiagnosisCode`:780.96(headache).

```
Patient_1 = {'PatientID':'A1234-B456', 
             'Admission ID':[12,34,15], 
             'PrimaryDiagnosisCode':[['E11.64','I25.812','I25.10'],
                                     ['E11.64','I25.812','I25.10','780.96','784.0'],
                                     ['E11.64','I25.812','I25.10','786.50','401.9','789.00']],
             'CodingSystem':['ICD-9','ICD-9','ICD-9'],
             'DiagnosisCodeDescription':[['Type 2 diabetes mellitus with hypoglycemia',
                                          'Atherosclerosis of bypass graft of coronary artery of transplanted heart without angina pectoris',
                                          'Atherosclerotic heart disease of native coronary artery without angina pectoris'],
                                         ['Type 2 diabetes mellitus with hypoglycemia',
                                          'Atherosclerosis of bypass graft of coronary artery of transplanted heart without angina pectoris',
                                          'Atherosclerotic heart disease of native coronary artery without angina pectoris',
                                          'Generalized Pain', 'Dizziness and giddiness'],
                                         ['Type 2 diabetes mellitus with hypoglycemia',
                                          'Atherosclerosis of bypass graft of coronary artery of transplanted heart without angina pectoris',
                                          'Atherosclerotic heart disease of native coronary artery without angina pectoris',
                                          'Chest pain, unspecified','Essential hypertension, unspecified',
                                          'Abdominal pain, unspecified site']]}
Patient_2 = {'PatientID':'B1234-C456', 
              'Admission ID':[13,34], 
              'PrimaryDiagnosisCode':[['M05.59','Z13.85','O99.35'],['M05.59','Z13.85','O99.35','D37.0']],
              'CodingSystem':['ICD-9','ICD-9'],
              'DiagnosisCodeDescription':[['Rheumatoid polyneuropathy with rheumatoid arthritis of multiple sites',
                                           'Encounter for screening for nervous system disorders',
                                           'Diseases of the nervous system complicating pregnancy, childbirth, and the puerperium'],
                                          ['Rheumatoid polyneuropathy with rheumatoid arthritis of multiple sites',
                                           'Encounter for screening for nervous system disorders',
                                           'Diseases of the nervous system complicating pregnancy, childbirth, and the puerperium',
                                           'Neoplasm of uncertain behavior of lip, oral cavity and pharynx']]}
```

<strong>Helper functions for parsing data from a dictionary to DataFrame</strong>

```
def process_ehr(Patient1,Patient2):
    pt_diagnosis_table = [Patient1,Patient2]
    pt_diagnosis_table = pd.concat([pd.DataFrame({k:v for k,v in d.items()}) for d in pt_diagnosis_table])
    
    pt_diagnosis_table = (pt_diagnosis_table.set_index(['PatientID', 'Admission ID','CodingSystem'])
              .apply(lambda x: x.apply(pd.Series).stack())
              .reset_index()
              .drop('level_3', 1))
    return pt_diagnosis_table
def hash_key(df):
    df['HashKey'] = df['PatientID'].\
    apply(lambda x: x.split('-')[0]) + '-' + df['Admission ID'].astype('str')
    cols = [df.columns[-1]] + [col for col in df if col != df.columns[-1]]
    print(cols)
    return df[cols]
diagnosis_table = process_ehr(Patient_1,Patient_2)
diagnosis_table.head()
```


![img](https://cdn-images-1.medium.com/max/1600/1*1TAij25y-5LyvK2b03BqrA.png)

<center>DataFrame of artificially generated EHR data</center>

<strong>Create a hashkey for Admission ID</strong>

Why do this step? Unless your EHR system has uniquely identifiable Admission IDs for each patients visit, it would be difficult to associate each patient ID with a unique `Admission ID`. To demonstrate this, we deliberately created double digit `Admission ID`s one of which was repeated ( `Admission ID`: 34) for both patients. To avoid this, we took a pre-cautionary step to create a hash key that is a unique combination of the first half of the the unique `PatientID`hyphenated with the patient's specific `Admission ID`.


![img](https://cdn-images-1.medium.com/max/1600/1*QZwPK8dyftYl3IAlkWQGRQ.png)

<strong>Final Admission and Diagnosis Tables generated with fake EHR data</strong>



![img](https://cdn-images-1.medium.com/max/1600/1*VyPnZzA4s1Zcr1MEwD8_og.png)

<center>Admission table with artificially generated data</center>



![img](https://cdn-images-1.medium.com/max/1600/1*vBBwqCQkGPcWpaPMh8YKkA.png)

<center>Diagnosis table with artificially generated</center>

<strong>Write tables to csv files</strong>


```
# Write files to data directory
diagnosis_table.to_csv('data/Diagnosis_Table.csv',encoding='UTF-8',index=False)
admission_table.to_csv('data/Admissions_Table.csv',encoding='UTF-8',index=False,date_format='%Y-%m-%d')

```

------

<strong>Part 2: Pre-processing artificially generated EHR data</strong>

In this section we will demonstrate how to process the data in preparation for modeling.The intent of this tutorial is to provide a detailed step through on how EHR data should be pre-processed for use in RNNs using Pytorch. This paper is one of the few papers that provide a code base to start taking a detailed look into how we can build generic models that leverages temporal models to predict future clinical events. However, while this highly cited paper is open sourced (written using Theano:[https://github.com/mp2893/doctorai)](https://github.com/mp2893/doctorai%29), it assumes quite a bit about its readers. As such, we have modernized the code for ease of use in python 3+ and provided a detailed explanation of each step to allow anyone, with a computer and access to healthcare data to begin trying to develop innovative solutions to solve healthcare challenges.

<strong>Important Disclaimer:</strong>

This data set was artificial created with two patients in Part 1 of this series to help provide readers with a clear understanding of the basic structure of EHR data. Please note that each EHR system is specifically designed to meet a specific providers needs and this is just a basic example of data that is typically contained in most systems. Additionally, it is also key to note that this tutorial begins after all of the desired exclusion and inclusion criteria related to your research question has been performed. Therefore, at this step your data would have been fully wrangled and cleaned.

<strong>Load data : A quick review of the artificial EHR data we created in Part 1:</strong>



![img](https://cdn-images-1.medium.com/max/1600/1*16Nb9p9M69U-Trsfj-EGIw.png)

<strong>tep 1: Create mappings of patient IDs</strong>

In this step we are going to create a dictionary that maps each patient with his or her specific visit or `Admission ID`.


![img](https://cdn-images-1.medium.com/max/1600/1*Njhot-skLFhjBupmuI0dqA.png)

```
import pandas as pd
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
from datetime import datetime
import torch
import pickle
from collections import defaultdict
warnings.filterwarnings('ignore')
sns.set(style='white')
%autosave 180

print('Creating visit date mapping')
patHashMap = dict(defaultdict(list))  # this creates a dictionary with a list of values for each patient:[number of visists]
visitMap = dict(defaultdict()) # this creates a dictionary with a mapping of the patientID : visitdates

data = open('data/Admissions_Table.csv','r')
data.readline()[1:] # read every line except the file header

for line in data:
    feature = line.strip().split(',') # split line on , and isolate columns
    visitDateID = datetime.strptime(feature[3],'%Y-%m-%d') 
    patHashMap.setdefault(feature[1], []).append(feature[0]) # create a mapping for each visit for a specific PatientID
    visitMap.setdefault(feature[0], []).append(visitDateID) # create a mapping for each visit for a specific Admission Date
    
#Patient ID- visit mapping
patHashMap

```
{'A1234-B456': ['A1234-12', 'A1234-34', 'A1234-15'],
 'B1234-C456': ['B1234-13', 'B1234-34']}
```

# Patient Admission ID- visit date mapping
visitMap

```
{'A1234-12': [datetime.datetime(2019, 1, 3, 0, 0)],
 'A1234-34': [datetime.datetime(2019, 2, 3, 0, 0)],
 'A1234-15': [datetime.datetime(2019, 4, 3, 0, 0)],
 'B1234-13': [datetime.datetime(2018, 1, 3, 0, 0)],
 'B1234-34': [datetime.datetime(2018, 2, 3, 0, 0)]}
```

```

<strong>Step 2: Create Diagnosis Code Mapped to each unique patient and visit</strong>

This step as with all subsequent steps is very important as it is important to keep the patient’s diagnosis codes in the correct visit order.


![img](https://cdn-images-1.medium.com/max/1600/1*w87koeQa6_xUTXp9Kplsug.png)

```
print('Creating Diagnosis-Visit mapping')
visitDxMap = dict(defaultdict(list))

data = open('data/Diagnosis_Table.csv', 'r')
data.readline()[1:]

for line in data:
    feature = line.strip().split(',')
    visitDxMap.setdefault(feature[0], []).append('D_' + feature[4].split('.')[0]) # add a unique identifier before the
    
visitDxMap # Mapping of each Admission ID to each diagnosis code assigned during that visit

```
{'A1234-12': ['D_E11', 'D_I25', 'D_I25'],
 'A1234-34': ['D_E11', 'D_I25', 'D_I25', 'D_780', 'D_784'],
 'A1234-15': ['D_E11', 'D_I25', 'D_I25', 'D_786', 'D_401', 'D_789'],
 'B1234-13': ['D_M05', 'D_Z13', 'D_O99'],
 'B1234-34': ['D_M05', 'D_Z13', 'D_O99', 'D_D37']}
```
```

<strong>Step 3: Embed diagnosis codes into visit mapping Patient-Admission mapping</strong>

This step essentially adds each code assigned to the patient directing into the dictionary with the patient-admission id mapping and the visit date mapping `visitMap`. Which allows us to have a list of list of diagnosis codes that each patient received during each visit.


![img](https://cdn-images-1.medium.com/max/1600/1*29uW_sXzFx-2UPsBOP72Sg.png)

```
print("Sorting visit mapping")
patDxVisitOrderMap = {}
for patid, visitDates in patHashMap.items():
    sorted_list = ([(visitMap[visitDateID], visitDxMap[visitDateID]) for visitDateID in visitDates])
    patDxVisitOrderMap[patid] = sorted_list 
  
patDxVisitOrderMap

```
{'A1234-B456': [([datetime.datetime(2019, 1, 3, 0, 0)],
   ['D_E11', 'D_I25', 'D_I25']),
  ([datetime.datetime(2019, 2, 3, 0, 0)],
   ['D_E11', 'D_I25', 'D_I25', 'D_780', 'D_784']),
  ([datetime.datetime(2019, 4, 3, 0, 0)],
   ['D_E11', 'D_I25', 'D_I25', 'D_786', 'D_401', 'D_789'])],
 'B1234-C456': [([datetime.datetime(2018, 1, 3, 0, 0)],
   ['D_M05', 'D_Z13', 'D_O99']),
  ([datetime.datetime(2018, 2, 3, 0, 0)],
   ['D_M05', 'D_Z13', 'D_O99', 'D_D37'])]}
```
```


<strong>Step 4a: Extract patient IDs, visit dates and diagnosis</strong>

In this step, we will create a list of all of the diagnosis codes, this will then be used in step 4b to convert these strings into integers for modeling.


![img](https://cdn-images-1.medium.com/max/1600/1*P7R81im_dheDqaq0SRgwmA.png)

```
print("Extracting patient IDs, visit dates and diagnosis codes into individual lists for encoding")
patIDs = [patid for patid, visitDate in patDxVisitOrderMap.items()]
datesList = [[visit[0][0] for visit in visitDate] for patid, visitDate in patDxVisitOrderMap.items()]
DxsCodesList = [[visit[1] for visit in visitDate] for patid, visitDate in patDxVisitOrderMap.items()]

patIDs

```
['A1234-B456', 'B1234-C456']

```

datesList

```
[[datetime.datetime(2019, 1, 3, 0, 0),
  datetime.datetime(2019, 2, 3, 0, 0),
  datetime.datetime(2019, 4, 3, 0, 0)],
 [datetime.datetime(2018, 1, 3, 0, 0), datetime.datetime(2018, 2, 3, 0, 0)]]
```

DxsCodesList

```
[[['D_E11', 'D_I25', 'D_I25'],
  ['D_E11', 'D_I25', 'D_I25', 'D_780', 'D_784'],
  ['D_E11', 'D_I25', 'D_I25', 'D_786', 'D_401', 'D_789']],
 [['D_M05', 'D_Z13', 'D_O99'], ['D_M05', 'D_Z13', 'D_O99', 'D_D37']]]
```
```


<strong>Step 4b: Create a dictionary of the unique diagnosis codes assigned at each visit for each unique patient</strong>

Here we need to make sure that the codes are not only converted to integers but that they are kept in the unique orders in which they were administered to each unique patient.


![img](https://cdn-images-1.medium.com/max/1600/1*poQbXNnKQlEPZq7q-ZWQFg.png)

```
('Encoding string Dx codes to integers and mapping the encoded integer value to the ICD-10 code for interpretation')
DxCodeDictionary = {}
encodedDxs = []
for patient in DxsCodesList:
    encodedPatientDxs = []
    for visit in patient:
        encodedVisit = []
        for code in visit:
            if code in DxCodeDictionary:
                encodedVisit.append(DxCodeDictionary[code])
            else:
                DxCodeDictionarprinty[code] = len(DxCodeDictionary)
                encodedVisit.append(DxCodeDictionary[code])
        encodedPatientDxs.append(encodedVisit)
    encodedDxs.append(encodedPatientDxs)
    
DxCodeDictionary # Dictionary of all unique codes in the entire dataset aka: Our Code Vocabulary

```
{'D_E11': 0,
 'D_I25': 1,
 'D_780': 2,
 'D_784': 3,
 'D_786': 4,
 'D_401': 5,
 'D_789': 6,
 'D_M05': 7,
 'D_Z13': 8,
 'D_O99': 9,
 'D_D37': 10}
```

encodedDxs # Converted list of list with integer converted diagnosis codes

```
[[[0, 1, 1], [0, 1, 1, 2, 3], [0, 1, 1, 4, 5, 6]], [[7, 8, 9], [7, 8, 9, 10]]]
```
```

<strong>Step 6: Dump the data into a pickled list of list</strong>

```
outFile = 'ArtificialEHR_Data'
print('Dumping files into a pickled list')
pickle.dump(patIDs, open(outFile+'.patIDs', 'wb'),-1)
pickle.dump(datesList, open(outFile+'.dates', 'wb'),-1)
pickle.dump(encodedDxs, open(outFile+'.encodedDxs', 'wb'),-1)
pickle.dump(DxCodeDictionary, open(outFile+'.Dxdictionary', 'wb'),-1)
```

<strong>Full Script</strong>

```
print('Creating visit date mapping')
patHashMap = dict(defaultdict(list))  # this creates a dictionary with a list of values for each patient:[number of visists]
visitMap = dict(defaultdict()) # this creates a dictionary with a mapping of the patientID : visitdates

data = open('data/Admissions_Table.csv','r')
data.readline()[1:] # read every line except the file header

for line in data:
    feature = line.strip().split(',')
    visitDateID = datetime.strptime(feature[4],'%Y-%m-%d')
    patHashMap.setdefault(feature[0], []).append(feature[1])
    visitMap.setdefault(feature[1], []).append(visitDateID)

print('Creating Diagnosis-Visit mapping')
visitDxMap = dict(defaultdict(list))

data = open('data/Diagnosis_Table.csv', 'r')
data.readline()[1:]

for line in data:
    feature = line.strip().split(',')
    visitDxMap.setdefault(feature[1], []).append('D_' + feature[7].split('.')[0])

print("Sorting visit mapping")
patDxVisitOrderMap = {}
for patid, visitDates in patHashMap.items():
    sorted_list = ([(visitMap[visitDateID], visitDxMap[visitDateID]) for visitDateID in visitDates])
    patDxVisitOrderMap[patid] = sorted_list 

print("Extracting patient IDs, visit dates and diagnosis codes into individual lists for encoding")
patIDs = [patid for patid, visitDate in patDxVisitOrderMap.items()]
datesList = [[visit[0][0] for visit in visitDate] for patid, visitDate in patDxVisitOrderMap.items()]
DxsCodesList = [[visit[1] for visit in visitDate] for patid, visitDate in patDxVisitOrderMap.items()]

print('Encoding string Dx codes to integers and mapping the encoded integer value to the ICD-10 code for interpretation')
DxCodeDictionary = {}
encodedDxs = []
for patient in DxsCodesList:
    encodedPatientDxs = []
    for visit in patient:
        encodedVisit = []
        for code in visit:
            if code in DxCodeDictionary:
                encodedVisit.append(DxCodeDictionary[code])
            else:
                DxCodeDictionary[code] = len(DxCodeDictionary)
                encodedVisit.append(DxCodeDictionary[code])
        encodedPatientDxs.append(encodedVisit)
    encodedDxs.append(encodedPatientDxs)
```

------

<strong>Part 3: Doctor AI Pytorch minimal implementation</strong>

We will now apply the knowledge gained from the [GRUs tutorial](https://towardsdatascience.com/gate-recurrent-units-explained-using-matrices-part-1-3c781469fc18) and [part 1](https://medium.com/@sparklerussell/using-electronic-health-records-ehr-for-predicting-future-diagnosis-codes-using-gated-recurrent-bcd0de7d7436) of this series to a larger publicly available EHR dataset.This study will utilize the [MIMIC III electronic health record (EHR) dataset](https://mimic.physionet.org/), which is comprised of over *58,000* hospital admissions for *38,645* adults and *7 ,875* neonates. This dataset is a collection of de-identified intensive care unit stays at the **Beth Israel Deaconess Medical Center** from June 2001- October 2012. Despite being de-identified, this EHR dataset contains information about the patients’ demographics, vital sign measurements made at the bedside (~1/hr), laboratory test results, billing codes, medications, caregiver notes, imaging reports, and mortality (during and after hospitalization). Using the pre-processing methods demonstrated on artificially generated dataset in (Part 1 & Part 2) we will create a companion cohort for use in this study.

------

<strong>Model Architecture</strong>



![img](https://cdn-images-1.medium.com/max/1600/1*NAT-F4V9OkG8e6uPpaoM1A.png)

<center>Doctor AI model architecture</center>

<strong>Checking for GPU availability</strong>

This model was trained on a GPU enabled system…highly recommended.


```
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np
import itertools
import pickle
import sys, random
np.random.seed(0)
torch.manual_seed(0)
%autosave 120

# check if GPU is available
if(torch.cuda.is_available()):
    print('Training on GPU!')
else: 
    print('Training on CPU!')
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```
<strong>Load data</strong>

The data pre-processed datasets will be loaded and split into a train, test and validation set at a `75%:15%:10%` ratio.

```
def load_data(sequences, labels):
    dataSize = len(labels)
    idx = np.random.permutation(dataSize)
    nTest = int(np.ceil(0.15 * dataSize))
    nValid = int(np.ceil(0.10 * dataSize))

    test_idx = idx[:nTest]
    valid_idx = idx[nTest:nTest+nValid]
    train_idx = idx[nTest+nValid:]

    train_x = sequences[train_idx]
    train_y = labels[train_idx]
    test_x = sequences[test_idx]
    test_y = labels[test_idx]
    valid_x = sequences[valid_idx]
    valid_y = labels[valid_idx]

    train_x = [sorted(seq) for seq in train_x]
    train_y = [sorted(seq) for seq in train_y]
    valid_x = [sorted(seq) for seq in valid_x]
    valid_y = [sorted(seq) for seq in valid_y]
    test_x = [sorted(seq) for seq in test_x]
    test_y = [sorted(seq) for seq in test_y]

    train = (train_x, train_y)
    test = (test_x, test_y)
    valid = (valid_x, valid_y)
    return (train, test, valid)
```

<strong>Padding the inputs</strong>

The input tensors were padded with zeros, note that the inputs are padded to allow the RNN to handle the variable length inputs. A mask was then created to provide the algorithm information about the padding. Note this can be done using Pytorch’s utility `pad_pack_sequence` function. However, given the nested nature of this dataset, the encoded inputs were first multi-one hot encoded. This off-course creates a high-dimenisonal sparse inputs, however the dimensionality was then projected into a lower-dimensional space using an embedding layer.

```
def padding(seqs, labels, vocab, n_classes):
    lengths = np.array([len(seq) for seq in seqs]) - 1 # remove the last list in each patient's sequences for labels
    n_samples = len(lengths)
    maxlen = np.max(lengths)

    x = torch.zeros(maxlen, n_samples, vocab) # maxlen = number of visits, n_samples = samples
    y = torch.zeros(maxlen, n_samples, n_classes)
    mask = torch.zeros(maxlen, n_samples)
    for idx, (seq,label) in enumerate(zip(seqs,labels)):
        for xvec, subseq in zip(x[:,idx,:], seq[:-1]):
            xvec[subseq] = 1.
        for yvec, subseq in zip(y[:,idx,:], label[1:]):
            yvec[subseq] = 1.
        mask[:lengths[idx], idx] = 1.
        
    return x, y, lengths, mask
```


<strong>GRU Class</strong>

This class contains randomly initiated weights needed to begin calculating the hidden states of the algorithms. Note that in this paper the author used embedding matrix (W_emb) generated using the skip-gram algorithm, which outperformed the randomly initialized approached shown in this step.

```
torch.manual_seed(1)
class EHRNN(nn.Module):
    def __init__(self, inputDimSize, hiddenDimSize,embSize, batchSize, numClass):
        super(EHRNN, self).__init__()

        self.hiddenDimSize = hiddenDimSize
        self.inputDimSize = inputDimSize
        self.embSize = embSize
        self.numClass = numClass
        self.batchSize = batchSize

        #Initialize random weights
        self.W_z = nn.Parameter(torch.randn(self.embSize, self.hiddenDimSize).cuda())
        self.W_r = nn.Parameter(torch.randn(self.embSize, self.hiddenDimSize).cuda())
        self.W_h = nn.Parameter(torch.randn(self.embSize, self.hiddenDimSize).cuda())

        self.U_z = nn.Parameter(torch.randn(self.hiddenDimSize, self.hiddenDimSize).cuda())
        self.U_r = nn.Parameter(torch.randn(self.hiddenDimSize, self.hiddenDimSize).cuda())
        self.U_h = nn.Parameter(torch.randn(self.hiddenDimSize, self.hiddenDimSize).cuda())

        self.b_z = nn.Parameter(torch.zeros(self.hiddenDimSize).cuda())
        self.b_r = nn.Parameter(torch.zeros(self.hiddenDimSize).cuda())
        self.b_h = nn.Parameter(torch.zeros(self.hiddenDimSize).cuda())

        
        self.params = [self.W_z, self.W_r, self.W_h, 
                       self.U_z, self.U_r, self.U_h,
                       self.b_z, self.b_r, self.b_h]

        
    def forward(self,emb,h):
        z = torch.sigmoid(torch.matmul(emb, self.W_z)  + torch.matmul(h, self.U_z) + self.b_z)
        r = torch.sigmoid(torch.matmul(emb, self.W_r)  + torch.matmul(h, self.U_r) + self.b_r)
        h_tilde = torch.tanh(torch.matmul(emb, self.W_h)  + torch.matmul(r * h, self.U_h) + self.b_h)
        h = z * h + ((1. - z) * h_tilde)
        return h
    
                           
    def init_hidden(self):
        return Variable(torch.zeros(self.batchSize,self.hiddenDimSize))
```

<strong>Custom Layer for handling two layer GRU</strong>

The purpose of this class is to perform the initially embedding followed by calculating the hidden states and performing dropout between the layers.

```
torch.manual_seed(1)
class build_EHRNN(nn.Module):
    def __init__(self, inputDimSize=4894, hiddenDimSize=[200,200], batchSize=100, embSize=200,numClass=4894, dropout=0.5,logEps=1e-8):
        super(build_EHRNN, self).__init__()
        
        self.inputDimSize = inputDimSize
        self.hiddenDimSize = hiddenDimSize
        self.numClass = numClass
        self.embSize = embSize
        self.batchSize = batchSize
        self.dropout = nn.Dropout(p=0.5)
        self.logEps = logEps
        
        
        # Embedding inputs
        self.W_emb = nn.Parameter(torch.randn(self.inputDimSize, self.embSize).cuda())
        self.b_emb = nn.Parameter(torch.zeros(self.embSize).cuda())
        
        self.W_out = nn.Parameter(torch.randn(self.hiddenDimSize, self.numClass).cuda())
        self.b_out = nn.Parameter(torch.zeros(self.numClass).cuda())
         
        self.params = [self.W_emb, self.W_out, 
                       self.b_emb, self.b_out] 
    
    def forward(self,x, y, h, lengths, mask):
        self.emb = torch.tanh(torch.matmul(x, self.W_emb) + self.b_emb)
        input_values = self.emb
        self.outputs = [input_values]
        for i, hiddenSize in enumerate([self.hiddenDimSize, self.hiddenDimSize]):  # iterate over layers
            rnn = EHRNN(self.inputDimSize,hiddenSize,self.embSize,self.batchSize,self.numClass) # calculate hidden states
            hidden_state = []
            h = self.init_hidden().cuda()
            for i,seq in enumerate(input_values): # loop over sequences in each batch
                h = rnn(seq, h)                    
                hidden_state.append(h)    
            hidden_state = self.dropout(torch.stack(hidden_state))    # apply dropout between layers
            input_values = hidden_state
       
        y_linear = torch.matmul(hidden_state, self.W_out)  + self.b_out # fully connected layer
        yhat = F.softmax(y_linear, dim=1)  # yhat
        yhat = yhat*mask[:,:,None]   # apply mask
        
        # Loss calculation
        cross_entropy = -(y * torch.log(yhat + self.logEps) + (1. - y) * torch.log(1. - yhat + self.logEps))
        last_step = -torch.mean(y[-1] * torch.log(yhat[-1] + self.logEps) + (1. - y[-1]) * torch.log(1. - yhat[-1] + self.logEps))
        prediction_loss = torch.sum(torch.sum(cross_entropy, dim=0),dim=1)/ torch.cuda.FloatTensor(lengths)
        cost = torch.mean(prediction_loss) + 0.000001 * (self.W_out ** 2).sum() # regularize
        return (yhat, hidden_state, cost)

    def init_hidden(self):
        return torch.zeros(self.batchSize, self.hiddenDimSize)  # initial state
```

<strong>Train model</strong>

This model is a minimal implementation for the Dr.AI algorithm created by Edward Choi, while functional it requires significant tuning. This will be demonstrated in a subsequent tutorial.

```
optimizer = torch.optim.Adadelta(model.parameters(), lr = 0.01, rho=0.90)
max_epochs = 10

loss_all = []
iteration = 0
        
for e in range(max_epochs):
    for index in random.sample(range(n_batches), n_batches):
        batchX = train[0][:n_batches*batchSize][index*batchSize:(index+1)*batchSize]
        batchY = train[1][:n_batches*batchSize][index*batchSize:(index+1)*batchSize]
        
        optimizer.zero_grad()
        
        x, y, lengths, mask = padding(batchX, batchY, 4894, 4894)
        
        if torch.cuda.is_available():
            x, y, lenghts, mask = x.cuda(), y.cuda(), lengths, mask.cuda()
        
        outputs, hidden, cost = model(x,y, h, lengths, mask)
        
        if torch.cuda.is_available():
            cost.cuda()
        cost.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        loss_all.append(cost.item())
        iteration +=1
        if iteration % 10 == 0:
            # Calculate Accuracy         
            losses = []
            model.eval()
            for index in random.sample(range(n_batches_valid), n_batches_valid):
                validX = valid[0][:n_batches_valid*batchSize][index*batchSize:(index+1)*batchSize]
                validY = valid[1][:n_batches_valid*batchSize][index*batchSize:(index+1)*batchSize]

                x, y, lengths, mask = padding(validX, validY, 4894, 4894)

                if torch.cuda.is_available():
                    x, y, lenghts, mask = x.cuda(), y.cuda(), lenghts, mask.cuda()

                outputs, hidden_val, cost_val = model(x,y, h, lengths, mask)
                losses.append(cost_val)
            model.train()

            print("Epoch: {}/{}...".format(e+1, max_epochs),
                          "Step: {}...".format(iteration),
                          "Training Loss: {:.4f}...".format(np.mean(loss_all)),
                          "Val Loss: {:.4f}".format(torch.mean(torch.tensor(losses))))
```
<strong>Final Notes/ Next Steps:</strong>

This should serve as starter code to get the model up and running. As noted before, a significant amount of tuning will be required as this was built using custom classes. We will walkthrough the process in a future tutorial.

<strong>References:</strong>

1. Doctor AI: Predicting Clinical Events via Recurrent Neural Networks (<https://arxiv.org/abs/1511.05942>)