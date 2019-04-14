

---
date: 2019-03-11T10:58:08-04:00
description: "Background: Detailed review of Doctor AI: Predicting Clinical Events via Recurrent Neural Nets (Choi et.al 2016)"
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

<script src="https://gist.github.com/sparalic/025e96e13824305b179d271cb3eed9b2.js"></script>

<strong>Patient Diagnosis Table</strong>

The diagnosis table is quite unique, as it can contain several diagnosis codes for the same visit. For example, Patient 1 was diagnosed with diabetes (`PrimaryDiagnosisCode`:E11.64) during his/her first visit (`Admission ID`:12). However, this code also shows up on subsequent visits (`Admission ID`:34, 15), why is that? Well if a patient is diagnosed with an uncurable condition he/she that code will always be associated all subsequent visits. On the other hand, codes associated with acute care, will come and go as seen with `PrimaryDiagnosisCode`:780.96(headache).

<script src="https://gist.github.com/sparalic/d9b58614f58826db95415ed941a12161.js"></script>

<strong>Helper functions for parsing data from a dictionary to DataFrame</strong>

<script src="https://gist.github.com/sparalic/b2d6bb353448a66b9bbd243cd31730ed.js"></script>


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


<script src="https://gist.github.com/sparalic/1595ea1406c53987a6f6ce9c5465ebcf.js"></script>

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

<script src="https://gist.github.com/sparalic/b792f6847240d2c37ad3035567a84763.js"></script>

<strong>Step 2: Create Diagnosis Code Mapped to each unique patient and visit</strong>

This step as with all subsequent steps is very important as it is important to keep the patient’s diagnosis codes in the correct visit order.


![img](https://cdn-images-1.medium.com/max/1600/1*w87koeQa6_xUTXp9Kplsug.png)

<script src="https://gist.github.com/sparalic/e9befa55a48f6b3f251d012692a85179.js"></script>

<strong>Step 3: Embed diagnosis codes into visit mapping Patient-Admission mapping</strong>

This step essentially adds each code assigned to the patient directing into the dictionary with the patient-admission id mapping and the visit date mapping `visitMap`. Which allows us to have a list of list of diagnosis codes that each patient received during each visit.


![img](https://cdn-images-1.medium.com/max/1600/1*29uW_sXzFx-2UPsBOP72Sg.png)

<script src="https://gist.github.com/sparalic/3c8c7986a680afeea2a42c9350533d4a.js"></script>


<strong>Step 4a: Extract patient IDs, visit dates and diagnosis</strong>

In this step, we will create a list of all of the diagnosis codes, this will then be used in step 4b to convert these strings into integers for modeling.


![img](https://cdn-images-1.medium.com/max/1600/1*P7R81im_dheDqaq0SRgwmA.png)

<script src="https://gist.github.com/sparalic/d84161f73b3732e3b58bf1beaa96a596.js"></script>


<strong>Step 4b: Create a dictionary of the unique diagnosis codes assigned at each visit for each unique patient</strong>

Here we need to make sure that the codes are not only converted to integers but that they are kept in the unique orders in which they were administered to each unique patient.


![img](https://cdn-images-1.medium.com/max/1600/1*poQbXNnKQlEPZq7q-ZWQFg.png)

<script src="https://gist.github.com/sparalic/371121f2f8d3dd6e0ba3813f88216046.js"></script>

<strong>Step 6: Dump the data into a pickled list of list</strong>

<script src="https://gist.github.com/sparalic/7a3be737b599c6750e437c317080e813.js"></script>

<strong>Full Script</strong>
<script src="https://gist.github.com/sparalic/8ffc12eececadea37872766be92e2052.js"></script>

------

<strong>Part 3: Doctor AI Pytorch minimal implementation</strong>

We will now apply the knowledge gained from the [GRUs tutorial](https://towardsdatascience.com/gate-recurrent-units-explained-using-matrices-part-1-3c781469fc18) and [part 1](https://medium.com/@sparklerussell/using-electronic-health-records-ehr-for-predicting-future-diagnosis-codes-using-gated-recurrent-bcd0de7d7436) of this series to a larger publicly available EHR dataset.This study will utilize the [MIMIC III electronic health record (EHR) dataset](https://mimic.physionet.org/), which is comprised of over *58,000* hospital admissions for *38,645* adults and *7 ,875* neonates. This dataset is a collection of de-identified intensive care unit stays at the **Beth Israel Deaconess Medical Center** from June 2001- October 2012. Despite being de-identified, this EHR dataset contains information about the patients’ demographics, vital sign measurements made at the bedside (~1/hr), laboratory test results, billing codes, medications, caregiver notes, imaging reports, and mortality (during and after hospitalization). Using the pre-processing methods demonstrated on artificially generated dataset in (Part 1 & Part 2) we will create a companion cohort for use in this study.

------

<strong>Model Architecture</strong>



![img](https://cdn-images-1.medium.com/max/1600/1*NAT-F4V9OkG8e6uPpaoM1A.png)

<center>Doctor AI model architecture</center>

<strong>Checking for GPU availability</strong>

This model was trained on a GPU enabled system…highly recommended.

<script src="https://gist.github.com/sparalic/3bbd374269d66c4f1450eb3607c5e435.js"></script>

<strong>Load data</strong>

The data pre-processed datasets will be loaded and split into a train, test and validation set at a `75%:15%:10%` ratio.

<script src="https://gist.github.com/sparalic/594c6bc345ad1efccd115a32bc89e776.js"></script>

<strong>Padding the inputs</strong>

The input tensors were padded with zeros, note that the inputs are padded to allow the RNN to handle the variable length inputs. A mask was then created to provide the algorithm information about the padding. Note this can be done using Pytorch’s utility `pad_pack_sequence` function. However, given the nested nature of this dataset, the encoded inputs were first multi-one hot encoded. This off-course creates a high-dimenisonal sparse inputs, however the dimensionality was then projected into a lower-dimensional space using an embedding layer.

<script src="https://gist.github.com/sparalic/326b60396cb6e7ac529db2c902d643ca.js"></script>


<strong>GRU Class</strong>

This class contains randomly initiated weights needed to begin calculating the hidden states of the algorithms. Note that in this paper the author used embedding matrix (W_emb) generated using the skip-gram algorithm, which outperformed the randomly initialized approached shown in this step.

<script src="https://gist.github.com/sparalic/b11d2c09bc8298ae8641e4038130d8a7.js"></script>

<strong>Custom Layer for handling two layer GRU</strong>

The purpose of this class is to perform the initially embedding followed by calculating the hidden states and performing dropout between the layers.

<script src="https://gist.github.com/sparalic/6a1ef076f1add3d1cefbf74592615f9f.js"></script>

<strong>Train model</strong>

This model is a minimal implementation for the Dr.AI algorithm created by Edward Choi, while functional it requires significant tuning. This will be demonstrated in a subsequent tutorial.

<script src="https://gist.github.com/sparalic/634235bc617d9bf749ed96d25c03bff5.js"></script>
<strong>Final Notes/ Next Steps:</strong>

This should serve as starter code to get the model up and running. As noted before, a significant amount of tuning will be required as this was built using custom classes. We will walkthrough the process in a future tutorial.

<strong>References:</strong>

1. Doctor AI: Predicting Clinical Events via Recurrent Neural Networks (<https://arxiv.org/abs/1511.05942>)