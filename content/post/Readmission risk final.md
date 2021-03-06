---
date: 2020-05-10T11:58:08-04:00
description: "Predicting 30-day Readmission risk for ICU patients using a multi-task deep learning framework"
author: "Sparkle Russell-Puleri"
featured_image: "/images/personalized.png"
tags: ["Healthcare", "Electronic Health Records","GRUs", "deep Learning", "RNNs", "deep learning", "machine learning", "personalized healthcare"]
title: "Predicting 30-day Readmission risk for ICU patients using a multi-task deep learning framework: Unsupervised clustering of patient sub-populations for multitask learning using electronic health records"
---
by: Sparkle Russell-Puleri 

<h3>Problem</h3>
<div style="text-align: justify;">Readmission whether planned or unplanned costs the U.S healthcare system billions of dollars annually. This, among other things,  is of great importance to healthcare systems, since about one-third of readmissions are preventable, providing a huge opportunity for hospital systems to improve their quality of care <sup>1</sup>. This compounded with the fact that  hospitals are also penalized with  up to 3% lower reimbursements from the Centers for Medicare & Medicaid Services(CMS)  for increased readmission rates, as a part of the 2012 affordable care act passed by the Obama administration <sup>2</sup>, creates a huge opportunity for the use of data to build algorithms that are capable of predicting which patients are likely to be readmitted if discharged early.</div>

<div style="text-align: justify;">In recent years, many studies using Electronic Health Records (EHR) data and machine learning techniques were executed to help determine whether or not big data can detect patients that are at high risk of being readmitted within 30 days. These studies utilized a plethora of statistical and machine learning techniques,  ranging from basic epidemiological models to deep learning  LSTM and CNN architectures with expertly crafted features<sup>3-12</sup>. Despite the progress, one shortcoming of many of the studies is the general one size fits all approach used to predict a patients risk of readmission mostly targeting specific phenotypes. As we move toward developing personalized medicines that takes into account each patient's individual risks factors, we should also apply this same mindset to creating algorithms that are tailored to the vast subpopulations that exists in complexed EHR data. Therefore, in this study I  explored the feasibility of using an unsupervised approach to cohort selection to first identify subpopulations within the dataset before predicting each group's risk of being readmitted in 30 days.</div>

<h3>Proposed Study</h3>
<div style="text-align: justify;">In this study, I will use single task and multi task models to predict 30 day readmission risk using the MIMIC III dataset. This will be done by first clustering patients using a data driven unsupervised approach to cluster patients using sociodemographic, operational, and clinical factors captured in the last 48 hours of their stay. This will then be used to predict the patient's risk of readmission within 30 days of discharge from the ICU using a multi-task framework. This approach was first introduced by Suresh et.al., (2018) <sup>13</sup> to predict in hospital mortality in the ICU. To the best of my knowledge this is the first study that uses this two step approach to predict 30 day readmission risks of heterogenous patients in the ICU. Additionally, it is key to note that this study focuses on the implementation and tests the feasibility of this modeling approach,  which if successful will then lead to possible follow-ups and improvements of this study in a subsequent post.</div>


<h3>Data Source</h3>
<div style="text-align: justify;">This project utilized the [MIMIC III](https://mimic.physionet.org/) electronic health record (EHR) dataset, which is comprised of over 58,000 hospital admissions for 38,645 adults and 7,875 neonates. This dataset is a collection of de-identified intensive care unit stays at the Beth Israel Deaconess Medical Center from June 2001- October 2012.</div>

<h3>Data Preprocessing</h3>
<div style="text-align: justify;">Sociodemographic, operational, and clinical factors have been shown to be associated with readmission. The vast majority of models built to assess 30 day readmission risk, rely mostly on the sociodemographic, operational and expertly crafted disease specific clinical variables(comorbidities)<sup>14-16</sup>, with a small portion including the time varying measurements such as labs and vitals to comprehensively capture a patients progress during their stay. Hence, to holistically assess this approach the following static and time varying variables were used in addition to the administrative and comorbidity features. The time varying features were windowed to capture the last 48 hours of a patients stay. To address the missing data challenges with the data, the data was backfilled from the last measure data point.</div>

{{< figure src="/images/feature_final.png" title="" >}}
<strong>Table 1.</strong> Features used to predict readmission risk based on physiological and vitals measured in the last 48 hours of a patients stay.

**Inclusion Criteria:**

1. Patients &ge; 18 
2. Patients first ICU stay as index (for patients with multiple hospital stays)

**Exclusion Criteria:**

1. Patients that died in the hospital
2. Transfers between units

The cohort was then created using the above inclusion and exclusion criteria in the following steps:

<strong>Step 1:</strong> Calculate age, remove patients under 18 years and create readmission flag

<div style="text-align: justify;">The SQL script below extracts patients first discharge date from the ICU and used that discharge date to estimate the time elapsed between the first discharge date and readmission back to the ICU. The age of each patient was calculated as the difference between the shifted date of admission to the ICU and the year of birth. Please do not be surprised when an age of 300 shows up in the dataset, this is caused by a replacement of all ages > 89 with 300 in the MIMIC III dataset.</div>

<script src="https://gist.github.com/sparalic/96f30a485fe41c687e6e58aaf4126b7a.js"></script>

**Step 2:** Using the four most common comorbidity risk factors, extract this information from each patient in the above SQL view. 

<script src="https://gist.github.com/sparalic/b1fb3ce6ed408297b827a8a5e31f8276.js"></script>

**Step 3:** Extract all of the time varying labs and vitals listed in Table 1. for each patient in the cohort

<script src="https://gist.github.com/sparalic/2b4fe7076b3b1dcb547591cbdf8b6e17.js"></script>

**Step 4:** Average labs that were measured in the same hour by lab and patient

<script src="https://gist.github.com/sparalic/017332a68e903c3628df451b9501d2a6.js"></script>


<h3>Methods</h3>
<div style="text-align: justify;">After preprocessing the data an autoencoder was used to first create a dense representation of the features, given their initial sparsity. The output of the autoencoder was then used to fit a Gaussian Mixture Model (GMM), which is a probabilistic model that can be thought of as a generalized version of the k-means by including information about the covariance structure of the data as well as the centers of the latent Gaussians.</div>

{{< figure src="/images/algo_flow.png" title="" >}}
**Figure 1:** Modeling approach demonstrating how cohort clusters were discovered prior to learning<sup>13</sup>

<h4>Global Model Architecture:</h4>
<div style="text-align: justify;">To help leverage the time varying variables for predicting 30-day readmission risk, a LSTM was used in each architecture. In the global model (Figure 2.), a single LSTM layer was used to train on all of the data, creating a global baseline with the intent of demonstrating the performance of a one size fits all model versus one that has a separate dense layer for each patient group.</div>   
<strong>Figure 2:</strong> Global multi-task model configuration with separate parameters for each cohort at the last output layer<sup>13</sup>

<h4>Multitask Model Architecture:</h4>
<div style="text-align: justify;">Similar to the global approach, this model used a single LSTM layer, however the final dense layer  was specific to each cohort. This presents an opportunity for the models to learn salient patterns that are only present in specific patient cohorts versus using a shared dense layer which does not account for the differences. The architecture of this model can be see in Figure 3 below:</div>


{{< figure src="/images/multiTT.png" title="" >}}
<strong>Figure 3:</strong> Seperate dense layer multi-task model configuration with separate parameters for each cohort at the last output layer<sup>13</sup>


<h3>Results</h3>
<h4>How many clusters (components)?</h4>
<div style="text-align: justify;">Using the embeddings created by the autoencoder, the <strong>Akaike information criterion (AIC)</strong> or  <strong>Bayesian information criterion (BIC)</strong> plot was created to determine the optimal number of components (clusters) that minimizes the AIC and BIC. The results of these plots illustrate that 5 clusters would be best. However, the larger the cluster size the smaller the cohort, and my experiments using 4 or 5 clusters with a highly imbalanced data set resulted in homogenous clusters with only negative cases. Therefore a cluster size of 3 was chosen as a compromise in order to create a meaningful cohort that had a consistent prevalence. It is also key to note that while the AIC and BIC do generate negative values it is not the absolute size of the AIC or BIC value that determines if the selected model is best, it is the relative values over the set of models considered<sup>18-19</sup></div>.</div>

{{< figure src="/images/n_comp.png" title="" >}}
<strong>Figure 4:</strong> AIC and BIC plot the illustrating the effect of the the number of components chosen on the performance of the density estimator of the GMM

<h3>Visualizing the unsupervised patient clusters</h3>
<h4>t- distributed Stohcastic Neighborhood Embedding representation of Clusters in 2D</h4>
<div style="text-align: justify;">The plot below(Figure 5) is a t-SNE representation of the output clusters predicted by GMM model. Here you can see that the clusters overlap quiet a bit which makes it difficult to get a better picture of the clusters in 2-dimensions. Thus a 3D t-SNE representation was created in Figure 6.</div>

{{< figure src="/images/2d_tsne.png" title="" >}}
<strong>Figure 5:</strong> 2D t-SNE plot of patient cohorts created by the GMM<br>

<h4>t-SNE representation of clusters in 3D</h4>
<div style="text-align: justify;">To help improve the visualization of the 4 clusters created, a t-SNE representation of the embeddings was created. The figure below shows the scatter plot and cluster plot of the clusters generated by the GMM model. Here we can see that a vast majority of the patients are in cluster 1, the cluster that contained majority of the readmissions, followed by cluster 2. We can see that both clusters 1 and 2 seemed to be more densely distributed than cluster 0, which seems to be must more dispersed.(Interactive plot, click on cluster number on top right to see individual clusters).</div>

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://sparalic.github.io/readmission/index.html" height="525" width="100%"></iframe>
<strong>Figure 6:</strong> Scatter plot of all patient clusters using a t-SNE 3D representation

{{< figure src="/images/clusters_indv.png" title="">}}

<strong>Figure 8:</strong>Individual 3D scatter plots of each patient cohort showing the spread of patients in each group

<h4>Readmission Prevalence by cluster</h4>

<div style="text-align: justify;">The prevalence rates varied significantly across each cluster, ranging from 5.6% in cluster 0, 13.04% in cluster 1, to 10.27% in cluster 2. The global multi-task model was trained on the entire cohort of patients with the final dense layer sharing the predictions of each patient cohort.</div>

{{< figure src="/images/prevalence.png" title="">}}
<strong>Table 2:</strong> Prevalence across all patient cohorts over the last 48 hours of stay

<h4>Patient Characteristics by Cluster and Readmission status</h4>
<div style="text-align: justify;">Using the clusters generated by the GMM model, Table 3 showed the baseline characteristics stratified by each patient cohort and readmission status of 33, 972 discharges and 3602(10.6%) readmissions. Overall, the vast majority of patients who were readmitted within 30 days were in cluster 1 ,  59% population in cluster were male, 67.7% white, with a median length of stay of 2.70 days and 48.7% covered by Medicare. Since the age was discretized, the majority of readmission across all clusters occurred in the 51-70 year old age group and were originally discharged to a skilled nursing facility(SNF). Patients who had existing conditions such as congestive heart failure also showed higher readmission rates over the other four comorbidities included in the dataset.</div>


{{< figure src="/images/table1.png" title="">}}
<strong>Table 3:</strong> Baseline characterics of patient cohorts by cluster and 30-day readmission status

<h4>Vitals and lab features by unsupervised cluster</h4>
<div style="text-align: justify;">The heatmaps below shows the change in z-score values of both the laboratory and vital sign features over the last 48 hours of a patient's stay. Positive,  negative or zero values of the vitals indicate elevated levels, decreased, or average levels of each feature over time, respectively. In the case of cluster 0, it showed consistently increasing blood pressure and respiratory rates over the last 48 hours while, cluster 1showed increased Glascow coma scores in the early hours of the day prior to discharge. Interestingly, cluster 2 showed quite a few missing measurements, which is expected since many of these labs will not be captured later in a patients stay.</div> 

{{< figure src="/images/cluster0_vitals.png" title="">}}
{{< figure src="/images/cluster1_vitals.png" title="">}}
{{< figure src="/images/cluster2_vitals.png" title="">}}
<strong>Figure 9:</strong> Vital sign variable trends over the last 48 hours prior to discharge from the ICU

<h4>Labs</h4>
<div style="text-align: justify;">In the case of the laboratory measurements within the last 48 hours prior to discharge; in cluster 1 where most of the patients were readmitted in we see many of the labs remained largely elevated over the course of the last 48 hours of stay, with the exception of Albumin, blood pH and pO2 which stayed relatively lower than average. Like the vitals heatmaps for the labs, demonstrated the same trend of missing lab values over the course of the last 48 hours increased as the frequency in which these labs were measured decreased later into the ICU stay.</div> 


{{< figure src="/images/cluster0_labs.png" title="">}}
{{< figure src="/images/cluster1_labs.png" title="">}}
{{< figure src="/images/cluster2_labs.png" title="">}}
<strong>Figure 10:</strong> Routine laboratory variable trends over the last 48 hours prior to discharge from the ICU

<h4>Model Results</h4>
<div style="text-align: justify;">To assess whether the clusters play a role in better predicting a patient's risk the patient cohorts were presented to a global multi-task model and the individual dense layer multi-task model for predicting a patients 30-day readmission risk. The results shown in Table 3. while promising come at a huge cost. While the algorithms were great at predicting negative cases, they did a poor job predicting positive cases. Hence the results below might be misleading, as they are heavily based on the majority class. This is can be attributed to several factors. Having several optimizers, network sizes and learning rates with no improvements in the PPV, I suggest that a possible cause for this is mostly likely the sparsity and reduced frequency in which many of the routine physiological data is collected at closer to discharge as illustrated in Figures 9-10. This led to the development of a feature set that is heavily influenced by missing values. However this approach can work using a different feature engineering approach. Note the single task model presented in the paper also performed significantly worse than the other which the author also experienced in her study, hence it was also not included.</div>

{{< figure src="/images/model_results.png" title="">}}
<strong>Table 4:</strong> Unsupervised modeling approach results on 30-day readmission task using global and multi-task models.

<h4>Conclusion/Limitations</h4>
<div style="text-align: justify;">This project demonstrates how the idea of personalized should be done in the real world setting. However, the method will need to be adapted to ensure that the selected features provide meaningful results for predicting the outcome at hand. Some of the findings when revisiting the work presented in this paper that should be addressed if this method is to be used on other research projects include:</div>

1. Performing z-score normalization after splitting the data into the train/test/val sets to avoid information leakage from the training set into the test set.
2. Filling the missing datasets with the normal values for these labs and providing the algorithm with a mask to identify missingness vs. dummifying the features and creating a larger set of sparse features.</div>

<h4>Future directions</h4>
<div style="text-align: justify;">This publication is a step in the right direction of personalizing healthcare, and with some changes this modeling approach can provide some interesting insights into specific sub-populations which are often overlooked when building health related models because of the limited cohort sizes available after applying the inclusion and exclusion criteria. However, while this is only a preliminary study that explored the modeling approach as is. With further research, with well crafted features and potentially using a more robust feature engineering strategy can further propel and evolve this approach. If we:</div>

1. Try training on a balanced the training<br>

2. Carefully craft lab features using values that are abnormal within the first day and last 24 hrs of stay<br>

3. Establish a baseline models with simpler models<br>

4. Then try using a channel-wise LSTM approach proposed by Harutyunyan et.al, 2017 ​ , to better account for missingness by explicitly showing which variables are missing as well as allowing the model to learn and store relevant information related to that specific variable before mixing with other variables with in each cluster.<br>

5. Try embedding the patient data prior to presentation to the LSTM layer, to help reduce the sparsity of the labs<br>

6. Add features extracted form clinical notes to extract prior drug use and add high–risk medications such as: steroids, narcotics, anticholinergics(7)<br>

[Code is available on github](https://github.com/sparalic)

<h4>Acknowledgements:</h4>
 Special thanks to Eraldi Skendaj for critically reading and providing feedback.

<h4>References:</h4>

1. Bates, D. W., Saria, S., Ohno-Machado, L., Shah, A. & Escobar, G. Big data in health care: using analytics to identify and manage high-risk and high-cost patients. Heal. Aff. 33, 1123–1131 (2014).<br>

2. https://www.cms.gov/Medicare/Medicare-Fee-for-Service-Payment/AcuteInpatientPPS/Readmissions-Reduction-Program<br>

3. Y. P. Tabak, X. Sun, C. M. Nunez, V. Gupta and R. S. Johannes, "Predicting readmission at early hospitalization using electronic clinical data: An early readmission risk score", *Med. Care*, vol. 55, no. 3, 2017.<br>

4. H. Wang, Z. Cui, Y. Chen, M. Avidan, A. B. Abdallah and A. Kronzer, "Predicting Hospital Readmission via Cost-Sensitive Deep Learning," in *IEEE/ACM Transactions on Computational Biology and Bioinformatics*, vol. 15, no. 6, pp. 1968-1978, 1 Nov.-Dec. 2018, doi: 10.1109/TCBB.2018.2827029.<br>

5. Marc D. Silverstein, Huanying Qin, S. Quay Mercer, Jaclyn Fong & Ziad Haydar (2008) Risk Factors for 30-Day Hospital Readmission in Patients ≥65 Years of Age, Baylor University Medical Center Proceedings, 21:4, 363-372<br>

6. Velibor V. Mišić, Eilon Gabel, Ira Hofer, Kumar Rajaram, Aman Mahajan; Machine Learning Prediction of Postoperative Emergency Department Hospital Readmission. *Anesthesiology* 2020;132(5):968-980. doi: https://doi.org/10.1097/ALN.0000000000003140.<br>

7. Vunikili, Ramya & Glicksberg, Benjamin & Johnson, Kipp & Dudley, Joel & Subramanian, Lakshminarayanan & Khader, Shameer. (2018). Predictive modeling of susceptibility to substance abuse, mortality and drug-drug interactions in opioid patients.<br>

8. McIntyre LK, Arbabi S, Robinson EF, Maier RV. Analysis of Risk Factors for Patient Readmission 30 Days Following Discharge From General Surgery. JAMA Surg. 2016;151(9):855‐861. doi:10.1001/jamasurg.2016.1258<br>

9. Press VG. Is It Time to Move on from Identifying Risk Factors for 30-Day Chronic Obstructive Pulmonary Disease Readmission? A Call for Risk Prediction Tools. *Ann Am Thorac Soc*. 2018;15(7):801‐803. doi:10.1513/AnnalsATS.201804-246ED<br>

10. M.J. Rothman et al./Journal of Biomedical Informatics 46 (2013) 837–848<br>

11. Ohnuma, T., Shinjo, D., Brookhart, A. *et al.* Predictors associated with unplanned hospital readmission of medical and surgical intensive care unit survivors within 30 days of discharge. *j intensive care* **6,** 14 (2018). https://doi.org/10.1186/s40560-018-0284-x<br>

12. Hrayr Harutyunyan, Hrant Khachatrian, David C Kale, and Aram Galstyan. 2017. Multitask Learning and Benchmarking with Clinical Time Series Data. arXiv preprint arXiv:1703.07771 (2017).<br>

13. Harini Suresh, Jen J Gong, and John Guttag. Learning tasks for multitask learning: Het- erogenous patient populations in the icu. arXiv preprint arXiv:1806.02878, 2018<br>

14. Keenan PS, Normand SL, Lin Z, et al. An administrative claims measure suitable for profiling hospital performance on the basis of 30- day all-cause readmission rates among patients with heart failure. *Circ Cardiovasc Qual Outcomes*. 2008;1:29–37<br>

15. Krumholz HM, Lin Z, Drye EE, et al. An administrative claims measure suitable for profiling hospital performance based on 30-day all-cause readmission rates among patients with acute myocardial infarction. *Circ Cardiovasc Qual Outcomes*. 2011;4:243–252.<br>

16. Lindenauer PK, Normand SL, Drye EE, et al. Development, validation, and results of a measure of 30-day readmission following hospitalization for pneumonia. *J Hosp Med*. 2011;6:142–150.<br>

17. Baguley, Thomas. Serious stats: A guide to advanced statistics for the behavioral sciences. Palgrave Macmillan, 2012. (page 402)<br>

18.  [Model Selection and Multi-model Inference: A Practical Information-theoretic Approach](https://books.google.ca/books?id=fT1Iu-h6E-oC&printsec=frontcover#v=onepage&q&f=false) (Burnham and Anderson, 2004)<br>

19. image source: https://audiotech.com/trends-magazine/building-the-foundation-for-personalized-medicine/