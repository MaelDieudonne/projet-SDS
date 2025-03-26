** Create a log file.
log using "C:\Users\maeld\Documents\Code\3A-MQS-project\log.smcl", replace

** Open the dataset.
use "C:\Users\maeld\Documents\Code\3A-MQS-project\data\data2004_i.parquet"

** Quick overview of the dataset
describe

** Models without covariates
qui gsem (clseusa ambornin amcit amlived amenglsh amchrstn amgovt amfeel amcitizn amshamed belikeus ambetter ifwrong proudsss proudgrp proudpol prouddem proudeco roudspt proudart proudhis proudmil proudsci <- ), lclass(C 1)
estimates store oneclass
qui gsem (clseusa ambornin amcit amlived amenglsh amchrstn amgovt amfeel amcitizn amshamed belikeus ambetter ifwrong proudsss proudgrp proudpol prouddem proudeco roudspt proudart proudhis proudmil proudsci  <- ), lclass(C 2)
estimates store twoclass
qui gsem (clseusa ambornin amcit amlived amenglsh amchrstn amgovt amfeel amcitizn amshamed belikeus ambetter ifwrong proudsss proudgrp proudpol prouddem proudeco roudspt proudart proudhis proudmil proudsci  <- ), lclass(C 3)
estimates store threeclass
qui gsem (clseusa ambornin amcit amlived amenglsh amchrstn amgovt amfeel amcitizn amshamed belikeus ambetter ifwrong proudsss proudgrp proudpol prouddem proudeco roudspt proudart proudhis proudmil proudsci  <- ), lclass(C 4)
estimates store fourclass
qui gsem (clseusa ambornin amcit amlived amenglsh amchrstn amgovt amfeel amcitizn amshamed belikeus ambetter ifwrong proudsss proudgrp proudpol prouddem proudeco roudspt proudart proudhis proudmil proudsci  <- ), lclass(C 5)
estimates store fiveclass
qui gsem (clseusa ambornin amcit amlived amenglsh amchrstn amgovt amfeel amcitizn amshamed belikeus ambetter ifwrong proudsss proudgrp proudpol prouddem proudeco roudspt proudart proudhis proudmil proudsci  <- ), lclass(C 6)
estimates store sixclass
qui gsem (clseusa ambornin amcit amlived amenglsh amchrstn amgovt amfeel amcitizn amshamed belikeus ambetter ifwrong proudsss proudgrp proudpol prouddem proudeco roudspt proudart proudhis proudmil proudsci  <- ), lclass(C 7)
estimates store sevenclass
qui gsem (clseusa ambornin amcit amlived amenglsh amchrstn amgovt amfeel amcitizn amshamed belikeus ambetter ifwrong proudsss proudgrp proudpol prouddem proudeco roudspt proudart proudhis proudmil proudsci  <- ), lclass(C 8)
estimates store eightclass

** Use BIC and AIC estimates for each model to compare and infer which best represents the data.
estimates stats oneclass twoclass threeclass fourclass fiveclass sixclass sevenclass eightclass



qui gsem (clseusa ambornin amcit amlived amenglsh amchrstn amgovt amfeel amcitizn amshamed belikeus ambetter ifwrong proudsss proudgrp proudpol prouddem proudeco roudspt proudart proudhis proudmil proudsci  <- ) (C <- sex age_f race_f educ_f born_usa realinc_f party_f party_fs religstr_f reltrad_f region_f), lclass(C 4)

estat lcprob
estat lcmean

qui gsem (clseusa ambornin amcit amlived amenglsh amchrstn amgovt amfeel amcitizn amshamed belikeus ambetter ifwrong proudsss proudgrp proudpol prouddem proudeco roudspt proudart proudhis proudmil proudsci  <- sex age_f race_f educ_f born_usa realinc_f party_f party_fs religstr_f reltrad_f region_f) (C <- sex age_f race_f educ_f born_usa realinc_f party_f party_fs religstr_f reltrad_f region_f), lclass(C 4)

estat lcprob
estat lcmean




** Close the file.
log close