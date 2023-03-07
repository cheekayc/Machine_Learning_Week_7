Assignment 7
================
Chee Kay Cheong
2023-02-29

``` r
knitr::opts_chunk$set(message = FALSE, warning = FALSE)

library(tidyverse)
library(caret)
library(gbm) 
library(pROC) # ROC curve
library(rpart.plot) # tree plots
library(rpart) # CaRT
library(e1071) # SVM 
```

Do 5-folds cv instead of 10-folds because the outcome is so so so small
(only 159 outcomes)

# Load and clean dataset

``` r
# Load and clean dataset
admission = read_csv("./Data/mi.data.csv") %>% 
  janitor::clean_names() %>% 
  select(-id) %>% 
  mutate(
    sex = as_factor(sex),
    pulm_adema = as_factor(pulm_adema),
    fc = as_factor(fc),
    arr = as_factor(arr),
    diab = as_factor(diab),
    obesity = as_factor(obesity),
    asthma = as_factor(asthma),
    readmission = as_factor(readmission),
    readmission = fct_recode(readmission, 'No' = '0', 'Yes' = '1'))

# Identify any rows that do not have complete cases (i.e. have missing data)
miss.rows = admission[!complete.cases(admission), ]
# No missing values detected.

# Set 'No readmission' as reference Level
admission$readmission = relevel(admission$readmission, ref = "No")

# Check data distribution
summary(admission)
```

    ##       age        sex          sodium           alt              wbc        
    ##  Min.   :26.00   0: 634   Min.   :117.0   Min.   :0.0300   Min.   : 2.000  
    ##  1st Qu.:54.00   1:1064   1st Qu.:133.0   1st Qu.:0.2300   1st Qu.: 6.400  
    ##  Median :63.00   2:   2   Median :136.0   Median :0.3800   Median : 8.000  
    ##  Mean   :61.87            Mean   :136.6   Mean   :0.4736   Mean   : 8.804  
    ##  3rd Qu.:70.00            3rd Qu.:140.0   3rd Qu.:0.6100   3rd Qu.:10.500  
    ##  Max.   :92.00            Max.   :169.0   Max.   :3.0000   Max.   :27.900  
    ##       esr              sbp             dbp        pulm_adema fc      arr     
    ##  Min.   :  1.00   Min.   :  0.0   Min.   :  0.0   0:1589     0:682   0:1658  
    ##  1st Qu.:  5.00   1st Qu.:120.0   1st Qu.: 70.0   1: 111     1: 49   1:  42  
    ##  Median : 10.00   Median :140.0   Median : 80.0              2:899           
    ##  Mean   : 13.48   Mean   :138.8   Mean   : 82.1              3: 59           
    ##  3rd Qu.: 19.00   3rd Qu.:160.0   3rd Qu.: 90.0              4: 11           
    ##  Max.   :140.00   Max.   :260.0   Max.   :190.0                              
    ##  diab     obesity  asthma   readmission
    ##  0:1472   0:1658   0:1662   No :1541   
    ##  1: 228   1:  42   1:  38   Yes: 159   
    ##                                        
    ##                                        
    ##                                        
    ## 

``` r
# Readmission: Yes = 159, No = 1541 (Very unbalanced)
```

# Data Partitioning

Partition the data into training and testing using a 70/30 split.

``` r
set.seed(123)

training.index = 
  admission$readmission %>% 
  createDataPartition(p = 0.7, list = F)

training = admission[training.index, ]
testing = admission[-training.index, ]

# I want to see if I have enough outcomes in my training dataset and if I should do "up" or "down" sample.
training %>% 
  select(readmission) %>% 
  group_by(readmission) %>% 
  count()
```

    ## # A tibble: 2 × 2
    ## # Groups:   readmission [2]
    ##   readmission     n
    ##   <fct>       <int>
    ## 1 No           1079
    ## 2 Yes           112

# Construct 3 prediction models, choose hyperparameters, and compare performance

### LASSO

### Support Vector Classifier

### Ensemble Method with Gradient Boosting
