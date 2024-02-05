 # Log

- Deadline: 30.09.2023 (defense, report, submission, etc.)
    - hard deadline: ~~29.09~~ 14.10 
    - better before ~~25.09~~ 14.10
    - great if we had the talk before 08.09? might me a little early
- requirements
    - presentation
    - report

# 2023-09-26

- agenda
    -review report
    -Presentation Discussion
- TODO:
    - see previous TODO and e-mail

## 2023-09-12

- agenda
    - grading
    - report
    - accuracy quality function
- todo
    - [ ] university overleaf
    - [ ] continue with last plan


## 2023-09-04

- next steps
    - [ ] overleaf with access for me
        - [ ] related work
            - [ ] some ideas: Scape, clustering, fairness
        - [ ] background section
            - [ ] subgroup discovery
            - [ ] ...
        - [ ] methodology
            - [ ] basic methods for finding predictable and not predictable subgroups
            - [ ] new methods: ARL, ROC AUC, PRC AUC (average precision)
            - [ ] refitted subgroups
                - [ ] general idea
                - [ ] efficiency
        - [ ] implementation
            - [ ] minimal code if any
            - [ ] current state, interfaces / classes
            - [ ] limitations
            - [ ] required changes 
        - [ ] experiments
            - [ ] classification
                - [ ] AUC / ACC, class imbalances (ratio)
                    - [ ] all QFs (ARL, ...) 
                - [ ] fix class imablance
            - [ ] regression
                - [ ] show it works
                - [ ] different results for refitting
                - [ ] we are faster with with warm_starts
    - show AUC / ACC (or MAE / MSE for regression) for every subgroup



## 2023-08-23

- next steps
    - [ ] make simple application notebooks
        - [ ] basic methods
            - [ ] classification
                - [ ] new column (correct / incorrect) [uses BinaryTarget]
                - [ ] loss (class/probability based) [uses NumericTarget]
            - [ ] regression
                - [ ] loss [uses NumericTarget]
        - [ ] New methods: binary Case
            - [ ] fix (**a: no 1) and normalize ARL
            - [ ] clean up code
        - [ ] New methods: refitting
            - [ ] refit model on subgroup after search and show statistics
            - [ ] use kfold
            - [ ] https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    - [ ] plan for analyses (and include why)
    - [ ] clean up 
    - [ ] outline for report
    - [ ] agenda + next steps
    - [ ] slides

## 2023-08-16

- next steps
    - [ ] make simple application notebooks
        - [ ] basic methods
            - [ ] classification
                - [ ] new column (correct / incorrect) [uses BinaryTarget]
                - [ ] loss (class/probability based) [uses NumericTarget]
            - [ ] regression
                - [ ] loss [uses NumericTarget]
        - [ ] New methods: binary Case
            - [ ] ARL (versions)
            - [ ] AUC
            - [ ] sklearn metrics
        - [ ] New methods: refitting
    - [ ] provide working natobooks for ARL on UCI / paper datasets
        - [ ] remove size factor for comparison
    - [ ] restructure code (src, import, ...)!
    - [ ] several normalized versions of ARL
        (see https://opus.bibliothek.uni-wuerzburg.de/opus4-wuerzburg/frontdoor/deliver/index/docId/9781/file/Dissertation-Lemmerich.pdf, page 20)
    - [ ] compare AUC / ARL 
    - [ ] implement linear regression refitting
    - [ ] plan for analyses (and include why) 
    - [ ] agenda + next steps
    - [ ] slides


## 2023-08-08

- agenda
    - implementing prediction target
- next steps
    - [ ] implement basic methods (regression, binary, etc.) 
    - [ ] classification
        - [ ] try to verify ARL on all paper datasets as best as possible (e.g., UCI)
            - [ ] https://sfb876.tu-dortmund.de/PublicPublicationFiles/duivesteijn_thaele_2014a.pdf
        - [ ] statistic: number/ratio of positive cases
        - [ ] compare AUC and ARL
            - [ ] same subgroups (in dataframe)
            - [ ] as separate QF (in subgroup search)
        - [ ] introduce size parameter for ARL and AUC (normalize ARL?)
        - [ ] analyze class imbalance in found subgroups
            - [ ] come up with a way to adjust for class imbalance in subgroups
    - [ ] implement linear regression refitting
    - [ ] plan for analyses (and include why) 
    - [ ] agenda + next steps

## 2023-08-01 (4/12)

- agenda
    -How to find subgroup, which performs worse
    -How to use pysubgroup discovery on incorrect_df 
- notes
    - through discussion on next steps
- next steps
    - [ ] implement basic methods (regression, binary, etc.) 
    - [ ] implement AUC/ARL
    - [ ] implement linear regression refitting
    - [ ] commit!

## 2023-07-11 (3/12)

- agenda
    - ARL
    - using ARL in subgroup discovery
- next steps
    - [ ] **work done + agenda + next steps**
    - [ ] fix (?) / test ARL
    - [ ] regression and simple classification with numeric target (give class imbalance in the subgroups for classification)
    - [ ] y_pred needs to come from out of sample prediction (k-fold or loo), only keep test prediction
    - [ ] implement ARL / AUC in subgroup discovery

## 2023-07-04 (2/12)

- next steps
    - [ ] double check if two people can be on the same pre-thesis
    - [ ] implement ARL
    - [ ] AUC and ARL (independent of subgroup discovery)
    - [ ] **run with subgroup discovery**
    - [ ] AUC as target
    - [ ] try to refit a model in each subgroup
    - [ ] **agenda + next steps**

## 2023-06-27 (1/12)

- TODO
    - [ ] agenda
    - [ ] double check if two people can be on the same pre-thesis
    - [ ] implement classification / prediction target (see paper: https://ieeexplore.ieee.org/abstract/document/7023405)

## 2023-06-07

- agenda
    - do we have to register the project?
- TODOs:
    - [ ] put papers in README
    - [ ] implement classification / prediction target (see paper: https://ieeexplore.ieee.org/abstract/document/7023405)
    - [ ] agenda
    - [x] take care registration and figure out deadline

## 2023-05-19

- try pysubgroup
- look for (large) data sets (predominately from the medical and biomedical domain)
- fit models (regression)
- use subgroup discovery to find subgroups where models work particularly well or badly (using a numeric target)
- think about how to modify pysubgroup to support binary prediction targets

If you want a more comprehensive exceptional model mining / subgroup discovery overview, look at this work: https://d-nb.info/110878075X/34
Also check for similar methods.
