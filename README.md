## Subject Harmonization of Digital Biomarkers: Improved Detection of Mild Cognitive Impairment from Language Markers
Official code for paper: "Subject Harmonization of Digital Biomarkers: Improved Detection of Mild Cognitive Impairment from Language Markers", Bao Hoang, Yijiang Pang, Hiroko H. Dodge, and Jiayu Zhou, PSB 2024.

## Overview 
Mild cognitive impairment (MCI) represents the early stage of dementia including Alzheimer’s disease (AD) and plays a crucial role in developing therapeutic interventions and treatment. Early detection of MCI offers opportunities for early intervention and significantly benefits cohort enrichment for clinical trials. Imaging markers and in vivo markers
in plasma and cerebrospinal fluid biomarkers have high detection performance, and yet their prohibitive costs and intrusiveness demand more affordable and accessible alternatives. The recent advances in digital biomarkers, especially language markers, have shown great potential, where variables informative to MCI are derived from linguistic and/or speech and later
used for predictive modeling. A major challenge in modeling language markers comes from the variability of how each person speaks. As the cohort size for language studies is usually
small due to extensive data collection efforts, the variability among persons makes language markers hard to generalize to unseen subjects. In this paper, we propose a novel subject harmonization tool to address the issue of distributional differences in language markers across subjects, thus enhancing the generalization performance of machine learning models. Our empirical results show that machine learning models built on our harmonized features have improved prediction performance on unseen data.

## Language Marker Extractor
To extract language marker from the transcripts, you need to extract syntactic complexity feature using [L2 Syntactic Complexity Analyzer](https://sites.psu.edu/xxl13/l2sca/). After that, put your syntactic complexity feature in file `rawdata/syntactic_complexity_measures.csv` and your transcripts data in folder `Transcriptions`, then run command ```python feature_extractor.py```

It will give you 99-dimensional language marker feature in `rawdata/id2feature.p`

## Harmonization Methods
Here we provide several demos of using harminzation commands. Remember to use your own transcripts data, the existing data in repo is just for demo.

- **None Harmonization:**

    - Run ```python main.py --solver Standard_solver```

- **Generalized least squares:**
  
    - Run ```python main.py --solver Baseline_confounder_solver```
 
- **Deep harmonization - subject (Proposed method):**
  
    - Run ```python main.py --solver whiting_solver```

- **Deep harmonization - confounder:**
  
    - Run ```python main.py --solver whiting_confounder_solver```

    - You can change confounder variable using variable ``config.training.confounder_var`` in ``configs/cfg.py``

## Data Request
The data is available upon request at [https://www.i-conect.org/](https://www.i-conect.org/)

## Acknowledgement
This material is based in part upon work supported by the National Science Foundation under
Grant IIS-2212174, IIS-1749940, Office of Naval Research N00014-20-1-2382, and National
Institute on Aging (NIA) RF1AG072449, R01AG051628, R01AG056102.

      



