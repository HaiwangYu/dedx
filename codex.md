

Task: consider the prior distribution of each particle distributions
1. add an option for the optional prior-distribution-dir
2. if this option is provided the program will generate the prior distributions for each PID, if the prior distributions for the PID already exists, use the existing one unless another option "prior-distribution-force-gen" is true (default is false)
3. after reading in the prior-distribution, at evaluation time use this to calculate the score for each PID hypothesis: prior-distribution (p) / abs((dedx - band_mean(p))/band_sigma(p)), then categorize that dedx as the highest score
4. ask me if anything not clear
    - a histogram, use the same binning as the band and also save as a csv
    - derive from the ROOT

Followup:
1. when prior-distribution-dir is given, draw png of the prior-distribution
2. the prior distribution of each PID is defined as: n-particles-of-specific-PID / n-all-particles-in-that-p-bin, re-write the code on related parts