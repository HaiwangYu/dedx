# add a ROI-like curve in addition to current simple efficiency and purity determined by the highest score
- score_frac_{pid} := score_{pid}/Sum(scores) (range: 0 - 1)
- categorize a reco track as PID when score_frac_{PID} is larger than a certain TH
- obtain the efficiency vs. purity curve for each PID with each point corresponding to a score_frac TH
  - the efficiency and purity should be integrated over a momentum range, which is analysis-momentum-range
- calculate the AUC for each PID and add in the ROI-like curve
