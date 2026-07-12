# matroid
Different matroids represent different laboratory rules:

* A partition matroid limits how many experiments come from one composition range, annealing schedule or cold-work level.
* A laminar matroid handles nested groups, such as base alloy, doped alloy, heat-treatment subgroup and testing subgroup.
* A transversal matroid assigns samples to available tests such as resistivity, Seebeck coefficient, fatigue and microscopy.
* A graphic matroid avoids redundant loops in process–property networks.
* A uniform matroid simply limits the total number of experiments. 
The optimisation then searches for a partition that performs well in the worst plausible scenario.
The basic workflow is:

Generate candidates → encode constraints → create uncertainty scenarios → evaluate candidates → partition them → calculate worst-case loss → select the strongest partition → test experimentally → update the model . 