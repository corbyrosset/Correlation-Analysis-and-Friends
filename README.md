# RL_final
RL final project

Look only at files in src/ directory. Particularly cca.m and kcca.m. Supporting files for kcca.m are isvd.m (courtesy of Dr.
Arora) and scalableKCCA.m
Other standalone software packages were tested, they didn't work, thus they are in the IGNORE directory. 

The logs.txt file contains console output. Lower in the file shows most recent progress: kcca achieves ~63% accuracy on the test
set even though the training/dev accuracy was only ~55%. 

You may run either cca.m or kcca.m, the latter will take nearly an hour. For faster runtimes, change lines 25-27 in kcca.m
to make the training, dev, and test sets smaller. Also see the figures/ directory to see plots of the train and test data onto
the top two principle directions. 
