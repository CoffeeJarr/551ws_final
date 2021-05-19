stop_sign_detector.py: is the main code file include all functions and main domain.

a.p: is the file created by "pickle" to store and load hand_labeled rgb matrix. It represents X in IID training dataset. Odd rows are stop-sign red color and even rows are similar colors such as yellow, brown. 

b.p: is another file created by "pickle" to store and load 1 or -1 corresponding to whether the rgb is red or not red color. Kt represents Y in IID training dataset.

acquire data.py: is the code file using roipoly to hand-label color regions and use pickle to store those data for the use of machine learning from main file.




