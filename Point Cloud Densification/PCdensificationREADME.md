In each folder, I have detailed in a text file what datatypes should be placed where

Check the code parameters and once the folders have the correct data, just run rgb2pc (ensuring you have the libraries at the top of the py file installed onto your environment.

The error correct vector (bestEC) is applying corrections to each of the angles (not projected x,y,z)

This error correction is found manually for each lane - the example bestEC in the code is for Lane 1 of A11 Red Lodge

Best do this using cloud compare, changing each angle, see what gives best result after lots of experimenting.

Or automate the process of using the projected images, and semantic segmentation using the lane lines in the raw and projected images to maximise the alignment that gives the highest IoU score (see report or speak to me or Diana)

