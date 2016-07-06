from __future__ import division

import pecdeeplearn as pdl
import data_path
import sys
import time
import numpy as np


# Load the experiment.
exp = pdl.utils.Experiment(data_path.get())
exp.load_experiment(sys.argv[1])

# List and load all volumes.
vol_list = exp.list_volumes()
for vol_name in ['VL00033', 'VL00034']:
    try:
        vol_list.remove(vol_name)
        vol_list.append(vol_name)
    except ValueError:
        pass
vols = [exp.load_volume(vol) for vol in vol_list]

# Standardise the data.
pdl.utils.standardise_volumes(vols)

# Get the test volumes.
testing_vols = vols[45:]

# Create an Extractor.
ext = pdl.extraction.Extractor()

# Add features.
ext.add_feature(
    feature_name='local_patch',
    feature_function=lambda volume, point:
    pdl.extraction.flat_patch(volume, point, [25, 25, 1])
)

# Load the network.
net = exp.unpickle_network('net')

# Perform predictions on all testing volumes in the set.
print('Beginning predictions.\n')
prediction_start_time = time.time()
for i, testing_vol in list(enumerate(testing_vols))[:5]:

    # Perform the prediction on the current testing volume.
    print("Predicting on volume " + testing_vol.name + ".")
    predicted_vol = ext.predict(
        net,
        testing_vol,
        1000,
        bounds=[
            (12, 12, 0),
            np.array(testing_vol.shape) - 1 - np.array((12, 12, 0))
        ]
    )

    # Calculate statistics for this prediction and record them.
    correct_positives, false_positives, false_negatives = \
        pdl.utils.prediction_stats(testing_vol.seg_data,
                                   predicted_vol.seg_data)
    dice = pdl.utils.dice_coefficient(testing_vol.seg_data,
                                      predicted_vol.seg_data)
    exp.add_result(testing_vol.name + '_correct_positives', correct_positives)
    exp.add_result(testing_vol.name + '_false_positives', false_positives)
    exp.add_result(testing_vol.name + '_false_negatives', false_negatives)
    exp.add_result(testing_vol.name + '_dice', dice)

    # Save the prediction for comparison.
    exp.export_nii(predicted_vol)

    # Print prediction progress.
    pdl.utils.print_progress(time.time() - prediction_start_time,
                             i + 1,
                             len(testing_vols))

# Record the parameters and results.
exp.record()
