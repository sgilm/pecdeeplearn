from __future__ import division

import pecdeeplearn as pdl
import data_path
import time
import numpy as np


# Create an experiment object to keep track of parameters and facilitate data
# loading and saving.
exp = pdl.utils.Experiment(data_path.get())
exp.load_experiment('single_a_dense_three_landmark_1')
exp.add_param('num_training_volumes', 45)
exp.add_param('max_points_per_volume', 25000)
exp.add_param('margins', (12, 12, 0))
exp.add_param('local_patch_shape', [25, 25, 1])
exp.add_param('local_patch_input_shape', [25 * 25])
exp.add_param('landmark_1', 'Sternal angle')
exp.add_param('landmark_2', 'Left nipple')
exp.add_param('landmark_3', 'Right nipple')
exp.add_param('join_dense1_num_units', 986)
exp.add_param('batch_size', 5000)
exp.add_param('update_learning_rate', 0.001)
exp.add_param('update_momentum', 0.9)
exp.add_param('max_epochs', 100)
exp.add_param('validation_prop', 0.2)

# List and load all volumes.
vol_list = exp.list_volumes()
test_vol_names = ['VL00027', 'VL00032', 'VL00033', 'VL00035', 'VL00042',
                  'VL00047', 'VL00049', 'VL00056', 'VL00066', 'VL00067',
                  'VL00070', 'VL00074', 'VL00080', 'VL00090', 'VL00096']
for vol_name in test_vol_names:
    try:
        vol_list.remove(vol_name)
        vol_list.append(vol_name)
    except ValueError:
        pass
vols = [exp.load_volume(vol) for vol in vol_list]

# Standardise the data.
pdl.utils.standardise_volumes(vols)

# Split into a training set and testing set.
training_vols = vols[:exp.params['num_training_volumes']]
testing_vols = vols[exp.params['num_training_volumes']:]

# Create training maps.
training_maps = [
    pdl.extraction.half_half_map(
        vol,
        max_points=exp.params['max_points_per_volume'],
        margins=exp.params['margins']
    )
    for vol in training_vols]

# Create an Extractor.
ext = pdl.extraction.Extractor()

# Add features.
ext.add_feature(
    feature_name='local_patch',
    feature_function=lambda volume, point:
    pdl.extraction.flat_patch(volume, point, exp.params['local_patch_shape'])
)
ext.add_feature(
    feature_name='landmark_1',
    feature_function=lambda volume, point:
    pdl.extraction.landmark_displacement(
        volume, point, exp.params['landmark_1'])
)
ext.add_feature(
    feature_name='landmark_2',
    feature_function=lambda volume, point:
    pdl.extraction.landmark_displacement(
        volume, point, exp.params['landmark_2'])
)
ext.add_feature(
    feature_name='landmark_3',
    feature_function=lambda volume, point:
    pdl.extraction.landmark_displacement(
        volume, point, exp.params['landmark_3'])
)

# Create the net.
net = exp.unpickle_network('net')

# Perform predictions on all testing volumes in the set.
print('Beginning predictions.\n')
prediction_start_time = time.time()
for i, testing_vol in list(enumerate(testing_vols))[-3:]:

    # Perform the prediction on the current testing volume.
    print("Predicting on volume " + testing_vol.name + ".")
    predicted_vol = ext.predict(
        net,
        testing_vol,
        exp.params['batch_size'],
        bounds=[
            exp.params['margins'],
            np.array(testing_vol.shape) - 1 - np.array(exp.params['margins'])
        ]
    )

    # Save the prediction probabilities for comparison.
    predicted_vol.name += "_prob"
    exp.export_nii(predicted_vol)

    # Print prediction progress.
    pdl.utils.print_progress(time.time() - prediction_start_time,
                             i + 1,
                             len(testing_vols))
