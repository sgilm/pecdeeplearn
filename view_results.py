import pecdeeplearn as pdl
import data_path

if __name__ == '__main__':

    experiment_names = {}
    experiment_names[1] = 'single_dense'
    experiment_names[2] = 'single_a_conv'
    experiment_names[3] = 'single_a_conv_single_landmark'
    experiment_names[4] = 'single_a_conv_triple_landmark'
    experiment_names[5] = \
        'single_local_a_conv_single_context_a_conv_triple_landmark'

    experiment_number = 4
    experiment_index = 1
    if experiment_index == 1:
        suffix = ''
    else:
        suffix = ' ' + str(experiment_index)

    exp = pdl.utils.Experiment(data_path.get(),
                               experiment_names[experiment_number] + suffix)

    test = exp.unpickle_volume('test_volume')
    test.show_slice(0)

    pred =exp.unpickle_volume('predicted_volume')
    pred.show_slice(0, seg_cmap='Blues')