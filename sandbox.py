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
    experiment_names[6] = 'single_acs_conv_triple_landmark'
    experiment_names[7] = 'double_a_conv_triple_landmark'

    experiment_number = 5
    experiment_index = 1

    exp = pdl.utils.Experiment(data_path.get())
    exp.load_experiment(experiment_names[experiment_number] +
                        "_" + str(experiment_number))

    test = exp.unpickle_volume('test_volume')
    test.show_slice(0)

    pred =exp.unpickle_volume('predicted_vol')
    pred.show_slice(0, seg_cmap='Blues')