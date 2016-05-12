import pecdeeplearn as pdl
import data_path

if __name__ == '__main__':
    exp = pdl.utils.Experiment(data_path.get(), 'single_hidden_layer')
    x = exp.unpickle_volume('predicted_volume')
    x.show_slice(0)