from pecdeeplearn.utils import Experiment

if __name__ == '__main__':
    exp = Experiment(r'C:\Users\sgilm\Documents\University\ENGSCI 700\data',
                     'test')
    exp.add_param('batch_size', 5000)
    exp.add_param('kernel', [1, 21, 21])
    exp.record_params()

