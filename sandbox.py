import data_path
import pecdeeplearn as pdl

exp = pdl.utils.Experiment(data_path.get(), 'test')

vol = exp.load_volume('VL00037')
hhmap = pdl.extraction.half_half_map(vol, margins=(20, 20, 20))
ext = pdl.extraction.Extractor()

ext.add_feature('test',
                lambda volume, point: pdl.extraction.point_offset(volume, point, [0, 0, 0]))
ext.find_feature_sizes(vol, point_map=hhmap)
y = 1