from simplecv import registry
from simplecv import dp_train as train
from data.segloader import SegDataLoader
from module import deeplab, deeplab_decoder, deeplab_encoder

if __name__ == '__main__':
    args = train.parser.parse_args()
    train.run('defalut', './tmp',
              after_construct_launcher_callbacks=[None])
