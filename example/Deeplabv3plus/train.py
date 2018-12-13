from simplecv import registry
from simplecv import train
from data.segloader import SegDataLoader
from module import deeplab, deeplab_decoder, deeplab_encoder

if __name__ == '__main__':
    args = train.parser.parse_args()
    train.run(0, 'defalut', './tmp')
