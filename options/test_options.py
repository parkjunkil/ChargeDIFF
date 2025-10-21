# Reference: The code has been modified from the LDM repo: https://https://github.com/yccyenchicheng/SDFusion

from .base_options import BaseOptions

class TestOptions(BaseOptions):

    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = False

        self.phase = 'test'

        self.parser.add_argument("--target",   default=None, help='conditional generation target')
        self.parser.add_argument("--n_gen",    type=int, default=100, help='number of generation attempts')
        self.parser.add_argument("--save_dir", type=str, default=None, help="repository to save generated samples")
        self.parser.add_argument("--num_batch", type=int, default=1, help="number of batches to sample")
        self.parser.add_argument("--sample_dataset", type=str, default='mp-20', help="dataset to sample atom num")
        
