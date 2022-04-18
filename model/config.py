
class Config:
    '''将所有参数封装一个类，方便调用'''
    def __init__(self):
        
        # two class
        self.cls = 2
        
        # 输入空间参数
        # patch 一般为方形，长宽相等
        self.patch_height = 1
        self.patch_width = 1
        
        self.image_height = 640
        self.image_width = 480
        self.channels = 3
        
        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0
        # patch 数量--序列长度
        self.patch_num = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)
        # patch 大小
        self.patch_dim_l = self.patch_height * self.patch_width * self.channels
        
        
        
        # 输出空间参数
        self.raster_patch_height = 1
        self.raster_patch_width = 1
        self.raster_height = 60
        self.raster_width = 80
        
        
        assert self.raster_height % self.raster_patch_height == 0 and self.raster_width % self.raster_patch_width == 0
        # patch 数量--序列长度
        self.raster_patch_num = (self.raster_height // self.raster_patch_height) * (self.raster_width // self.raster_patch_width)
        # patch 大小 - 最终输出维度
        self.raster_patch_dim = self.raster_patch_height * self.raster_patch_width * self.channels
        
        # embedding长度--自己设计
        self.embed_dim = 64
        # 注意头数量,以及每个头内的编码长度
        self.n_heads = 8 
        self.head_dim = 32
        # k、v多头注意总长度
        self.ma_dim = self.head_dim * self.n_heads
        # mlp 隐藏层长度
        self.hidden_dim = 512
        
        # encoder-decoder 数量
        self.N = 6
        
        self.dropout = 0.1
        
        self.MAX_EPOCH = 10
        self.BATCH_SIZE = 8
        self.LR = 0.1
        self.log_interval = 5
        self.val_interval = 1
        
        self.log_dir = 'model_log/'
        self.train_dir = 'data/apartment_0/train/' 
        self.valid_dir = 'data/apartment_0/valid/'
        
        self.test_dir = 'data/test/'
        self.output_dir = 'output/'
        