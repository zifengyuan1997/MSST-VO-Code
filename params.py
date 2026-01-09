import os
from modelUsed import modelUsed
from datasetUsed import datasetUsed

class Parameters():
    def __init__(self):
        self.n_processors = 8
        # Path
        if datasetUsed == 'KITTI':
            self.data_dir =  'KITTI'
        elif datasetUsed == 'VOD':
            self.data_dir = 'VOD'

        self.image_dir = self.data_dir + "\\images\\"
        self.pose_dir = self.data_dir + '\\pose_GT\\'

        if datasetUsed == 'KITTI':
            self.train_video = ['00', '01', '02', '06', '08', '03']
            self.valid_video = ['04', '05', '07', '09', '10']
        elif datasetUsed == 'VOD':
            self.train_video = ['00', '02', '03', '01', '05', '09', '10', '11', '13', '14', '15', '16', '17', '18', '20', '21', '22', '23']
            self.valid_video = ['12', '06', '07', '04', '19']
        self.partition = None  # partition videos in 'train_video' to train / valid dataset  #0.8
        # self.inc_down = [32, 64, 128, 256, 512]
        self.inc_down = [16, 32, 64, 128, 256]

        # Data Preprocessing
        self.resize_mode = 'rescale'  # choice: 'crop' 'rescale' None
        if datasetUsed == 'KITTI':
            if modelUsed == 'STDNVO':
                self.img_w = 224   # original size is about 1226
                self.img_h = 224   # original size is about 370
            elif modelUsed == 'TSformer':
                self.img_w = 640
                self.img_h = 192
            elif modelUsed == 'SWformer':
                self.img_w = 678
                self.img_h = 224
            elif modelUsed == 'TartanVO':
                self.img_w = 608*2
                self.img_h = 184*2
            else:
                self.img_w = 608
                self.img_h = 184
        elif datasetUsed == 'VOD':
            if modelUsed == 'STDNVO':
                self.img_w = 224   # original size is about 1226
                self.img_h = 224   # original size is about 370
            elif modelUsed == 'TSformer':
                self.img_w = 640
                self.img_h = 192
            elif modelUsed == 'SWformer':
                self.img_w = 678
                self.img_h = 224
            elif modelUsed == 'TartanVO':
                self.img_w = 484*2
                self.img_h = 384*2
            else:
                self.img_w = 484
                self.img_h = 304
        self.img_means =  (0.19007764876619865, 0.15170388157131237, 0.10659445665650864)
        self.img_stds =  (0.2610784009469139, 0.25729316928935814, 0.25163823815039915)
        self.minus_point_5 = True

        if modelUsed == 'TSformer' or modelUsed == 'SWformer':
            self.seq_len = (4, 4)
        elif modelUsed == 'STDNVO' or modelUsed == 'CNNVO' or modelUsed == 'TartanVO':
            self.seq_len = (2, 2)
        elif modelUsed == 'DVOAM':
            self.seq_len = (8, 8)
        else:
            self.seq_len = (8, 8)

        if self.seq_len[0] !=2:
            self.sample_times = self.seq_len[0]//2
        else:
            self.sample_times = self.seq_len[0]//2

        # Data info path
        self.train_data_info_path = 'datainfo\\train_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(''.join(self.train_video), ''.join(self.valid_video), self.partition, self.seq_len[0], self.seq_len[1], self.sample_times)
        self.valid_data_info_path = 'datainfo\\valid_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(''.join(self.train_video), ''.join(self.valid_video), self.partition, self.seq_len[0], self.seq_len[1], self.sample_times)


        # Model
        self.backBoneLayer = 5
        self.activeFunc = 'lrelu'
        self.rnn_hidden_size = 1000
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.rnn_dropout_out = 0.5
        self.rnn_dropout_between = 0   # 0: no dropout
        self.clip = None
        self.batch_norm = True
        # Training
        if modelUsed == 'MSSTVO':
            self.epochs = 100
        if modelUsed == 'DeepVO':
            self.epochs = 250
        if modelUsed == 'PoseConvGRU' or modelUsed == 'TSformer' or modelUsed == 'SWformer' or modelUsed == 'TartanVO':
            self.epochs = 100
        if modelUsed == 'DVOAM':
            self.epochs = 150
        if modelUsed == 'CEGVO' or modelUsed == 'STDNVO' or modelUsed == 'CNNVO':
            self.epochs = 200

        self.batch_size = 2
        self.pin_mem = True
        self.optim = {'opt': 'Adam'}
        # {'opt': 'Adagrad', 'lr': 0.0005, 'eps': 1e-8}
        			# Choice:
        			# {'opt': 'Adagrad', 'lr': 0.001}

        			# {'opt': 'Cosine', 'T': 100 , 'lr': 0.001}

        # Pretrain, Resume training
        self.pretrained_flownet = None
                                # Choice:
                                # None
                                # './pretrained/flownets_bn_EPE2.459.pth.tar'
                                # './pretrained/flownets_EPE1.951.pth.tar'
        self.resume = True  # resume training
        self.resume_t_or_v = '.train'
        self.load_model_path = 'models\\t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model{}'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]), self.resume_t_or_v)
        self.load_optimizer_path = 'models\\t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer{}'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]), self.resume_t_or_v)

        self.record_path = 'records\\t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.txt'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
        self.save_model_path = 'models\\t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
        self.save_optimzer_path = 'models\\t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))


        if not os.path.isdir(os.path.dirname(self.record_path)):
            os.makedirs(os.path.dirname(self.record_path))
        if not os.path.isdir(os.path.dirname(self.save_model_path)):
            os.makedirs(os.path.dirname(self.save_model_path))
        if not os.path.isdir(os.path.dirname(self.save_optimzer_path)):
            os.makedirs(os.path.dirname(self.save_optimzer_path))
        if not os.path.isdir(os.path.dirname(self.train_data_info_path)):
            os.makedirs(os.path.dirname(self.train_data_info_path))

par = Parameters()

