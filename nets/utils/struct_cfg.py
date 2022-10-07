class yolov5_cfg:
    def __init__(self,
                 num_classes1:int=10,
                 num_classes2:int=10,
                 num_classes3:int=10):
        self.num_cls_1 = num_classes1
        self.num_cls_2 = num_classes2
        self.num_cls_3 = num_classes3
        
    def yolov5_s(self):
        """
        0    1      3520  models.common.Conv         [3, 32, 6, 2, 2]
        1    1     18560  models.common.Conv         [32, 64, 3, 2]
        2    1     18816  models.common.C3           [64, 64, 1]
        3    1     73984  models.common.Conv         [64, 128, 3, 2]
        4    2    115712  models.common.C3           [128, 128, 2]
        5    1    295424  models.common.Conv         [128, 256, 3, 2]
        6    3    625152  models.common.C3           [256, 256, 3]
        7    1   1180672  models.common.Conv         [256, 512, 3, 2]
        8    1   1182720  models.common.C3           [512, 512, 1]
        """
        cfg = {
                "feature_map":[
                                [[1],[3,32,6,2,2]],     # conv  0
                                [[1],[32,64,3,2]],      # conv  1
                                [[3],[64,64,1]],        # c3    2
                                [[1],[64,128,3,2]],     # conv  3
                                [[6],[128,128,2]],      # c3    4
                                [[1],[128,256,3,2]],    # conv  5
                                [[9],[256,256,3]],      # c3    6
                                [[1],[256,512,3,2]],    # conv  7
                                [[3],[512,512,1]],      # c3    8
                                ],
               "classify":{"1":[512,self.num_cls_1],
                           "2":[512,self.num_cls_2],
                           "3":[512,self.num_cls_3]}
               }
        return cfg
    def yolov5_n(self):
        """
        0    1      1760  models.common.Conv         [3, 16, 6, 2, 2]
        1    1      4672  models.common.Conv         [16, 32, 3, 2]
        2    1      4800  models.common.C3           [32, 32, 1]
        3    1     18560  models.common.Conv         [32, 64, 3, 2]
        4    2     29184  models.common.C3           [64, 64, 2]
        5    1     73984  models.common.Conv         [64, 128, 3, 2]
        6    3    156928  models.common.C3           [128, 128, 3]
        7    1    295424  models.common.Conv         [128, 256, 3, 2]
        8    1    296448  models.common.C3           [256, 256, 1]
        """
        cfg = {
                "feature_map":[
                                [[1],[3,16,6,2,2]],     # conv  0
                                [[1],[16,32,3,2]],      # conv  1
                                [[1],[32,32,1]],        # c3    2
                                [[1],[32,64,3,2]],      # conv  3
                                [[2],[64,64,2]],        # c3    4
                                [[1],[64,128,3,2]],     # conv  5
                                [[3],[128,128,3]],      # c3    6
                                [[1],[128,256,3,2]],    # conv  7
                                [[1],[256,256,1]],      # c3    8
                                ],
               "classify":{"1":[256,self.num_cls_1],
                           "2":[256,self.num_cls_2],
                           "3":[256,self.num_cls_3]}
               }
        return cfg
    def yolov5_l(self):  # 弃用
        """
        0   1      7040  models.common.Conv    [3, 64, 6, 2, 2]
        1   1     73984  models.common.Conv    [64, 128, 3, 2]
        2   3    156928  models.common.C3      [128, 128, 3]
        3   1    295424  models.common.Conv    [128, 256, 3, 2]
        4   6   1118208  models.common.C3      [256, 256, 6]
        5   1   1180672  models.common.Conv    [256, 512, 3, 2]
        6   9   6433792  models.common.C3      [512, 512, 9]
        7   1   4720640  models.common.Conv    [512, 1024, 3, 2]
        8   3   9971712  models.common.C3      [1024, 1024, 3]
        """
        cfg = {
                "feature_map":[
                                [[1],[3,64,6,2,2]],     # conv  0
                                [[1],[64,128,3,2]],      # conv  1
                                [[1],[128,128,3]],        # c3    2
                                [[1],[128,256,3,2]],      # conv  3
                                [[2],[256,256,6]],        # c3    4
                                [[1],[256,512,3,2]],     # conv  5
                                [[3],[512,512,9]],      # c3    6
                                [[1],[512,1024,3,2]],    # conv  7
                                [[1],[1024,1024,3]],      # c3    8
                                ],
               "classify":{"1":[1024,self.num_cls_1],
                           "2":[1024,self.num_cls_2],
                           "3":[1024,self.num_cls_3]}
               }
        return cfg
    def yolov5_m(self): # 弃用
        """
        0    1      5280  models.common.Conv     [3, 48, 6, 2, 2]
        1    1     41664  models.common.Conv     [48, 96, 3, 2]
        2    2     65280  models.common.C3       [96, 96, 2]
        3    1    166272  models.common.Conv     [96, 192, 3, 2]
        4    4    444672  models.common.C3       [192, 192, 4]
        5    1    664320  models.common.Conv     [192, 384, 3, 2]
        6    6   2512896  models.common.C3       [384, 384, 6]
        7    1   2655744  models.common.Conv     [384, 768, 3, 2]
        8    2   4134912  models.common.C3       [768, 768, 2]
        """
        cfg = {
                "feature_map":[
                                [[1],[3,48,6,2,2]],     # conv  0
                                [[1],[48,96,3,2]],      # conv  1
                                [[1],[96,96,2]],        # c3    2
                                [[1],[96,192,3,2]],      # conv  3
                                [[2],[192,192,4]],        # c3    4
                                [[1],[192,384,3,2]],     # conv  5
                                [[3],[384,384,6]],      # c3    6
                                [[1],[384,768,3,2]],    # conv  7
                                [[1],[768,768,2]],      # c3    8
                                ],
               "classify":{"1":[768,self.num_cls_1],
                           "2":[768,self.num_cls_2],
                           "3":[768,self.num_cls_3]}
               }
        return cfg


# class yolov5_cls_base():
#     def __init__(self,a,b,c,func,name):
#         self.a = a
#         self.b = b
#         self.c = c
#         self.func = func
#         self.name = name
#     def run(self):
#         return self.func(num_classesself.a,self.b,self.c)
#     def get_name(self):
#         return self.name