HOME ='/root/autodl-tmp/SMENet/SMENet'
# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (86, 91, 82)

voc = {
    'num_classes': 5,
    'lr_steps': (1000,2000,3000,4000,5000,6000,7740,9000,10000,11000,12000,13000,14000,15000,16000,
                 17000,18000,19000,20000,25000,30000,35000,40000,45000,50000,55000,60000,75000),
    'max_iter': 20000,
    'feature_maps': [50, 25, 13, 7, 5, 3],  # scale of feature map
    'min_dim': 400,   # input size
    'steps': [8, 16, 30, 57, 80, 133],  # The mapping relationship between the feature map  and the input img
    'min_sizes': [25, 65, 116, 167, 218, 269],
    'max_sizes': [65, 116, 167, 218, 269, 320],
    'aspect_ratios': [[2, 0.65], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
