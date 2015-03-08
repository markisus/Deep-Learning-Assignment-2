unlabeled_data_path = '/scratch/courses/DSGA1008/A2/binary/unlabeled_X.bin'
train_data_path = '/scratch/courses/DSGA1008/A2/binary/train_X.bin'
train_labels_path = '/scratch/courses/DSGA1008/A2/binary/train_y.bin'

unlabeled_image_count = 100000
train_image_count = 500

image_width = 96
image_height = 96
image_channels = 3
receptive_field_size = 6
patch_size = image_channels*receptive_field_size*receptive_field_size

random_sample_count = 10000

num_centroids = 1600