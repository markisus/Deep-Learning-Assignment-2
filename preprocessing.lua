require 'nn'
require 'torch'
require 'math'

data_path = '/scratch/courses/DSGA1008/A2/binary/unlabeled_X.bin'
image_count = 100000
image_width = 96
image_height = 96
image_channels = 3

random_sample_count = 1000
receptive_field_size = 10

data = torch.DiskFile(data_path, 'r', true)
data:binary():littleEndianEncoding()
tensor = torch.ByteTensor(image_count, image_channels, image_width, image_height)
data:readByte(tensor:storage())
tensor:float()

-- Random Patch Selection 

random_patches = torch.FloatTensor(random_sample_count, image_channels, receptive_field_size, receptive_field_size)
for i = 1, random_sample_count do
    image = tensor[math.random(1, image_count)]
    patch_x = math.random(1, image_width - receptive_field_size + 1)
    patch_y = math.random(1, image_height - receptive_field_size + 1)
    patch = image[{{}, {patch_x, patch_x + receptive_field_size - 1}, {patch_y, patch_y + receptive_field_size - 1}}]
    random_patches[i] = patch
end