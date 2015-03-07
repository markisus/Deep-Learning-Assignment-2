require 'nn'
require 'torch'
require 'math'
require 'unsup'

data_path = '/scratch/courses/DSGA1008/A2/binary/unlabeled_X.bin'
image_count = 100000
image_width = 96
image_height = 96
image_channels = 3

random_sample_count = 1000
receptive_field_size = 10

num_centroids = 10

data = torch.DiskFile(data_path, 'r', true)
data:binary():littleEndianEncoding()
tensor = torch.ByteTensor(image_count, image_channels, image_width, image_height)
data:readByte(tensor:storage())
tensor:float()

-- Random Patch Selection 

random_patches = torch.FloatTensor(random_sample_count, image_channels, receptive_field_size, receptive_field_size)
patch_count = 0
while patch_count < random_sample_count do
    image_index = math.random(1, image_count)
    image = tensor[image_index]
    patch_x = math.random(1, image_width - receptive_field_size + 1)
    patch_y = math.random(1, image_height - receptive_field_size + 1)
    patch = image[{{}, {patch_x, patch_x + receptive_field_size - 1}, {patch_y, patch_y + receptive_field_size - 1}}]
    patch_count = patch_count + 1
    random_patches[patch_count] = patch
    mean = random_patches[{patch_count, {}, {}, {}}]:mean()
    std = random_patches[{patch_count, {}, {}, {}}]:std()
    if std == 0 then
       print(string.format('Discarding patch from img %d, x:%d, y:%d for 0 std', image_index, patch_x, patch_y))
       goto continue
    end
    random_patches[{patch_count, {}, {}, {}}]:add(-mean)
    random_patches[{patch_count, {}, {}, {}}]:div(std)
    ::continue::
end

-- Whitening

patch_size = image_channels*receptive_field_size*receptive_field_size
random_patches_flattened = torch.reshape(random_patches, random_sample_count, patch_size)
means = torch.mean(random_patches_flattened, 1)
random_patches_flattened:add(means:expand(random_sample_count, patch_size))
covariances = torch.mm(random_patches_flattened:transpose(1,2), random_patches_flattened):div(random_sample_count)
Q, D, Q_T = torch.svd(covariances)
D_inv_sqrt = torch.pow(D, -1):sqrt():resize(patch_size, 1)
D_isQ_t = Q_T:cmul(D_inv_sqrt:expand(patch_size, patch_size))
W_zca = torch.mm(Q, D_isQ_t)
whitened = torch.mm(W_zca, random_patches_flattened:transpose(1,2)):transpose(1,2)

-- k-means

kmeans = unsup.kmeans(whitened:double(), num_centroids, 100, 100)

