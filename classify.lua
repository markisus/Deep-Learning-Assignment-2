require 'preprocessing'
require 'torch'
require 'math'
require 'constants'
require 'xlua'

print("Loading test data")
data = torch.DiskFile(test_data_path, 'r', true)
data:binary():littleEndianEncoding()
tensor = torch.ByteTensor(test_image_count, image_channels, image_width, image_height)
data:readByte(tensor:storage())
test_data = tensor:float()

print("Loading trained neural net")
net = torch.load(trained_model_path, 'ascii')

print("Classifying")
guesses = torch.IntTensor(test_image_count)

for i = 1, test_image_count do
    xlua.progress(i, test_image_count)
    image = test_data[i]
    patches = patchify(image)
    features = extract_features(patches)
    features:resize(1, features:size()[1], features:size()[2]) --batch size 1
    classification = net:forward(features)

    best_index = 1
    best_delta = math.huge
    for guess = 1, num_categories do
    	guess_as_vector = torch.DoubleTensor(num_categories):fill(0)
	guess_as_vector[guess] = 1
    	delta = criterion:forward(classification, guess_as_vector) 
	if delta < best_delta then
	   best_deta = delta
	   best_index = guess
	end
    end
    guesses[i] = best_index
end

print("Saving classifications")
torch.save("classifications.data", guesses, 'ascii')