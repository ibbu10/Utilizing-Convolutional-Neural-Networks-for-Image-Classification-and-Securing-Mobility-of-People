
import numpy as np 
from keras.preprocessing import image
from keras.models import load_model

model=load_model('my_model.h5',custom_objects={'f1_score': f1_score,'Precision': Precision, 'recall': recall, 'tp' : tp, 'fn' : fn, 'fp' : fp})

test_image = image.load_img('' ,target_size = (128,128))
test_image = image.img_to_array(test_image)

test_batch = np.expand_dims(test_image,axis=0)

print(test_batch.shape)
layer_outputs = [layer.output for layer in model.layers[:2]] 
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
activations = activation_model.predict(test_batch) 
first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
plt.axis('off')
plt.savefig("Filtered.png",quality=95, dpi= 300)



from matplotlib.pyplot import figure

layer_names = []
for layer in model.layers[:7]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
images_per_row = 15
for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.axis('off')
    plt.savefig("Filtered.png",quality=95, dpi= 100)
