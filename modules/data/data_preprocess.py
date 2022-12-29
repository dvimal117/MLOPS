from keras.preprocessing.image import ImageDataGenerator
import pickle

class preprocess:
  
  def train_preprocess():

    file_path = 'C:/Users/vimal.dhanapal/Downloads/project1.0/project/Car-Bike-Dataset'
    # classes = sorted(os.listdir(file_path))
    # n_classes = len(classes)
    batch_size = 32
    img_height = 256
    img_width = 256

    print("Preprocessing the data...!")
    data_generator = ImageDataGenerator(rescale=1/255., # Normalisaion
                               rotation_range = 90,
                               width_shift_range = 0.1,
                               horizontal_flip = True,
                               vertical_flip = True, 
                               validation_split = 0.2) #portion of 80% : 20%

    train_set = data_generator.flow_from_directory(file_path, 
                                class_mode = 'binary',
                                target_size = (img_height,img_width), 
                                shuffle = True, 
                                batch_size = batch_size,
                                save_to_dir='file_path/train',
                                save_format="png", 
                                subset = 'training')

    test_set = data_generator.flow_from_directory(file_path, 
                                class_mode = 'binary', 
                                target_size = (img_height,img_width), 
                                shuffle = False, 
                                batch_size = batch_size, 
                                subset = 'validation')

    

    return (train_set,test_set)
print (preprocess.train_preprocess())
#preprocessing.train_preprocess()






