import cxr_dataset as CXR
import eval_model as E
import model as M


# you will need to customize PATH_TO_IMAGES to where you have uncompressed
# NIH images
if __name__ == '__main__':
    PATH_TO_IMAGES = "C:/Users/User/Desktop/Hibah Dikti/DB/images"
    WEIGHT_DECAY = 1e-4
    LEARNING_RATE = 0.01
    preds, aucs = M.train_cnn(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY)