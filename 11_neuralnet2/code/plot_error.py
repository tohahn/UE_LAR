import pickle
import numpy as np
from mnist import read, scale, generate_targets, model_accuracy
import matplotlib.pyplot as plt

if __name__ == "__main__":
    categories = 10
    test_images, test_labels = read(range(categories), "testing")
    test_images = scale(test_images)[:, None, :, :]
    test_targets = generate_targets(test_labels)

    mlnn = pickle.load(open("mnist_model.pickle", "rb"))

    accuracy = 100 * model_accuracy(mlnn, test_images, test_labels)
    error = mlnn.error(test_images, test_targets)
    print("Accuracy on test set: %.2f %%" % accuracy)
    print("Error = %.3f" % error)
    
    predict = np.argmax(mlnn.predict(test_images), axis=1)
    errors = []
    for i, x in enumerate(test_labels):
        if (x != predict[i]):
            errors.append(i)
    
    for row in range(8):
        for col in range(16):
            if (row == 7 and col == 12):
                break
            plt.subplot(8,16,row*16+col+1)
            plt.title("T:{0}/P:{1}".format(test_labels[errors[row*16+col],0], predict[errors[row*16+col]]))
            plt.imshow(test_images[errors[row*16+col],0])
            plt.axis('off')
    plt.show()
