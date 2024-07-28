import numpy as np
import scipy.io
from pathlib import Path
from libsvm.svmutil import svm_train, svm_predict

class HandWritten_Digit_Dataset():
    def __init__(self, mat_file:Path) -> None:
        self.trainX:np.ndarray = None
        self.trainY:np.ndarray = None
        self.testX:np.ndarray = None
        self.testY:np.ndarray = None
        self._read_handwritten_digit(mat_file=mat_file)

    def _read_handwritten_digit(self, mat_file:Path, to_float:bool=True) -> None:
        mat_data = scipy.io.loadmat(mat_file)
        self.trainX = mat_data['train']
        self.trainY = mat_data['train_label'].reshape(-1).astype(np.int32)
        self.testX = mat_data['test']
        self.testY = mat_data['test_label'].reshape(-1).astype(np.int32)
        if to_float:
            self.trainX = self.trainX.astype(np.float32)
            self.testX = self.testX.astype(np.float32)
    
    def get_digits_XY(self, digits:list) -> tuple:
        train_mask = np.isin(self.trainY, digits)
        test_mask = np.isin(self.testY, digits)
        return self.trainX[train_mask], self.trainY[train_mask], self.testX[test_mask], self.testY[test_mask]

if __name__ == "__main__":

    handwritten_digit_dataset = HandWritten_Digit_Dataset(mat_file="usps.mat")
    
    digits = [6, 9]
    trainX, trainY, testX, testY = handwritten_digit_dataset.get_digits_XY(digits)

    trainY = np.where(trainY == 6, 1, -1)
    testY = np.where(testY == 6, 1, -1)

    trainX_list = trainX.tolist()
    trainY_list = trainY.tolist()
    testX_list = testX.tolist()
    testY_list = testY.tolist()

    C_range = [2**i for i in range(-5, 6, 2)]
    gamma_range = [2**i for i in range(-15, -4, 2)]

    best_accuracy = -1
    best_params = {}

    for C in C_range:
        for gamma in gamma_range:
            param = f'-v 5 -c {C} -g {gamma} -q'
            cv_accuracy = svm_train(trainY_list, trainX_list, param)
            if cv_accuracy > best_accuracy:
                best_accuracy = cv_accuracy
                best_params['C'] = C
                best_params['gamma'] = gamma

    best_C = best_params['C']
    best_gamma = best_params['gamma']
    print(f"Best C: {best_C}, Best gamma: {best_gamma}")

    final_param = f'-c {best_C} -g {best_gamma} -q'
    model = svm_train(trainY_list, trainX_list, final_param)

    p_label_train, p_acc_train, p_val_train = svm_predict(trainY_list, trainX_list, model)
    train_accuracy = p_acc_train[0]

    p_label_test, p_acc_test, p_val_test = svm_predict(testY_list, testX_list, model)
    test_accuracy = p_acc_test[0]

    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Testing Accuracy: {test_accuracy:.2f}%")

    min_vals = np.min(trainX, axis=0)
    max_vals = np.max(trainX, axis=0)
    trainX_scaled = (trainX - min_vals) / (max_vals - min_vals)
    testX_scaled = (testX - min_vals) / (max_vals - min_vals)

    trainX_scaled_list = trainX_scaled.tolist()
    testX_scaled_list = testX_scaled.tolist()

    model_scaled = svm_train(trainY_list, trainX_scaled_list, final_param)

    p_label_train_scaled, p_acc_train_scaled, p_val_train_scaled = svm_predict(trainY_list, trainX_scaled_list, model_scaled)
    train_accuracy_scaled = p_acc_train_scaled[0]

    p_label_test_scaled, p_acc_test_scaled, p_val_test_scaled = svm_predict(testY_list, testX_scaled_list, model_scaled)
    test_accuracy_scaled = p_acc_test_scaled[0]

    print(f"Training Accuracy with scaling: {train_accuracy_scaled:.2f}%")
    print(f"Testing Accuracy with scaling: {test_accuracy_scaled:.2f}%")
