import subprocess
import os

if __name__ == "__main__":

    train_script_path = "train.py"
    test_script_path = "test.py"
    base_path = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(base_path, "weights")
    img_path = os.path.join(base_path, "result")

    # 不同epochs
    epochs = [20,40,80]
    for epoch in epochs:
        img1 = os.path.join(base_path, "result","Change_epoch",f"{epoch}")
        weight_path_1 = os.path.join(base_path, "weights","Change_epoch",f"{epoch}")
        weight1 = os.path.join(weight_path_5,"weight.pth")
        if not os.path.exists(img1):
            os.makedirs(img1)
        if not os.path.exists(weight_path_1):
            os.makedirs(weight_path_1)
        command = f"python {train_script_path} --epochs {epoch} --img_path {img1} --weight_path {weight1}"
        test = f"python {test_script_path} --img_path {img1} --weight_path {weight1}"

        subprocess.run(command, shell=True)
        subprocess.run(test, shell=True)
    
   
    # 不同 learning rate
    learning_rates = [0.1,0.01,0.001]
    names = ['0_1','0_01','0_001']
    for learning_rate, name in zip(learning_rates,names):
        img2 = os.path.join(base_path, "result","Change_learning_rate",f"{name}")
        weight_path_2 = os.path.join(base_path, "weights","Change_learning_rate",f"{name}")
        weight2 = os.path.join(weight_path_2,"weight.pth")
        if not os.path.exists(img2):
            os.makedirs(img2)
        if not os.path.exists(weight_path_2):
            os.makedirs(weight_path_2)
        command = f"python {train_script_path} --learning_rate {learning_rate} --img_path {img2} --weight_path {weight2}"
        test = f"python {test_script_path} --img_path {img2} --weight_path {weight2}"

        subprocess.run(command, shell=True)
        subprocess.run(test, shell=True)

    # 不同 batchsize
    batch_sizes = [8,16,32]
    for batch_size in batch_sizes:
        img6 = os.path.join(base_path, "result","Change_batchsize",f"{batch_size}")
        weight_path_6 = os.path.join(base_path, "weights","Change_learning_rate",f"{batch_size}")
        weight6 = os.path.join(weight_path_6,"weight.pth")
        if not os.path.exists(img6):
            os.makedirs(img6)
        if not os.path.exists(weight_path_6):
            os.makedirs(weight_path_6)
        command = f"python {train_script_path} --img_path {img6} --weight_path {weight6} --batch_size {batch_size}"
        test = f"python {test_script_path} --img_path {img6} --weight_path {weight6} --batch_size {batch_size}"
        subprocess.run(command, shell=True)
        subprocess.run(test, shell=True)


    # 不同 loss function
    loss_functions = ['MSELoss','L1Loss','FocalLoss','SmoothL1Loss', 'CrossEntropyLoss']

    for loss_func in loss_functions:
        img5 = os.path.join(base_path, "result","Change_loss_function",f"{loss_func}")
        weight_path_5 = os.path.join(base_path, "weights","Change_loss_function",f"{loss_func}")
        weight5 = os.path.join(weight_path_5,"weight.pth")
        if not os.path.exists(img5):
            os.makedirs(img5)
        if not os.path.exists(weight_path_5):
            os.makedirs(weight_path_5)
        command = f"python {train_script_path} --loss_function {loss_func} --img_path {img5} --weight_path {weight5}"
        test = f"python {test_script_path} --img_path {img5} --weight_path {weight5}"

        subprocess.run(command, shell=True)
        subprocess.run(test, shell=True)