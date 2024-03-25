import numpy as np
import argparse
from Perceptron import LinearPerceptron
import os
import time
import random

import numpy as np

def PLA(perceptron: LinearPerceptron, iter_num) -> np.ndarray:
    """
    Do the Pocket algorithm here on your own.
    weight_matrix -> 3 * 1 resulted weight matrix  
    
    """
    start_time = time.time()
    weight_matrix = np.zeros(3)       
    ############START##########
    iter_cnt = 0
    while iter_cnt < iter_num:

        data = np.array(perceptron.data)
        labels = perceptron.label
        
        predictions = np.sign(np.dot(data, weight_matrix))  # 計算所有樣本的預測值       
        errors = np.sum(predictions != labels)  # 計算錯誤的點數
        
        # 如果所有資料都正確分類，則結束迴圈
        if errors == 0:
            print(iter_cnt)
            break
        
        # 隨機選擇一個錯誤的點，更新權重
        misclassified_indices = np.argwhere(predictions != labels)
        random_value = random.choice(misclassified_indices)[0]
        weight_matrix += data[random_value] * labels[random_value]
        
        iter_cnt += 1
    
    end_time = time.time()
    excution_time = end_time - start_time
    print("程式執行時間: {:.7f} 秒".format(end_time - start_time))

    ############END############
    return  weight_matrix, iter_cnt, excution_time

def main(args):
    try:
        if args.path == None or not os.path.exists(args.path):
            raise
    except:
        print("File not found, please try again")
        exit()

    perceptron = LinearPerceptron(args.path)
    updated_weight , iter_num , excution_time= PLA(perceptron=perceptron, iter_num=2000)

    name_parts = args.path.split(".")
    name = name_parts[0]  # 選取第一部分

    #############################################
    perceptron.draw(weight=updated_weight, title=f"PLA {name}", execution_time=excution_time, iter_num=iter_num, Issave=args.save_img)
    #############################################

if __name__ == '__main__':
    
    parse = argparse.ArgumentParser(description='Place the .txt file as your path input')
    parse.add_argument('--path', type=str, help='Your file path')
    parse.add_argument('--save_img', type=bool, help='Save image or not')
    args = parse.parse_args()
    main(args)
