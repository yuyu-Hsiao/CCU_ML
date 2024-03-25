import numpy as np
import argparse
from Perceptron import LinearPerceptron
import os
import time
import random

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
        # 初始化錯誤計數
        errors = 0
        misclassified_indices = []
        data = np.array(perceptron.data)
        labels = perceptron.label
        
        # 遍歷資料集
        for i in range(len(perceptron.data)):
            # 提取資料和標籤
            x = np.array(data[i])
            y = labels[i]

            # 進行預測
            prediction = np.sign(np.dot(weight_matrix, x))
            
            # 如果預測錯誤，則更新權重
            if prediction != y:
                errors += 1
                misclassified_indices.append(i)

        # 如果所有資料都正確分類，則結束迴圈
        if errors == 0:
            print(iter_cnt)
            break
        random_value = random.choice(misclassified_indices)
        weight_matrix += data[random_value] * labels[random_value]
        
        iter_cnt += 1
    
    end_time = time.time()
    print("程式執行時間: {:.7f} 秒".format(end_time - start_time))

    ############END############
    return  weight_matrix

def main(args):
    try:
        if args.path == None or not os.path.exists(args.path):
            raise
    except:
        print("File not found, please try again")
        exit()

    perceptron = LinearPerceptron(args.path)
    updated_weight = PLA(perceptron=perceptron, iter_num=2000)

    #############################################
    perceptron.draw(weight=updated_weight, title=f"PLA {args.path}")
    #############################################

if __name__ == '__main__':
    
    parse = argparse.ArgumentParser(description='Place the .txt file as your path input')
    parse.add_argument('--path', type=str, help='Your file path')
    args = parse.parse_args()
    main(args)
