import numpy as np
import matplotlib.pyplot as plt

class LinearPerceptron():

    def __init__(self, path: str) -> None:
        """
        The meaning of varibles:
        data -> list, n * 3 matrix, n = nums of data points
        label -> list, n * 1, label of coordinate point(x,y)
        cor_x_pos -> list, (n/2) * 1,coordinate point of x axis with postive label 
        cor_y_pos -> list, (n/2) * 1,coordinate point of y axis with postive label
        cor_x_neg -> list, (n/2) * 1,coordinate point of x axis with negative label
        cor_y_neg -> list, (n/2) * 1,coordinate point of y axis with negative label

        """
        self.data, self.label, self.cor_x_pos, self.cor_y_pos, self.cor_x_neg, self.cor_y_neg = self.read_data(path)


    def read_data(self, path: str) -> list:
        """
        To read data coordinate points and put them into 'list' data type

        """
        with open(path, 'r') as raw_data:
            lines = raw_data.readlines()
            lines_num = len(lines)
            data = []       # dataset
            label = []      # label
            cor_x_pos = []  # data with +1 label in x axis
            cor_y_pos = []  # data with +1 label in y axis
            cor_x_neg = []  # data with -1 label in x axis
            cor_y_neg = []  # data with -1 label in y axis
            data_cnt = 0
            
            for i in lines: #
                data_cnt += 1
                i = i.strip().split(' ')
                for num in range(4):
                    i[num] = float(i[num])
                    if num == 1:
                        if int(i[3]) == 1:
                            cor_x_pos.append(i[num])
                        elif int(i[3]) == -1:
                            cor_x_neg.append(i[num])
                    if num == 2:
                        if int(i[3]) == 1:
                            cor_y_pos.append(i[num])
                        elif int(i[3]) == -1:
                            cor_y_neg.append(i[num])
                            
                data.append(i[0:3])
                label.append(i[3])

            return data, label, cor_x_pos, cor_y_pos, cor_x_neg, cor_y_neg 
        
    def draw(self, weight, title='PLA',  execution_time=None, accuracy=None, iter_num=None, Issave=None):

        plt.scatter(self.cor_x_pos, self.cor_y_pos, color='blue', label='+1 label', s=5)
        plt.scatter(self.cor_x_neg, self.cor_y_neg, color='red', label='-1 label', s=5)

        # 繪製分類邊界
        x_vals = np.linspace(min(self.cor_x_pos + self.cor_x_neg), max(self.cor_x_pos + self.cor_x_neg), 100)
        y_vals = -(weight[1] * x_vals + weight[0]) / weight[2]
        plt.plot(x_vals, y_vals, color='green', label='Decision Boundary')

        baseline_x = np.linspace(-1000, 1000, 100)
        baseline_y = 3 * baseline_x + 5
        plt.plot(baseline_x, baseline_y, color='black', linestyle='--', label='Baseline: y=3x+5')

        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title(title)

        if execution_time is not None:
            plt.text(600, 1450, f'Execution Time: {execution_time:.7f} seconds', ha='left',fontsize=8)
        if iter_num is not None:
            plt.text(600, 1700, f'Iterate Numbers: {iter_num}', ha='left',fontsize=8)
        if accuracy is not None:
            plt.text(600, 1200, f'Accuracy: {accuracy:.2f}', ha='left',fontsize=8)


        plt.legend()
        plt.grid(True)

        plt.xlim([-1500, 1500])  
        plt.ylim([-2000, 2000])         
        if Issave==True:
            plt.savefig(f'img/{title}.png')
        plt.show()
        





