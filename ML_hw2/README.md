# CCU Machine Learning Homework2

## PLA
1. prediction=sign(wT ⋅ x)
    wT 是權重矩陣的轉置，表示一個行向量，包含了每個特徵的權重
    x 是特徵向量；
    sign(⋅) 將內積的結果取符號，即大於等於0時為+1，小於0時為-1。

2. 預測錯誤，則根據感知器學習規則進行權重的更新。將錯誤分類的樣本的特徵向量乘以其標籤（即誤差），然後加到權重向量上：
    w = w + y⋅x