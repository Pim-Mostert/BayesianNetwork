```math

\begin{align}

d_8(y_2) & = 
    \begin{cases}
        1 & \text{if } y_2 = \hat{y_2} \\
        0 & \text{if } y_2 \ne \hat{y_2} \\
    \end{cases}
    & P(\hat{Y_2}|Y_2) \\

a_8(y_2) & = 
    d_8(y_2) \\

b_8(y_2) & =
    \sum_{q_2}{f_4(q_2, y_2)a_6(q_2)} \\

b_6(y_2) & =
    \sum_{y_2}{f_4(y_2, q_2)a_8(y_2)} \\

a_6(q_2) & = 
    b_5(q_2)b_7(q_2) \\

d_9(y_3) & = 
    \begin{cases}
        1 & \text{if } y_3 = \hat{y_3} \\
        0 & \text{if } y_3 \ne \hat{y_3} \\
    \end{cases}
    & P(\hat{Y_3}|Y_3) \\

a_9(y_3) & = 
    d_9(y_3) \\

b_9(y_3) & =
    \sum_{q_2}{f_5(q_2, y_3)a_7(q_2)} \\

a_7(q_2) & = 
    b_5(q_2)b_6(q_2) \\

b_7(y_3) & =
    \sum_{y_3}{f_5(y_3, q_2)a_9(y_3)} \\

a_5(q_2) & = 
    b_6(q_2)b_7(q_2) \\

b_5(q_2) & =
    \sum_{q_1}{f_3(q_1, q_2)a_3(q_1)} \\

a_3(q_2) & = 
    b_1(q_1)b_2(q_1) \\

b_3(q_1) & =
    \sum_{q_2}{f_3(q_1, q_1)a_5(q_2)} \\

d_4(y_1) & = 
    \begin{cases}
        1 & \text{if } y_1 = \hat{y_1} \\
        0 & \text{if } y_1 \ne \hat{y_1} \\
    \end{cases}
    & P(\hat{Y_1}|Y_1) \\
        
a_4(y_1) & = 
    d_4(y) \\
    
b_4(y_1) & =
    \sum_{q_1}{f_2(q_1, y_1)a_2(q_1)} \\
    
b_2(q_1) & =
    \sum_{y_1}{f_2(q_1, y_1)a_4(q_1)} \\

a_2(q_1) & = 
    b_1(q_1)b_3(q_1) \\
    
a_1(q_1) & = 
    b_2(q_1)b_3(q_1) \\
    
b_1(q_1) & = 
    f_1(q_1) \\
        

\end{align}

```