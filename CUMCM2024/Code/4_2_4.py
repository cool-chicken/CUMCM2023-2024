import random
import numpy as np
from scipy.special import comb

def Solve(Element_S1, Element_S2, Test_Mark_S1, Test_Mark_S2, Test_Mark_Finished_Product, Dismantle_Mark_Finished_Product, Exchange_Quantity):

    Profit = 0
    Cost = 0
    Test_Price_S1 = 1
    Test_Price_S2 = 1
    Finished_Product_Defect_Rate = 0.2
    Finished_Product_Assemble_Price = 6
    Finished_Product_Test_Price = 2
    Finished_Product_Sell_Price = 56
    Finished_Product_Exchange_Loss = 30
    Finished_Product_Dismantle_cost = 5
    Perturbed_Finished_Product_Defect_Rate = Finished_Product_Defect_Rate + np.random.normal(0, 0.01001)

    # 零件阶段
    if Test_Mark_S1 == 1:
        Cost += Test_Price_S1 * len(Element_S1)
        Element_S1 = [x for x in Element_S1 if x != 0]
    if Test_Mark_S2 == 1:
        Cost += Test_Price_S2 * len(Element_S2)
        Element_S2 = [x for x in Element_S2 if x != 0]

    random.shuffle(Element_S1)
    random.shuffle(Element_S2)

    if len(Element_S1) == len(Element_S2):
        Finished_Product = [int(a * b) for a, b in zip(Element_S1, Element_S2)]
        Finished_Product_Form = np.array([Element_S1, Element_S2])
        Cost += Finished_Product_Assemble_Price * len(Finished_Product)
        Qualified_Finished_Products_Indices = [index for index, value in enumerate(Finished_Product) if value == 1]
        Num_To_Select = int(len(Qualified_Finished_Products_Indices) * Perturbed_Finished_Product_Defect_Rate)
        Qualified_Finished_Products_Selected_Indices = np.random.choice(Qualified_Finished_Products_Indices, size=Num_To_Select, replace=False)
        for index in Qualified_Finished_Products_Selected_Indices:
            Finished_Product[index] = 0
        Element_S1 = []
        Element_S2 = []
    elif len(Element_S1) < len(Element_S2):
        Selected_Indices_S2 = random.sample(range(len(Element_S2)), len(Element_S1))
        Selected_Element_S2 = [Element_S2[i] for i in Selected_Indices_S2]
        Finished_Product = [int(a * b) for a, b in zip(Element_S1, Selected_Element_S2)]
        Finished_Product_Form = np.array([Element_S1, Selected_Element_S2])
        Cost += Finished_Product_Assemble_Price * len(Finished_Product)
        Qualified_Finished_Products_Indices = [index for index, value in enumerate(Finished_Product) if value == 1]
        Num_To_Select = int(len(Qualified_Finished_Products_Indices) * Perturbed_Finished_Product_Defect_Rate)
        Qualified_Finished_Products_Selected_Indices = np.random.choice(Qualified_Finished_Products_Indices, size=Num_To_Select, replace=False)
        for index in Qualified_Finished_Products_Selected_Indices:
            Finished_Product[index] = 0
        Element_S1 = []
        Element_S2 = [Element_S2[i] for i in range(len(Element_S2)) if i not in Selected_Indices_S2]
    elif len(Element_S1) > len(Element_S2):
        Selected_Indices_S1 = random.sample(range(len(Element_S1)), len(Element_S2))
        Selected_Element_S1 = [Element_S1[i] for i in Selected_Indices_S1]
        Finished_Product = [int(a * b) for a, b in zip(Selected_Element_S1, Element_S2)]
        Finished_Product_Form = np.array([Selected_Element_S1, Element_S2])
        Cost += Finished_Product_Assemble_Price * len(Finished_Product)
        Qualified_Finished_Products_Indices = [index for index, value in enumerate(Finished_Product) if value == 1]
        Num_To_Select = int(len(Qualified_Finished_Products_Indices) * Perturbed_Finished_Product_Defect_Rate)
        Qualified_Finished_Products_Selected_Indices = np.random.choice(Qualified_Finished_Products_Indices, size=Num_To_Select, replace=False)
        for index in Qualified_Finished_Products_Selected_Indices:
            Finished_Product[index] = 0
        Element_S1 = [Element_S1[i] for i in range(len(Element_S1)) if i not in Selected_Indices_S1]
        Element_S2 = []

    # 成品阶段
    if Test_Mark_Finished_Product == 1:
        Cost += Finished_Product_Test_Price * len(Finished_Product)
        Qualified_Finished_Products_Indices = [index for index, value in enumerate(Finished_Product) if value == 1]
        Unqualified_Finished_Products_Indices = [index for index, value in enumerate(Finished_Product) if value == 0]
        Profit += Finished_Product_Sell_Price * (len(Qualified_Finished_Products_Indices) - Exchange_Quantity)
        Exchange_Quantity = 0
        if Dismantle_Mark_Finished_Product == 1:
            Cost += Finished_Product_Dismantle_cost * len(Unqualified_Finished_Products_Indices)
            Unqualified_Finished_Product_Form = Finished_Product_Form[:, Unqualified_Finished_Products_Indices]
            Element_S1.extend(Unqualified_Finished_Product_Form[0])
            Element_S2.extend(Unqualified_Finished_Product_Form[1])
    else:
        Qualified_Finished_Products_Indices = [index for index, value in enumerate(Finished_Product) if value == 1]
        Unqualified_Finished_Products_Indices = [index for index, value in enumerate(Finished_Product) if value == 0]
        Profit += Finished_Product_Sell_Price * (len(Finished_Product) - Exchange_Quantity)
        Exchange_Quantity = 0
        Cost += Finished_Product_Exchange_Loss * len(Unqualified_Finished_Products_Indices)
        Exchange_Quantity = len(Unqualified_Finished_Products_Indices)
        if Dismantle_Mark_Finished_Product == 1:
            Cost += Finished_Product_Dismantle_cost * len(Unqualified_Finished_Products_Indices)
            Unqualified_Finished_Product_Form = Finished_Product_Form[:, Unqualified_Finished_Products_Indices]
            Element_S1.extend(Unqualified_Finished_Product_Form[0])
            Element_S2.extend(Unqualified_Finished_Product_Form[1])

    return Profit, Cost, Element_S1, Element_S2, Exchange_Quantity

S1 = 1000
S2 = 1000
Defect_Rate_S1 = 0.2
Defect_Rate_S2 = 0.2

Time = 1
Simulation_Number = 100
Profits = np.zeros(16)
Costs = np.zeros(16)
Purchase_Price_S1 = 4
Purchase_Price_S2 = 18

Decision_Matrix = np.zeros((16, 4), dtype=int)
for i in range(16):
    Binary_Representation = format(i, '04b')
    for j in range(4):
        Decision_Matrix[i, j] = int(Binary_Representation[j])

for i in range(0, 16):
    Test_Mark_S1 = Decision_Matrix[i][0]
    Test_Mark_S2 = Decision_Matrix[i][1]
    Test_Mark_Finished_Product = Decision_Matrix[i][2]
    Dismantle_Mark_Finished_Product = Decision_Matrix[i][3]
    for j in range(0, Time):
        Element_S1 = []
        Element_S2 = []
        Exchange_Quantity = 0
        k = 0
        while k < Simulation_Number:
            random.seed(j * Time + k)
            np.random.seed(j * Time + k)
            Perturbed_Defect_Rate_S1 = Defect_Rate_S1 + np.random.normal(0, 0.01022)
            Perturbed_Defect_Rate_S2 = Defect_Rate_S2 + np.random.normal(0, 0.01024)
            Bad_S1 = int(Perturbed_Defect_Rate_S1 * S1)
            Bad_S2 = int(Perturbed_Defect_Rate_S2 * S2)
            Prepared_Element_S1 = np.ones(S1)
            Random_Indices_S1 = np.random.choice(S1, size=Bad_S1, replace=False)
            Prepared_Element_S1[Random_Indices_S1] = 0
            Element_S1.extend(Prepared_Element_S1)
            Prepared_Element_S2 = np.ones(S2)
            Random_Indices_S2 = np.random.choice(S2, size=Bad_S2, replace=False)
            Prepared_Element_S2[Random_Indices_S2] = 0
            Element_S2.extend(Prepared_Element_S2)
            Costs[i] += Purchase_Price_S1 * len(Prepared_Element_S1) + Purchase_Price_S2 * len(Prepared_Element_S2)
            Profit, Cost, Element_S1, Element_S2, Exchange_Quantity = Solve(Element_S1, Element_S2, Test_Mark_S1, Test_Mark_S2, Test_Mark_Finished_Product, Dismantle_Mark_Finished_Product, Exchange_Quantity)
            Profits[i] += Profit
            Costs[i] += Cost
            k += 1

Profits = Profits / (Simulation_Number * Time)
Costs = Costs / (Simulation_Number * Time)
Pure_Profits = (Profits - Costs) / S1
print(Pure_Profits)
Max_Index = np.argmax(Pure_Profits)
Binary_Max_Index = format(Max_Index, '04b')
print("最大值是:", Pure_Profits[Max_Index])
print("最大值的索引是:", Max_Index)
print("最大值的索引的二进制表示是:", Binary_Max_Index)
