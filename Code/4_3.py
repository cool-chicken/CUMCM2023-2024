import random
import numpy as np
from scipy.special import comb

def Solve(Element_S1, Element_S2, Element_S3, Element_S4, Element_S5, Element_S6, Element_S7, Element_S8, Semi_Finished_Product_S1, Semi_Finished_Product_S2, Semi_Finished_Product_S3, Semi_Finished_Product_S1_Form, Semi_Finished_Product_S2_Form, Semi_Finished_Product_S3_Form, Decision_Vector, Exchange_Quantity):

    Profit = 0
    Cost = 0
    Test_Price_S1 = 1
    Test_Price_S2 = 1
    Test_Price_S3 = 2
    Test_Price_S4 = 1
    Test_Price_S5 = 1
    Test_Price_S6 = 2
    Test_Price_S7 = 1
    Test_Price_S8 = 2
    Semi_Finished_Product_S1_Defect_Rate = 0.1
    Semi_Finished_Product_S1_Assemble_Price = 8
    Semi_Finished_Product_S1_Test_Price = 4
    Semi_Finished_Product_S1_Dismantle_cost = 6
    Semi_Finished_Product_S2_Defect_Rate = 0.1
    Semi_Finished_Product_S2_Assemble_Price = 8
    Semi_Finished_Product_S2_Test_Price = 4
    Semi_Finished_Product_S2_Dismantle_cost = 6
    Semi_Finished_Product_S3_Defect_Rate = 0.1
    Semi_Finished_Product_S3_Assemble_Price = 8
    Semi_Finished_Product_S3_Test_Price = 4
    Semi_Finished_Product_S3_Dismantle_cost = 6
    Finished_Product_Defect_Rate = 0.1
    Finished_Product_Assemble_Price = 8
    Finished_Product_Test_Price = 6
    Finished_Product_Dismantle_cost = 10
    Finished_Product_Sell_Price = 200
    Finished_Product_Exchange_Loss = 40

    Perturbed_Semi_Finished_Product_S1_Defect_Rate = Semi_Finished_Product_S1_Defect_Rate + np.random.normal(0, 0.01)
    Perturbed_Semi_Finished_Product_S2_Defect_Rate = Semi_Finished_Product_S2_Defect_Rate + np.random.normal(0, 0.01)
    Perturbed_Semi_Finished_Product_S3_Defect_Rate = Semi_Finished_Product_S3_Defect_Rate + np.random.normal(0, 0.01)

    Test_Mark_S1 = Decision_Vector[0]
    Test_Mark_S2 = Decision_Vector[1]
    Test_Mark_S3 = Decision_Vector[2]
    Test_Mark_S4 = Decision_Vector[3]
    Test_Mark_S5 = Decision_Vector[4]
    Test_Mark_S6 = Decision_Vector[5]
    Test_Mark_S7 = Decision_Vector[6]
    Test_Mark_S8 = Decision_Vector[7]
    Test_Mark_Semi_Finished_Product_S1 = Decision_Vector[8]
    Dismantle_Mark_Semi_Finished_Product_S1 = Decision_Vector[9]
    Test_Mark_Semi_Finished_Product_S2 = Decision_Vector[10]
    Dismantle_Mark_Semi_Finished_Product_S2 = Decision_Vector[11]
    Test_Mark_Semi_Finished_Product_S3 = Decision_Vector[12]
    Dismantle_Mark_Semi_Finished_Product_S3 = Decision_Vector[13]
    Test_Mark_Finished_Product = Decision_Vector[14]
    Dismantle_Mark_Finished_Product = Decision_Vector[15]

    # 零件阶段
    if Test_Mark_S1 == 1:
        Cost += Test_Price_S1 * len(Element_S1)
        Element_S1 = [x for x in Element_S1 if x != 0]
    if Test_Mark_S2 == 1:
        Cost += Test_Price_S2 * len(Element_S2)
        Element_S2 = [x for x in Element_S2 if x != 0]
    if Test_Mark_S3 == 1:
        Cost += Test_Price_S3 * len(Element_S3)
        Element_S3 = [x for x in Element_S3 if x != 0]
    if Test_Mark_S4 == 1:
        Cost += Test_Price_S4 * len(Element_S4)
        Element_S4 = [x for x in Element_S4 if x != 0]
    if Test_Mark_S5 == 1:
        Cost += Test_Price_S5 * len(Element_S5)
        Element_S5 = [x for x in Element_S5 if x != 0]
    if Test_Mark_S6 == 1:
        Cost += Test_Price_S6 * len(Element_S6)
        Element_S6 = [x for x in Element_S6 if x != 0]
    if Test_Mark_S7 == 1:
        Cost += Test_Price_S7 * len(Element_S7)
        Element_S7 = [x for x in Element_S7 if x != 0]
    if Test_Mark_S8 == 1:
        Cost += Test_Price_S8 * len(Element_S8)
        Element_S8 = [x for x in Element_S8 if x != 0]

    Min_S1_S2_S3 = min(len(Element_S1), len(Element_S2), len(Element_S3))
    random.shuffle(Element_S1)
    random.shuffle(Element_S2)
    random.shuffle(Element_S3)
    Selected_Element_S1 = Element_S1[:Min_S1_S2_S3]
    Selected_Element_S2 = Element_S2[:Min_S1_S2_S3]
    Selected_Element_S3 = Element_S3[:Min_S1_S2_S3]
    Semi_Finished_Product_S1.extend([int(a * b * c) for a, b, c in zip(Selected_Element_S1, Selected_Element_S2, Selected_Element_S3)])
    if len(Semi_Finished_Product_S1_Form) == 0 or (len(Semi_Finished_Product_S1_Form) == 3 and all(len(sublist) == 0 for sublist in Semi_Finished_Product_S1_Form)):
        Semi_Finished_Product_S1_Form = np.array([Selected_Element_S1, Selected_Element_S2, Selected_Element_S3])
    else:
        Semi_Finished_Product_S1_Form = np.concatenate((np.array(Semi_Finished_Product_S1_Form), np.array([Selected_Element_S1, Selected_Element_S2, Selected_Element_S3])), axis=1)
    Cost += Semi_Finished_Product_S1_Assemble_Price * Min_S1_S2_S3
    Qualified_Semi_Finished_Product_S1_Indices = [index for index, value in enumerate(Semi_Finished_Product_S1) if value == 1]
    Num_To_Select = int(len(Qualified_Semi_Finished_Product_S1_Indices) * Perturbed_Semi_Finished_Product_S1_Defect_Rate)
    Qualified_Semi_Finished_Product_S1_Selected_Indices = np.random.choice(Qualified_Semi_Finished_Product_S1_Indices, size=Num_To_Select, replace=False)
    for index in Qualified_Semi_Finished_Product_S1_Selected_Indices:
        Semi_Finished_Product_S1[index] = 0
    Element_S1 = Element_S1[Min_S1_S2_S3:]
    Element_S2 = Element_S2[Min_S1_S2_S3:]
    Element_S3 = Element_S3[Min_S1_S2_S3:]

    Min_S4_S5_S6 = min(len(Element_S4), len(Element_S5), len(Element_S6))
    random.shuffle(Element_S4)
    random.shuffle(Element_S5)
    random.shuffle(Element_S6)
    Selected_Element_S4 = Element_S4[:Min_S4_S5_S6]
    Selected_Element_S5 = Element_S5[:Min_S4_S5_S6]
    Selected_Element_S6 = Element_S6[:Min_S4_S5_S6]
    Semi_Finished_Product_S2.extend([int(a * b * c) for a, b, c in zip(Selected_Element_S4, Selected_Element_S5, Selected_Element_S6)])
    if len(Semi_Finished_Product_S2_Form) == 0 or (len(Semi_Finished_Product_S2_Form) == 3 and all(len(sublist) == 0 for sublist in Semi_Finished_Product_S2_Form)):
        Semi_Finished_Product_S2_Form = np.array([Selected_Element_S4, Selected_Element_S5, Selected_Element_S6])
    else:
        Semi_Finished_Product_S2_Form = np.concatenate((np.array(Semi_Finished_Product_S2_Form), np.array([Selected_Element_S4, Selected_Element_S5, Selected_Element_S6])), axis=1)
    Cost += Semi_Finished_Product_S2_Assemble_Price * Min_S4_S5_S6
    Qualified_Semi_Finished_Product_S2_Indices = [index for index, value in enumerate(Semi_Finished_Product_S2) if value == 1]
    Num_To_Select = int(len(Qualified_Semi_Finished_Product_S2_Indices) * Perturbed_Semi_Finished_Product_S2_Defect_Rate)
    Qualified_Semi_Finished_Product_S2_Selected_Indices = np.random.choice(Qualified_Semi_Finished_Product_S2_Indices, size=Num_To_Select, replace=False)
    for index in Qualified_Semi_Finished_Product_S2_Selected_Indices:
        Semi_Finished_Product_S2[index] = 0
    Element_S4 = Element_S4[Min_S4_S5_S6:]
    Element_S5 = Element_S5[Min_S4_S5_S6:]
    Element_S6 = Element_S6[Min_S4_S5_S6:]

    Min_S7_S8= min(len(Element_S7), len(Element_S8))
    random.shuffle(Element_S7)
    random.shuffle(Element_S8)
    Selected_Element_S7 = Element_S7[:Min_S7_S8]
    Selected_Element_S8 = Element_S8[:Min_S7_S8]
    Semi_Finished_Product_S3.extend([int(a * b) for a, b in zip(Selected_Element_S7, Selected_Element_S8)])
    if len(Semi_Finished_Product_S3_Form) == 0 or (len(Semi_Finished_Product_S3_Form) == 2 and all(len(sublist) == 0 for sublist in Semi_Finished_Product_S3_Form)):
        Semi_Finished_Product_S3_Form = np.array([Selected_Element_S7, Selected_Element_S8])
    else:
        Semi_Finished_Product_S3_Form = np.concatenate((np.array(Semi_Finished_Product_S3_Form), np.array([Selected_Element_S7, Selected_Element_S8])), axis=1)
    Cost += Semi_Finished_Product_S3_Assemble_Price * Min_S7_S8
    Qualified_Semi_Finished_Product_S3_Indices = [index for index, value in enumerate(Semi_Finished_Product_S3) if value == 1]
    Num_To_Select = int(len(Qualified_Semi_Finished_Product_S3_Indices) * Perturbed_Semi_Finished_Product_S3_Defect_Rate)
    Qualified_Semi_Finished_Product_S3_Selected_Indices = np.random.choice(Qualified_Semi_Finished_Product_S3_Indices, size=Num_To_Select, replace=False)
    for index in Qualified_Semi_Finished_Product_S3_Selected_Indices:
        Semi_Finished_Product_S3[index] = 0
    Element_S7 = Element_S7[Min_S7_S8:]
    Element_S8 = Element_S8[Min_S7_S8:]

    # 半成品阶段
    if Test_Mark_Semi_Finished_Product_S1 == 1:
        Cost += Semi_Finished_Product_S1_Test_Price * len(Semi_Finished_Product_S1)
        Qualified_Semi_Finished_Product_S1_Indices = [index for index, value in enumerate(Semi_Finished_Product_S1) if value == 1]
        Unqualified_Semi_Finished_Product_S1_Indices = [index for index, value in enumerate(Semi_Finished_Product_S1) if value == 0]
        Qualified_Semi_Finished_Product_S1 = [Semi_Finished_Product_S1[i] for i in range(len(Semi_Finished_Product_S1)) if i in Qualified_Semi_Finished_Product_S1_Indices]
        Qualified_Semi_Finished_Product_S1_Form = Semi_Finished_Product_S1_Form[:, Qualified_Semi_Finished_Product_S1_Indices]
        Passed_Semi_Finished_Product_S1 = Qualified_Semi_Finished_Product_S1
        Passed_Semi_Finished_Product_S1_Form = Qualified_Semi_Finished_Product_S1_Form

        if Dismantle_Mark_Semi_Finished_Product_S1 == 1:
            Cost += Semi_Finished_Product_S1_Dismantle_cost * len(Unqualified_Semi_Finished_Product_S1_Indices)
            Unqualified_Semi_Finished_Product_S1_Form = Semi_Finished_Product_S1_Form[:, Unqualified_Semi_Finished_Product_S1_Indices]
            Element_S1.extend(Unqualified_Semi_Finished_Product_S1_Form[0])
            Element_S2.extend(Unqualified_Semi_Finished_Product_S1_Form[1])
            Element_S3.extend(Unqualified_Semi_Finished_Product_S1_Form[2])
            Semi_Finished_Product_S1 = []
            Semi_Finished_Product_S1_Form = [[], [], []]
        else:
            Semi_Finished_Product_S1 = []
            Semi_Finished_Product_S1_Form = [[], [], []]
    else:
        Passed_Semi_Finished_Product_S1 = Semi_Finished_Product_S1
        Passed_Semi_Finished_Product_S1_Form = Semi_Finished_Product_S1_Form
        Semi_Finished_Product_S1 = []
        Semi_Finished_Product_S1_Form = [[], [], []]

    if Test_Mark_Semi_Finished_Product_S2 == 1:
        Cost += Semi_Finished_Product_S2_Test_Price * len(Semi_Finished_Product_S2)
        Qualified_Semi_Finished_Product_S2_Indices = [index for index, value in enumerate(Semi_Finished_Product_S2) if value == 1]
        Unqualified_Semi_Finished_Product_S2_Indices = [index for index, value in enumerate(Semi_Finished_Product_S2) if value == 0]
        Qualified_Semi_Finished_Product_S2 = [Semi_Finished_Product_S2[i] for i in range(len(Semi_Finished_Product_S2)) if i in Qualified_Semi_Finished_Product_S2_Indices]
        Qualified_Semi_Finished_Product_S2_Form = Semi_Finished_Product_S2_Form[:, Qualified_Semi_Finished_Product_S2_Indices]
        Passed_Semi_Finished_Product_S2 = Qualified_Semi_Finished_Product_S2
        Passed_Semi_Finished_Product_S2_Form = Qualified_Semi_Finished_Product_S2_Form

        if Dismantle_Mark_Semi_Finished_Product_S2 == 1:
            Cost += Semi_Finished_Product_S2_Dismantle_cost * len(Unqualified_Semi_Finished_Product_S2_Indices)
            Unqualified_Semi_Finished_Product_S2_Form = Semi_Finished_Product_S2_Form[:, Unqualified_Semi_Finished_Product_S2_Indices]
            Element_S4.extend(Unqualified_Semi_Finished_Product_S2_Form[0])
            Element_S5.extend(Unqualified_Semi_Finished_Product_S2_Form[1])
            Element_S6.extend(Unqualified_Semi_Finished_Product_S2_Form[2])
            Semi_Finished_Product_S2 = []
            Semi_Finished_Product_S2_Form = [[], [], []]
        else:
            Semi_Finished_Product_S2 = []
            Semi_Finished_Product_S2_Form = [[], [], []]
    else:
        Passed_Semi_Finished_Product_S2 = Semi_Finished_Product_S2
        Passed_Semi_Finished_Product_S2_Form = Semi_Finished_Product_S2_Form
        Semi_Finished_Product_S2 = []
        Semi_Finished_Product_S2_Form = [[], [], []]

    if Test_Mark_Semi_Finished_Product_S3 == 1:
        Cost += Semi_Finished_Product_S3_Test_Price * len(Semi_Finished_Product_S3)
        Qualified_Semi_Finished_Product_S3_Indices = [index for index, value in enumerate(Semi_Finished_Product_S3) if value == 1]
        Unqualified_Semi_Finished_Product_S3_Indices = [index for index, value in enumerate(Semi_Finished_Product_S3) if value == 0]
        Qualified_Semi_Finished_Product_S3 = [Semi_Finished_Product_S3[i] for i in range(len(Semi_Finished_Product_S3)) if i in Qualified_Semi_Finished_Product_S3_Indices]
        Qualified_Semi_Finished_Product_S3_Form = Semi_Finished_Product_S3_Form[:, Qualified_Semi_Finished_Product_S3_Indices]
        Passed_Semi_Finished_Product_S3 = Qualified_Semi_Finished_Product_S3
        Passed_Semi_Finished_Product_S3_Form = Qualified_Semi_Finished_Product_S3_Form

        if Dismantle_Mark_Semi_Finished_Product_S3 == 1:
            Cost += Semi_Finished_Product_S3_Dismantle_cost * len(Unqualified_Semi_Finished_Product_S3_Indices)
            Unqualified_Semi_Finished_Product_S3_Form = Semi_Finished_Product_S3_Form[:, Unqualified_Semi_Finished_Product_S3_Indices]
            Element_S7.extend(Unqualified_Semi_Finished_Product_S3_Form[0])
            Element_S8.extend(Unqualified_Semi_Finished_Product_S3_Form[1])
            Semi_Finished_Product_S3 = []
            Semi_Finished_Product_S3_Form = [[], []]
        else:
            Semi_Finished_Product_S3 = []
            Semi_Finished_Product_S3_Form = [[], []]
    else:
        Passed_Semi_Finished_Product_S3 = Semi_Finished_Product_S3
        Passed_Semi_Finished_Product_S3_Form = Semi_Finished_Product_S3_Form
        Semi_Finished_Product_S3 = []
        Semi_Finished_Product_S3_Form = [[], []]

    # 成品阶段
    Finished_Product_Number = min(len(Passed_Semi_Finished_Product_S1), len(Passed_Semi_Finished_Product_S2), len(Passed_Semi_Finished_Product_S3))
    random.shuffle(Passed_Semi_Finished_Product_S1)
    random.shuffle(Passed_Semi_Finished_Product_S2)
    random.shuffle(Passed_Semi_Finished_Product_S3)
    Selected_Indices_Semi_Finished_Product_S1 = random.sample(range(len(Passed_Semi_Finished_Product_S1)), Finished_Product_Number)
    Selected_Semi_Finished_Product_S1 = [Passed_Semi_Finished_Product_S1[i] for i in Selected_Indices_Semi_Finished_Product_S1]
    Selected_Semi_Finished_Product_S1_Form = Passed_Semi_Finished_Product_S1_Form[:, Selected_Indices_Semi_Finished_Product_S1]
    Selected_Indices_Semi_Finished_Product_S2 = random.sample(range(len(Passed_Semi_Finished_Product_S2)), Finished_Product_Number)
    Selected_Semi_Finished_Product_S2 = [Passed_Semi_Finished_Product_S2[i] for i in Selected_Indices_Semi_Finished_Product_S2]
    Selected_Semi_Finished_Product_S2_Form = Passed_Semi_Finished_Product_S2_Form[:, Selected_Indices_Semi_Finished_Product_S2]
    Selected_Indices_Semi_Finished_Product_S3 = random.sample(range(len(Passed_Semi_Finished_Product_S3)), Finished_Product_Number)
    Selected_Semi_Finished_Product_S3 = [Passed_Semi_Finished_Product_S3[i] for i in Selected_Indices_Semi_Finished_Product_S3]
    Selected_Semi_Finished_Product_S3_Form = Passed_Semi_Finished_Product_S3_Form[:, Selected_Indices_Semi_Finished_Product_S3]
    Finished_Product = [int(a * b * c) for a, b, c in zip(Selected_Semi_Finished_Product_S1, Selected_Semi_Finished_Product_S2, Selected_Semi_Finished_Product_S3)]
    Finished_Product_Form = np.concatenate((Selected_Semi_Finished_Product_S1_Form, Selected_Semi_Finished_Product_S2_Form, Selected_Semi_Finished_Product_S3_Form), axis=0)
    Cost += Finished_Product_Assemble_Price * Finished_Product_Number
    Qualified_Finished_Product_Indices = [index for index, value in enumerate(Finished_Product) if value == 1]
    Num_To_Select = int(len(Qualified_Finished_Product_Indices) * Finished_Product_Defect_Rate)
    Qualified_Finished_Product_Selected_Indices = np.random.choice(Qualified_Finished_Product_Indices, size=Num_To_Select, replace=False)
    for index in Qualified_Finished_Product_Selected_Indices:
        Finished_Product[index] = 0
    Semi_Finished_Product_S1.extend([Passed_Semi_Finished_Product_S1[i] for i in range(len(Passed_Semi_Finished_Product_S1)) if i not in Selected_Indices_Semi_Finished_Product_S1])
    Semi_Finished_Product_S2.extend([Passed_Semi_Finished_Product_S2[i] for i in range(len(Passed_Semi_Finished_Product_S2)) if i not in Selected_Indices_Semi_Finished_Product_S2])
    Semi_Finished_Product_S3.extend([Passed_Semi_Finished_Product_S3[i] for i in range(len(Passed_Semi_Finished_Product_S3)) if i not in Selected_Indices_Semi_Finished_Product_S3])
    Semi_Finished_Product_S1_Form = np.concatenate((Semi_Finished_Product_S1_Form, Passed_Semi_Finished_Product_S1_Form[:, [Passed_Semi_Finished_Product_S1[i] for i in range(len(Passed_Semi_Finished_Product_S1)) if i not in Selected_Indices_Semi_Finished_Product_S1]]), axis=1)
    Semi_Finished_Product_S2_Form = np.concatenate((Semi_Finished_Product_S2_Form, Passed_Semi_Finished_Product_S2_Form[:, [Passed_Semi_Finished_Product_S2[i] for i in range(len(Passed_Semi_Finished_Product_S2)) if i not in Selected_Indices_Semi_Finished_Product_S2]]), axis=1)
    Semi_Finished_Product_S3_Form = np.concatenate((Semi_Finished_Product_S3_Form, Passed_Semi_Finished_Product_S3_Form[:, [Passed_Semi_Finished_Product_S3[i] for i in range(len(Passed_Semi_Finished_Product_S3)) if i not in Selected_Indices_Semi_Finished_Product_S3]]), axis=1)

    if Test_Mark_Finished_Product == 1:
        Cost += Finished_Product_Test_Price * len(Finished_Product)
        Qualified_Finished_Product_Indices = [index for index, value in enumerate(Finished_Product) if value == 1]
        Unqualified_Finished_Product_Indices = [index for index, value in enumerate(Finished_Product) if value == 0]
        Profit += Finished_Product_Sell_Price * (len(Qualified_Finished_Product_Indices) - Exchange_Quantity)
        Exchange_Quantity = 0
        if Dismantle_Mark_Finished_Product == 1:
            Cost += Finished_Product_Dismantle_cost * len(Unqualified_Finished_Product_Indices)
            Unqualified_Finished_Product_Form = Finished_Product_Form[:, Unqualified_Finished_Product_Indices]
            Semi_Finished_Product_S1.extend([int(a * b * c) for a, b, c in zip(Unqualified_Finished_Product_Form[0], Unqualified_Finished_Product_Form[1], Unqualified_Finished_Product_Form[2])])
            Semi_Finished_Product_S1_Form = np.concatenate((Semi_Finished_Product_S1_Form, np.array([Unqualified_Finished_Product_Form[0].tolist(), Unqualified_Finished_Product_Form[1].tolist(), Unqualified_Finished_Product_Form[2].tolist()])), axis=1)
            Semi_Finished_Product_S2.extend([int(a * b * c) for a, b, c in zip(Unqualified_Finished_Product_Form[3], Unqualified_Finished_Product_Form[4], Unqualified_Finished_Product_Form[5])])
            Semi_Finished_Product_S2_Form = np.concatenate((Semi_Finished_Product_S2_Form, np.array([Unqualified_Finished_Product_Form[3].tolist(), Unqualified_Finished_Product_Form[4].tolist(), Unqualified_Finished_Product_Form[5].tolist()])), axis=1)
            Semi_Finished_Product_S3.extend([int(a * b) for a, b in zip(Unqualified_Finished_Product_Form[6], Unqualified_Finished_Product_Form[7])])
            Semi_Finished_Product_S3_Form = np.concatenate((Semi_Finished_Product_S3_Form, np.array([Unqualified_Finished_Product_Form[6].tolist(), Unqualified_Finished_Product_Form[7].tolist()])), axis=1)

    else:
        Qualified_Finished_Product_Indices = [index for index, value in enumerate(Finished_Product) if value == 1]
        Unqualified_Finished_Product_Indices = [index for index, value in enumerate(Finished_Product) if value == 0]
        Profit += Finished_Product_Sell_Price * (len(Finished_Product) - Exchange_Quantity)
        Exchange_Quantity = 0
        Cost += Finished_Product_Exchange_Loss * len(Unqualified_Finished_Product_Indices)
        Exchange_Quantity = len(Unqualified_Finished_Product_Indices)

        if Dismantle_Mark_Finished_Product == 1:
            Cost += Finished_Product_Dismantle_cost * len(Unqualified_Finished_Product_Indices)
            Unqualified_Finished_Product_Form = Finished_Product_Form[:, Unqualified_Finished_Product_Indices]
            Semi_Finished_Product_S1.extend([int(a * b * c) for a, b, c in zip(Unqualified_Finished_Product_Form[0], Unqualified_Finished_Product_Form[1], Unqualified_Finished_Product_Form[2])])
            Semi_Finished_Product_S1_Form = np.concatenate((Semi_Finished_Product_S1_Form, np.array([Unqualified_Finished_Product_Form[0].tolist(), Unqualified_Finished_Product_Form[1].tolist(), Unqualified_Finished_Product_Form[2].tolist()])), axis=1)
            Semi_Finished_Product_S2.extend([int(a * b * c) for a, b, c in zip(Unqualified_Finished_Product_Form[3], Unqualified_Finished_Product_Form[4], Unqualified_Finished_Product_Form[5])])
            Semi_Finished_Product_S2_Form = np.concatenate((Semi_Finished_Product_S2_Form, np.array([Unqualified_Finished_Product_Form[3].tolist(), Unqualified_Finished_Product_Form[4].tolist(), Unqualified_Finished_Product_Form[5].tolist()])), axis=1)
            Semi_Finished_Product_S3.extend([int(a * b) for a, b in zip(Unqualified_Finished_Product_Form[6], Unqualified_Finished_Product_Form[7])])
            Semi_Finished_Product_S3_Form = np.concatenate((Semi_Finished_Product_S3_Form, np.array([Unqualified_Finished_Product_Form[6].tolist(), Unqualified_Finished_Product_Form[7].tolist()])), axis=1)

    return Profit, Cost, Element_S1, Element_S2, Element_S3, Element_S4, Element_S5, Element_S6, Element_S7, Element_S8, Semi_Finished_Product_S1, Semi_Finished_Product_S2, Semi_Finished_Product_S3, Semi_Finished_Product_S1_Form, Semi_Finished_Product_S2_Form, Semi_Finished_Product_S3_Form, Exchange_Quantity

class Genetic_Algorithm:
    def __init__(self, Objective_Function, Initial_Solution, Population_Size=100, Mutation_Rate=0.1, Crossover_Rate=0.5, Max_Iter=100, Elitism_Rate=0.05):
        self.Objective_Function = Objective_Function
        self.Population_Size = Population_Size
        self.Mutation_Rate = Mutation_Rate
        self.Crossover_Rate = Crossover_Rate
        self.Max_Iter = Max_Iter
        self.Elitism_Rate = Elitism_Rate
        self.Dim = len(Initial_Solution)
        self.Population = np.random.randint(2, size=(Population_Size, self.Dim))

    def Select(self, Fitness):
        Fitness = np.array(Fitness)
        Fitness = Fitness - np.min(Fitness) + 1e-10
        Probabilities = Fitness / np.sum(Fitness)
        Cumulative_Probabilities = np.cumsum(Probabilities)
        Selected_Indices = []
        for _ in range(self.Population_Size):
            r = random.random()
            for i, cp in enumerate(Cumulative_Probabilities):
                if r < cp:
                    Selected_Indices.append(i)
                    break
        return self.Population[Selected_Indices]

    def Crossover(self, Parent1, Parent2):
        if random.random() < self.Crossover_Rate:
            Cross_Point = random.randint(0, self.Dim - 1)
            Child1 = np.concatenate((Parent1[:Cross_Point], Parent2[Cross_Point:]))
            Child2 = np.concatenate((Parent2[:Cross_Point], Parent1[Cross_Point:]))
            return Child1, Child2
        else:
            return Parent1, Parent2

    def Mutate(self, Individual):
        for i in range(self.Dim):
            if random.random() < self.Mutation_Rate:
                Individual[i] = 1 - Individual[i]
        return Individual

    def Optimize(self):
        for Generation in range(self.Max_Iter):
            Fitness = [self.Objective_Function(x) for x in self.Population]

            # 选择精英个体
            Elite_Size = int(self.Population_Size * self.Elitism_Rate)
            Elite_Indices = np.argsort(Fitness)[-Elite_Size:]
            Elite_Population = self.Population[Elite_Indices]

            Selected_Population = self.Select(Fitness)
            New_Population = []
            for i in range(0, self.Population_Size - Elite_Size, 2):
                Parent1 = Selected_Population[i]
                Parent2 = Selected_Population[i + 1]
                Child1, Child2 = self.Crossover(Parent1, Parent2)
                Child1 = self.Mutate(Child1)
                Child2 = self.Mutate(Child2)
                New_Population.append(Child1)
                New_Population.append(Child2)

            # 将精英个体加入新种群
            New_Population.extend(Elite_Population)
            self.Population = np.array(New_Population)
            Best_Solution = self.Population[np.argmax(Fitness)]
            Best_Fitness = np.max(Fitness)

            print(f"Generation {Generation}: Best_Solution = {Best_Solution}, Best Fitness = {Best_Fitness}")

        return Best_Solution, Best_Fitness

def Objective_Function(X):

    S1 = 100
    S2 = 100
    S3 = 100
    S4 = 100
    S5 = 100
    S6 = 100
    S7 = 100
    S8 = 100
    Defect_Rate_S1 = 0.1
    Defect_Rate_S2 = 0.1
    Defect_Rate_S3 = 0.1
    Defect_Rate_S4 = 0.1
    Defect_Rate_S5 = 0.1
    Defect_Rate_S6 = 0.1
    Defect_Rate_S7 = 0.1
    Defect_Rate_S8 = 0.1

    Simulation_Number = 100
    Time = 1
    Profits = 0
    Costs = 0
    Purchase_Price_S1 = 2
    Purchase_Price_S2 = 8
    Purchase_Price_S3 = 12
    Purchase_Price_S4 = 2
    Purchase_Price_S5 = 8
    Purchase_Price_S6 = 12
    Purchase_Price_S7 = 8
    Purchase_Price_S8 = 12

    for i in range(0, Time):
        Decision_Vector = X
        Element_S1 = []
        Element_S2 = []
        Element_S3 = []
        Element_S4 = []
        Element_S5 = []
        Element_S6 = []
        Element_S7 = []
        Element_S8 = []
        Semi_Finished_Product_S1 = []
        Semi_Finished_Product_S2 = []
        Semi_Finished_Product_S3 = []
        Semi_Finished_Product_S1_Form = [[], [], []]
        Semi_Finished_Product_S2_Form = [[], [], []]
        Semi_Finished_Product_S3_Form = [[], []]
        Exchange_Quantity = 0
        k = 0
        while k < Simulation_Number:
            random.seed(i * Time + k)
            np.random.seed(i * Time + k)

            Perturbed_Defect_Rate_S1 = Defect_Rate_S1 + np.random.normal(0, 0.01123)
            Perturbed_Defect_Rate_S2 = Defect_Rate_S2 + np.random.normal(0, 0.01012)
            Perturbed_Defect_Rate_S3 = Defect_Rate_S3 + np.random.normal(0, 0.01021)
            Perturbed_Defect_Rate_S4 = Defect_Rate_S4 + np.random.normal(0, 0.01156)
            Perturbed_Defect_Rate_S5 = Defect_Rate_S5 + np.random.normal(0, 0.01382)
            Perturbed_Defect_Rate_S6 = Defect_Rate_S6 + np.random.normal(0, 0.01122)
            Perturbed_Defect_Rate_S7 = Defect_Rate_S7 + np.random.normal(0, 0.01302)
            Perturbed_Defect_Rate_S8 = Defect_Rate_S8 + np.random.normal(0, 0.01241)
            Bad_S1 = int(Perturbed_Defect_Rate_S1 * S1)
            Bad_S2 = int(Perturbed_Defect_Rate_S2 * S2)
            Bad_S3 = int(Perturbed_Defect_Rate_S3 * S3)
            Bad_S4 = int(Perturbed_Defect_Rate_S4 * S4)
            Bad_S5 = int(Perturbed_Defect_Rate_S5 * S5)
            Bad_S6 = int(Perturbed_Defect_Rate_S6 * S6)
            Bad_S7 = int(Perturbed_Defect_Rate_S7 * S7)
            Bad_S8 = int(Perturbed_Defect_Rate_S8 * S8)

            Prepared_Element_S1 = np.ones(S1)
            Random_Indices_S1 = np.random.choice(S1, size=Bad_S1, replace=False)
            Prepared_Element_S1[Random_Indices_S1] = 0
            Element_S1.extend(Prepared_Element_S1)
            Prepared_Element_S2 = np.ones(S2)
            Random_Indices_S2 = np.random.choice(S2, size=Bad_S2, replace=False)
            Prepared_Element_S2[Random_Indices_S2] = 0
            Element_S2.extend(Prepared_Element_S2)
            Prepared_Element_S3 = np.ones(S3)
            Random_Indices_S3 = np.random.choice(S3, size=Bad_S3, replace=False)
            Prepared_Element_S3[Random_Indices_S3] = 0
            Element_S3.extend(Prepared_Element_S3)
            Prepared_Element_S4 = np.ones(S4)
            Random_Indices_S4 = np.random.choice(S4, size=Bad_S4, replace=False)
            Prepared_Element_S4[Random_Indices_S4] = 0
            Element_S4.extend(Prepared_Element_S4)
            Prepared_Element_S5 = np.ones(S5)
            Random_Indices_S5 = np.random.choice(S5, size=Bad_S5, replace=False)
            Prepared_Element_S5[Random_Indices_S5] = 0
            Element_S5.extend(Prepared_Element_S5)
            Prepared_Element_S6 = np.ones(S6)
            Random_Indices_S6 = np.random.choice(S6, size=Bad_S6, replace=False)
            Prepared_Element_S6[Random_Indices_S6] = 0
            Element_S6.extend(Prepared_Element_S6)
            Prepared_Element_S7 = np.ones(S7)
            Random_Indices_S7 = np.random.choice(S7, size=Bad_S7, replace=False)
            Prepared_Element_S7[Random_Indices_S7] = 0
            Element_S7.extend(Prepared_Element_S7)
            Prepared_Element_S8 = np.ones(S8)
            Random_Indices_S8 = np.random.choice(S8, size=Bad_S8, replace=False)
            Prepared_Element_S8[Random_Indices_S8] = 0
            Element_S8.extend(Prepared_Element_S8)
            Costs += Purchase_Price_S1 * len(Prepared_Element_S1) + Purchase_Price_S2 * len(Prepared_Element_S2)
            Costs += Purchase_Price_S3 * len(Prepared_Element_S3) + Purchase_Price_S4 * len(Prepared_Element_S4)
            Costs += Purchase_Price_S5 * len(Prepared_Element_S5) + Purchase_Price_S6 * len(Prepared_Element_S6)
            Costs += Purchase_Price_S7 * len(Prepared_Element_S7) + Purchase_Price_S8 * len(Prepared_Element_S8)
            Profit, Cost, Element_S1, Element_S2, Element_S3, Element_S4, Element_S5, Element_S6, Element_S7, Element_S8, Semi_Finished_Product_S1, Semi_Finished_Product_S2, Semi_Finished_Product_S3, Semi_Finished_Product_S1_Form, Semi_Finished_Product_S2_Form, Semi_Finished_Product_S3_Form, Exchange_Quantity = Solve(Element_S1, Element_S2, Element_S3, Element_S4, Element_S5, Element_S6, Element_S7, Element_S8, Semi_Finished_Product_S1, Semi_Finished_Product_S2, Semi_Finished_Product_S3, Semi_Finished_Product_S1_Form, Semi_Finished_Product_S2_Form, Semi_Finished_Product_S3_Form, Decision_Vector, Exchange_Quantity)
            Profits += Profit
            Costs += Cost
            k += 1

    Profits = Profits / (Simulation_Number * Time)
    Costs = Costs / (Simulation_Number * Time)
    Pure_Profits = Profits - Costs

    return Pure_Profits

Period = 1
Initial_Solution = np.zeros(16 * Period)
Genetic_Algorithm = Genetic_Algorithm(Objective_Function, Initial_Solution)
Best_Solution, Best_Fitness = Genetic_Algorithm.Optimize()
print(f"Best Solution: {Best_Solution}")
print(f"Best Fitness: {Best_Fitness}")
