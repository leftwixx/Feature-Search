import math
import pandas as pd

def eval(df, features):
    class_df = df.iloc[:, [0]]
    features_df = df.iloc[:, features]
    currentfeatures = []
    totalcorrect = 0
    total = 0
    id = 0 #id of the iterator
    drop = 0 #id of the current feature subset
    min = 100 #initial value of minimum distance, gets replaced later
    currentclass = 0
    nearestclass = 0
    #print(class_df)
    #print(features_df)


    for i in class_df[0]: # i is the value 1 or 2
        id = 0
        currentclass = i
        new_df = class_df.drop([drop]) #new df is all the instances without the current instance
        currentfeatures = []
        for x in range(0, len(features)):
            currentfeatures.append(features_df.iloc[drop][features[x]]) #[3.2423, 1,31314, 5.56321]
        
        #print("==============================")
        #print("finding classication of id", drop, "class", i)

        for x in new_df[0]: # x is the value 1 or 2
            if(id == drop):
                id = id + 1

            comparefeatures = []
            for j in range(0, len(features)):
                comparefeatures.append(features_df.iloc[id][features[j]])
            
            if(calculate(currentfeatures, comparefeatures) < min):
                min = calculate(currentfeatures, comparefeatures)
                nearestclass = x
            
            id = id + 1
        drop = drop + 1
        min = 100
        if(nearestclass == currentclass):
            totalcorrect = totalcorrect + 1
        total = total + 1
        #print(totalcorrect ,"/", total)
        #print("accuracy", totalcorrect / total)
    return round(((totalcorrect/total) * 100), 3)

def forwardselection(df, allcombos):
    max = 0
    localmax = 0
    level = 1
    previousposition = 0
    currentposition = 0
    stop = 2 #if the accuracies decrease twice in a row, the algorithm stops
    flag = False
    bestfeature = []
    bestaccuracy = 0
    currentaccuracy = 0

    for x in allcombos:
        if len(x) == level + 1:
            level = level + 1
            previousposition = currentposition
            #localmax = 0
            if(flag):
                print("Feature set", allcombos[currentposition], "was best, accuracy ", max, "%\n")
                bestfeature = allcombos[currentposition]
                bestaccuracy = max
                flag = False
                stop = 2
            else:
                print("Accuracy has decreased!!")
                print("The best feature to continue expanding is", allcombos[currentposition], " with accuracy ", localmax, "%\n")
                stop = stop - 1
                if(stop == 0):
                    break
            localmax = 0
        if(set(allcombos[previousposition]).issubset(set(x))):
            if(x == []):
                continue

            currentaccuracy = eval(df, x)
            print("Using feature(s)", x, " accuracy is", currentaccuracy, "%")

            if(currentaccuracy > localmax):
                currentposition = allcombos.index(x)
                localmax = currentaccuracy
                if(currentaccuracy > max):
                    max = currentaccuracy
                    flag = True
        
        if(x == allcombos[-1]):
            newlists = expandlist(df, allcombos[currentposition])
            for x in newlists:
                allcombos.append(x)
    if(stop == 0):
        print("Accuracy keeps decreasing, search has stopped")
    print("The best feature set is", bestfeature, "with accuracy", bestaccuracy, "%\n")

def backwardelimination(df, allcombos):
    max = 0
    localmax = 0
    level = len(allcombos[0])
    previousposition = 0
    currentposition = 0
    stop = 2 #if the accuracies decrease twice in a row, the algorithm stops
    flag = False
    bestfeature = []
    bestaccuracy = 0
    currentaccuracy = 0

    for x in allcombos:
        if len(x) == level - 1:
            level = level - 1
            previousposition = currentposition
            
            if(flag):
                print("Feature set", allcombos[currentposition], "was best, accuracy ", max, "%\n")
                bestfeature = allcombos[currentposition]
                bestaccuracy = max
                flag = False
                stop = 2
            else:
                print("Accuracy has decreased!!")
                print("The best feature to continue expanding is", allcombos[currentposition], " with accuracy ", localmax, "%\n")
                stop = stop - 1
                if(stop == 0):
                    break
            localmax = 0
        if(set(x).issubset(set(allcombos[previousposition]))):
            currentaccuracy = eval(df, x)
            print("Using feature(s)", x, " accuracy is", currentaccuracy, "%")

            if(currentaccuracy > localmax):
                currentposition = allcombos.index(x)
                localmax = currentaccuracy
                if(currentaccuracy > max):
                    max = currentaccuracy
                    flag = True
        
        if(x == allcombos[-1]):
            newlists = shrinklist(df, allcombos[currentposition])
            for x in newlists:
                allcombos.append(x)
    if(stop == 0):
        print("Accuracy keeps decreasing, search has stopped")
    print("The best feature set is", bestfeature, "with accuracy", bestaccuracy, "%\n")

def calculate(point1,point2):
    sum = 0
    position = 0
    for x in point1:
        y = point2[position]
        dif = abs(x - y)
        sum = sum + (dif * dif) 
        position = position + 1
    return math.sqrt(sum)

def expandlist(df, currentlist):
    amount_of_features = len(df.columns) - 1
    newp = []
    for x in range(1, amount_of_features + 1):
        p = currentlist.copy()
        if(x not in currentlist):
            p.append(x)
            newp.append(p)
        p = currentlist
    #print(newp)
    return newp

def shrinklist(df, currentlist):
    amount_of_features = len(df.columns) - 1
    newp = []
    for x in range(1, amount_of_features + 1):
        p = currentlist.copy()
        if(x in currentlist):
            p.remove(x)
            newp.append(p)
        p = currentlist
    #print(newp)
    return newp

def normalize(df):
    x = df.iloc[:,1:]
    y = df.iloc[:,0]
    normalizedx = (x-x.mean())/x.std()
    normalized_df = pd.concat([y, normalizedx], axis=1).reindex(y.index)
    return normalized_df


invalid = True
while(invalid):
    selection = int(input("Welcome to Alvin Chan Feature Selection Algorithm. Type “1” to use a small dataset, or “2” to use a large dataset\n"))
    if selection == 1:
        file_name = 'CS170_Spring_2024_Small_data__8.txt'
        #file_name = 'small-test-dataset.txt'
        df1 = pd.read_csv(file_name,sep=r'\s+',header=None)
        df = normalize(df1)
        invalid = False
    elif selection == 2:
        file_name = 'CS170_Spring_2024_Large_data__8.txt'
        #file_name = 'large-test-dataset.txt'
        df1 = pd.read_csv(file_name,sep=r'\s+',header=None)
        df = normalize(df1)
        invalid = False
    else :
        print("invalid")
    

invalid = True
while(invalid):
    selection = int(input("Type the number of the algorithm you want to run. \nForward Selection (1)\n Backward Elimination (2)\n"))

    #forward selection
    if selection == 1:
        amount_of_features = len(df.columns) - 1
        allcombos = [[]]
        for i in range(1, amount_of_features + 1):
            allcombos.append([i])
        forwardselection(df, allcombos)
        invalid = False

    #backward elimination
    elif selection == 2:
        amount_of_features = len(df.columns) - 1
        allcombos = []
        for i in range(1, amount_of_features + 1):
            allcombos.append(i)
        allcombos = [allcombos]
        
        newlists = shrinklist(df, allcombos[0])
        for x in newlists:
            allcombos.append(x)
        
        backwardelimination(df, allcombos)
        invalid = False

    else :
        print("invalid")

