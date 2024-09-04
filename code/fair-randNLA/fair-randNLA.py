import cvxpy as cp
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time

#Iterations
#numIters = 35
numIters = 50
fairTime = 0
normalTime = 0
# Define the variables
n = 2 # Dimension of the coefficient vector for law school dataset
#n = 3 #Synthetic dataset
ALLOPTS = []
ALLNONFAIR = []
ALLRATIOS = []
ALLMINS = []
ALLMEANS = []
ALLMAXS = []
ALLSTATS = []
probs = [1/256,1/128,1/64,1/32,1/16,1/8,1/4,1/2,1]
#ks = [50,100,150,200,250,300,350]
#ks = [3,4,5,6,7,8,9,10,11]
for p in probs:
#for k in ks:
    all_opt_objs = []
    all_nonfair_objs = []
    ratios = []
    for numIter in range(numIters):
        x = cp.Variable(n)

        # Open the file in read mode
        A1 = np.empty((0, n))
        A2 = np.empty((0, n))
        A3 = np.empty((0, n))
        A4 = np.empty((0, n))
        A5 = np.empty((0, n))
        b1 = np.array([])
        b2 = np.array([])
        b3 = np.array([])
        b4 = np.array([])
        b5 = np.array([])


        #####BEGIN LAW DATASET
        with open('bar-pass-prediction.csv', 'r') as file:
            alltext = file.readlines()
            #lines = alltext[:-1]
            lines = alltext[1:]
            for line in lines:
                v = line.split(",")
                #LSAT/ugrad GPA
                #features = [float(v[7]),float(v[8])]
                if(v[9]==""):
                    continue
                features = [float(v[8]),float(v[30])]
                            #int(v[2][:-1])
                #v[4] gender
                #if(v[4]=='1'):
                #v[7] LSAT score
                #v[8] ugrad GPA
                #v[9] fygpa (first year grad GPA)
                #v[30] whether they pass the bar
                if(v[23]=='hisp'):
                    A1=np.vstack((A1, features))
                    b1=np.append(b1, float(v[9]))
                elif(v[23]=='black'):
                    A2=np.vstack((A2, features))
                    b2=np.append(b2, float(v[9]))
                elif(v[23]=='asian'):
                    A3=np.vstack((A3, features))
                    b3=np.append(b3, float(v[9]))
                elif(v[23]=='white'):
                    A4=np.vstack((A4, features))
                    b4=np.append(b4, float(v[9]))
                elif(v[23]=='other'):
                    A5=np.vstack((A5, features))
                    b5=np.append(b5, float(v[9]))           


        sample_size=100
        random_row_indices = random.sample(range(min(len(A1),len(A2),len(A3),len(A4),len(A5))), sample_size)
        M1=A1
        M2=A2
        M3=A3
        M4=A4
        M5=A5
        bb1=b1
        bb2=b2
        bb3=b3
        bb4=b4
        bb5=b5
        sample_size=math.floor(len(A1)*p)
        random_row_indices = random.sample(range(len(A1)), sample_size)
        #random_row_indices = random.sample(range(len(A1)), k)
        A1=A1[random_row_indices]
        b1=b1[random_row_indices]
        sample_size=math.floor(len(A2)*p)
        random_row_indices = random.sample(range(len(A2)), sample_size)
        #random_row_indices = random.sample(range(len(A2)), k)
        A2=A2[random_row_indices]
        b2=b2[random_row_indices]
        sample_size=math.floor(len(A3)*p)
        random_row_indices = random.sample(range(len(A3)), sample_size)
        #random_row_indices = random.sample(range(len(A3)), k)
        A3=A3[random_row_indices]
        b3=b3[random_row_indices]
        sample_size=math.floor(len(A4)*p)
        random_row_indices = random.sample(range(len(A4)), sample_size)
        #random_row_indices = random.sample(range(len(A4)), k)
        A4=A4[random_row_indices]
        b4=b4[random_row_indices]
        sample_size=math.floor(len(A5)*p)
        random_row_indices = random.sample(range(len(A5)), sample_size)
        #random_row_indices = random.sample(range(len(A5)), k)
        A5=A5[random_row_indices]
        b5=b5[random_row_indices]
        ####NORMALIZE ALL MATRICES
        A1=A1/math.sqrt(len(A1))
        A2=A2/math.sqrt(len(A2))
        A3=A3/math.sqrt(len(A3))
        A4=A4/math.sqrt(len(A4))
        A5=A5/math.sqrt(len(A5))
        b1=b1/math.sqrt(len(b1))
        b2=b2/math.sqrt(len(b2))
        b3=b3/math.sqrt(len(b3))
        b4=b4/math.sqrt(len(b4))
        b5=b5/math.sqrt(len(b5))
        M1=M1/math.sqrt(len(M1))
        M2=M2/math.sqrt(len(M2))
        M3=M3/math.sqrt(len(M3))
        M4=M4/math.sqrt(len(M4))
        M5=M5/math.sqrt(len(M5))
        bb1=bb1/math.sqrt(len(bb1))
        bb2=bb2/math.sqrt(len(bb2))
        bb3=bb3/math.sqrt(len(bb3))
        bb4=bb4/math.sqrt(len(bb4))
        bb5=bb5/math.sqrt(len(bb5))
        #####END LAW DATASET



        #######BEGIN ADULT DATASET
        ##with open('adult.data', 'r') as file:
        ##    alltext = file.readlines()
        ##    lines = alltext[:-1]
        ##    for line in lines:
        ##        v = line.split()
        ##        features = [int(v[0][:-1]),int(v[4][:-1]),int(v[11][:-1]),int(v[12][:-1])]
        ##                    #int(v[2][:-1])
        ##        if(v[9]=='Male,'):
        ##            A1=np.vstack((A1, features))
        ##            if(v[14]=='<=50K'):
        ##                b1=np.append(b1, 0)
        ##            else:
        ##                b1=np.append(b1, 1)
        ##        else:
        ##            A2=np.vstack((A2, features))
        ##            if(v[14]=='<=50K'):
        ##                b2=np.append(b2, 0)
        ##            else:
        ##                b2=np.append(b2, 1)
        #######END ADULT DATASET



        ##mean_values = np.mean(A1, axis=0)
        ##std_deviation = np.std(A1, axis=0)
        ##A1 = (A1 - mean_values) / std_deviation
        ##A1 = A1/len(A1)
        ##
        ##mean_values = np.mean(A2, axis=0)
        ##std_deviation = np.std(A2, axis=0)
        ##A2 = (A2 - mean_values) / std_deviation
        ##A2 = A2/len(A2)

        ####NORMALIZATION OF VECTOR b
        ##mean_values = np.mean(b1, axis=0)
        ##std_deviation = np.std(b1, axis=0)
        ##b1 = (b1 - mean_values) / std_deviation
        ##b1 = b1/len(b1)
        ##
        ##mean_values = np.mean(b2, axis=0)
        ##std_deviation = np.std(b2, axis=0)
        ##b2 = (b2 - mean_values) / std_deviation
        ##b2 = b2/len(b2)





##        ## Define the matrices A and C, vectors b1, b2, y1, and y2
##        ## SYNTHETIC DATASET
##        A1 = np.array([[1, 2, 1],
##                      [2, 1, 3],
##                      [3, 2, 1]])
##        A1 = np.random.randint(1, k, size=(10, 3))
##        b1 = np.array([1, 2, 3])
##        b1 = np.random.randint(1, k, size=10)
##        
##        A2 = np.array([[2, 1, 3],
##                      [1, 2, 1],
##                      [3, 3, 2]])
##        A2 = np.random.randint(1, k, size=(10, 3))
##        b2 = np.array([2, 1, 4])
##        b2 = np.random.randint(1, k, size=10)
##
##        M1 = A1
##        M2 = A2
##        bb1 = b1
##        bb2 = b2
        
        # Define the optimization problem
        #obj = cp.Minimize(cp.maximum(cp.norm(A1@x-b1, 2), cp.norm(A2@x-b2, 2)))
        obj = cp.Minimize(cp.maximum(cp.norm(A1@x-b1, 2),
                                     cp.norm(A2@x-b2, 2),
                                     cp.norm(A3@x-b3, 2),
                                     cp.norm(A4@x-b4, 2),
                                     cp.norm(A5@x-b5, 2),
                                     ))

        # Create the optimization problem
        problem = cp.Problem(obj)

        startTime = time.time()
        # Solve the problem
        problem.solve()
        endTime = time.time()
        fairTime = fairTime + (endTime - startTime)

        # Get the optimal value and the optimal x (coefficient vector)
        optimal_value = problem.value
        optimal_x = x.value

        optimal_value_all = max(np.linalg.norm(M1@optimal_x-bb1, 2),
                             np.linalg.norm(M2@optimal_x-bb2, 2),
                             np.linalg.norm(M3@optimal_x-bb3, 2),
                             np.linalg.norm(M4@optimal_x-bb4, 2),
                             np.linalg.norm(M5@optimal_x-bb5, 2)
                             )
        all_opt_objs.append(optimal_value_all)

        #print("Value of Fair Solution:", optimal_value_all)
        #print("Optimal Coefficient Vector x:", optimal_x)

        ####SECOND PROBLEM

        # Stack the matrices vertically
        #A = np.vstack((A1, A2))
        #b = np.concatenate((b1, b2))
        A = np.vstack((A1, A2, A3, A4, A5))
        b = np.concatenate((b1, b2, b3, b4, b5))

        # Define the optimization problem
        obj = cp.Minimize(cp.norm(A@x-b,2))

        # Create the optimization problem
        problem = cp.Problem(obj)

        startTime = time.time()
        # Solve the problem
        problem.solve()
        endTime = time.time()
        normalTime = normalTime + (endTime - startTime)

        # Get the optimal value and the optimal x (coefficient vector)
        optimal_value = problem.value
        optimal_x = x.value

        #non_fair_value = max(np.linalg.norm(A1@optimal_x-b1, 2), np.linalg.norm(A2@optimal_x-b2, 2))
        non_fair_value_all = max(np.linalg.norm(M1@optimal_x-bb1, 2),
                             np.linalg.norm(M2@optimal_x-bb2, 2),
                             np.linalg.norm(M3@optimal_x-bb3, 2),
                             np.linalg.norm(M4@optimal_x-bb4, 2),
                             np.linalg.norm(M5@optimal_x-bb5, 2)
                             )


        #print("Optimal Value:", optimal_value)
        all_nonfair_objs.append(non_fair_value_all)
        ratio = optimal_value_all/non_fair_value_all
        #linear least squares
        ratio = math.pow(ratio,2)
        ratios.append(ratio)
        #print("Value of Approximate Solution on Fair Objective:", non_fair_value_all)
        #print("Optimal Coefficient Vector x:", optimal_x)
    ALLSTATS.append((np.min(ratios),np.mean(ratios),np.max(ratios)))
    ALLMINS.append(np.min(ratios))
    ALLMEANS.append(np.mean(ratios))
    ALLMAXS.append(np.max(ratios))
    ALLOPTS.append(all_opt_objs)
    ALLNONFAIR.append(all_nonfair_objs)
    ALLRATIOS.append(ratios)


# Sample data
x = np.array(probs)
#x = np.array(ks)
y1 = np.array(ALLMINS)
y2 = np.array(ALLMEANS)
y3 = np.array(ALLMAXS)

# Customized labels for the x-axis
#x_labels = ['2','3','4','5','6','7','8','9','10']
x_labels = ['1/256','1/128','1/64','1/32','1/16','1/8','1/4','1/2','1']
#x_labels=['50','100','150','200','250','300','350']

# Plot the line graphs for y1 and y2
plt.plot(x, y1, marker='o', linestyle='-', color='b', label='Min')
plt.plot(x, y2, marker='s', linestyle='--', color='r', label='Mean')
#plt.plot(x, y3, marker='s', linestyle='--', color='g', label='Line 3')

#plt.xlabel('Sampling Probabilities (Log-Scale)')
plt.xlabel('Range of Entries')
plt.ylabel('Relative Regression Loss')
plt.xscale('log')
plt.xlim(0, 1)
#plt.xlim(3,10)
#plt.ylim(0.40, 1.2)
#plt.ylim(0.85, 1.00)
plt.xticks(x, x_labels)  # Set customized x-axis labels
plt.fill_between(x, y1, y2, where=(y1 < y2), interpolate=True, color='gray', alpha=0.5)#, label='Shaded Area')
#plt.fill_between(x, y2, y3, where=(y2 < y3), interpolate=True, color='gray', alpha=0.5)#, label='Shaded Area')
plt.title('Relative Regression Loss for Synthetic Dataset')
#plt.title('Relative Regression Loss for Law School Dataset')
plt.legend()  # Show legend for each line
plt.grid(True)
plt.show()

print(fairTime)
print(normalTime)
