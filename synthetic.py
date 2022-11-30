import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from learners_spline import *

np.random.seed(6746)

def mu(x, t, factor):
    if x<-0.5:
        return np.power(x+2,2)/2 + t*factor
    elif x<0:
        return x/2+0.875
    elif x<0.5:
        return -5*np.power(x-0.2,2)+1.075 + t*factor
    else:
        return x+0.125 + t*factor

def e(x):
    return 0.5+0.4*np.sign(x)

def gen_data(factor, n = 1000):

    xs = np.zeros(n)
    treats = np.zeros(n)
    outcomes = np.zeros(n)

    for i in range(n):
        xs[i] = np.random.uniform(-1,1)
        treats[i]= np.random.binomial(1, e(xs[i]))
        outcomes[i] = mu(xs[i], treats[i], factor) + np.random.normal(scale = 0.2-0.1*np.cos(2*np.pi*xs[i]))

    xs_test = np.random.uniform(-1, 1, n)

    return xs, treats, outcomes, xs_test
    
FACTOR = 0

num_rep = 100

mse_s = np.zeros(num_rep)
mse_t = np.zeros(num_rep)
mse_r = np.zeros(num_rep)
mse_dr = np.zeros(num_rep)

for i in tqdm(range(num_rep)):    
    xs, treats, outcomes, xs_test = gen_data(FACTOR, n = 1000)
    cates_test = FACTOR * np.ones(len(xs))
    mse_s[i] = s_learner(xs, treats, outcomes, xs_test, cates_test)
    mse_t[i] = t_learner(xs, treats, outcomes, xs_test, cates_test)
    mse_r[i] = r_learner(xs, treats, outcomes, xs_test, cates_test)
    mse_dr[i] = dr_learner(xs, treats, outcomes, xs_test, cates_test)

print(np.mean(mse_s), np.std(mse_s)*1.96/np.sqrt(num_rep))
print(np.mean(mse_t), np.std(mse_t)*1.96/np.sqrt(num_rep))
print(np.mean(mse_r), np.std(mse_r)*1.96/np.sqrt(num_rep))
print(np.mean(mse_dr), np.std(mse_dr)*1.96/np.sqrt(num_rep))




# plt.scatter(xs[treats==0], outcomes[treats==0], s = 5, label = 'Control Outcomes')
# plt.scatter(xs[treats==1], outcomes[treats==1], s = 5, label = 'Treated Outcomes')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()



# function_x = np.linspace(-1,1,100)
# function_y = [mu(x,1,0) for x in function_x]

# plt.plot(function_x,function_y, c = 'black')

# plt.show()


