from sklearn.model_selection import train_test_split

from .cost_function import cost_function
import numpy as np
import time


def gda(X, y):
     #split test and train as usual
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #denote phi as the probability of y==1
    #we just calculate the number of samples where y==1
    #divided by the total number samples
    phi=len(x_train[y_train==1])/len(x_train)

    #calculate the 'mean vector' for two scenarios
    #y==0 and y==1
    #note that we use pandas dataframe
    #so the results are not a vector
    #and we dont need to transpose it
    #cuz later when we need to input a transposed vector
    #we can directly input mean0/mean1
    mean0=np.mean(x_train[y_train==0],0)
    mean1=np.mean(x_train[y_train==1],0)

    #calculate the difference between x and mean for two scenarios
    #concatenate x for two scenarios together
    dif=pd.concat([x_train[y_train==0]-mean0,x_train[y_train==1]-mean1])

    #calculate the covariance matrix
    #we use a list to append all covariance/variance
    #later we reshape it into a 4 by 4 matrix
    #our x is four-dimensional so we should have 4 by 4
    temp=list(dif.columns)
    cov=[]

    for i in range(len(dif.columns)):
        for j in range(len(dif.columns)):
            cov.append( \
                       ( \
                        np.mat(dif[temp[i]])*np.mat(dif[temp[j]]).T).item()/len(temp) \
                      )

    cov=np.mat(cov).reshape(4,4)


    #now we have mean for two scenarios, covariance matrix and phi
    #we use bayesian conditional probability formula
    #to calculate the probability of y==0 and y==1 for each x
    #and we use the larger probability of the two to forecast y
    p0=[]
    p1=[]

    #the bayes' theorem is p(y|x)=p(x|y)*p(y)/p(x)
    #in our case, x is always independent of y
    #which gives us p(x)=1
    #and we shall simplify it into p(y|x)=p(x|y)*p(y)
    #where p(y)=phi**y+(1-phi)**(1-y) is the probability density function of bernoulli distribution
    #p(x|y) is the pdf of multivariate gaussian distribution
    #note that our pandas dataframe is not transposed
    #so the formulas here are a lil bit different from the original
    #plz refer to wikipedia for the details of formulas
    # https://en.wikipedia.org/wiki/Bayes%27_theorem
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    # https://en.wikipedia.org/wiki/Covariance_matrix
    # https://see.stanford.edu/materials/aimlcs229/cs229-notes2.pdf

    #we calculate the probability for each element in x matrix
    #if we use the whole matrix to do algebra, lets see
    #lets forget anything before np.e as we can see determinant there
    #everything before np.e becomes scalar
    #for (x-mean).T*cov.I*(x-mean)
    #(x-mean).T is n by 4 matrix
    #cov.I is 4 by 4 matrix
    #we get n by 4 matrix
    #then multiplied by (x-mean) which is 4 by n matrix
    #we would end up with n by n matrix in the end
    #oh, god, thats not what we want
    #when (x-mean).T is 1 by 4 matrix
    #after multiplication by cov.I which is 4 by 4 matrix
    #now we get 1 by 4 matrix
    #eventually times (x-mean).T which is 4 by 1 matrix
    #we get scalar, a 1 by 1 matrix!!!
    for k in range(len(x_test)):

        probability0=phi/( \
                          np.linalg.det(2*np.pi*cov)**(0.5) \
                         ) \
        *np.exp( \
                -0.5*(np.mat(x_test.iloc[k]-mean0))*cov.I*(np.mat(x_test.iloc[k]-mean0)).T \
               )


        probability1=phi/( \
                          np.linalg.det(2*np.pi*cov)**(0.5) \
                         ) \
        *np.exp( \
                -0.5*(np.mat(x_test.iloc[k]-mean1))*cov.I*(np.mat(x_test.iloc[k]-mean1)).T \
               )


        p0.append(probability0.item())
        p1.append(probability1.item())

    #here we use numpy sign
    #numpy sign treats positive number as 1
    #it treats zero as 0
    #however, it treats negative number as -1
    #we use a map function to convert -1 to 0
    forecast=np.sign(np.subtract(p1,p0))
    forecast=list(map(lambda x: 0 if x<0 else int(x),forecast))

    print('test accuracy: {}%'.format(len(y_test[forecast==y_test])/len(y_test)*100))

    #just too lazy to write codes for plotting

    return
