
# coding: utf-8

# In[372]:

"""
Program to calculate 5-year risk of mortality.

    Parameters
    ----------
    1. The name of the file that contains the results of the questionnaire. 
       Should be in the same folder of the program.
    
    Data Needed (All data should be in the same folder of the program)
    ----------
    1. CoefM.txt (coefficients for males)
    2. CoefF.txt (coefficients for females)
    3. wwMage.csv (weights for males)
    4. wwFage.csv (weights for females)
    5. MM095.csv (5-year survival probabilities from lifetables, males)
    6. FF095.csv (5-year survival probabilities from lifetables, females)
    7. varcovM.csv (file with the variance-covariance matrix for males)
    8. varcovF.csv (file with the variance-covariance matrix for females)
    
    Returns
    -------
    1. A string containing the following parameters separated by ;
        1. Age
        2. Sex
        3. UBBLE age with 95% C.I.
        4. Risk from lifetable corresponding to the real age
        5. Predicted risk
        6. Number of deaths within 100 individuals with same risk profile
        7. Number of alives within 100 individuals with same risk profile
        8. Values used in the tornado plot with the format 'name variable = sum[beta(x-M)] [95% C.I.]'

"""

import numpy as np
import math as mt
from scipy.stats import norm
import sys
import os.path
import itertools

def skip_d(d):
    '''
    This function replace the 'skip' value with the expected value 
    and delete the remaining 'skip' values. 
    Also delete the 'Sex' variable, which is not useful anymore.
    '''
    L1=[]
    L2=[]
    for x in d:
        if x[0] == 'Sex':
            continue
            
        elif x[0] in ['f.6141.0.l1','f.6141.0.l2','f.6141.0.l3','f.6141.0.l4','f.6141.0.l5','f.6141.0.l6','f.6141.0.l7','f.6141.0.l8'] and x[1]=='skip':
            if d[1,1] == "Male":
                x[1] = 'Live alone'
                x[0] = x[0]
            else:
                continue
    
        elif x[0] == 'f.1249.0.0' and x[1] == 'skip':
            x[1] = "I'm a smoker"
            x[0] = x[0]
    
        elif x[1] == 'skip':
            continue
        
        else:
            x[1] = x[1]
            x[0] = x[0]
        L1.append(x[1])
        L2.append(x[0])

    d=np.column_stack((L2,L1))
    return d

def add_interaction(interact_var,d):
    '''
    Function to create variables that interact with age
    and assign the right value
    '''
    for i in interact_var:
        newi = i.split (".", 2)[1] 
        newvar = 'f.age:'+newi
        newpar = d[d[:,0] == i,1]
        for k in newpar:
            newx = np.array((newvar,k))
            d = np.vstack((d,newx))
    return d

def deal_with_zeros(d):
    '''
    This function replace the '0' with the right values making the data structure 
    compatible with the functions
    '''
    L1=[]
    L2=[]
    for x in d:
        if x[1] == '0':
            continue

        elif x[0][len(x[0])-2:len(x[0])-1] == 'l':
            x[1] = x[1]
            x[0] = x[0][0:len(x[0])-3]

        else:
            x[1] = x[1]
            x[0] = x[0]
        L1.append(x[1])
        L2.append(x[0])
    d=np.column_stack((L2,L1))
    return d

def calculate_f709(d):
    '''
    This function calculate the right string interval for the variable f.709:
    number of people in the household
    '''
    if d[d[:,0] == ['f.709.0.0'],1] == '[1,2]' or d[d[:,0] == ['f.709.0.0'],1] == '(2,4]' or d[d[:,0] == ['f.709.0.0'],1] == '(4,100]':
        return d
    else:
        if np.int_(d[d[:,0] == ['f.709.0.0'],1]) < 3:
            d[d[:,0] == ['f.709.0.0'],1] = ['[1,2]']
        elif np.int_(d[d[:,0] == ['f.709.0.0'],1]) < 5:
            d[d[:,0] == ['f.709.0.0'],1] = ['(2,4]']
        else:
            d[d[:,0] == ['f.709.0.0'],1] = ['(4,100]']
        return d 

def calculate_f2734(d):
    '''
    This function calculate the right string interval for the variable f.2734:
    number of children
    '''
    if d[d[:,0] == ['f.2734.0.0'],1] == '[0,1]' or d[d[:,0] == ['f.2734.0.0'],1] == '(1,2]' or d[d[:,0] == ['f.2734.0.0'],1] == '(2,4]' or d[d[:,0] == ['f.2734.0.0'],1] == '(4,22]':
        return d
    else:
        if np.int_(d[d[:,0] == ['f.2734.0.0'],1]) < 2:
            d[d[:,0] == ['f.2734.0.0'],1] = ['[0,1]']
        elif np.int_(d[d[:,0] == ['f.2734.0.0'],1]) < 3:
            d[d[:,0] == ['f.2734.0.0'],1] = ['(1,2]']
        elif np.int_(d[d[:,0] == ['f.2734.0.0'],1]) < 5:
            d[d[:,0] == ['f.2734.0.0'],1] = ['(2,4]']
        else:
            d[d[:,0] == ['f.2734.0.0'],1] = ['(4,22]']
        return d 

'''
Following three functions are used to extract
the right values from the file containing coefficients
'''

def reform_array_par(name_var,coef):
    newarray=coef.view(np.recarray)
    return newarray.f1[newarray.f0 == name_var]

def reform_array_coef(name_var,coef):
    newarray=coef.view(np.recarray)
    return newarray.f2[newarray.f0 == name_var]

def reform_array_mean(name_var,coef):
    newarray=coef.view(np.recarray)
    return newarray.f3[newarray.f0 == name_var]

def calculate_lp(pars,means,coefs,name_var,d,age):
    '''
    Function to calculate the linear predictor [coef*(x-mean)] and standardized value (x-mean) for each variable
    '''
    name_par = d[d[:,0] == name_var,1]
    lp = 0
    xmean = []
    for i,j in enumerate(pars):   
        f = 0
        for k in name_par:
            if j == k:
                f = f+1
        if f == 0:
            if coefs[i] == 0:
                continue       
            xmean0 = 0-means[i]
            lp0 = xmean0*coefs[i]
        else:
            if name_var[0:5] == 'f.age': #If age interaction then use formula (age-mean) * coef
                if coefs[i] == 0:
                    continue       
                xmean0 = age-means[i]
                lp0 = xmean0*coefs[i]
            else:
                if coefs[i] == 0:
                    continue       
                xmean0 = 1-means[i]
                lp0 = xmean0*coefs[i]
        xmean.append(xmean0)
        lp += lp0
    return lp, xmean

def age_lp(age,coef):
    
    '''
    Function to calculate the linear predictor for age
    '''
    xmean=[age-coef[0,][3]]
    lp=xmean[0]*coef[0,][2]
    return lp, xmean


def delta_method(gd,varcovSel):
    '''
    Delta method. Since the first derivative (gd) of coef*(x-m) = x-m, 
    not formal derivative function is used.
    '''
    assert len(varcovSel)==len(gd), 'ERROR'
    varcovSel = varcovSel.astype(float)
    deriv1 = np.dot(np.transpose(gd),varcovSel) # Delta-method: gd %*% vcov %*% gd
    assert np.dot(gd,deriv1) > 0, 'ERROR'
    stderr = np.sqrt(np.dot(gd,deriv1))
    assert stderr > 0, 'ERROR'
    return stderr


class Predscore_final(object):
    
    def __init__ (self, d, age, sex, directory):
        self.d = d
        self.age=age
        self.sex=sex
        self.directory=directory 
        
    def data_load_and_process(self):
        '''
        Initial data process, output a clean dataset that is used to calculate the linear predictors
        Output: Clean orginal data
        '''
        if  self.sex == 'Male':
            self.coef = np.loadtxt(fname=self.directory +'/coefM.txt', delimiter='\t',dtype={'names': ('f0', 'f1', 'f2','f3'),'formats': ('S40', 'S100', '<f8','<f8')}, usecols=(0,1,2,3))
            self.basehaz=0.0127703  
            self.wwage = np.loadtxt(fname=self.directory + '/wwMage.csv', delimiter=',', dtype='<f8')
            self.S095 = np.loadtxt(fname=self.directory + '/MM095.csv', delimiter=',', dtype='<f8')
            self.varcov = np.loadtxt(fname=self.directory + '/varcovM.csv', delimiter=',', dtype='S100')
            # Determine which variable are interacting with age
            interact_var = ['f.709.0.0','f.6141.0','f.924.0.0','f.2443.0.0','f.2453.0.0','f.6150.0','f.6146.0']
            # These functions set the data in the right format
            self.d=skip_d(self.d)
            self.d=calculate_f709(self.d)   
            self.d=deal_with_zeros(self.d)
            self.d=add_interaction(interact_var,self.d)
        else:    
            self.coef = np.loadtxt(fname=self.directory +'/coefF.txt', delimiter='\t',dtype={'names': ('f0', 'f1', 'f2','f3'),'formats': ('S40', 'S100', '<f8','<f8')}, usecols=(0,1,2,3))
            self.basehaz=0.007218876
            self.wwage = np.loadtxt(fname=self.directory + '/wwFage.csv', delimiter=',', dtype='<f8')
            self.S095 = np.loadtxt(fname=self.directory + '/FF095.csv', delimiter=',', dtype='<f8')
            self.varcov = np.loadtxt(fname=self.directory + '/varcovF.csv', delimiter=',', dtype='S100')
            # Determine which variable are interacting with age
            interact_var = ['f.2453.0.0','f.6146.0']
            # These functions set the data in the right format
            self.d=skip_d(self.d)
            self.d=calculate_f2734(self.d)
            self.d=deal_with_zeros(self.d)
            self.d=add_interaction(interact_var,self.d)
        return self.d     

    
    def _assertion_data(self):
        '''
        Function that does checks on the processed data
        also check that all the parameters in the answer file are among the elegible values
        and that the variables are in the same order as in the coef file
        finally, it output if some questions need to be skipped in the tornado plot
        '''
        
        varnamed = self.d[:,0]
        varnamecoef = self.coef['f0']
        
        assert len(self.d) > 1, 'ERROR'
        assert varnamed[0] == 'age' , 'ERROR'
         
        indexes1 = np.unique(varnamed, return_index=True)[1]
        indexes2 = np.unique(varnamecoef, return_index=True)[1]
        assert np.array_equal([varnamed[index1] for index1 in sorted(indexes1)],[varnamecoef[index2] for index2 in sorted(indexes2)]), 'ERROR'
        
        for i,j in enumerate(self.d[:,1]):
            if varnamed[i] != 'age':    
                assert j in self.coef['f1'],'ERROR'        

        if self.d[varnamed == ['f.1239.0.0'],1]=='Yes, on most or all days':
            self.indicator1239=1
        else:
            self.indicator1239=0
        if 'Live alone' in self.d[varnamed == ['f.6141.0'],1]:
            self.indicator6141=1
        else:
            self.indicator6141=0
            
            
    def _calculate_LP_xmean(self):
        '''
        Calculate linear predictors and xmean for subsequent use
        '''
        
        self._assertion_data() # Run checking and assign some values needed later
        

        _ , idx=np.unique(self.d[:,0],return_index=True) #Get unique names, but preserving the order
        names=self.d[np.sort(idx),0] 
        
        self.LP = []
        self.Xmean = []
        
        for i in names:
            if i =='age':
                lpstd= age_lp(self.age,self.coef)
                lp, xmean  = lpstd
            else:
                coefs=reform_array_coef(i, self.coef)
                means=reform_array_mean(i, self.coef)
                pars=reform_array_par(i, self.coef)
                lpstd=calculate_lp(pars,means,coefs,i,self.d,self.age)
                lp, xmean = lpstd # Linear predictors coef*(x-mean), standardized value = 1 derivative = x-mean   
            self.LP.append(lp)
            self.Xmean.append(xmean)  
            assert len(self.LP) == len(self.Xmean) , 'ERROR'
 

    def calculate_risk(self):
        '''
        Calculate the predicted risk and the confidence intervales using delta method
        Output: 1. Predicted risk 2.3: Predicted risk C.I.
        
        '''
                
        self._calculate_LP_xmean() # to extract qunatities needed
        
        Xmean = list(itertools.chain.from_iterable(self.Xmean))
        
        stderr = delta_method(Xmean,self.varcov[1:,1:]) # Get standard error for linear predictor
        
        lp_risk=sum(self.LP)
        w = self.wwage[self.wwage[:,0] == self.age ,1] # Obtain the right weights
        MMt = mt.exp(-self.basehaz * w) # Weight the baseline hazard
        #MMt=mt.exp(-self.basehaz)
        self.predscore = 1-(MMt**mt.exp(lp_risk)) #
        
        self.predscoreL = 1-(MMt**mt.exp(lp_risk-norm.ppf(0.975)*stderr))
        self.predscoreU = 1-(MMt**mt.exp(lp_risk+norm.ppf(0.975)*stderr))
        
        return self.predscore, self.predscoreL, self.predscoreU
    
    
    def bioage(self):
        '''
        Find closest value to values in array:
        Output: 1.biological age, 2,3.biological age C.I.
        4.the risk from lifetables corresponding to the true age
        '''
        idx = (np.abs(self.S095[:,1]-(1-self.predscore))).argmin()
        idxL = (np.abs(self.S095[:,1]-(1-self.predscoreL))).argmin()
        idxU = (np.abs(self.S095[:,1]-(1-self.predscoreU))).argmin()
        
        realriskage = 1-self.S095[self.S095[:,0]==self.age,1]
        
        bioage = self.S095[idx,0]
        bioageL = self.S095[idxL,0]
        bioageU = self.S095[idxU,0]
        
        return bioage, bioageL, bioageU, realriskage

    
    def _namesnewfun(self):
        '''
        Function to extract the name of the variable even in case of interaction
        '''
        _ , idx=np.unique(self.d[:,0],return_index=True) #Get unique names, but preserving the order
        names=self.d[np.sort(idx),0]
        self.namesnew=[]
        for i in names:
            if i[0:3] != 'age':
                names1=i.translate(None, 'age:')
                names2=names1.split (".", 2)[1]        
            else:
                names2=i
            self.namesnew.append(names2)
        assert len(self.namesnew)==len(names), 'ERROR'

    
    
    def _unique_lp_stderr_tornado(self):
        '''
         1. It finds the questions with same ID (normally one with and one without interaction with age) 
            and:
            a. sums the linear predictors within questions
            b. founds the correct variance covariance matrix subset
         2. It calculates the standard errors based on the delta-method
        '''
        
        self._namesnewfun() # to extract the variables names
        
        namesnew=np.array(self.namesnew)
        LP=np.array(self.LP, dtype=object)
        Xmean=np.array(self.Xmean, dtype=object)
        
        _, idx = np.unique(namesnew, return_index=True) #Get unique names, but preserving the order
        self.unique_namesnew = namesnew[np.sort(idx)].tolist()
        
        self.unique_LP=[]
        self.unique_stder=[]
        
        for group in self.unique_namesnew:
            sumgroup = sum(LP[namesnew == group])
            self.unique_LP.append(sumgroup) # Sum LP within each variable
            
            # Obtain correct subset of varcovar matrix
            b = self.varcov[:,0]==group
            rowsel = self.varcov[b,:]
            varcovSel = rowsel[:,b]

            # Obtain standard errors
            if group != 'age': 
                gd = np.concatenate(Xmean[namesnew == group])
            else:
                gd = Xmean[namesnew == group][0]
                
            stderr = delta_method(gd,varcovSel) #Delta-method
            self.unique_stder.append(stderr)
        
    
    def list_for_plot(self,exclude_age):
        '''
        Skip some question if needed 
        Output: 1. New variable name; 2. Linear predictor; 3.Standard errors 
        
        '''     
        self._unique_lp_stderr_tornado() # Creates names, lp and stderrors
        # Exclude age is requested
        if exclude_age == 'Yes':
            index_age=self.unique_namesnew.index('age')
            del self.unique_namesnew[index_age]
            del self.unique_LP[index_age]
            del self.unique_stder[index_age]
        # Exclude f.1249 if current smoker
        if self.indicator1239 == 1:
            index_1249=self.unique_namesnew.index('1249')
            del self.unique_namesnew[index_1249]
            del self.unique_LP[index_1249]
            del self.unique_stder[index_1249]
        # Exclude f.6141 is living alone
        if self.indicator6141 == 1:
            index_6141=self.unique_namesnew.index('6141')
            del self.unique_namesnew[index_6141]
            del self.unique_LP[index_6141]
            del self.unique_stder[index_6141]
            
        assert len(self.unique_namesnew)==len(self.unique_LP), 'ERROR'
        return self.unique_namesnew, self.unique_LP, self.unique_stder



def show ():
    ''' 
    Function that reads the data and the required files;
    run the functions to obtain the predicted score and performs checks
    '''
    
    # Read data and define main variables
    directory = os.getcwd()
    assert os.path.isdir(directory), 'ERROR'
    
    fname = directory + '/' + sys.argv[1]
    assert os.path.isfile(fname), 'ERROR'
    
    d = np.loadtxt(fname=fname, delimiter='\t',dtype='S100') # Read external file containing questionnaire results
    age=np.int_(d[d[:,0] == ['age'],1][0])
    assert issubclass(type(age), np.integer) and (age > 39 and age < 71), 'ERROR'
    
    sex=np.str_(d[d[:,0] == ['Sex'],1][0])
    assert sex == 'Male' or sex == 'Female', 'ERROR'
  
    # Define functions and process data
    fun_to_run = Predscore_final(d,age,sex,directory)
    
    clean_data=fun_to_run.data_load_and_process() # Clean data
    
    predscore, predscoreL, predscoreU = fun_to_run.calculate_risk() # Prediction score
    assert predscore < 1 and predscore > 0, 'ERROR'
    
    bioage, bioageL, bioageU, realriskage = fun_to_run.bioage() # Biological age
    assert bioage > 14 and bioage < 96, 'ERROR'
    
    categoryname, stdLP, STDERR = fun_to_run.list_for_plot(exclude_age='Yes') # Values for plot

    risk = np.int_(np.round( predscore*100.0, 0 ))
    invrisk = 100- risk
    if risk < 1:
        riskout = "less than 1"
        invriskout = "more than 99"
    else:
        riskout=risk
        invriskout=invrisk
    
    # Below the output
    sys.stdout.write(str(age) + ';' + sex + ';' + str(np.int_(bioage)) + "[" + str(np.int_(bioageL)) + ";" 
                     + str(np.int_(bioageU)) + "]" + ';' + str(np.float_(realriskage)) + ';' + str(predscore) + ';' 
                     + str(riskout) + ';' + str(invriskout) + ';')
    for i in range(0,len(categoryname)):
        sys.stdout.write(categoryname[i] + ' = ' + str(stdLP[i]) + "[" + str(stdLP[i]-norm.ppf(0.975)*STDERR[i]) + ";" 
                         + str(stdLP[i]+norm.ppf(0.975)*STDERR[i]) +  "]" + ';')

show()

