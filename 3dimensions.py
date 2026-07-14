import pandas as pd
import numpy as np
import yfinance as yf
import scipy
import warnings


#check Pandas version - treating keys as positions.
warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


#Variables (:
returnsArrayHold = []
returnsArrayT = []
returnsArrayGaussian = []

cola = yf.Ticker("KO")
pepsi = yf.Ticker("PEP")
mnst = yf.Ticker("MNST")
v = 3 #manually set v here, or use the function to estimate v
time = "10y"
percentSplit = 0.7
sensitivity = 0.25

#tidy pls
def trainingAndRealData(data,df=v):
    df = data.history(period=time, auto_adjust = True)
    df = df.dropna()
    splitPoint = int(len(df)*percentSplit)
    trainingData = df.iloc[:splitPoint]
    realData = df.iloc[splitPoint:]
    trainingData.loc[:,"Log Returns"] = np.log(trainingData.loc[:,"Close"]/trainingData.loc[:,"Close"].shift(1))
    trainingData = trainingData.dropna()
    n = trainingData["Log Returns"].count()
    trainingData["Rank"] = trainingData["Log Returns"].rank(method="average")
    trainingData["Percentile"] = trainingData["Rank"] / (n + 1)
    trainingData.loc[:,"Normal Score"] = scipy.stats.norm.ppf(trainingData.loc[:,"Percentile"])
    trainingData.loc[:, "T Score"] = scipy.stats.t.ppf(trainingData.loc[:, "Percentile"],v)
    realData.loc[:,"Log Returns"] = np.log(realData.loc[:,"Close"]/realData.loc[:,"Close"].shift(1))
    realData = realData.dropna()
    sortedTrainingData = np.sort(trainingData.loc[:,"Log Returns"])
    n =len(sortedTrainingData) 
    percentiles = []
    for r in realData.loc[:,"Log Returns"]:
        count = np.searchsorted(sortedTrainingData,r, side="right")
        percentile = (count+1)/(n+2)
        percentiles.append(percentile)
    realData.loc[:,"Percentile"] = [p for p in percentiles] 
    realData.loc[:,"Normal Score"] = scipy.stats.norm.ppf(realData.loc[:,"Percentile"])
    realData["T Score"] = scipy.stats.t.ppf(realData.loc[:, "Percentile"],v)
    return trainingData, realData 
    
trainingCola, realCola = trainingAndRealData(cola)
trainingPepsi, realPepsi= trainingAndRealData(pepsi)
trainingMnst, realMnst = trainingAndRealData(mnst)

print(trainingCola)
print(realCola)

def correlationMatrix(dataX, dataY, dataZ, target ='X'): #update to not do manually, training data
    zx = dataX["Normal Score"]
    zy = dataY["Normal Score"]
    zz = dataZ["Normal Score"]
    pzx = zz.corr(zx)
    pyx = zy.corr(zx)
    pxy = zx.corr(zy)
    pzy = zz.corr(zy)
    pxz = zx.corr(zz)
    pyz = zy.corr(zz)
    #R = np.array([[1,pxy,pxz],[[pyx, 1, pyz], [pzx, pzy,1]]])  #total one, don't really need
    Rxo = np.array([pxy, pxz])
    Rox = np.array([[1, pyz], [pzy,1]])
    Ryo = np.array([pyx, pyz])
    Roy = np.array([[1, pxz], [pzx,1]])
    Rzo = np.array([pzx, pzy])
    Roz = np.array([[1, pxy], [pxy,1]])
    if target == 'Y':
        return Ryo, Roy
    elif target == 'Z':
        return Rzo, Roz
    else:
        return Rxo, Rox

#global
Rvox, Rovx = correlationMatrix(trainingCola,trainingPepsi, trainingMnst, target = 'X')
Rvoy, Rovy = correlationMatrix(trainingCola,trainingPepsi, trainingMnst, target = 'Y')
Rvoz, Rovz = correlationMatrix(trainingCola,trainingPepsi, trainingMnst, target = 'Z')


def getNormalScores(dataX, dataY, dataZ, i,  target ='X'): #real data input
    zDictionary = {
        'X': dataX["Normal Score"],
        'Y': dataY["Normal Score"],
        'Z': dataZ["Normal Score"]
    } 
    toStack = [] #to get Zvo, had trouble with np.append, like array appends 
    for temp in zDictionary:
        if temp != target:
            toStack.append(zDictionary[temp][i])
    #Rvo, Rov = correlationMatrix(dataX, dataY, dataZ, target)
    Zvo = np.stack(toStack)
    #uConditional = Rvo @ np.linalg.inv(Rov) @ np.linalg(Zvo)
    return Zvo

def getHx(dataX, dataY, dataZ,i,Rvo, Rov, target = 'X'): #real data input, don't forget Rvo and Rov from training
    #duplicating this ffs man
    zDictionary = {
        'X': dataX["Normal Score"],
        'Y': dataY["Normal Score"],
        'Z': dataZ["Normal Score"]
    }
    zTarget = zDictionary[target][i]
    Zvo = getNormalScores(dataX, dataY, dataZ, i, target=target)
    muConditional = Rvo @np.linalg.inv(Rov) @ Zvo 
    stdConditional = np.sqrt(1- Rvo @ np.linalg.inv(Rov)@ Rvo)
    hx = scipy.stats.norm.cdf((zTarget-muConditional)/stdConditional) 
    return hx

def handleHx(hx):
    if hx <= sensitivity:
        var = 1
    elif hx >= 1-sensitivity:
        var = -1
    else:
        var = 0
    return var

def checkPositiveAndNegative(arr):
    negative = False
    positive = False
    for i in arr:
        if i>0:
            positive = True
        elif i<0:
            negative = True
    if positive and negative:
        return True
    return False

def validatePositionVector(arr): #arr: array of hx's
    h = [handleHx(temp) for temp in arr]
    hasLong = any(x > 0 for x in h)
    hasShort = any(x < 0 for x in h)
    if not hasLong or not hasShort:
        return [0, 0, 0]
    longExposure = sum(x for x in h if x > 0)
    shortExposure = -sum(x for x in h if x < 0)

    balanced = []

    for x in h:
        if x > 0:
            balanced.append(x / longExposure)
        elif x < 0:
            balanced.append(x / shortExposure)
        else:
            balanced.append(0)

    return balanced 

def getPositionVectorForADay(i):
    #handle if i not proper day
    hxx= getHx(realCola,realPepsi, realMnst, i,Rvox,Rovx, target = 'X')
    hxy =getHx(realCola,realPepsi, realMnst, i,Rvoy,Rovy, target = 'Y')
    hxz = getHx(realCola,realPepsi, realMnst, i,Rvoz,Rovz, target = 'Z')
    positionVector =  validatePositionVector([hxx, hxy, hxz]) 
    return positionVector    



def getReturn(data,i):
    return data["Close"][i]/data["Close"][i-1]-1    

def getTotalReturn(wPrevious,i):     
    colaReturn = getReturn(realCola,i)
    pepsiReturn = getReturn(realPepsi,i)
    mnstReturn = getReturn(realMnst,i)
    returnArray = [colaReturn,pepsiReturn, mnstReturn]
    totalReturn = np.dot(wPrevious,returnArray)
    return totalReturn 


def strategyGaussian():
    totalReturns = 1
    for i in range(1,len(realCola)):
        wt = getPositionVectorForADay(i-1)
        dailyReturn = getTotalReturn(wt,i)
        totalReturns *= 1+dailyReturn 
        returnsArrayGaussian.append(dailyReturn)
    return totalReturns-1




def equalWeightBuyAndHold():
    totalReturns = 1

    for i in range(1, len(realCola)):
        colaReturn = realCola["Close"].iloc[i] / realCola["Close"].iloc[i-1] - 1
        pepsiReturn = realPepsi["Close"].iloc[i] / realPepsi["Close"].iloc[i-1] - 1
        monsterReturn = realMnst["Close"].iloc[i] / realMnst["Close"].iloc[i-1] - 1

        portfolioReturn = (1/3)*colaReturn + (1/3)*pepsiReturn + (1/3)*monsterReturn
        returnsArrayHold.append(portfolioReturn)
        totalReturns *= 1 + portfolioReturn

    return totalReturns - 1


#t:
def correlationMatrixT(dataX, dataY, dataZ, target="X"):
    tx = dataX["T Score"]
    ty = dataY["T Score"]
    tz = dataZ["T Score"]

    def getRho(a, b):
        tau = a.corr(b, method="kendall")
        return np.sin(np.pi * tau / 2)

    pxy = getRho(tx, ty)
    pxz = getRho(tx, tz)
    pyz = getRho(ty, tz)

    if target == "X":
        Rvo = np.array([pxy, pxz])
        Rov = np.array([
            [1, pyz],
            [pyz, 1]
        ])

    elif target == "Y":
        Rvo = np.array([pxy, pyz])
        Rov = np.array([
            [1, pxz],
            [pxz, 1]
        ])

    elif target == "Z":
        Rvo = np.array([pxz, pyz])
        Rov = np.array([
            [1, pxy],
            [pxy, 1]
        ])

    return Rvo, Rov

def getTScores(dataX, dataY, dataZ, i, target="X"):
    tDictionary = {
        "X": dataX["T Score"],
        "Y": dataY["T Score"],
        "Z": dataZ["T Score"]
    }

    toStack = []

    for temp in tDictionary:
        if temp != target:
            toStack.append(tDictionary[temp].iloc[i])

    Tvo = np.array(toStack)

    return Tvo

Rtox, Rotx = correlationMatrixT(trainingCola, trainingPepsi, trainingMnst, target="X")
Rtoy, Roty = correlationMatrixT(trainingCola, trainingPepsi, trainingMnst, target="Y")
Rtoz, Rotz = correlationMatrixT(trainingCola, trainingPepsi, trainingMnst, target="Z")

def getHxT(dataX, dataY, dataZ, i, Rvo, Rov, target="X"):
    tDictionary = {
        "X": dataX["T Score"],
        "Y": dataY["T Score"],
        "Z": dataZ["T Score"]
    }

    tTarget = tDictionary[target].iloc[i]
    Tvo = getTScores(dataX, dataY, dataZ, i, target=target)

    invRov = np.linalg.inv(Rov)

    q = len(Tvo)  # number of conditioning variables, so here q = 2

    muConditional = Rvo @ invRov @ Tvo

    baseVariance = 1 - Rvo @ invRov @ Rvo.T

    tailScale = (v + Tvo @ invRov @ Tvo) / (v + q)

    stdConditional = np.sqrt(tailScale * baseVariance)

    hx = scipy.stats.t.cdf(
        (tTarget - muConditional) / stdConditional,
        df=v + q
    )

    return hx

def strategyT():
    totalReturns = 1

    for i in range(1, len(realCola)):
        wt = getPositionVectorForADayT(i-1)

        dailyReturn = getTotalReturn(wt, i)
        returnsArrayT.append(dailyReturn)
        totalReturns *= 1 + dailyReturn

    return totalReturns - 1

def getPositionVectorForADayT(i):
    hxx = getHxT(realCola, realPepsi, realMnst, i, Rtox, Rotx, target="X")
    hxy = getHxT(realCola, realPepsi, realMnst, i, Rtoy, Roty, target="Y")
    hxz = getHxT(realCola, realPepsi, realMnst, i, Rtoz, Rotz, target="Z")

    positionVector = validatePositionVector([hxx, hxy, hxz])

    return positionVector

def performanceStats(name, returnsArray):
    returnsArray = np.array(returnsArray)

    totalReturn = np.prod(1 + returnsArray) - 1
    dailyVolatility = np.std(returnsArray)
    annualisedVolatility = dailyVolatility * np.sqrt(252)

    if dailyVolatility != 0:
        sharpeRatio = np.mean(returnsArray) * np.sqrt(252) / dailyVolatility
    else:
        sharpeRatio = np.nan

    cumulative = np.cumprod(1 + returnsArray)
    runningMax = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / runningMax - 1
    maxDrawdown = np.min(drawdowns)

    winRate = np.mean(returnsArray > 0)
    averageDailyReturn = np.mean(returnsArray)

    activeDays = np.sum(returnsArray != 0)
    exposure = activeDays / len(returnsArray)

    print(f"---{name}---")
    print("Total return:", totalReturn * 100, "%")
    print("Average daily return:", averageDailyReturn)
    print("Annualised volatility:", annualisedVolatility)
    print("Sharpe ratio:", sharpeRatio)
    print("Max drawdown:", maxDrawdown * 100, "%")
    print("Win rate:", winRate * 100, "%")
    print("Active days:", activeDays)
    print("Exposure:", exposure * 100, "%")
    print()

def main():
    #performanceStats('Hold', returnsArrayHold)
    strategyGaussian()
    strategyT()
    equalWeightBuyAndHold()
    performanceStats('Hold', returnsArrayHold)
    performanceStats('T', returnsArrayT)
    performanceStats('Gaussian', returnsArrayGaussian)
main()