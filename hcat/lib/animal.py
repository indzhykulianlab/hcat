from typing import Callable
# https://en.wikipedia.org/wiki/Greenwood_function
class Animal:
    NAME: str
    A: float  # Scaling constant
    K: float  # constant of integration
    B: float  # slope of straight-line portion of the frequency-position curve
    BASE: float # exponent
    FN: Callable[[float], float]


"""
public double fmouse(double d){ // d is fraction total distance
    //f(Hz) = (10 ^((1-d)*0.92) - 0.680)* 9.8
    return (Math.pow(10, (1-d)*0.92) - 0.680) * 9.8;
}
"""
class Mouse(Animal):
    NAME: str = 'mouse'
    A: float = 9.8
    B: float = 0.92
    K: float = 0.680
    BASE: float = 10
    FN = None


class Cat(Animal):
    NAME: str = 'cat'
    A: float = 0.456
    B: float = 2.1
    K: float = 0.80
    BASE: float = 10

class Chinchilla(Animal):
    NAME: str = 'chinchilla'
    A: float = 0.456
    B: float = 5.1
    K: float = 0.80
    BASE: float = 2.718281828 # e

class RhesysMonkey(Animal):
    NAME: str = 'rhesys monkey'
    A: float = 360/1000
    B: float = 2.1
    K: float = 0.85
    BASE: float = 2.718281828 # e

class GuineaPig(Animal):
    FN = lambda d: 10 ** ((66.4 - (100*d))/38.2)


"""
public double fCat(double d)	//where d is the fraction of the total distance from base (0 - 1)
{
    //(f(Hz) = 10 ^((100-d)*0.021) - 0.80)* 0.456
    d = d * 100;
    double res = (Math.pow(10, (100-d)*0.021) - 0.80) * 0.456;
    return res;
}

public double fGuinea_pig(double d)
{
    d = d * 100;
    //f(Hz) = 10 ^((66.4-d)/38.2)
    return Math.pow(10, (66.4-d)/38.2);
    //return (Math.pow(10, (1-d)*0.92) - 0.680) * 9.8;
}

public double fChinchilla(double d)
{
    d = d * 100;
    // (f(KHz) = e ^((100-d)*0.051) *125)/1000
    return (Math.exp((100-d)*0.051) * 125) / 1000;
}

public double fHuman(double d)
{
    d = d * 100;
    // f(KHz) = (10 ^(((100-d)/100)*2) - 0.4)*200/1000
    return (Math.pow(10, ((100-d)/100)*2) - 0.4) * 200 / 1000;
}

public double fMouse(double d)
{
    //f(Hz) = (10 ^((1-d)*0.92) - 0.680)* 9.8
    return (Math.pow(10, (1-d)*0.92) - 0.680) * 9.8;
}

//added 10/2014
//f=-(LN((x+4.632)/102.048))/0.04357
public double fRat(double d)
{
    return Math.log(((d*100+4.632)/102.048))*-1/0.04357;
}

//f  (kHz) =  360 * (10 ^((1-d)*2.1) - 0.85)    where d is the fraction of the total distance from base (0 - 1)
public double fRhesusMonkey(double d)
{
    return (Math.pow(10, (1-d)*2.1) - 0.85) * 360 / 1000;
}

// Added 10/2015 by Brad Buran. Based on Muller, Hearing Research 94 (1996)
// pp. 148-156.
// CF(pos) = 0.398*(10^((100-pos)*0.022)-0.631)
// Since this plugin provides distance as a fraction rather than percent,
// modified the equation to
// CF(pos) = 0.398*(10^((1-d)*2.2)-0.631)
public double fGerbil(double d)
{
    return 0.398*(Math.pow(10, (1-d)*2.2)-0.631);
}

//Marmoset: f(Hz) = 255.7*((10^((1-d)*2.1))-0.686)
public double fMarmoset(double d)
{
    return (255.7*(Math.pow(10, ((1-d)*2.1))-0.686)/1000);	//return kHz
}
"""

