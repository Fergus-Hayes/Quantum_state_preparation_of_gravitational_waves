import numpy as np

def CxnNOT_U(n):
    if n>4:
        return 3*(2**n) - 1
    elif n==4:
        return 69
    elif n==3:
        return 17
    elif n==2:
        return 9
    elif n==1:
        return 0
    
def CxnNOT_CX(n):
    if n>4:
        return 3*(2**n) - 4
    elif n==4:
        return 36
    elif n==3:
        return 14
    elif n==2:
        return 6
    elif n==1:
        return 1
    
def CxnNOT_depth(n):
    if n>4:
        return 5*(2**n) - 4
    elif n==4:
        return 81
    elif n==3:
        return 27
    elif n==2:
        return 11
    elif n==1:
        return 1

def QFT_U(n):
    return (3*n*(n-1)/2) + n

def QFT_CX(n):
    return n*(n-1)

def CXL_U(n, nl):
    return n*(2**nl)*(CxnNOT_U(nl)) + 2**(nl+1) - 2

def CXL_CX(n, nl):
    return n*(2**nl)*CxnNOT_CX(nl)

def IntComp_U(n):
    return 14*(n-1)

def IntComp_CX(n):
    return 6*(n-1) + 1

def CInc_U(n):
    return 3*n
#    if n>2:
#        return 3*n
#    elif n==1:
#        return 0
#    elif n==2:
#        return 9
#    elif n==3:
#        return 94 #!?!

def CInc_CX(n):
    return 2*n
#    if n>2:
#        return 2*n
#    elif n==1:
#        return 1
#    elif n==2:
#        return 7
#    #elif n==3:
#    #    return 61

def Label_U(n, nl):
    return 2 + 2*QFT_U(nl) + (2**nl)*(2*IntComp_U(n+1) + CInc_U(nl))

def Label_CX(n, nl):
    return 2*QFT_CX(nl) + (2**nl)*(2*IntComp_CX(n+1) + CInc_CX(nl))

def Multold_U(n1, n2, n3):
    return 2*QFT_U(n3) + 198*(n1-1)*n2*n3 + 2

def Multold_CX(n1, n2, n3):
    return 2*QFT_CX(n3) + 132*(n1-1)*n2*n3

def Add_U(n1, n2):
    return 3*(n1*n2)+2*QFT_U(n2)

def Add_CX(n1, n2):
    return 2*(n1*n2)+2*QFT_CX(n2)

#def CTwoComp_U(n):
#    return 9*QFT_U(n) + (2*n)

def CTwoComp_CX(n):
    return 2*(9*QFT_CX(n) + (3*n)) + n

#def Mult_U(n1,n2,n3):
#    return 2*QFT_U(n3) + 9*(n1-1)*n2*n3 + TwoComp_U(n3)

def Mult_CX(n1,n2,n3):
    return 2*QFT_CX(n3) + 8*(n1-1)*n2*n3 + CTwoComp_CX(n3)

def LPFold_U(nx, na, nc, nl):
    return Label_U(nx, nl) + CXL_U(nc, nl) + CXL_U(na, nl) + Multold_U(nc, nx, na)

def LPFold_CX(nx, na, nc, nl):
    return Label_CX(nx, nl) + CXL_CX(nc, nl) + CXL_CX(na, nl) + Multold_CX(nc, nx, na)

#def LPF_U(nx, na, nc, nl):
#    return Label_U(nx, nl) + CXL_U(nc, nl) + CXL_U(na, nl) + Mult_U(nc, nx, na)

#def LPF_CX(nx, na, nc, nl):
#    return Label_CX(nx, nl) + 3*CXL_CX(nc, nl) + Mult_CX(nc, nx, na) + Add_CX(nc, na)

def LPF_CX(nx, na, nc, nl):
    #print('Label:',Label_CX(nx, nl),'Mult:', Mult_CX(nc, nx, na),'CXL:', 3*CXL_CX(nc, nl),'Add:', Add_CX(nc, na))
    return Label_CX(nx, nl) + 3*CXL_CX(nc, nl) + Mult_CX(nc, nx, na) + Add_CX(nc, na)

def CxnRY_U(n):
    #if n>2:
    #    return 4*(2**n) - 4
    #elif n==2:
    #    return 20 
    #elif n==1:
    #    return 2
    #elif n==0:
    #    return 1
    return np.where(n>2, 4*(2**n) - 4, np.where(n==2, 20, np.where(n==1, 2, 0)))

def CxnRY_CX(n):
    #if n>2:
    #    return 3*(2**n) - 4
    #elif n==2:
    #    return 12
    #elif n==1:
    #    return 2
    #elif n==0:
    #    return 0
    return np.where(n>2,3*(2**n) - 4,np.where(n==2, 12, np.where(n==1, 2, 0)))

def GRlow_U(n):
    if n==1:
        return 1
    elif n==0:
        return 0
    else:
        ms = np.arange(1,n)
        return np.sum((2**(ms+1))-2+(2**(ms))*CxnRY_U(ms)) + 1

def GRlow_CX(n):
    if n==1:
        return 0
    elif n==0:
        return 0
    else:
        ms = np.arange(1,n)
        return np.sum((2**ms)*CxnRY_CX(ms))

def GRhigh_U(n, na, nc, nl, mc=1):
    if n<=mc:
        return GRlow_U(n)
    if n==1:
        return 1
    if n==0:
        return 0
    else:
        ms = np.arange(mc,n)
        return GRlow_U(mc) + np.sum(2*LPF_U(ms, na, nc, nl) + 2*na)

def GRhigh_CX(n, na, nc, nl, mc=1):
    if n<=mc:
        return GRlow_CX(n)

    if n==1:
        return 0
    if n==0:
        return 0
    else:
        ms = np.arange(mc,n)
        return GRlow_CX(mc) + np.sum(2*LPF_CX(ms, na, nc, nl) + 2*na)

def GRhighold_CX(n, na, nc, nl, mc=1):
    if n<=mc:
        return GRlow_CX(n)

    if n==1:
        return 0
    if n==0:
        return 0
    else:
        ms = np.arange(mc,n)
        return GRlow_CX(mc) + np.sum(2*LPFold_CX(ms, na, nc, nl) + 2*na)

def GW_U(n, na, nc, nl, mc=1):
    return GRhigh_U(n, int(na+nc)//2, int(na+nc)-int(na+nc)//2, nl, mc=mc) + 2*LPF_U(n, na, nc, nl) + na

def GW_CX(n, na, nc, nl, mc=1, mmax=None):
    if mmax==None:
        mmax=n
    return GRhigh_CX(mmax, na, nc, nl, mc=mc) + 2*LPF_CX(n, na, nc, nl)


def GWold_CX(n, na, nc, nl, mc=1):
    return GRhighold_CX(n, int(na+nc)//2, int(na+nc)-int(na+nc)//2, nl, mc=mc) + 2*LPFold_CX(n, na, nc, nl)


def GW_CX_ML(n, na, nc, nl, R=1):
    return 2*LPF_CX(n, na, nc, nl) + R*n
