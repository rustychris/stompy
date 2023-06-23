from __future__ import print_function

# Pure python implementation of J.R. Shewchuk's robust geometric predicates.
# This is a straightforward translation of the predicates in triangle.c into
# python.

# THIS CODE IS NOT ROBUST!
# it does not attempt to set the rounding and precision mode of the FPU
# As such it does not properly detect FP roundoff and gives incorrect
# results.


## Initialization:
#  There is a bit of dynamic work which happens on import to figure out
#  a few magic floating point values

# skipping weird FPU stuff. hope that isn't necessary ....   probably wrong about that.

#    cword = 4722;                 /* set FPU control word for double precision */
#    _FPU_SETCW(cword);
    
every_other = 1
half = 0.5
epsilon = 1.0
splitter = 1.0
check = 1.0

while 1:
    lastcheck = check
    epsilon *= half
    if every_other:
        splitter *= 2.0

    every_other = 1-every_other
    check = 1.0 + epsilon
    if not ((check != 1.0) and (check != lastcheck)):
        break

splitter += 1.0
# Error bounds for orientation and incircle tests.
resulterrbound = (3.0 + 8.0 * epsilon) * epsilon;
ccwerrboundA = (3.0 + 16.0 * epsilon) * epsilon;
ccwerrboundB = (2.0 + 12.0 * epsilon) * epsilon;
ccwerrboundC = (9.0 + 64.0 * epsilon) * epsilon * epsilon;
iccerrboundA = (10.0 + 96.0 * epsilon) * epsilon;
iccerrboundB = (4.0 + 48.0 * epsilon) * epsilon;
iccerrboundC = (44.0 + 576.0 * epsilon) * epsilon * epsilon;
o3derrboundA = (7.0 + 56.0 * epsilon) * epsilon;
o3derrboundB = (3.0 + 28.0 * epsilon) * epsilon;
o3derrboundC = (26.0 + 288.0 * epsilon) * epsilon * epsilon;



## these are all macros in the C version:

def Fast_Two_Sum(a,b):
    x = a + b
    bvirt = x-a
    y = b-bvirt
    return x,y

def Two_Sum(a,b):
    x = (a + b)
    bvirt = x - a
    avirt = x - bvirt
    bround = b - bvirt
    around = a - avirt
    y = around + bround
    return x,y


def Split(a):
    c = splitter * a
    abig = c - a
    ahi = c - abig
    alo = a - ahi
    return ahi,alo

def Two_Product_Presplit(a, b, bhi, blo):
    x = a * b
    ahi,alo = Split(a)
    err1 = x - (ahi * bhi)
    err2 = err1 - (alo * bhi)
    err3 = err2 - (ahi * blo)
    y = (alo * blo) - err3
    return x,y

def Two_Product(a, b):
    x = a * b
    ahi,alo = Split(a)
    bhi,blo = Split(b)
    err1 = x - (ahi * bhi)
    err2 = err1 - (alo * bhi) 
    err3 = err2 - (ahi * blo) 
    y = (alo * blo) - err3
    return x,y

def Two_Two_Diff(a1, a0, b1, b0):
    _j, _0, x0 = Two_One_Diff(a1, a0, b0)
    x3, x2, x1 = Two_One_Diff(_j, _0, b1)
    return x3, x2, x1, x0

def Two_Two_Sum(a1, a0, b1, b0):
    _j, _0, x0 = Two_One_Sum(a1, a0, b0)
    x3, x2, x1 = Two_One_Sum(_j, _0, b1)
    return x3, x2, x1, x0

def Two_One_Sum(a1, a0, b):
    _i, x0 = Two_Sum(a0, b)
    x2, x1 = Two_Sum(a1, _i)
    return x2, x1, x0


def Two_One_Diff(a1, a0, b):
    _i, x0 = Two_Diff(a0, b)
    x2, x1 = Two_Sum(a1, _i)
    return x2,x1,x0

def Two_Diff(a, b):
    x = a - b
    bvirt = a - x
    avirt = x + bvirt
    bround = bvirt - b
    around = a - avirt
    y = around + bround
    return x,y


def Two_Diff_Tail(a, b, x):
    bvirt = (a - x)
    avirt = x + bvirt
    bround = bvirt - b
    around = a - avirt
    y = around + bround
    return y
  
def Absolute(a):
    if a >= 0.0:
        return a
    else:
        return -a


def Square(a):
    x = (a * a)
    ahi,alo = Split(a)
    err1 = x - (ahi * ahi)
    err3 = err1 - ((ahi + ahi) * alo)
    y = (alo * alo) - err3
    return x,y


## Operations on lists of floating point values
def fast_expansion_sum_zeroelim(e, f):
    # where e, f and h are lists of floats (for now just lists, not numpy arrays)
    enow = e[0]
    fnow = f[0]
    
    elen = len(e)
    flen = len(f)
    
    eindex = findex = 0
    if (fnow > enow) == (fnow > -enow):
        Q = enow
        eindex += 1
        if eindex < len(e):
            enow = e[eindex]
        else:
            enow = 0.0 # not sure what to do here
    else:
        Q = fnow
        findex += 1
        if findex < len(f):
            fnow = f[findex]
        else:
            fnow = 0.0 # or here...

    hindex = 0
    h=[]
    if (eindex < elen) and (findex < flen):
        if (fnow > enow) == (fnow > -enow):
            Qnew, hh = Fast_Two_Sum(enow, Q)
            eindex += 1
            if eindex < len(e):
                enow = e[eindex]
            else:
                enow = 0.0
        else:
            Qnew, hh = Fast_Two_Sum(fnow, Q)
            findex += 1
            if findex < len(f):
                fnow = f[findex]
            else:
                fnow = 0.0

        Q = Qnew
        if hh != 0.0:
            h.append(hh)
            
        while (eindex < elen) and (findex < flen):
            if (fnow > enow) == (fnow > -enow):
                Qnew,hh = Two_Sum(Q, enow)
                eindex+=1
                if eindex < len(e):
                    enow = e[eindex]
                else:
                    enow = 0.0
            else:
                Qnew, hh = Two_Sum(Q, fnow)
                findex += 1
                if findex< len(f):
                    fnow = f[findex]
                else:
                    fnow = 0.0
            Q = Qnew
            if hh != 0.0:
                h.append(hh)

    while eindex < elen:
        Qnew, hh = Two_Sum(Q, enow)
        eindex+=1
        if eindex < len(e):
            enow = e[eindex]
        else:
            enow = 0.0
            
        Q = Qnew
        if hh != 0.0:
            h.append(hh)

    while findex < flen:
        Qnew, hh = Two_Sum(Q, fnow)
        findex += 1
        if findex < len(f):
            fnow = f[findex]
        else:
            fnow = 0.0
            
        Q = Qnew
        if hh != 0.0:
            h.append(hh)
            
    if (Q != 0.0) or (len(h) == 0):
        h.append(Q)
    return h

# test - seems to work, though it puts the smallest term first in the
# output.
# h = fast_expansion_sum_zeroelim([12.0],[1.])
 

def scale_expansion_zeroelim(e, b):
    h = []
    bhi,blo = Split(b)
    Q, hh = Two_Product_Presplit(e[0], b, bhi, blo)

    if hh != 0:
        h.append(hh)

    for eindex in range(1,len(e)): # (eindex = 1; eindex < elen; eindex++):
        enow = e[eindex]
        product1, product0 = Two_Product_Presplit(enow, b, bhi, blo)
        sum, hh = Two_Sum(Q, product0)
        if hh != 0:
            h.append(hh)

        Q, hh = Fast_Two_Sum(product1, sum)
        if hh != 0:
            h.append(hh)

    if (Q != 0.0) or (len(h) == 0):
        h.append(Q)

    return h



 
def estimate(e):
    Q = e[0]
    for eindex in range(1,len(e)):
        Q += e[eindex]
    return Q


def counterclockwiseadapt(pa, pb, pc, detsum):
    B = [0]*4
    
    acx = pa[0] - pc[0]
    bcx = pb[0] - pc[0]
    acy = pa[1] - pc[1]
    bcy = pb[1] - pc[1]
  
    detleft, detlefttail =  Two_Product(acx, bcy)
    detright, detrighttail = Two_Product(acy, bcx)
  
    B[3], B[2], B[1], B[0] = Two_Two_Diff(detleft, detlefttail, detright, detrighttail )
  
    det = estimate(B)
    errbound = ccwerrboundB * detsum
    if ((det >= errbound) or (-det >= errbound)):
        return det

    acxtail = Two_Diff_Tail(pa[0], pc[0], acx)
    bcxtail = Two_Diff_Tail(pb[0], pc[0], bcx)
    acytail = Two_Diff_Tail(pa[1], pc[1], acy)
    bcytail = Two_Diff_Tail(pb[1], pc[1], bcy)
  
    if ((acxtail == 0.0) and (acytail == 0.0) and (bcxtail == 0.0) and (bcytail == 0.0)):
        return det

    errbound = ccwerrboundC * detsum + resulterrbound * Absolute(det)
    det += (acx * bcytail + bcy * acxtail) - (acy * bcxtail + bcx * acytail)
    
    if ((det >= errbound) or (-det >= errbound)):
        return det

    s1, s0 = Two_Product(acxtail, bcy)
    t1, t0 = Two_Product(acytail, bcx)
    u = [0]*4
    u[3], u[2], u[1], u[0] = Two_Two_Diff(s1, s0, t1, t0, )
    
    C1 = fast_expansion_sum_zeroelim(B, u)
  
    s1, s0 = Two_Product(acx, bcytail)
    t1, t0 = Two_Product(acy, bcxtail)
    u[3], u[2], u[1], u[0] = Two_Two_Diff(s1, s0, t1, t0)
    C2 = fast_expansion_sum_zeroelim(C1, u)
  
    s1, s0 = Two_Product(acxtail, bcytail)
    t1, t0 = Two_Product(acytail, bcxtail)
    u[3], u[2], u[1], u[0] = Two_Two_Diff(s1, s0, t1, t0)

    D = fast_expansion_sum_zeroelim(C2, u)
  
    return D[-1]

  
def counterclockwise(pa, pb, pc):
    detleft = (pa[0] - pc[0]) * (pb[1] - pc[1])
    detright = (pa[1] - pc[1]) * (pb[0] - pc[0])
    det = detleft - detright
  
    if detleft > 0.0:
        if detright <= 0.0:
            return det
        else:
            detsum = detleft + detright
    elif detleft < 0.0:
        if detright >= 0.0:
            return det
        else:
            detsum = -detleft - detright
    else:
        return det

    errbound = ccwerrboundA * detsum
    if ((det >= errbound) or (-det >= errbound)):
        return det
  
    return counterclockwiseadapt(pa, pb, pc, detsum)


  

def incircleadapt(pa, pb, pc, pd, permanent):
    bc = [0.0]*4 # needed.
    ca = [0.0]*4
    ab = [0.0]*4
    aa = [0.0]*4
    bb = [0.0]*4
    cc = [0.0]*4
    u = [0.0]*4
    v = [0.0]*4
    abtt = [0.0]*4
    bctt = [0.0]*4
    catt = [0.0]*4

    # print "incircle going to adaptive precision"
    
    adx = pa[0] - pd[0]
    bdx = pb[0] - pd[0]
    cdx = pc[0] - pd[0]
    ady = pa[1] - pd[1]
    bdy = pb[1] - pd[1]
    cdy = pc[1] - pd[1]
  
    bdxcdy1, bdxcdy0 = Two_Product(bdx, cdy)
    cdxbdy1, cdxbdy0 = Two_Product(cdx, bdy)
    bc[3], bc[2], bc[1], bc[0] = Two_Two_Diff(bdxcdy1, bdxcdy0, cdxbdy1, cdxbdy0)

    axbc = scale_expansion_zeroelim(bc, adx)
    axxbc = scale_expansion_zeroelim(axbc, adx)
    aybc = scale_expansion_zeroelim(bc, ady)
    ayybc = scale_expansion_zeroelim(aybc, ady)
    adet = fast_expansion_sum_zeroelim(axxbc, ayybc)
  
    cdxady1, cdxady0 = Two_Product(cdx, ady)
    adxcdy1, adxcdy0 = Two_Product(adx, cdy)
    
    ca[3], ca[2], ca[1], ca[0] = Two_Two_Diff(cdxady1, cdxady0, adxcdy1, adxcdy0)
    
    bxca = scale_expansion_zeroelim(ca, bdx)
    bxxca = scale_expansion_zeroelim(bxca, bdx)
    byca = scale_expansion_zeroelim(ca, bdy)
    byyca = scale_expansion_zeroelim(byca, bdy)
    bdet = fast_expansion_sum_zeroelim(bxxca, byyca)
  
    adxbdy1, adxbdy0 = Two_Product(adx, bdy)
    bdxady1, bdxady0 = Two_Product(bdx, ady)
    ab[3], ab[2], ab[1], ab[0] = Two_Two_Diff(adxbdy1, adxbdy0, bdxady1, bdxady0)

    cxab = scale_expansion_zeroelim(ab, cdx)
    cxxab = scale_expansion_zeroelim(cxab, cdx)
    cyab = scale_expansion_zeroelim(ab, cdy)
    cyyab = scale_expansion_zeroelim(cyab, cdy)
    cdet = fast_expansion_sum_zeroelim(cxxab, cyyab)
  
    abdet = fast_expansion_sum_zeroelim(adet, bdet)
    fin1 = fast_expansion_sum_zeroelim(abdet, cdet)
  
    det = estimate(fin1)
    errbound = iccerrboundB * permanent
    if ((det >= errbound) or (-det >= errbound)):
        return det

    # print "incircle: maybe going to higher precision"
    
    adxtail = Two_Diff_Tail(pa[0], pd[0], adx)
    adytail = Two_Diff_Tail(pa[1], pd[1], ady)
    bdxtail = Two_Diff_Tail(pb[0], pd[0], bdx)
    bdytail = Two_Diff_Tail(pb[1], pd[1], bdy)
    cdxtail = Two_Diff_Tail(pc[0], pd[0], cdx)
    cdytail = Two_Diff_Tail(pc[1], pd[1], cdy)
    
    if ((adxtail == 0.0) and (bdxtail == 0.0) and (cdxtail == 0.0) and (adytail == 0.0) and (bdytail == 0.0) and (cdytail == 0.0)):
        return det

    # print "really going to higher precision"

    errbound = iccerrboundC * permanent + resulterrbound * Absolute(det)

    det += ((adx * adx + ady * ady) * ((bdx * cdytail + cdy * bdxtail) - (bdy * cdxtail + cdx * bdytail)) \
            + 2.0 * (adx * adxtail + ady * adytail) * (bdx * cdy - bdy * cdx))  \
         + ((bdx * bdx + bdy * bdy) * ((cdx * adytail + ady * cdxtail)          \
                                       - (cdy * adxtail + adx * cdytail))       \
            + 2.0 * (bdx * bdxtail + bdy * bdytail) * (cdx * ady - cdy * adx))  \
         + ((cdx * cdx + cdy * cdy) * ((adx * bdytail + bdy * adxtail)          \
                                       - (ady * bdxtail + bdx * adytail))       \
            + 2.0 * (cdx * cdxtail + cdy * cdytail) * (adx * bdy - ady * bdx))
    
    if (det >= errbound) or (-det >= errbound):
        return det

    # print "incircle: going all the way"
  
    finnow = fin1
  
    if (bdxtail != 0.0) or (bdytail != 0.0) or (cdxtail != 0.0) or (cdytail != 0.0):
        adxadx1, adxadx0 = Square(adx)
        adyady1, adyady0 = Square(ady)
        aa[3], aa[2], aa[1], aa[0] = Two_Two_Sum(adxadx1, adxadx0, adyady1, adyady0)

    if (cdxtail != 0.0) or (cdytail != 0.0) or (adxtail != 0.0) or (adytail != 0.0):
        bdxbdx1, bdxbdx0 = Square(bdx)
        bdybdy1, bdybdy0 = Square(bdy)
        bb[3], bb[2], bb[1], bb[0] = Two_Two_Sum(bdxbdx1, bdxbdx0, bdybdy1, bdybdy0)

    if (adxtail != 0.0) or (adytail != 0.0) or (bdxtail != 0.0) or (bdytail != 0.0):
        cdxcdx1, cdxcdx0 = Square(cdx)
        cdycdy1, cdycdy0 = Square(cdy)
        cc[3], cc[2], cc[1], cc[0] = Two_Two_Sum(cdxcdx1, cdxcdx0, cdycdy1, cdycdy0)

    if adxtail != 0.0:
        axtbc = scale_expansion_zeroelim(bc, adxtail)
        temp16a = scale_expansion_zeroelim(axtbc, 2.0 * adx)
               
        axtcc = scale_expansion_zeroelim(cc, adxtail)
        temp16b = scale_expansion_zeroelim(axtcc, bdy)
    
        axtbb = scale_expansion_zeroelim(bb, adxtail)
        temp16c = scale_expansion_zeroelim(axtbb, -cdy)
    
        temp32a = fast_expansion_sum_zeroelim(temp16a, temp16b)
        temp48 = fast_expansion_sum_zeroelim(temp16c, temp32a)
        finother = fast_expansion_sum_zeroelim(finnow, temp48)
        finnow,finother = finother,finnow

    if adytail != 0.0:
        aytbc = scale_expansion_zeroelim(bc, adytail)
        temp16a = scale_expansion_zeroelim(aytbc, 2.0 * ady)
    
        aytbb = scale_expansion_zeroelim(bb, adytail)

        temp16b = scale_expansion_zeroelim( aytbb, cdx)

        aytcc = scale_expansion_zeroelim( cc, adytail)
        

        temp16c = scale_expansion_zeroelim( aytcc, -bdx)

        temp32a = fast_expansion_sum_zeroelim( temp16a, temp16b)
        temp48 = fast_expansion_sum_zeroelim( temp16c, temp32a)
        
        finother = fast_expansion_sum_zeroelim( finnow,
                                                temp48)
        finnow,finother = finother,finnow
        
    if bdxtail != 0.0:
        bxtca = scale_expansion_zeroelim( ca, bdxtail)

        temp16a = scale_expansion_zeroelim( bxtca, 2.0 * bdx)
  
        bxtaa = scale_expansion_zeroelim( aa, bdxtail)
        temp16b = scale_expansion_zeroelim( bxtaa, cdy)
  
        bxtcc = scale_expansion_zeroelim( cc, bdxtail)
        temp16c = scale_expansion_zeroelim( bxtcc, -ady)
  
        temp32a = fast_expansion_sum_zeroelim( temp16a, temp16b)
        temp48 = fast_expansion_sum_zeroelim( temp16c, temp32a)
        finother = fast_expansion_sum_zeroelim( finnow,
                                              temp48)
        finnow,finother = finother,finnow

    if bdytail != 0.0:
        bytca = scale_expansion_zeroelim( ca, bdytail)

        temp16a = scale_expansion_zeroelim( bytca, 2.0 * bdy)
  
        bytcc = scale_expansion_zeroelim( cc, bdytail)
        temp16b = scale_expansion_zeroelim( bytcc, adx)
  
        bytaa = scale_expansion_zeroelim( aa, bdytail)
        temp16c = scale_expansion_zeroelim( bytaa, -cdx)
  
        temp32a = fast_expansion_sum_zeroelim( temp16a, temp16b)
        temp48 = fast_expansion_sum_zeroelim( temp16c, temp32a)
        finother = fast_expansion_sum_zeroelim( finnow,
                                              temp48)
        finnow,finother = finother,finnow

    if cdxtail != 0.0:
        cxtab = scale_expansion_zeroelim( ab, cdxtail)

        temp16a = scale_expansion_zeroelim( cxtab, 2.0 * cdx)
  
        cxtbb = scale_expansion_zeroelim( bb, cdxtail)
        temp16b = scale_expansion_zeroelim( cxtbb, ady)
  
        cxtaa = scale_expansion_zeroelim( aa, cdxtail)
        temp16c = scale_expansion_zeroelim( cxtaa, -bdy)
  
        temp32a = fast_expansion_sum_zeroelim( temp16a, temp16b)
        temp48 = fast_expansion_sum_zeroelim( temp16c, temp32a)
        finother = fast_expansion_sum_zeroelim( finnow,
                                              temp48)
        finnow,finother = finother,finnow

    if cdytail != 0.0:
        cytab = scale_expansion_zeroelim( ab, cdytail)

        temp16a = scale_expansion_zeroelim( cytab, 2.0 * cdy)
  
        cytaa = scale_expansion_zeroelim( aa, cdytail)
        temp16b = scale_expansion_zeroelim( cytaa, bdx)
  
        cytbb = scale_expansion_zeroelim( bb, cdytail)
        temp16c = scale_expansion_zeroelim( cytbb, -adx)
  
        temp32a = fast_expansion_sum_zeroelim( temp16a, temp16b)
        temp48 = fast_expansion_sum_zeroelim( temp16c, temp32a)
        finother = fast_expansion_sum_zeroelim( finnow,
                                              temp48)
        finnow,finother = finother,finnow

    if (adxtail != 0.0) or (adytail != 0.0): 
        if (bdxtail != 0.0) or (bdytail != 0.0) or (cdxtail != 0.0) or (cdytail != 0.0):
            ti1, ti0 = Two_Product(bdxtail, cdy )
            tj1, tj0 = Two_Product(bdx, cdytail)
            u[3], u[2], u[1], u[0] = Two_Two_Sum(ti1, ti0, tj1, tj0)

            negate = -bdy
            ti1, ti0 = Two_Product(cdxtail, negate)
            negate = -bdytail
            tj1, tj0 = Two_Product(cdx, negate)
            v[3], v[2], v[1], v[0] = Two_Two_Sum(ti1, ti0, tj1, tj0)

            bct = fast_expansion_sum_zeroelim( u, v)
      
            ti1, ti0 = Two_Product(bdxtail, cdytail)
            tj1, tj0 = Two_Product(cdxtail, bdytail)
            bctt[3], bctt[2], bctt[1], bctt[0] = Two_Two_Diff(ti1, ti0, tj1, tj0)
        else:
            bct= [0.0]
            bctt = [0.0]

        if (adxtail != 0.0):
            temp16a = scale_expansion_zeroelim( axtbc, adxtail)
            axtbct = scale_expansion_zeroelim( bct, adxtail)
  
            temp32a = scale_expansion_zeroelim( axtbct, 2.0 * adx)
            temp48 = fast_expansion_sum_zeroelim( temp16a, temp32a)
            finother = fast_expansion_sum_zeroelim( finnow,temp48)
            finnow,finother = finother,finnow
            
            if bdytail != 0.0:
                temp8 = scale_expansion_zeroelim( cc, adxtail)
    
                temp16a = scale_expansion_zeroelim( temp8, bdytail)
                finother = fast_expansion_sum_zeroelim( finnow, temp16a)
                finnow,finother = finother,finnow

            if cdytail != 0.0:
                temp8 = scale_expansion_zeroelim( bb, -adxtail)
                temp16a = scale_expansion_zeroelim( temp8, cdytail)
                finother = fast_expansion_sum_zeroelim( finnow, temp16a)
                finnow,finother = finother,finnow

            temp32a = scale_expansion_zeroelim( axtbct, adxtail)
            axtbctt = scale_expansion_zeroelim( bctt, adxtail)
  
            temp16a = scale_expansion_zeroelim( axtbctt, 2.0 * adx)
  
            temp16b = scale_expansion_zeroelim( axtbctt, adxtail)
            temp32b = fast_expansion_sum_zeroelim( temp16a, temp16b)
            temp64 = fast_expansion_sum_zeroelim( temp32a, temp32b)
            finother = fast_expansion_sum_zeroelim( finnow, temp64)
            finnow,finother = finother,finnow

        if adytail != 0.0:
            temp16a = scale_expansion_zeroelim( aytbc, adytail)
            aytbct = scale_expansion_zeroelim( bct, adytail)
  
            temp32a = scale_expansion_zeroelim( aytbct, 2.0 * ady)
            temp48 = fast_expansion_sum_zeroelim( temp16a, temp32a)
            finother = fast_expansion_sum_zeroelim( finnow,temp48)
            finnow,finother = finother,finnow
    
            temp32a = scale_expansion_zeroelim( aytbct, adytail)
            aytbctt = scale_expansion_zeroelim( bctt, adytail)
  
            temp16a = scale_expansion_zeroelim( aytbctt, 2.0 * ady)
  
            temp16b = scale_expansion_zeroelim( aytbctt, adytail)
            temp32b = fast_expansion_sum_zeroelim( temp16a, temp16b)
            temp64 = fast_expansion_sum_zeroelim( temp32a, temp32b)
            finother = fast_expansion_sum_zeroelim( finnow,temp64)
            finnow,finother = finother,finnow

    if (bdxtail != 0.0) or (bdytail != 0.0):
        if (cdxtail != 0.0) or (cdytail != 0.0) or (adxtail != 0.0) or (adytail != 0.0):
            ti1, ti0 = Two_Product(cdxtail, ady)
            tj1, tj0 =Two_Product(cdx, adytail)
            u[3], u[2], u[1], u[0] = Two_Two_Sum(ti1, ti0, tj1, tj0)
            negate = -cdy
            ti1, ti0 = Two_Product(adxtail, negate)
            negate = -cdytail
            tj1, tj0 = Two_Product(adx, negate)
            v[3], v[2], v[1], v[0] = Two_Two_Sum(ti1, ti0, tj1, tj0)
            cat = fast_expansion_sum_zeroelim(u, v)
        
            ti1, ti0 = Two_Product(cdxtail, adytail)
            tj1, tj0 = Two_Product(adxtail, cdytail)
            catt[3], catt[2], catt[1], catt[0] = Two_Two_Diff(ti1, ti0, tj1, tj0)
        else:
            cat = [0.0]
            catt = [0.0]
        
        if bdxtail != 0.0:
            temp16a = scale_expansion_zeroelim( bxtca, bdxtail)
            bxtcat = scale_expansion_zeroelim( cat, bdxtail)
        
            temp32a = scale_expansion_zeroelim( bxtcat, 2.0 * bdx)
            temp48 = fast_expansion_sum_zeroelim( temp16a, temp32a)
            finother = fast_expansion_sum_zeroelim( finnow, temp48)
            finnow,finother = finother,finnow
            if cdytail != 0.0:
                temp8 = scale_expansion_zeroelim( aa, bdxtail)
                temp16a = scale_expansion_zeroelim( temp8, cdytail)
                finother = fast_expansion_sum_zeroelim( finnow, temp16a)
                finnow,finother = finother,finnow

            if adytail != 0.0:
                temp8 = scale_expansion_zeroelim( cc, -bdxtail)
                temp16a = scale_expansion_zeroelim( temp8, adytail)
                finother = fast_expansion_sum_zeroelim( finnow,temp16a)
                finnow,finother = finother,finnow
        
            temp32a = scale_expansion_zeroelim( bxtcat, bdxtail)
            bxtcatt = scale_expansion_zeroelim( catt, bdxtail)
        
            temp16a = scale_expansion_zeroelim( bxtcatt, 2.0 * bdx)
        
            temp16b = scale_expansion_zeroelim( bxtcatt, bdxtail)
            temp32b = fast_expansion_sum_zeroelim( temp16a, temp16b)
            temp64 = fast_expansion_sum_zeroelim( temp32a, temp32b)
            finother = fast_expansion_sum_zeroelim( finnow,temp64)
            finnow,finother = finother,finnow

        if bdytail != 0.0:
            temp16a = scale_expansion_zeroelim( bytca, bdytail)
            bytcat = scale_expansion_zeroelim( cat, bdytail)
            
            temp32a = scale_expansion_zeroelim( bytcat, 2.0 * bdy)
            temp48 = fast_expansion_sum_zeroelim( temp16a, temp32a)
            finother = fast_expansion_sum_zeroelim( finnow,temp48)
            finnow,finother = finother,finnow
        
            temp32a = scale_expansion_zeroelim( bytcat, bdytail)
            bytcatt = scale_expansion_zeroelim( catt, bdytail)
        
            temp16a = scale_expansion_zeroelim( bytcatt, 2.0 * bdy)
        
            temp16b = scale_expansion_zeroelim( bytcatt, bdytail)
            temp32b = fast_expansion_sum_zeroelim( temp16a, temp16b)
            temp64 = fast_expansion_sum_zeroelim( temp32a, temp32b)
            finother = fast_expansion_sum_zeroelim( finnow,temp64)
            finnow,finother = finother,finnow

    ###
    if (cdxtail != 0.0) or (cdytail != 0.0):
        if (adxtail != 0.0) or (adytail != 0.0) or (bdxtail != 0.0) or (bdytail != 0.0):
            ti1, ti0 = Two_Product(adxtail, bdy)
            tj1, tj0 = Two_Product(adx, bdytail)
            u[3], u[2], u[1], u[0] = Two_Two_Sum(ti1, ti0, tj1, tj0)
            negate = -ady
            ti1, ti0 = Two_Product(bdxtail, negate)
            negate = -adytail
            tj1, tj0 = Two_Product(bdx, negate)
            v[3], v[2], v[1], v[0] = Two_Two_Sum(ti1, ti0, tj1, tj0)

            abt = fast_expansion_sum_zeroelim(u, v)
      
            ti1, ti0 = Two_Product(adxtail, bdytail)
            tj1, tj0 = Two_Product(bdxtail, adytail)
            abtt[3], abtt[2], abtt[1], abtt[0] = Two_Two_Diff(ti1, ti0, tj1, tj0)
        else:
            abt = [0.0]
            abtt = [0.0] 

        if cdxtail != 0.0:
            temp16a = scale_expansion_zeroelim( cxtab, cdxtail)
            cxtabt = scale_expansion_zeroelim( abt, cdxtail)
  
            temp32a = scale_expansion_zeroelim( cxtabt, 2.0 * cdx)
            temp48 = fast_expansion_sum_zeroelim( temp16a, temp32a)
            finother = fast_expansion_sum_zeroelim( finnow,temp48)
            finnow,finother = finother,finnow
            if adytail != 0.0:
                temp8 = scale_expansion_zeroelim( bb, cdxtail)
                temp16a = scale_expansion_zeroelim( temp8, adytail)
                finother = fast_expansion_sum_zeroelim( finnow,temp16a)
                finnow,finother = finother,finnow

            if bdytail != 0.0:
                temp8 = scale_expansion_zeroelim( aa, -cdxtail)
                temp16a = scale_expansion_zeroelim( temp8, bdytail)
                finother = fast_expansion_sum_zeroelim( finnow,temp16a)
                finnow,finother = finother,finnow
  
            temp32a = scale_expansion_zeroelim( cxtabt, cdxtail)
            cxtabtt = scale_expansion_zeroelim( abtt, cdxtail)
  
            temp16a = scale_expansion_zeroelim( cxtabtt, 2.0 * cdx)
  
            temp16b = scale_expansion_zeroelim( cxtabtt, cdxtail)
            temp32b = fast_expansion_sum_zeroelim( temp16a, temp16b)
            temp64 = fast_expansion_sum_zeroelim( temp32a, temp32b)
            finother = fast_expansion_sum_zeroelim( finnow,temp64)
            finnow,finother = finother,finnow

        if cdytail != 0.0:
            temp16a = scale_expansion_zeroelim( cytab, cdytail)
            cytabt = scale_expansion_zeroelim( abt, cdytail)
  
            temp32a = scale_expansion_zeroelim( cytabt, 2.0 * cdy)
            temp48 = fast_expansion_sum_zeroelim( temp16a, temp32a)
            finother = fast_expansion_sum_zeroelim( finnow,temp48)
            finnow,finother = finother,finnow
  
            temp32a = scale_expansion_zeroelim( cytabt, cdytail)
            cytabtt = scale_expansion_zeroelim( abtt, cdytail)
  
            temp16a = scale_expansion_zeroelim( cytabtt, 2.0 * cdy)
  
            temp16b = scale_expansion_zeroelim( cytabtt, cdytail)
            temp32b = fast_expansion_sum_zeroelim( temp16a, temp16b)
            temp64 = fast_expansion_sum_zeroelim( temp32a, temp32b)
            finother = fast_expansion_sum_zeroelim( finnow, temp64)
            finnow,finother = finother,finnow
  
    return finnow[-1]


def incircle(pa, pb, pc, pd):
    adx = pa[0] - pd[0]
    bdx = pb[0] - pd[0]
    cdx = pc[0] - pd[0]
    ady = pa[1] - pd[1]
    bdy = pb[1] - pd[1]
    cdy = pc[1] - pd[1]
  
    bdxcdy = bdx * cdy
    cdxbdy = cdx * bdy
    alift = adx * adx + ady * ady
  
    cdxady = cdx * ady
    adxcdy = adx * cdy
    blift = bdx * bdx + bdy * bdy
  
    adxbdy = adx * bdy
    bdxady = bdx * ady
    clift = cdx * cdx + cdy * cdy
  
    det = alift * (bdxcdy - cdxbdy) \
        + blift * (cdxady - adxcdy) \
        + clift * (adxbdy - bdxady)
  
    permanent = (Absolute(bdxcdy) + Absolute(cdxbdy)) * alift \
              + (Absolute(cdxady) + Absolute(adxcdy)) * blift \
              + (Absolute(adxbdy) + Absolute(bdxady)) * clift
    
    errbound = iccerrboundA * permanent
    if (det > errbound) or  (-det > errbound):
        return det
  
    return incircleadapt(pa, pb, pc, pd, permanent)

def orientation(a,b,c):
    """ maybe there are faster ways when all we care about is 
    yes no, zero.
    """
    # float cast seems extraneous, but otherwise these are numpy
    # floats, which make numpy bools, and the cmp hack will return
    # a numpy bool, where python floats will return an integer
    ccw=float(counterclockwise(a,b,c))
    # hack for missing cmp in python3
    return (ccw>0)-(ccw<0)


if __name__ == '__main__':
    ## Some testing:

    print("incircle, for cocircular:",incircle([0,0],[1,0],[1,1],[0,1]))


    print("incircle, for inside:",incircle([0,0],[1,0],[1,1],[1e-30,1]))

    print("incircle, for outside:",incircle([0,0],[1,0],[1,1],[-1e-30,1]))
    # simple test
    # h = scale_expansion_zeroelim([pi],pi)
    
    # seems to work, even when it has to go to full precision.
    # print counterclockwise(pa=[-pi*1e20,-pi*1e20], pb=[0,1e-5], pc=[pi*1e20,pi*1e20])
