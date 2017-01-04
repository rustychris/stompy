c
c**********************************************************************
        PROGRAM  GSWOX
C**********************************************************************
C
c       winds on critical streamline surfaces (wocss/wox) program finds
c       topographically induced winds by making the originally analyzed
c       winds nondivergent within flow surfaces that are defined using 
c       a concept analagous to the critical streamline when the 
c       atmosphere is stably stratified. flow surfaces can intersect 
c       the terrain resulting in zero winds in these 'underground 
c       cells.'  adjustment toward nondivergence in bal5 causes flow 
c       around the obstacles.  geosig reads in wind soundings and 
c       surface data.  direct vector alterations are used in subroutine
c       bal5. wind components at specified points can be held constant
c       or adjusted at a fraction of the adjustments in other cells in 
c       bal5.  the model has been described in:
c     
c       Ludwig, F. L., J. M. Livingston, and R. M. Endlich, 1991: 
c        'Use of Mass Conservation and Dividing Streamline Concepts 
c        for Efficient Objective Analysis of Winds in Complex Terrain,' 
c                  J. Appl. Meteorol., Vol. 30, pp. 1490-1499.
c
C       this version has removed several options found in some earlier 
c       versions.  only one grid is used, no nesting.  potential 
c       temperature lapse rates are not estimated.  at least one 
c       sounding is required; the geostrophic wind provision is no 
c       longer available.  the low altitude grid points to be used
c       for defining critical streamline winds are now input.  
c
c       new provisions include a parameter input file so that 
c       adjustment factors, flow surface compression and other factors 
c       can be changed without recompiling.  the number of labeled 
c       commons has also been substatially  reduced  through 
c       consolidation.  
c
c       April 1997   
c                    F. LUDWIG
c                    Environmental Fluid Mechanics Lab
c                    Dept. of Civil Engineering
c                    Stanford University
c                    Stanford, CA 94305-4020
c
c	based in large part on earlier work with:
c		R. ENDLICH, A. BECKER, D. SINTON, K. NITZ, 
c                     J. LIVINGSTON, B. MORLEY and C. BHUMRALKAR
c
c  more changes 12/2005 by fludwig
c
c  this version further modified to remove debugg flags and provide 
c  only those outputs used on the US Geological Survey website:
c             http://sfports.wr.usgs.gov/cgi-bin/wind/windbin.cgi
c   
c	fludwig, 7/2005
c
C**********************************************************************
c
c  ADJMAX       the fraction of the usual iterative adjustment toward 
c               nondivergence that is made at grid points near 
c	        observations in subroutine bal5.
c  AVTHK        height agl of top surface over lowest terrain.
c  CMPRES       maximum 'compression' of surfaces -- 0 means that lower
c               sfcs must parallel those above, 1 means that the low
c               sfc can touch the next level -- defines influence of 
c               upper stable layers on less stable lower ones.
c  D2MIN        the minimum distance allowed in the inverse weighting 
c               denominator.
c  dptmin	the minimum allowed potent temp lapse rate (deg/m) --
c               limits instability and hence rise of flow sfcs.
c  DS           grid size in meters 
c  DSCRS        grid size (km)
c  DTWT         distance weight power --weight=1/(distance**DTWT)
c  DZMAX(jt,jz) maximum rise for jzth flow sfc as determined from 
c               t-sonde jt 
c  GRDHI        height msl of highest terrain on grid.
c  GRDLO        height msl of lowest terrain on grid.
c  MDATE        day of month        
c  IDOP(jw)     position in surface observation list of wind sounding
c               jw winds at pt jx,jy
c  ITMP(jt)     position in surface observation list of temperature 
c    	        sounding jt
c  KGRIDX       x index of reference point           
c  KGRIDY       y index of reference point
c  LOWIX(jl,jt) item jl (of 5) in the list x indices of lowest pts
c               around t-sonde jt
c  LOWIY(jl,jt) item jl (of 5) in the list y indices of lowest pts 
c	        around t-sonde jt
c  NCOL         number of columns (x index) in grid             
c  NCOLM1       NCOL-1
c  NLVL         number of flow surfaces
c  NROW         number of rows (y index) in grid
c  NROWM1       NROW-1
c  NSITES       maximum number of sites (including upper air).
c  NSNDHT       maximum number sounding (wind or temp) levels.
c  NTSITE       maximum number t-sondes to be used.
c  NUMDOP       number of upper wind sites
c  NUMNWS       number of surface wind sites
c  NUMTMP       number of temeprature sounding sites
c  NUMTOT       total number of observing wind sites
c  NWSITE       maximum number wind soundings to be used.
c  NXGRD        dimension for grids in W-E direction.
c  NYGRD        dimension for grids in S-N direction.
c  NZGRD        dimension for flow surfaces.
c  PWR          power to which separation is raised for inverse 
c	        distance wts.
c  RHS(jx,jy,jz) height (m) above terrain of jzth flow sfc above
c	         pt jx,jy
c  RHSLO(jt,jz) height (m) of jzth flow sfc above the lowest terrain 
c	        near t-sonde jt  
c  SFCHT(jx,jy) terrain height (m msl) at pt jx,jy
c  SFCLOW       terrain height (m msl) of lowest pt in the domain
c  SIGMA(jz)    fraction (over lowest terrain) of the height of the 
c	        top surface for sfc jz
c  SLFAC        controls the degree to which the 1st guess surfaces
c               follow the terrain -- 0 gives flat sfcs & 1 gives 
c	        terrain-following sfcs &
c  SPDCNV,HTCNV conversion factor -- input units to m/s & heights to m
c  TDSI         1.0/(2.0*DS) 
c  U(jx,jy,jz)  westerly component (m/s) at (jx,jy,jz)
c  UCOMP(job)   observed westerly component (m/s) at site job
c  USIG(jws,jz) v component at flow sfc jz on wind sounding jws
c  UTMAPX       utm easting coord (km) of ref. pt (KGRIDX,KGRIDY).
c  UTMAPY       utm northing coord (km) of ref. pt (KGRIDX,KGRIDY).
c  V(jx,jy,jz)  southerly component (m/s) at (jx,jy,jz)
c  VCOMP(job)   observed southerly component (m/s) at site job
c  VSIG(jws,jz) v component at flow sfc jz on wind sounding jws
c  XG(jsite)    coordinate (grid units) of observation site jsite.
c  XORIG        utm easting coordinate (km) of the grid origin. 
c  YG(jsite)    coordinate (grid units) of observation site jsite.
c  YORIG        utm northing coordinate (km) of the grid origin.
c                       (NOTE: xorig,yorig are at 1,1)
c  ZRISE        difference between sfchi -sfclow.
c  ZZERO        roughness length (m) for log profile estimates.
c	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
c
	logical JGOOD(NSITES)
C
	integer nread,iuplot(NXGRD),ivplot(NXGRD)     
c
	common /buggsy/ nskip
        COMMON /ANCHOR/ SLAT,SLNG,UTMAPX,UTMAPY,MDATE,
     $                  IMO,IHOUR,NEND,XORIG,YORIG
        COMMON /compnt/ U(NXGRD,NYGRD,NZGRD),V(NXGRD,NYGRD,NZGRD),z0,
     $        USIG(NSITES,NZGRD),VSIG(NSITES,NZGRD),UCOMP(NSITES),
     $        VCOMP(NSITES),IUGRAF(NXGRD,NYGRD,NHORIZ),Z10,zzero,
     $	      IVGRAF(NXGRD,NYGRD,NHORIZ),SPDMET(NXGRD,NYGRD,NZGRD),
     $        DIRMET(NXGRD,NYGRD,NZGRD),SPDCNV
	COMMON /CSFC/ SFCHT(NXGRD,NYGRD),SIGMA(NZGRD),
     $                SFCLOW,SFCHI,ZRISE
        COMMON /CVOS/ RCM,RMF,IV,DSCRS,kgridx,kgridy,D2MIN,
     $	              HT2DIS,ADJMAX,dtwt
	COMMON /FLOWER/ GRDHI,GRDLO,DZMAX(NTSITE,NZGRD),
     $                  RHS(NXGRD,NYGRD,NZGRD),AVTHK,SLFAC,
     $                  RHSLO(NTSITE,NZGRD),CMPRES,DPTMIN,NFLAT,
     $                  ZCHOOZ(NZGRD),levbot(NXGRD,NYGRD)
	COMMON /LIMITS/ NCOL,NROW,NLVL,NCOLM1,NROWM1,
     $                  LOWIX(5,ntsite),LOWIY(5,ntsite)
	COMMON /NUMOBS/ NUMDOP,NUMNWS,NUMTOT,NUMTMP
	COMMON /PARMS/ ZTOP,DS,NLVLM1,TDSI,TYM	 
	COMMON /STALOC/ XG(NSITES),YG(NSITES),ZG(NSITES),
     $                  WTDOP(NXGRD,NYGRD,NWSITE),JJJ,IDOP(NWSITE),
     $                  ITMP(NTSITE),WTEMP(NXGRD,NYGRD,NTSITE),JGOOD
C
C  SET INITIAL LOW HTS FOR DIFFERENT TSONDE/LEVEL COMBOS & MAXIMUM
C  COMPRESSION OF SEPARATION BETWEEN FLOW SURFACES
C
	do 18 isite=1,NTSITE
	   do 17 iz=1,NZGRD
	      RHSLO(isite,iz)=0.0
17	   continue
18	continue
c
c    Open input parameter file
c
        call runrd (nit)
	nread=0
c
	write (16,*) 'starting'
	write (*,*) 'starting'
c
C    OPEN FILE 11 (TOPOGRAPHY) & READ TERRAIN HEIGHTS 
C
	OPEN(11,FILE='terrain.dat',STATUS='OLD',
     $                              form='formatted')
	write (*,*)  'opened terrain.dat'
	CALL TOPO
        DS=DSCRS*1.0E3
        NCOLM1=NCOL-1
        NROWM1=NROW-1
        NLVLM1=NLVL-1
        TDSI=1./(2.0*DS)
c
	close (11)
	write (16,*) 'closed terrain.dat'
	write (16,*) 'opening SFBfiles'
c
44	continue
c
	 open (12,file='winds.dat',status='old',form='formatted')
c
	 write (16,*) 'opened winds.dat'
c
C  READ & ANALYZE WIND DATA USING WXANAL MAKE INITIAL WIND ANALYSIS
C
	read (12,6001,end=8686) ihour,imo,mdate,iyear
c
	nread=nread+1
	if (nread .gt. nend)  go to 8586
c
	khr=ihour
	kmon=imo
	kday=mdate
	lastmo=imo
c
c  geosig reads meteorological inputs & calls routines for
c  interpolation and defining surface shapes. also write files
c  for plotting and querying observations.
c
        CALL geosig
C
C CALL SUBROUTINE TO MAKE WINDS NONDIVERGENT
C
        CALL BAL5(NIT)
C
C  INTERPOLATE reduced divergence winds TO ANEMOMETER HEIGHT
C
	call levwnd
c
c  at this point we have hts of flow surfaces at each grid
c  point and wind speed and direction on at 10m above sfc
c  so we write ascii output files
c
	open(46,file='windsuv.out',
     $	         form='formatted', status='unknown')
C
	write (16,*) 'ucomps', NFLAT,NROW,NCOL,NSKIP
c
	   do 75 IY=NROW,1,-NSKIP
c	
c  CONVERT from meters/second to {deci}KNOTS and check to see if 
c  components are too large for input to COMET.  write u components
c  first then v. units of iuplot & ivplot are deciknots.
c
	      do 72 ix=1,ncol
	         iuplot(ix)=nint(19.4*u(ix,iy,1))
	         if (iuplot(ix).gt.582) iuplot(ix)=582
72	      continue
	      write(46,6008) (iuplot(ix),IX=1,NCOL,NSKIP)
75	   continue
c
c  now do v compnents
c
	   do 79 IY=NROW,1,-NSKIP
	      do 77 ix=1,ncol
	        ivplot(ix)=nint(19.4*v(ix,iy,1))
	        if (ivplot(ix).gt.582) ivplot(ix)=582
77	      continue
	      write(46,6008) (ivplot(ix),IX=1,NCOL,NSKIP)
79	   continue
c
	close (46)
	write(*,*) ' finished '
c
	close (12)
	go to 44
8586    write(16,*) 'reached maximum number specified ', nread-1
8686	continue
c
	write (16,*) 'closing file '
	write (*,*) 'closing file'
c
	if (nread .gt. nend)  go to 8786
c
6001	format (4i2,1x,i4)
6002	format (a16)		
6003	format (a17)
6004	format (1x,45f7.1)
6006	format (1x,i2,'00z ',i2,'/',i2,'/96 completed ','   'a8)
6008	format  (1X,150I5)
6009	format (a1)
8786	continue
c		
	write (16,*) 'finished '		
	write (*,*) 'finished '
	pause
c
	STOP
c
	END
c
c*********************************************************************
        subroutine runrd(niter)
c*********************************************************************
c
c  allen becker added this to read values of most parameters from file 
c  'rundat.dat'.  This replaces the old block data file.  
c   
c     fludwig revised 5/97
c
c	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
c
	logical JGOOD(NSITES)     
c
	common /buggsy/ nskip
        COMMON /ANCHOR/ SLAT,SLNG,UTMAPX,UTMAPY,MDATE,
     $                  IMO,IHOUR,NEND,XORIG,YORIG
        COMMON /compnt/ U(NXGRD,NYGRD,NZGRD),V(NXGRD,NYGRD,NZGRD),z0,
     $        USIG(NSITES,NZGRD),VSIG(NSITES,NZGRD),UCOMP(NSITES),
     $        VCOMP(NSITES),IUGRAF(NXGRD,NYGRD,NHORIZ),Z10,zzero,
     $	      IVGRAF(NXGRD,NYGRD,NHORIZ),SPDMET(NXGRD,NYGRD,NZGRD),
     $        DIRMET(NXGRD,NYGRD,NZGRD),SPDCNV
        COMMON/CSFC/ SFCHT(NXGRD,NYGRD),SIGMA(NZGRD),
     $               SFCLOW,SFCHI,ZRISE
        COMMON /CVOS/ RCM,RMF,IV,DSCRS,kgridx,kgridy,D2MIN,
     $	              HT2DIS,ADJMAX,dtwt
        COMMON /FLOWER/ GRDHI,GRDLO,DZMAX(NTSITE,NZGRD),
     $                  RHS(NXGRD,NYGRD,NZGRD),AVTHK,SLFAC,
     $                  RHSLO(NTSITE,NZGRD),CMPRES,DPTMIN,NFLAT,
     $                  ZCHOOZ(NZGRD),levbot(NXGRD,NYGRD)
        COMMON /LIMITS/ NCOL,NROW,NLVL,NCOLM1,NROWM1,
     $                  LOWIX(5,ntsite),LOWIY(5,ntsite)
        COMMON /NUMOBS/ NUMDOP,NUMNWS,NUMTOT,NUMTMP
	COMMON/PARMS/ ZTOP,DS,NLVLM1,TDSI,TYM	 
	COMMON /STALOC/ XG(NSITES),YG(NSITES),ZG(NSITES),
     $                  WTDOP(NXGRD,NYGRD,NWSITE),JJJ,IDOP(NWSITE),
     $                  ITMP(NTSITE),WTEMP(NXGRD,NYGRD,NTSITE),JGOOD
c
	save
c
c  open run descriptors.
c
	write (*,*) 'opening rundat.dat'
        open (22,file='rundat.dat',status='old',form='formatted')
        read (22,*)
	open(16,file='outfyl',form='formatted', status='unknown')
c
c  read grid pt interval for ascii output -- 1 is all
c  grid pts, 2, every other point etc.
c
        read (22,*) nskip
c
c  read number of vertical flow surfaces 
c
        read (22,*) nlvl
	if (nlvl.gt.NZGRD .or. nlvl .lt. 1) then
	   write (*,*) 'bad number of levels', nlvl, NZGRD
	   pause
	   stop
	end if
c
c  read flow levels (fraction of top level)
c
        read(22,*) (sigma(k),k=1,nlvl)
	write(16,*) nlvl,' flow levels at:'
	write(16,6011) (sigma(k),k=1,nlvl)
6011	format (1x, 'sigmas: ',20f8.4)
c
c  read number of horizontal surfaces desired
c
        read (22,*) nflat
	if (nflat.gt.NHORIZ .or. nflat .lt. 1) then
	   write (*,*) 'bad number of levels', nhoriz, nflat
	   pause
	   stop
	end if
c
c  read horizontal levels in m asl
c
        read(22,*) (zchooz(k),k=1,nflat)
	write(16,*) nflat-1,' horizontal levels at:'
	write(16,6006) (zchooz(k),k=1,nflat)
	write(*,6006) (zchooz(m),m=1,nflat)
6006	format (1x, 'flat sfcs: ',20f8.1)
c
c  read max # of stations: sfc, wind & temp.sonde
c
        read(22,*) numnws,numtmp,numdop
	if (numtmp.le.0) then
	   write(*,*) 'wocss needs a temperature sounding to work'
	   stop
	end if
	write(16,*) 'number of upper wind & temp sites= ',numtmp
	write(16,*) 'max total observation sites= ',numnws
c
c  read factor for converting to m/s
c
	read (22,*) spdcnv,htcnv
	write(16,*) 'spd & ht conv. = ',spdcnv,htcnv
c
c  read no. cols,rows for grid
c
        read (22,*) nrow,ncol
	if (nrow .gt. NYGRD .or. ncol .gt. NXGRD) then
	   write (*,*) 'bad x,y dimensions are ',ncol,nrow
	   write (*,*) 'they should not be > ', NXGRD,NYGRD
	   pause
	   stop
	end if
	write(16,*) 'x & y grid dimensions are ',ncol,nrow
c
c  read grid intervals in km
c
        read (22,*) dscrs
	write(16,*) 'grid spacing (km) = ',dscrs
c
c  read utm coordinate of reference point (1,1)--(x,y) and height asl
c
        read (22,*) utmapx,utmapy
	write(16,*)  'anchor point utms = ',utmapx,utmapy
c
c  read anchor point grid indices (ix,jy) 
c
        read (22,*) kgridx,kgridy
	write(16,*) 'anchor point indices = ',kgridx,kgridy
	xorig=utmapx-dscrs*float(kgridx)
	yorig=utmapy-dscrs*float(kgridy)
c
c  constants used to define surface shapes subroutine resig
c  redefines surface shapes using critical streamline methodology.
c
        read (22,*) avthk
	write(16,*) 'ht of top surface over lowest pt = ',avthk
        read (22,*) slfac
	write(16,*)  '1st guess terrain-following factor = ',slfac
        read (22,*) cmpres
	write(16,*) 'sfc compression factor = ',cmpres
        read (22,*) dptmin
	write(16,*) 'minimum allowable potent. temp lapse (deg/m) ',
     $                      dptmin
c
c  	constants used for interpolation -- d2min is the minimum 
c       distance allowed in the inverse weighting denominator.  ht2dis
c       defines the relative import of changes in elevation versus 
c       horizontal distance in the inverse distance weighting.  when 
c       set to zero, vertical terrain effect is not included. roughness
c       length (zzero) and anemometer height (z10)
c
        read (22,*) zzero,z10
	z0=zzero
	zchooz(1)=z10
	write (16,*) 'roughness length & anemometer ht = ',zzero,z10
        read (22,*) d2min
        read (22,*) dtwt
	write (16,*) 'min inverse dist denominator & ',
     $              'relative import z to r= ', d2min,ht2dis
c
c       niter is the upper limit on iterations toward nondivergence in
c	bal5 adjmax is the fraction of the usual iterative adjustment 
c	toward nondivergence that is made at grid points near 
c	observations in subroutine bal5.
c
        read (22,*) niter
	write(16,*)  'number of iterations in bal5 = ',niter
        read (22,*) adjmax
	write(16,*)   'max adjustment near obs = ',adjmax
c
c  read indices of low points to be used in critical streamline 
c  calculations for each temperature sounding.
c
	do 101 isite=1,numtmp
	   do 97 lows=1,5 
	      read (22,*) lowix(lows,isite),lowiy(lows,isite)
97	   continue
	   write(16,*)  't-sonde site no. ',isite, ' low pts.: '
	   do 98 lo=1,5  
	       write(16,*) lowix(lo,isite),lowiy(lo,isite)
98	   continue
101	continue
        read (22,*) nend
	write(16,*) 'quit after ',nend,' cases.'
c
	return
c
        end
C
C*********************************************************************
        SUBROUTINE LGNTRP(Y,X0,X1,X,Y0,Y1)
C*********************************************************************
C
C DOES LOG-LINEAR INTERPOLATION OF Y VS. LOG X.
C
        IF (X1.EQ.X0 .or. x.eq.x0) THEN
		   y=y0   
        ELSE if (x.eq.x1) then
		   y=y1
        ELSE
           RATIO = ALOG10(X/X0)/ALOG10(X1/X0)
           Y = Y0 + RATIO * (Y1-Y0)
        END IF
C
        RETURN
        END
C
c**********************************************************************
        REAL FUNCTION WNDWT(X,Y,XOBS,YOBS)
c**********************************************************************
C   this version determines the squared distance between
c   the 2 pts (X,Y) & (XOBS,YOBS).  
c
c		f. l. ludwig,  10/97
C	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
c
        COMMON /CSFC/ SFCHT(NXGRD,NYGRD),SIGMA(NZGRD),
     $                SFCLOW,SFCHI,ZRISE
        COMMON /CVOS/ RCM,RMF,IV,DSCRS,kgridx,kgridy,D2MIN,
     $	              HT2DIS,ADJMAX,dtwt
c
	xdif=xobs-x
	ydif=yobs-y
        dist=sqrt((xdif)**2 + (ydif)**2)
c
        IF (dist.LT.D2MIN) dist=D2MIN
c
        WNDWT=1.0/(dist**dtwt)
c
        RETURN
        END
c
c*********************************************************************
        SUBROUTINE SETLOG(LOGVAL,LARRAY,NUM1,NUM2)
c*********************************************************************
c
C       INITIALIZES ALL ELEMENTS OF a logical ARRAY TO LOGVAL (THIS 
C       IS IDENTICAL TO 'SETMAT,' EXCEPT WITH LOGICAL ARGUMENTS)
C       REVISED 11/87
C	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
c
        logical LARRAY(NXGRD,NYGRD),LOGVAL
c
        DO 10 I=1,NUM1
        DO 10 J=1,NUM2
           LARRAY(I,J)=LOGVAL
   10   CONTINUE
        RETURN
        END
c
c*********************************************************************
        SUBROUTINE SETINT(IVALUE,IARRAY,NUM1,NUM2)
c*********************************************************************
c
C       INITIALIZES ALL ELEMENTS OF ARRAY TO VALUE. (THIS SUBPROGRAM
C       IS IDENTICAL TO 'SETMAT,' EXCEPT WITH INTEGER ARGUMENTS)
C       REVISED 11/87
C
c	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
c
        integer IARRAY(NXGRD,NYGRD)
C
        DO 10 I=1,NUM1
        DO 10 J=1,NUM2
           IARRAY(I,J)=IVALUE
   10   CONTINUE
        RETURN
        END
C
C*******************************************************************
        SUBROUTINE SETMAT(VALUE,ARRAY,NUM1,NUM2)
c*******************************************************************
C
C       INITIALIZES ALL ELEMENTS OF ARRAY TO VALUE.(THIS SUBPROGRAM IS
C       IDETICAL TO 'SETINT,' EXCEPT WITH REAL ARGUMENTS.
C       REVISED 11/87
C
c	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
c
        REAL ARRAY(NXGRD,NYGRD)
c
        DO 10 I=1,NUM1
        DO 10 J=1,NUM2
           ARRAY(I,J)=VALUE
   10   CONTINUE
        RETURN
        END
c
c**********************************************************************
          SUBROUTINE BAL5(NITER)
c**********************************************************************
C
C  THIS IS A MODIFIED VERSION OF ROY ENDLICH'S OCTOBER 1984 CODE. THE
C  OPTIONS FOR VORTICITY CONSERVATION & NONZERO DIVERGENCE HAVE BEEN
C  REMOVED. THE CODE WAS ALSO MODIFIED TO USE MORE
C  IF-THEN-ELSE STRUCTURE BY LUDWIG DECEMBER 1987.
C
C  THIS ROUTINE BALANCES DIVERGENCE TOWARD ZERO. DIV IS SCALED TO UNITS
C  OF 10**-6/SECOND. THE METHOD USES DIRECT VECTOR ALTERATIONS.
C  THIS FORM IS FOR A SQUARE GRID AND OMITS TRIGONOMETRIC FUNCTIONS.
C  THE FLUX FORMULATION IS USED SO THAT WIND COMPONENTS ARE WEIGHTED BY
C  THE THICKNESS OF THE LAYER WHEN THE FINITE DIFFERENCE SCHEME IS USED
C  TO REDUCE THE DIVERGENCE. INDICES IN ARRAYS (I,J,K) ARE I=COLUMN,
C  J=ROW, K=LEVEL; PT (1,1,1) IS SW CORNER AT GROUND. FOR COMPUTATION
C  BOXES, INDICES REFER TO SW CORNER OF BOX.
C______________________________________________________________________
c
c  Modified march 1996 so that below ground and near-station points are 
c  treated separately.  adjustments at grid points near observation
c  sites are adjusted by less (the factor adjmax) than other points. 
c  below ground are left unadjusted as before.
c
c	f. l. ludwig 3/96
C______________________________________________________________________
C
C  modififed so the adjustments over overridden at the end of each
c  iteration all points are adjusted, then those near observations
c  or surrounded by 3 or 4 subsurface point are set back to original,e
c  or near-original values befor starting next iteration.  
c
c	fludwig 3/2000
C______________________________________________________________________
c	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
c
	logical JGOOD(NSITES)
       	LOGICAL doadj,IFXPT(NXGRD,NYGRD),isdop
c
        REAL DI(NXGRD,NYGRD),U1(NXGRD,NYGRD),V1(NXGRD,NYGRD)
	REAL UN(NXGRD,NYGRD),VN(NXGRD,NYGRD),THK(NXGRD,NYGRD)
	REAL Ustart(NXGRD,NYGRD),Vstart(NXGRD,NYGRD)     
c
        COMMON /compnt/ U(NXGRD,NYGRD,NZGRD),V(NXGRD,NYGRD,NZGRD),z0,
     $        USIG(NSITES,NZGRD),VSIG(NSITES,NZGRD),UCOMP(NSITES),
     $        VCOMP(NSITES),IUGRAF(NXGRD,NYGRD,NHORIZ),Z10,zzero,
     $	      IVGRAF(NXGRD,NYGRD,NHORIZ),SPDMET(NXGRD,NYGRD,NZGRD),
     $        DIRMET(NXGRD,NYGRD,NZGRD),SPDCNV
        COMMON /CVOS/ RCM,RMF,IV,DSCRS,kgridx,kgridy,D2MIN,
     $	              HT2DIS,ADJMAX,dtwt
        COMMON /FLOWER/ GRDHI,GRDLO,DZMAX(NTSITE,NZGRD),
     $                  RHS(NXGRD,NYGRD,NZGRD),AVTHK,SLFAC,
     $                  RHSLO(NTSITE,NZGRD),CMPRES,DPTMIN,NFLAT,
     $                  ZCHOOZ(NZGRD),levbot(NXGRD,NYGRD)
        COMMON /LIMITS/ NCOL,NROW,NLVL,NCOLM1,NROWM1,
     $                  LOWIX(5,ntsite),LOWIY(5,ntsite)
	COMMON/PARMS/ ZTOP,DS,NLVLM1,TDSI,TYM	 
	COMMON /STALOC/ XG(NSITES),YG(NSITES),ZG(NSITES),
     $                  WTDOP(NXGRD,NYGRD,NWSITE),JJJ,IDOP(NWSITE),
     $                  ITMP(NTSITE),WTEMP(NXGRD,NYGRD,NTSITE),JGOOD
c
	save
c
        DATA doadj, ENFRAC /.FALSE.,0.5/
c
        GS=DS*1.0E-05
C
C  USE GRID SPACING IN 100'S OF KM.  DS IS IN M. FOR PROPER SCALING.
C
        GSI=10.0/GS
c
        DO 800 L=2,NLVL
C
c  get components for this level
c
	  DO 35 J=1,NROW
             DO 33 I=1,NCOL
               UN(I,J)=U(I,J,L)
               VN(I,J)=V(I,J,L)
33	     continue
35         CONTINUE
C
C  IDENTIFY PTS NOT TO BE CHANGED
C
           CALL FIXWND (IFXPT,L)
C
           CALL SETMAT(0.0,DI,NCOL,NROW)
C
C COMPUTE LAYER THICKNESS AND MULTIPLY WIND COMPONENTS
C
           DO 40 J=1,NROW
           DO 40 I=1,NCOL
                LA=L +1
                IF (LA.GT.NLVL) LA=NLVL
                HTA=RHS(I,J,LA)
                LB=L-1
                IF (LB.LT.1) LB=1
                HTB=RHS(I,J,LB)
                IF (HTB.LT.-1.0) HTB=-1.0
                THK(I,J)=0.5*(HTA-HTB)*0.01
C
C  FOR NEG (OR VERY SMALL) RHS.
C
                IF (THK(I,J).LE.0.01) THK(I,J)=0.01
C
C  UNITS OF THICKNESS ARE HUNDREDS OF M FOR CONVENIENCE. 
C  SET INITIAL WINDS BEFORE ALTERATIONS.
C
                U1(I,J)=UN(I,J)
                V1(I,J)=VN(I,J)
C
C  WEIGHT WINDS WITH THICKNESS OF LAYER.
C
                UN(I,J)=U1(I,J)*THK(I,J)
                VN(I,J)=V1(I,J)*THK(I,J)
 40       CONTINUE
C
C  COMPUTE DIVERGENCE (DI=DUE+DVN) FROM FLUX DIFFERENCES BETWEEN
C  V COMPONENTS AT NORTH (VNO) & SOUTH (VSO) SIDES OF BOX & U
C  COMPONENTS AT EAST (UE) & WEST (UW) SIDES OF BOX. DDIJ IS ZERO
C  TO PRODUCE NONDIVERGENT WINDS; OTHER VALUES COULD BE USED TO
C  ACCOMODATE AN AREA-WIDE GENERAL DIVERGENCE. RA=RELAXATION FACTOR.
C
	   DDIJ=0.0
           RA=0.7
           DO 240 LG=1,NITER
c
c  save values at start of iteration so the values at 
c  near-oservation and other fixed points can be changed back at 
c  end of ech iterative step -- added 3/2000 by fludwig
c
	      do 70 j=1,NROW
	         do 60 i=1,NCOL
		    ustart(i,j)=UN(i,j)
		    vstart(i,j)=VN(i,j)
60		continue
70	     continue
c
             DO 150 J=1,NROWM1
                DO 140 I=1,NCOLM1
                   UE=0.5*(UN(I+1,J)+UN(I+1,J+1))
                   UW=0.5*(UN(I,J)  +UN(I,J+1))
                   VSO=0.5*(VN(I+1,J)+VN(I,J))
                   VNO=0.5*(VN(I,J+1)+VN(I+1,J+1))
                   DUE=GSI*(UE-UW)
                   DVN=GSI*(VNO-VSO)
                   DI(I,J)=DUE+DVN
                   CUIJ=0.05*GS*(DDIJ-DI(I,J))*RA
                   CVIJ=0.05*GS*(DDIJ-DI(I,J))*RA
C
C  LIMIT CHANGES TO LESS THAN 1 FOR NUMERICAL STABILITY.
C
                   IF (CUIJ .LT.-1.0) CUIJ=-1.0
                   IF (CUIJ .GT. 1.0) CUIJ=1.0
                   IF (CVIJ .LT.-1.0) CVIJ=-1.0
                   IF (CVIJ .GT. 1.0) CVIJ=1.0
C
                   UN(I+1,J)=UN(I+1,J)+CUIJ
                   UN(I+1,J+1)=UN(I+1,J+1) +CUIJ
                   UN(I,J)=UN(I,J) -CUIJ
                   UN(I,J+1)=UN(I,J+1) -CUIJ
                   VN(I+1,J)=VN(I+1,J)-CVIJ
                   VN(I,J)=VN(I,J)-CVIJ
                   VN(I,J+1)=VN(I,J+1)+CVIJ
                   VN(I+1,J+1)=VN(I+1,J+1)+CVIJ
140		continue
C
150          CONTINUE
c
c  go back and substitute original values, corrected for maximum
c  specified adjustment at fixed points.  insert zeros for subsurface 
c  points.  increase adjustments for higher levels.
c
c     added by fludwig 3/2000
c
	     do 170 j=1,NROW
	        do 160 i=1,NCOL
c
		   if (ifxpt(i,j)) then
c
c  for upper wind site use minimum adjustment at all levels
c
		        isdop=.false.
			do 153 jd=1,NUMDOP
			   if (i.eq. nint(xg(idop(jd))) .and. 
     $                      j.eq. nint(yg(idop(jd)))) isdop=.true.
153			continue
c
			if (L.eq.levbot(i,j) .or. isdop) then
				         UN(i,j)=ustart(i,j)+
     $                     adjmax*(UN(i,j)-ustart(i,j))
				         VN(i,j)=vstart(i,j)+
     $                     adjmax*(UN(i,j)-ustart(i,j))
c
			else if (L.gt.levbot(i,j)) then
c
c  increase adjustment allowed with height at other fixed points
c
			   adjles=adjmax+(1.0-adjmax)*
     $                     (rhs(i,j,L)-rhs(i,j,levbot(i,j)))/
     $                     (rhs(i,j,NLVL)-rhs(i,j,levbot(i,j)))
c
			   UN(i,j)=ustart(i,j)+
     $                         adjles*(UN(i,j)-ustart(i,j))
				          VN(i,j)=vstart(i,j)+
     $                         adjles*(UN(i,j)-ustart(i,j))
			end if
		     end if
160		  continue
170	     continue
c
c  end of iteration loop
c
240	CONTINUE
C
        SUM1=0.0
        SUM2=0.0
        Q1=0.0
        DO 350 J=1,NROW
           DO 340 I=1,NCOL
C
C  INCLUDE ONLY POINTS THAT ARE ABOVE THE SURFACE and 
c  are not nearest to an observation site.
C
               IF (RHS(I,J,L).GT.0.0 ) then
                    UN(I,J)=UN(I,J)/THK(I,J)
                    VN(I,J)=VN(I,J)/THK(I,J)
                    U1(I,J)=U1(I,J)-UN(I,J)
                    V1(I,J)=V1(I,J)-VN(I,J)
                    Q1=Q1+1.0
                    SUM1=SUM1+U1(I,J)
                    SUM2=SUM2+V1(I,J)
               END IF
340	    continue
350	continue
c
        SUM1=SUM1/Q1
        SUM2=SUM2/Q1
C
C  NORMALIZE ORIGINAL AVERAGE VALUES FOR ABOVE GROUND POINTS
c  that are not nearest observation sites.
C
         DO 450 J=1,NROW
            DO 445 I=1,NCOL
                IF (RHS(I,J,L) .GT.0.0) THEN
                    UN(I,J)=UN(I,J)+SUM1
                    VN(I,J)=VN(I,J)+SUM2
                ELSE
                    UN(I,J)=0.0
                    VN(I,J)=0.0
                END IF
445	     continue
450	continue
C
C  CHANGE BACK TO 3D ARRAYS 
C
        DO 590 J=1,NROW
           DO 580 I=1,NCOL
               U(I,J,L)=UN(I,J)
               V(I,J,L)=VN(I,J)
580        CONTINUE
590	continue
c
c  end flow level loop
c
800     CONTINUE
C
6001	format (1x, 26f6.1)
c
        RETURN
c
        END
C
C******************************************************************
        SUBROUTINE FIXWND (IFXPT,LVL)
C******************************************************************
C
C  THIS ROUTINE IDENTIFIES PTS NEAR OBSERVATION SITES SO THAT
C  WIND ADJUSTMENTS CAN BE RESTRAINED IN SUBROUTINE BAL5.
C
c___________________________________________________________________
c
c  modified so that all levels above an upper wind site are
c  identified, but only the first above ground level for surface 
c  wind sites.
c
c		f. l. ludwig, 3/96
c___________________________________________________________________
c
c  further modified so that above ground points on a surface are
c  flagged for restrained adjustment if they have 3 or more of the
c  4 surrounding points are below ground.
c
c		fludwig 3/2000
c___________________________________________________________________
C
c	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
c
	logical JGOOD(NSITES)
        LOGICAL IFXPT(NXGRD,NYGRD)     
C
        COMMON /FLOWER/ GRDHI,GRDLO,DZMAX(NTSITE,NZGRD),
     $                  RHS(NXGRD,NYGRD,NZGRD),AVTHK,SLFAC,
     $                  RHSLO(NTSITE,NZGRD),CMPRES,DPTMIN,NFLAT,
     $                  ZCHOOZ(NZGRD),levbot(NXGRD,NYGRD)
        COMMON /NUMOBS/ NUMDOP,NUMNWS,NUMTOT,NUMTMP	 
	COMMON /STALOC/ XG(NSITES),YG(NSITES),ZG(NSITES),
     $                  WTDOP(NXGRD,NYGRD,NWSITE),JJJ,IDOP(NWSITE),
     $                  ITMP(NTSITE),WTEMP(NXGRD,NYGRD,NTSITE),JGOOD
c
	save
c
c  intialize ifxpt values to false
c
	do 22 ix=1,NXGRD
	   do 20 iy=1,NYGRD
	      ifxpt(ix,iy)=.false.
20         continue
22	continue
c
c  check all grid points to see if they should be adjusted
c  if the point itself, or 3 or 4 of the surrounding points 
c  are subterrainian, restrict adjustments, i.e. set ifxpt to .true.  
C
	do 50 ix=2,ncol-1
	   do 48 iy=2,nrow-1
	      if (rhs(ix,iy,lvl).le. 0.0) then
	         ifxpt(ix,iy)=.true.
	      else
	         nsubtr=0
	         do  44 jx=ix-1,ix+1,2
	            do  40 jy=iy-1,iy+1,2
	               if (rhs(jx,jy,lvl).le. 0.0) nsubtr=nsubtr+1 
40	            continue
44	         continue
	         if (nsubtr .ge. 3)  then
	            ifxpt(ix,iy)=.true.
	         else
	            ifxpt(ix,iy)=.false.
	         end if
	      end if
48	   continue
50	continue
		      
c
c  set flag for limited adjustment at points around obs site
c
	DO 200 I=1,NSITES
           if (I .le. numnws) then
c
c  check to see if this site had an observation
c
	      if (JGOOD(I)) then
                 ix=NINT(XG(I))
                 iy=NINT(YG(I))
                 if (ix.gt.0 .and. ix.le.ncol   
     $                 .and. iy.gt.0 .and.    
     $                          iy .le.nrow) then
c
c  check to see if this is first level above surface for 
c  surface observations.  
c
	            if (rhs(ix,iy,lvl) .gt. 0.0) 
     $                               IFXPT(iX,iY)=.true.
	         end if
	      end if
	   end if
c
200	CONTINUE
c
        RETURN
        END
c
c******************************************************************
        SUBROUTINE BETWIN
C******************************************************************
C
C  THIS SUBROUTINE ESTIMATES WINDS BETWEEN THE TOP AND BOTTOM LEVELS.
C  THE DEVIATION OF OBSERVED WINDS FROM A LOG PROFILE IS FIRST
C  DETERMINED.THEN THE DEVIATIONS AT EACH LEVEL ARE INTERPOLATED BY
C  AN INVERSE DISTANCE TO A POWER (PWR) WEIGHTING SCHEME.  THE
C  INTERPOLATED DEVIATIONS ARE USED TO CORRECT  THE CALCULATED LOG 
c  PROFILES AT THE GRID POINTS.
C               --F  LUDWIG  12/87
C	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)

        PARAMETER (PWR=-1.0)
c
	logical JGOOD(NSITES)
        REAL DDOPU(NWSITE,NZGRD),DDOPV(NWSITE,NZGRD),SEPWT(NWSITE)
c
        COMMON /compnt/ U(NXGRD,NYGRD,NZGRD),V(NXGRD,NYGRD,NZGRD),z0,
     $        USIG(NSITES,NZGRD),VSIG(NSITES,NZGRD),UCOMP(NSITES),
     $        VCOMP(NSITES),IUGRAF(NXGRD,NYGRD,NHORIZ),Z10,zzero,
     $	      IVGRAF(NXGRD,NYGRD,NHORIZ),SPDMET(NXGRD,NYGRD,NZGRD),
     $        DIRMET(NXGRD,NYGRD,NZGRD),SPDCNV
        COMMON/CSFC/ SFCHT(NXGRD,NYGRD),SIGMA(NZGRD),
     $               SFCLOW,SFCHI,ZRISE
        COMMON /FLOWER/ GRDHI,GRDLO,DZMAX(NTSITE,NZGRD),
     $                  RHS(NXGRD,NYGRD,NZGRD),AVTHK,SLFAC,
     $                  RHSLO(NTSITE,NZGRD),CMPRES,DPTMIN,NFLAT,
     $                  ZCHOOZ(NZGRD),levbot(NXGRD,NYGRD)
        COMMON /LIMITS/ NCOL,NROW,NLVL,NCOLM1,NROWM1,
     $                     LOWIX(5,ntsite),LOWIY(5,ntsite)
        COMMON /NUMOBS/ NUMDOP,NUMNWS,NUMTOT,NUMTMP
	COMMON /PARMS/  ZTOP,DS,NLVLM1,TDSI,TYM	 
	COMMON /STALOC/ XG(NSITES),YG(NSITES),ZG(NSITES),
     $                  WTDOP(NXGRD,NYGRD,NWSITE),JJJ,IDOP(NWSITE),
     $                  ITMP(NTSITE),WTEMP(NXGRD,NYGRD,NTSITE),JGOOD
C
	save
c
        IF (NLVL .LE. 3) THEN
           write(3, *) ' ONLY ',NLVL-1,' FLOW SURFACES'
           RETURN
        END IF
C
C GET LOG PROFILES AT OBSERVATION POINTS AND DEVIATIONS FROM THEM. Z0=
C ROUGHNESS HT.
C
        IF (NUMDOP .GT. 0) THEN
           DO 50 JDOP = 1,NUMDOP
              IX=MAX(1,NINT(XG(IDOP(JDOP))))
              IX=MIN(NCOL,IX)
              IY=MAX(1,NINT(YG(IDOP(JDOP))))
              IY=MIN(NROW,IY)
              UTOP = USIG(JDOP,NLVL)
              VTOP = VSIG(JDOP,NLVL)
              ZTOP = RHS(IX,IY,NLVL)
              U0=USIG(JDOP,levbot(IX,IY))
              V0=VSIG(JDOP,levbot(IX,IY))
              H0=RHS(IX,IY,levbot(IX,IY))
              DDOPU(JDOP,NLVL)=0.0
              DDOPV(JDOP,NLVL)=0.0
              DO 50 LL = 1,NLVL-1
                 IF(LL .GT. levbot(IX,IY)) THEN
                    ZZ = RHS(IX,IY,LL)
                    CALL LGNTRP(UU,H0,ZTOP,ZZ,U0,UTOP)
                    CALL LGNTRP(VV,H0,ZTOP,ZZ,V0,VTOP)
                    DDOPU(JDOP,LL) = USIG(JDOP,LL)-UU
                    DDOPV(JDOP,LL) = VSIG(JDOP,LL)-VV
                 ELSE
                    DDOPU(JDOP,LL)=0.0
                    DDOPV(JDOP,LL)=0.0
                 END IF
50            CONTINUE
        END IF
C
C  GET LOG PROFILES AT EACH GRID POINT-- FROM 1ST ABOVE-GROUND LEVEL.
C
        DO 100 IX = 1,NCOL
           DO 90 IY = 1,NROW
              UTOP = U(IX,IY,NLVL)
              VTOP = V(IX,IY,NLVL)
              ZTOP = RHS(IX,IY,NLVL)
              H0 = RHS(IX,IY,levbot(IX,IY))
              U0 = U(IX,IY,levbot(IX,IY))
              V0 = V(IX,IY,levbot(IX,IY))
              DO 65 LL=levbot(IX,IY),NLVL-1
                 ZZ=RHS(IX,IY,LL)
                 CALL LGNTRP(U(IX,IY,LL),H0,ZTOP,ZZ,U0,UTOP)
                 CALL LGNTRP(V(IX,IY,LL),H0,ZTOP,ZZ,V0,VTOP)
 65           CONTINUE
C
C IF THERE ARE UPPER WIND OBSERVATIONS DO WEIGHTED INTERPOLATION OF
C CORRECTIONS TO THE LOG PROFILE.
C
              IF (NUMDOP.GT.0) THEN
C
C GET WEIGHTS FOR THIS GRID POINT & EACH WIND SITE USING SEPARATION
C BETWEEN OBS SITE & GRID PT. TO AVOID OVER-CORRECTION THE MINIMUM
C SEPARATION CONSIDERED IS 1 GRID UNIT.
C
                 DO 73 JDOP=1,NUMDOP
                    JT = IDOP(JDOP)
                    SEPR=(XG(JT)-FLOAT(IX))**2
     $                         + (YG(JT)-FLOAT(IY))**2
                    IF(SEPR .LT. 1.0) SEPR=1.0
                    SEPWT(JDOP)=(SEPR**(0.5*PWR))
 73              CONTINUE
                 DO 80 LL=levbot(IX,IY),NLVL-1
                    DU = 0.0
                    DV = 0.0
                    SUM=0.0
                    DO 78 JDOP = 1,NUMDOP
                       SUM=SUM+SEPWT(JDOP)
                       DU = DU + DDOPU(JDOP,LL)*SEPWT(JDOP)
                       DV = DV + DDOPV(JDOP,LL)*SEPWT(JDOP)
 78                 CONTINUE
C
C  CHECK TO AVOID OVERCORRECTING
C
                    IF (SUM .GT. 1.0 ) THEN
                       DU=DU/SUM
                       DV=DV/SUM
                    END IF
C
C GETTING THE ADJUSTED WIND ESTIMATE.
C
                    U(IX,IY,LL) = DU + U(IX,IY,LL)
                    V(IX,IY,LL) = DV + V(IX,IY,LL)
80               CONTINUE
              END IF
 90        CONTINUE
100     CONTINUE
C
        RETURN
        END
C
C******************************************************************
        SUBROUTINE DOPSIG(ICALL)
C******************************************************************
C
C  ASSIGN WND STA WIND PROFILES TO FLOW SURFACES.MISSING WINDS ARE
C  DENOTED BY -999. IF SOUNDING IS NOT COMPLETE THE LAST REPORTED WIND
C  IS USED AT THE HIGHEST ALTITUDES. AFTER FLOW SURFACES ARE REDEFINED
C  (RESIG), DOPSIG IS RECALLED (ICALL >1) AND THE SOUNDINGS ARE
C  REINTERPOLATED TO THE SURFACES. TOPWIND AND BETWIN ARE RECALLED TO
C  PROVIDE THE WINDS THAT ARE TO BE BALANCED. THE ORIGINAL VERSION OF
C  THIS SUBROUTINE WAS WRITTEN BY R.M. ENDLICH, SRI INTN'L, MENLO PARK 
c  CA 94025. IT WAS LARGELY REWRITTEN BY LUDWIG IN NOV,1987. THIS 
c  VERSION INCLUDES CHANGES MADE IN APRIL, 1989.
c  further modified may 1989 to provide for a second call where winds
c  are reinterpolated to the newly defined flow surfaces; data reading
c  and other parts of the routine are skipped -- f. ludwig
c	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
c
	logical JGOOD(NSITES)     
	real DPHT(NSNDHT,NWSITE), DPUC(NSNDHT,NWSITE)
	real DPVC(NSNDHT,NWSITE),RHS1(NZGRD)
	integer NHTS(NWSITE)
c
        COMMON /ANCHOR/ SLAT,SLNG,UTMAPX,UTMAPY,MDATE,
     $                  IMO,IHOUR,NEND,XORIG,YORIG
        COMMON /compnt/ U(NXGRD,NYGRD,NZGRD),V(NXGRD,NYGRD,NZGRD),z0,
     $        USIG(NSITES,NZGRD),VSIG(NSITES,NZGRD),UCOMP(NSITES),
     $        VCOMP(NSITES),IUGRAF(NXGRD,NYGRD,NHORIZ),Z10,zzero,
     $	      IVGRAF(NXGRD,NYGRD,NHORIZ),SPDMET(NXGRD,NYGRD,NZGRD),
     $        DIRMET(NXGRD,NYGRD,NZGRD),SPDCNV
        COMMON /CSFC/ SFCHT(NXGRD,NYGRD),SIGMA(NZGRD),
     $                SFCLOW,SFCHI,ZRISE
        COMMON /CVOS/ RCM,RMF,IV,DSCRS,kgridx,kgridy,D2MIN,
     $	              HT2DIS,ADJMAX,dtwt
        COMMON /FLOWER/ GRDHI,GRDLO,DZMAX(NTSITE,NZGRD),
     $                  RHS(NXGRD,NYGRD,NZGRD),AVTHK,SLFAC,
     $                  RHSLO(NTSITE,NZGRD),CMPRES,DPTMIN,NFLAT,
     $                  ZCHOOZ(NZGRD),levbot(NXGRD,NYGRD)
        COMMON /LIMITS/ NCOL,NROW,NLVL,NCOLM1,NROWM1,
     $                  LOWIX(5,ntsite),LOWIY(5,ntsite)
        COMMON /NUMOBS/ NUMDOP,NUMNWS,NUMTOT,NUMTMP
	COMMON/PARMS/ ZTOP,DS,NLVLM1,TDSI,TYM	 
	COMMON /STALOC/ XG(NSITES),YG(NSITES),ZG(NSITES),
     $                  WTDOP(NXGRD,NYGRD,NWSITE),JJJ,IDOP(NWSITE),
     $                  ITMP(NTSITE),WTEMP(NXGRD,NYGRD,NTSITE),JGOOD
        COMMON /TSONDS/ ZSND(NSNDHT,NTSITE),TSND(NSNDHT,NTSITE),
     $     PSND(NSNDHT,NTSITE),ZMIDS(NSNDHT,NTSITE),NTLEV(NTSITE),
     $     DPTDZS(NSNDHT,NTSITE),POTEMP(NSNDHT,NTSITE),
     $     PTLAPS(NTSITE,NZGRD),T0(NTSITE,NZGRD),
     $     ZSIGL(NTSITE,NZGRD),DPWD(NSNDHT),DPWS(NSNDHT)
c
c  make sure values are here upon return.
c
	save
c
	DATA UVO/0.0/
C
C  VARIABLES ARE:
C    DPUC=U COMPONENT OF WND STA WIND IN MPS
C    DPVC=V COMPONENT OF WND STA WIND IN MPS
C    NHTS=NUMBER OF POINTS IN VERTICAL WIND PROFILE
C    NLVL=NUMBER OF FLOW LEVELS
C    RHS=HT OF FLOW SURFACES ABOVE TERRAIN (M)
C    XG,YG=STA. DIST IN X,Y IN GRID UNITS FROM 0,0 (SW CORNER)
C    Z0, UVO = ROUGHNESS HT AND ZERO-LEVEL WIND FOR INTERPOLATING
C  on 1st call at each time, read in wind profiles.
c
	if (icall .eq.1) then
	   DO 45 II=1,NUMDOP
              IT=IDOP(II)
              READ (12,*)
              READ (12,*)  dopx,dopy
C
C  locate grid point nearest the sounding (note origin at 1,1).
C
	      xg(it)=1.0+(dopx-XORIG)/DSCRS
	      yg(it)=1.0+(dopy-YORIG)/DSCRS
              IX=MAX(1,NINT(XG(IT)))
              JY=MAX(1,NINT(YG(IT)))
              IX=MIN(IX,NCOL)
              JY=MIN(JY,NROW)
	      zzg=sfcht(ix,jy)
              READ (12,*) NHTS(II)
c
              IF (NHTS(II) .LT.0) THEN
                 JGOOD(IT)=.false.
              ELSE
                 JGOOD(IT)=.true.
                 DO 15 LL=1,NHTS(II)
                    READ (12,*) ZZHT,ZZWD,ZZWS
c
c  convert ht to m & sounding info to meters, m/s 
c
		    ZZHT=htcnv*ZZHT
		    ZZWS=SPDCNV*ZZWS
                    IF (LL .LT. NSNDHT) THEN
                       DPHT(LL,II)=ZZHT
                       DPWD(LL)=ZZWD
                       DPWS(LL)=ZZWS
                    ELSE
                       DPHT(NSNDHT,II)=ZZHT
                       DPWD(NSNDHT)=ZZWD
                       DPWS(NSNDHT)=ZZWS
                    END IF
15               CONTINUE
                 IF (NHTS(II) .GT. NSNDHT) NHTS(II)=NSNDHT
C
C  CONVERT WIND MEASUREMENT HEIGHTS IN METERS (MSL) TO METERS (AGL)
C
                 DO 25 LL=1,NLVL
                    RHS1(LL)=RHS(IX,JY,LL)
25               CONTINUE
                 DO 30 LL=1,NHTS(II)
                    DPHT(LL,II)=DPHT(LL,II)-zzg
30               CONTINUE
C
C  CHANGE DIRECTION AND SPEED (MPS) TO U AND V; CHECK FOR MISSING DATA OR
C  BELOW GROUND HEIGHTS (AFTER CONVERSION FROM MSL) ON 1ST CALL.
C
                NEWLL=0
                DO 40 LL=1,NHTS(II)
                   IF (DPWD(LL).NE.999.0) THEN
                      IF (DPHT(LL,II) .GT. Z0) THEN
                         NEWLL=NEWLL+1
                         DPUC(NEWLL,II)=
     $                          -DPWS(LL)*SIN(DPWD(LL)/57.295)
                         DPVC(NEWLL,II)=
     $                          -DPWS(LL)*COS(DPWD(LL)/57.295)
                         DPHT(NEWLL,II)=DPHT(LL,II)
                      END IF
                   END IF
40              CONTINUE
                NHTS(II)=NEWLL
              END IF
45	   CONTINUE
	end if
C
C  INTERPOLATE TO ORIGINAL, OR NEWLY DEFINED, FLOW SURFACE HEIGHTS
C
	DO 450 II=1,NUMDOP
C
           IT=IDOP(II)
           IX=MAX(1,NINT(XG(IT)))
           JY=MAX(1,NINT(YG(IT)))
           IX=MIN(IX,NCOL)
           JY=MIN(JY,NROW)
           DO 71 LL=1,NLVL
              RHS1(LL)=RHS(IX,JY,LL)
71	   CONTINUE
           IF (NHTS(II) .GE. 2) THEN
              DO 400 K=1,NLVL
                 ZF=RHS1(K)
C
C  ZERO WIND WHEN FLOW SFC BELOW GROUND.
C
                 IF (ZF .LE. Z0) THEN
                    USIG(II,K)=0.0
                    VSIG(II,K)=0.0
                 ELSE IF (ZF.GT.Z0 .AND. ZF.LE.DPHT(1,II)) THEN
C
C  FLOW SFC BELOW 1ST OBSERVATION HT.
C
                    LNCALL=LNCALL+1
                    CALL LGNTRP(USIG(II,K),Z0,DPHT(1,II),
     $                                  ZF,UVO,DPUC(1,II))
                    LNCALL=LNCALL+1
                    CALL LGNTRP(VSIG(II,K),Z0,DPHT(1,II),
     $                                  ZF,UVO,DPVC(1,II))
                 ELSE IF (ZF .GT. DPHT(NHTS(II),II)) THEN
C
C  FLOW SFC ABOVE TOP OBSERVATION--USE TOP OBSERVATION
C
                    USIG(II,K)=DPUC(NHTS(II),II)
                    VSIG(II,K)=DPVC(NHTS(II),II)
                 ELSE IF (ZF.LT.DPHT(NHTS(II),II)
     $                       .AND. ZF.GT.DPHT(1,II)) THEN
                    DO 90 JHT=1,NHTS(II)-1
                       ZW1=DPHT(JHT,II)
                       ZW2=DPHT(JHT+1,II)
                       IF (ZF.GT.ZW1 .AND. ZF.LE.ZW2) THEN
C
C FLOW SFC BETWEEN OBSERVATION HTS
C
                          IFBR=2
                          CALL LGNTRP(USIG(II,K),ZW1,ZW2,ZF,
     $                           DPUC(JHT,II),DPUC(JHT+1,II))
                          CALL LGNTRP(VSIG(II,K),ZW1,ZW2,ZF,
     $                                    DPVC(JHT,II),DPVC(JHT+1,II))
                          GO TO 400
                       END IF
90                  CONTINUE
                 END IF
400           CONTINUE
           ELSE IF (NHTS(II) .EQ. 1) THEN
C
C  IF WIND MEASURED AT ONLY 1 HT. USE IT FOR HIGHER LEVELS, INTERP BELOW
C
              DO 410 K=1,NLVL
                 ZF=RHS1(K)
                 IF (ZF .GE. DPHT(1,II)) THEN
                    USIG(II,K)=DPUC(1,II)
                    VSIG(II,K)=DPVC(1,II)
                 ELSE IF (ZF.GT.Z0) THEN
                    CALL LGNTRP(USIG(II,K),Z0,DPHT(1,II),
     $                              ZF,UVO,DPUC(1,II))
                    CALL LGNTRP(VSIG(II,K),Z0,DPHT(1,II),
     $                              ZF,UVO,DPVC(1,II))
                 ELSE
                    USIG(II,K)=0.0
                    VSIG(II,K)=0.0
                 END IF
410           CONTINUE
           END IF
C
450  	CONTINUE
C
        RETURN
        END
c
C******************************************************************
        SUBROUTINE FLOWHT
C******************************************************************
C
C  THIS SUBROUTINE DETERMINES THE WEIGHTED AVERAGE FLOW SURFACE HEIGHTS
C  BASED ON THE VALUES THAT WOULD BE OBTAINED FROM THE INDIVIDUAL
C  TEMPERATURE PROFILES--F LUDWIG, JANUARY 1988
C
c	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
c
	logical JGOOD(NSITES)
c
        COMMON/CSFC/ SFCHT(NXGRD,NYGRD),SIGMA(NZGRD),
     $               SFCLOW,SFCHI,ZRISE
        COMMON /FLOWER/ GRDHI,GRDLO,DZMAX(NTSITE,NZGRD),
     $                  RHS(NXGRD,NYGRD,NZGRD),AVTHK,SLFAC,
     $                  RHSLO(NTSITE,NZGRD),CMPRES,DPTMIN,NFLAT,
     $                  ZCHOOZ(NZGRD),levbot(NXGRD,NYGRD)
        COMMON /LIMITS/ NCOL,NROW,NLVL,NCOLM1,NROWM1,
     $                  LOWIX(5,ntsite),LOWIY(5,ntsite)
        COMMON /NUMOBS/ NUMDOP,NUMNWS,NUMTOT,NUMTMP	 
	COMMON /STALOC/ XG(NSITES),YG(NSITES),ZG(NSITES),
     $                  WTDOP(NXGRD,NYGRD,NWSITE),JJJ,IDOP(NWSITE),
     $                  ITMP(NTSITE),WTEMP(NXGRD,NYGRD,NTSITE),JGOOD
        COMMON /TSONDS/ ZSND(NSNDHT,NTSITE),TSND(NSNDHT,NTSITE),
     $     PSND(NSNDHT,NTSITE),ZMIDS(NSNDHT,NTSITE),NTLEV(NTSITE),
     $     DPTDZS(NSNDHT,NTSITE),POTEMP(NSNDHT,NTSITE),
     $     PTLAPS(NTSITE,NZGRD),T0(NTSITE,NZGRD),
     $     ZSIGL(NTSITE,NZGRD),DPWD(NSNDHT),DPWS(NSNDHT)
C
	save
c
C  DZMAX(IT,IZ) = MAXIMUM RISE FOR IZth FLOW SFC AS DETERMINED FROM
C                 ITth T-SONDE
C
        DO 220 IX = 1,NCOL
           XX = FLOAT(IX)
           DO 200 IY = 1,NROW
	      YY=float(iy)
              HERE=SFCHT(IX,IY)
              DO 175 L = 1,NLVL
	         if (numtmp .gt. 1) then
                    SUMRYZ = 0.0
                    SUMWT=0.0
C
C  IF THE SITE HAD VALID DATA (NTLEV .)) THEN GET WEIGHTED AVERAGE RISE
C  AFTER GETTING RELATION FOR LOCAL TOPOGRAPHY HEIGHT VERSUS MAXIMUM
C  RISE FROM FUNCTION SLOPER.
C
                    DO 150 II = 1,NUMTMP
                      IT=ITMP(II)
                      IF (NTLEV(II) .GT. 0) THEN
                         ZRATIO=SLOPER(HERE,SFCLOW,zrise)
                         RISE = ZRATIO*DZMAX(II,L)
                         SUMRYZ = SUMRYZ+
     $                      WNDWT(XX,YY,XG(IT),YG(IT))*RISE
                         SUMWT= 
     $                      SUMWT+WNDWT(XX,YY,XG(IT),YG(IT))
                      END IF
150                 CONTINUE
                    IF (SUMWT .GT.0.0) THEN
		       rhere=sumryz/sumwt
                       RHS(IX,IY,L) =rhere+avthk*sigma(L)+SFCLOW-HERE
	            else
                       RHS(IX,IY,L)=avthk*sigma(L)-0.25*(HERE-SFCLOW)
	            end if
                 ELSE if (numtmp .lt.1) then
C
C  IF NO SOUNDING USE SFC THAT RISES 3/4 AS FAST AS THE TERRAIN.
C
                       RHS(IX,IY,L)=avthk*sigma(L)-0.25*(HERE-SFCLOW)
	         else if (numtmp.eq.1) then
		    rhere=SLOPER(HERE,SFCLOW,zrise)*DZMAX(1,L)
                    RHS(IX,IY,L) = rhere+avthk*sigma(L)+SFCLOW-HERE
	         end if
175           CONTINUE
200        CONTINUE
220	CONTINUE
C
        RETURN
        END
c
c**********************************************************************
        subroutine geosig
c**********************************************************************
c
c  prepare sfc station reports of wind direction, wind speed
c  (m/s), sea level pressure (mb), and temperature (deg celsius)
c  for input to wind analysis if no upper winds.
c  compute geos wind from pressure at three stations and
c  correct it for thermal wind component (if desired).
c
c  by r m  endlich, sri intn'l, menlo park ca 94025 dec '84.
c  variables.geostrophic wind calculations were put in the subroutine
c  geostr and a  different method of wind interpolation was introduced 
c  january 1988 by f. ludwig.
c
c  further modified may 1989 to reinterpolate to flow surfaces with
c  second calls to dopsig, topwnd, sfctrp and betwin --f. ludwig
c
c  further modified november 1989 to interpolate sfc temperatures and
c   upper levelpotent temp lapse rates to flow surfaces  -- f. ludwig
c
c  further modified march 1997 to read a long sequence of inputs from
c  the same file (unit 12), not winds and upper temperatures from
c  separate files.
c
c             -- f. ludwig  2/2000
c
c  variables:
c    numnws = number of sfc reports
c    wd = wind direction (deg cw from n-- meteorol convent.)
c    sp = wind speed (m/s)
c    stlt = station latitude in degs and hundredths
c    stln = station longitude in degs and hundredths
c    press = station sea level pressure in mb
c    temp = station temp in deg celsius
c    spdcnv=conversion factor to convert spds to m/s, if in other units
c    deg2r= factor to convert degrees to radians
c	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
	parameter (PI=3.1415927,DEG2R=PI/180.0)
c
	logical JGOOD(NSITES)
	INTEGER ncallz
c
        COMMON /ANCHOR/ SLAT,SLNG,UTMAPX,UTMAPY,MDATE,
     $                  IMO,IHOUR,NEND,XORIG,YORIG
        COMMON /compnt/ U(NXGRD,NYGRD,NZGRD),V(NXGRD,NYGRD,NZGRD),z0,
     $        USIG(NSITES,NZGRD),VSIG(NSITES,NZGRD),UCOMP(NSITES),
     $        VCOMP(NSITES),IUGRAF(NXGRD,NYGRD,NHORIZ),Z10,zzero,
     $	      IVGRAF(NXGRD,NYGRD,NHORIZ),SPDMET(NXGRD,NYGRD,NZGRD),
     $        DIRMET(NXGRD,NYGRD,NZGRD),SPDCNV 
        COMMON /CSFC/ SFCHT(NXGRD,NYGRD),SIGMA(NZGRD),
     $                SFCLOW,SFCHI,ZRISE
	COMMON /CVOS/ RCM,RMF,IV,DSCRS,kgridx,kgridy,D2MIN,
     $	              HT2DIS,ADJMAX,dtwt
        COMMON /FLOWER/ GRDHI,GRDLO,DZMAX(NTSITE,NZGRD),
     $                  RHS(NXGRD,NYGRD,NZGRD),AVTHK,SLFAC,
     $                  RHSLO(NTSITE,NZGRD),CMPRES,DPTMIN,NFLAT,
     $                  ZCHOOZ(NZGRD),levbot(NXGRD,NYGRD)
	COMMON /LIMITS/ NCOL,NROW,NLVL,NCOLM1,NROWM1,
     $                  LOWIX(5,ntsite),LOWIY(5,ntsite)
	COMMON /NUMOBS/ NUMDOP,NUMNWS,NUMTOT,NUMTMP
	COMMON /PARMS/ ZTOP,DS,NLVLM1,TDSI,TYM	 
	COMMON /STALOC/ XG(NSITES),YG(NSITES),ZG(NSITES),
     $                  WTDOP(NXGRD,NYGRD,NWSITE),JJJ,IDOP(NWSITE),
     $                  ITMP(NTSITE),WTEMP(NXGRD,NYGRD,NTSITE),JGOOD
c
       	save
c
	data ncallz /0/
c
c  read surface station data: station id, station coordinates (utm)
c  pressure, temperature, wind direction, wind speed (mps) 
c  read pressure, temperature, wind direction & wind speed.
c
	read (12,*)
	read (12,*) (idop(i), i = 1,nwsite)
	read (12,*)
	read (12,*) (itmp(i), i = 1,ntsite)
c
	read (12,*)
        read (12,*) numnws
c
	if (numnws.le.0) then
	   write(16,*)  'you need at least 1 sfc ob for wocss to work'
	    pause
	   stop
	end if
c
c  for USGS plotting purposes write nearest grid for each site to 
c  file windssuv.stations
c
	OPEN (38, FILE='windsuv.stations', STATUS='UNKNOWN',
     $               form='formatted')
c
c  read surface station data  
c
        do 75 jt=1,numnws
c
c if no. of sites gt than dimensioned for adjust index
c to skip those from the dimensioned value to end.
c
	   if (jt.gt.NSITES) then
	      it=NSITES
	   else
	      it=jt
	   end if
           read (12,6004) xs,ys,press,tfaren,wd,ws
6004	   format (f5.1,2f8.1,f4.0,2f8.1)
c
c  converting utm coordinates (km) and elev. (m) to grid coordinates
c  (km). note that origin is at point 1,1
c
	   xg(it)=1.0+(xs-xorig)/dscrs
           yg(it)=1.0+(ys-yorig)/dscrs
	   ixg=nint(xg(it))
	   iyg=nint(yg(it))
c
C 	write wind grid pts a la Jon C. Cate / January 1997
c
           WRITE(38,4) ixg,iyg
  4	   FORMAT(1X,2I5)	
c
c  use ht of nearest grid point for off-grid pts.
c
	   if (ixg.lt.1) ixg=1
	   if (iyg.lt.1) iyg=1
	   if (ixg.gt.ncol) ixg=ncol
	   if (iyg.gt.nrow) iyg=nrow
	   iyg=min(nrow,iyg)
           ztopo=sfcht(ixg,iyg)
	   zg(it)=0.001*ztopo/dscrs
c
c converting units & getting components
c
	   if (nint(ws) .ne. nint(999.0)) then
	      jgood(it)=.true.
              ws=ws*SPDCNV
              ucomp(it)=-ws*sin(wd*DEG2R)
              vcomp(it)=-ws*cos(wd*DEG2R)
           else
	      jgood(it)=.false.
              ucomp(it)=ws
              vcomp(it)=999.0
           end if
   75   continue
   	close (38)
c
c  call routine to write obs to files for plotting and querying.
c
	call cateob(numnws,xg,yg,ucomp,vcomp) 
c
	ncallz=ncallz+1
c
c  interpolate to lowest above ground grid points, using sfc wind data.
c
	nsfc=1
        call sfctrp(nsfc)
c
c get top winds by interpolation between upper soundings if available
c
        icall=1
        call dopsig(icall)
        call topwnd
c
c  read temp profile & get lapse rates at flow levels
c
        call strat
c
        call betwin 
c
c  reshape the flow surfaces using the 1st estimate winds &
c  temperature profiles.
c
        call resig
c
c  go back and reinterpolate to new surfaces
c
c  -- the following code added may 1989 to give better values 
c     for 1st guess field, f. ludwig
c
        icall=2
c
        call dopsig (icall)
        call topwnd
	nsfc=2
        call sfctrp(nsfc)
        call betwin
c
        DO 350 J=1, NROW
           DO 325 I=1, NCOL
              DO 300 LV=2,NLVL
C
C  WHEN RHS NEGATIVE (BELOW TERRAIN) MAKE WINDS 0.
C
                 IF (RHS(I,J,LV).le.z0) THEN
                    U(I,J,LV)=0.0
                    V(I,J,LV)=0.0
                 END IF
300 	      CONTINUE
325 	   CONTINUE
350 	CONTINUE
c
	return
        end
C
C*********************************************************************
        SUBROUTINE RESIG
C*********************************************************************
C
C  REDEFINES THE HEIGHTS OF THE FLOW SURFACES BASED ON WIND SPEED OVER
C  THE LOWEST TERRRAIN HEIGHTS & LAPSE RATES AT THE VARIOUS FLOW LEVELS
C  --THE UNDERLYING CONCEPT IS SIMILAR TO THAT OF THE 'CRITICAL STREAM-
C  LINE'. WEIGHTED AVERAGES ARE USED TO CALCULATE FLOW SURFACE HEIGHTS 
c  AT EACH GRID POINT, GIVING GREATEST WEIGHT TO VALUES APPROPRIATE TO 
c  THE NEAREST SOUNDINGS. 
c
c    --LUDWIG, JANUARY 1988.
C
c	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
c
	real COMPDZ
c
        COMMON /compnt/ U(NXGRD,NYGRD,NZGRD),V(NXGRD,NYGRD,NZGRD),z0,
     $        USIG(NSITES,NZGRD),VSIG(NSITES,NZGRD),UCOMP(NSITES),
     $        VCOMP(NSITES),IUGRAF(NXGRD,NYGRD,NHORIZ),Z10,zzero,
     $	      IVGRAF(NXGRD,NYGRD,NHORIZ),SPDMET(NXGRD,NYGRD,NZGRD),
     $        DIRMET(NXGRD,NYGRD,NZGRD),SPDCNV
        COMMON/ CSFC/ SFCHT(NXGRD,NYGRD),SIGMA(NZGRD),
     $                SFCLOW,SFCHI,ZRISE
        COMMON /FLOWER/ GRDHI,GRDLO,DZMAX(NTSITE,NZGRD),
     $                  RHS(NXGRD,NYGRD,NZGRD),AVTHK,SLFAC,
     $                  RHSLO(NTSITE,NZGRD),CMPRES,DPTMIN,NFLAT,
     $                  ZCHOOZ(NZGRD),levbot(NXGRD,NYGRD)
        COMMON /LIMITS/ NCOL,NROW,NLVL,NCOLM1,NROWM1,
     $                  LOWIX(5,ntsite),LOWIY(5,ntsite)
        COMMON /NUMOBS/ NUMDOP,NUMNWS,NUMTOT,NUMTMP
        COMMON /TSONDS/ ZSND(NSNDHT,NTSITE),TSND(NSNDHT,NTSITE),
     $     PSND(NSNDHT,NTSITE),ZMIDS(NSNDHT,NTSITE),NTLEV(NTSITE),
     $     DPTDZS(NSNDHT,NTSITE),POTEMP(NSNDHT,NTSITE),
     $     PTLAPS(NTSITE,NZGRD),T0(NTSITE,NZGRD),
     $     ZSIGL(NTSITE,NZGRD),DPWD(NSNDHT),DPWS(NSNDHT)
C
C  MAKE SURE STUFF IS HERE WHEN YOU COME BACK
C
        SAVE
C
C  OVERS(DPTDZ,DZTOP) IS A STATEMENT FUNCTION THAT CAN BE USED TO
C  DEFINE MAXIMUM RISE OF A FLOW SURFACE IN TERMS OF THE POTENTIAL
C  TEMPERATURELAPSE RATE (DPTDZ) RATE AND THE MAXIMUM DIFFERENCE
C  IN TERRAIN ELEVATION (DZMAX). THE VERSION INCLUDED HERE ASSUMES 
C  THAT FLOW FOLLOWS UNDERLYING TERRAIN FOR NEUTRAL OR UNSTABLE
C  CONDITIONS & THAT THERE IS NO OVERSHOOT OR UNDERSHOOT.
C
        OVERS(DPTDZ,DZTOP)=DZTOP
C
C  GETTING LOWEST POINT AND ROOT-MEAN-SQUARE AVERAGE WIND SPEED OVER
C  LOW TERRAIN GRID POINTS.
C
	DO 200 IT = 1,NUMTMP
C
C  CHECK THAT THERE WAS DATA FROM T-SONDE STATION (NTLEV >0)
C
           IF (NTLEV(IT).GT.0) THEN
              LX=LOWIX(1,IT)
              LY=LOWIY(1,IT)
              DO 100 L=1,NLVL
                 RHSLO(IT,L)=RHS(LX,LY,L)
                 ZSIGL(IT,L)=RHS(LX,LY,L)
                 RMSSPD=0.0
                 DO 50 J=1,5
                    JX=LOWIX(J,IT)
                    JY=LOWIY(J,IT)
                    RMSSPD=RMSSPD+U(JX,JY,L)**2+
     $                          V(JX,JY,L)**2
 50              CONTINUE
                 SPD=SQRT(RMSSPD/5.0)
C
C  PUT LOWER LIMIT ON POTENTIAL TEMPERATURE LAPSE RATE AND CALCULATE
C  MAX RISE CORRESPONDING TO THE WIND SPEED & LAPSE FOR THIS T-SONDE.
C
                 DTHETA=PTLAPS(IT,L)
                 IF (DTHETA.LE.dptmin) DTHETA=dptmin
                 DZMAX(IT,L)=SPD/SQRT(9.8*DTHETA/T0(IT,L))
C
C   FLOW AMPLITUDES ARE LIMITED using A STATEMENT FUNCTION (OVERS)
C   OF THE POT TMP LAPSE & TERR. AMPLITUDE.
C
                 IF (DZMAX(IT,L).GT.OVERS(DTHETA,zrise))
     $                          DZMAX(IT,L)=OVERS(DTHETA,zrise)
 100          CONTINUE
C
C  DEFINE MINIMUM ALLOWABLE FLOW SURFACE SEPARATIONS FOR EACH T-SONDE'S
C  DOMAIN OF INFLUENCE. MINIMUM SEPARATION DEFINED AS A FRACTION
C  (CMPRES) OF THEIR SEPARATION OVER THE TERRAIN LOW POINT FOR THE
C  T-SONDE. DO NOT LET SEPARATION BETWEEN SURFACES COMPRESS BY MORE
C  THAN CMPRES FACTOR OR BE GREATER THAN DEFINED BY OVERS(DPTDZ,DZMAX).
C
              DO 150 L=NLVL-1,1,-1
                 DTHETA=PTLAPS(IT,L)
                 COMPDZ=CMPRES*(RHSLO(IT,L+1)-RHSLO(IT,L))
                 IF (DZMAX(IT,L).GT.(DZMAX(IT,L+1)+COMPDZ))
     $                         DZMAX(IT,L)=DZMAX(IT,L+1) +COMPDZ
 150          CONTINUE
C
C  MAKE ONE MORE PASS TO ENSURE THAT UPPER SURFACES DO NOT RISE
C  MORE RAPIDLY THAN THE SURFACE OR TERRAIN BELOW, WHICHEVER IS HIGHER.
C
              DO 155 L=2,NLVL
	         if (L.eq.2) then
	            IF (DZMAX(IT,L).GT.OVERS(DTHETA,zrise))
     $                 DZMAX(IT,L)=OVERS(DTHETA,zrise)
		    else
		       diff=zrise-rhs(lx,ly,L)
                       TOTRYZ=DZMAX(IT,L-1)
                       IF (TOTRYZ .LT. DIFF) TOTRYZ=DIFF
                       IF(DZMAX(IT,L).GT.TOTRYZ) DZMAX(IT,L)=TOTRYZ
		    end if
 155	      CONTINUE
C
           END IF
C
200	CONTINUE
c
c  NOW CALL FLOWHT TO DETERMINE THE WEIGHTED AVERAGE HTS FOR EACH FLOW 
c  SFC ABOVE EACH GRID POINT & REDEFINE LOW FLOW SFC HEIGHTS ON GRID
C
	CALL FLOWHT
C
        RETURN
        END
C
C********************************************************************
        SUBROUTINE SFCTRP(ncall)
C********************************************************************
C
C  THIS SUBROUTINE GETS FIRST ESTIMATE OF WINDS AT LOWEST GRID POINTS.
c  inverse distance-squared.
c
c   this version fludwig 2/2000
C	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
c
	logical JGOOD(NSITES)
	real utemp(NXGRD,NYGRD),vtemp(NXGRD,NYGRD)  
c
        COMMON /compnt/ U(NXGRD,NYGRD,NZGRD),V(NXGRD,NYGRD,NZGRD),z0,
     $        USIG(NSITES,NZGRD),VSIG(NSITES,NZGRD),UCOMP(NSITES),
     $        VCOMP(NSITES),IUGRAF(NXGRD,NYGRD,NHORIZ),Z10,zzero,
     $	      IVGRAF(NXGRD,NYGRD,NHORIZ),SPDMET(NXGRD,NYGRD,NZGRD),
     $        DIRMET(NXGRD,NYGRD,NZGRD),SPDCNV
        COMMON /FLOWER/ GRDHI,GRDLO,DZMAX(NTSITE,NZGRD),
     $                  RHS(NXGRD,NYGRD,NZGRD),AVTHK,SLFAC,
     $                  RHSLO(NTSITE,NZGRD),CMPRES,DPTMIN,NFLAT,
     $                  ZCHOOZ(NZGRD),levbot(NXGRD,NYGRD)
        COMMON /LIMITS/ NCOL,NROW,NLVL,NCOLM1,NROWM1,
     $                  LOWIX(5,ntsite),LOWIY(5,ntsite)
        COMMON /NUMOBS/ NUMDOP,NUMNWS,NUMTOT,NUMTMP
	COMMON/PARMS/ ZTOP,DS,NLVLM1,TDSI,TYM
	COMMON /STALOC/ XG(NSITES),YG(NSITES),ZG(NSITES),
     $                  WTDOP(NXGRD,NYGRD,NWSITE),JJJ,IDOP(NWSITE),
     $                  ITMP(NTSITE),WTEMP(NXGRD,NYGRD,NTSITE),JGOOD
C
	save
c
        DATA ZERO /0.0/
C
C  FIRST, GET A VALUE FOR EACH X,Y GRID POINT. interpolate to get 
c  values at 10 m utemp & vtemp above each grid point on 1st call
c
	if (ncall.le.1) then
c
	   call guess1 (ucomp,utemp,numnws)
	   call guess1 (vcomp,vtemp,numnws)
c
c  now assign interpolated values to level 1
c
	   do 24 ix=1,ncol
	      do 22 iy=1,nrow
	         u(ix,iy,1)=utemp(ix,iy)
	         v(ix,iy,1)=vtemp(ix,iy)
c
c  extrapolate or interpolate to next surface
c
                 CALL LGNTRP(U(IX,IY,2),Z0,Z10,
     $                    RHS(IX,IY,2),ZERO,utemp(ix,iy))
                 CALL LGNTRP(V(IX,IY,2),Z0,Z10,
     $                    RHS(IX,IY,2),ZERO,vtemp(ix,iy))
	         levbot(ix,iy)=2
22	      continue
24	   continue
c
	else
c
c  on second call use values from 1st call to 
c  determine components at 1st above ground level
c
	   DO 50 IX=1,NCOL
	     DO 50 IY=1,NROW
	        if (RHS(IX,IY,NLVL).LE.Z0) THEN
                   DO 40 L=2,NLVL
                      U(IX,IY,L)=0.0
                      V(IX,IY,L)=0.0
40	           continue
	           levbot(ix,iy)=NLVL
	        else
                   UU =u(IX,IY,1)
                   VV =v(IX,IY,1)
C
C  NOW INTER(EXTRA)POLATE TO LOWEST ABOVE-GROUND FLOW SURFACE.
C
                   DO 45 L=2,NLVL
                      IF (RHS(IX,IY,L).LE.Z0) THEN
                         U(IX,IY,L)=0.0
                         V(IX,IY,L)=0.0
                      ELSE if (RHS(IX,IY,L-1).LE.Z0 .and.
     $                   RHS(IX,IY,L) .GT. Z0) then
                            CALL LGNTRP(U(IX,IY,L),Z0,Z10,
     $                             RHS(IX,IY,L),ZERO,UU)
                            CALL LGNTRP(V(IX,IY,L),Z0,Z10,
     $                             RHS(IX,IY,L),ZERO,VV)
	 	            levbot(ix,iy)=L
                      END IF
45                 CONTINUE
	        end if
50           CONTINUE
	end if
c
6001	format (1x,i4,15f7.2)
6002	format (1x,i4,15(2x,i3,2x))
6004	format (1x,45f7.1)
6011	format (1x,23i5)
C
        RETURN
        END
C
C********************************************************************
	subroutine guess1(val,valtrp,nmbr)
C********************************************************************
c  
c  this program uses inverse distance interpolation
c
c		fludwig, oct 97, rev 12/05
C
c
c  xg,yg,val	coordinates (grid units) & value at observing sites
c  valtrp	interpolated values at grid points
c  nmbr		number of observations
c  	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
c
	logical JGOOD(NSITES)
	integer nmbr
	real val(NSITES),valtrp(NXGRD,NYGRD) 
c
	COMMON /LIMITS/ NCOL,NROW,NLVL,NCOLM1,NROWM1,
     $                  LOWIX(5,ntsite),LOWIY(5,ntsite)
	COMMON /PARMS/ ZTOP,DS,NLVLM1,TDSI,TYM	 
	COMMON /STALOC/ XG(NSITES),YG(NSITES),ZG(NSITES),
     $                  WTDOP(NXGRD,NYGRD,NWSITE),JJJ,IDOP(NWSITE),
     $                  ITMP(NTSITE),WTEMP(NXGRD,NYGRD,NTSITE),JGOOD
c
	save
c
	do 370 ix=1,ncol
	   x2=float(ix)
	   do 365 iy=1,nrow
	      y2=float(iy)
	      sumwt=0.0
	      sumval=0.0
	      do 355 is=1,nmbr
	         dist2=(xg(is)-x2)**2+(yg(is)-y2)**2
	         if (dist2 .lt. 0.1) dist=0.1
	         recdis=1.0/dist2
	         sumwt=sumwt+recdis
	         sumval=sumval+recdis*val(is)
355	      continue
	      if (sumwt.eq.0.0) then
		  write(*,*) 'bad observed coordinates'
	          pause
	          stop
	       else
	          valtrp(ix,iy)=sumval/sumwt
	       end if
365	   continue
370	continue
c
c
6001	format(1x,31e10.2)
6002	format (1x,31i5)
6003	format(1x,31i4)
6004	format(1x,i4,15f7.2)
c
	return
	end
C
C**********************************************************************
        REAL FUNCTION SLOPER(HERE,SFCMIN,HIRISE)
C**********************************************************************
C
C  DETERMINES RATIO OF HEIGHT ABOVE LOWEST TOPOGRAPHY TO MAXIMUM
C  HEIGHT ABOVE LOWEST POINT. OTHER RELATIONSHIPS CAN BE SUBSTITUTED
C  TO GIVE DIFFERENT FLOW SURFACE HEIGHTS.  -- LUDWIG 11/87
C
        IF ((HIRISE-SFCMIN) .EQ. 0.0 .OR. HIRISE .EQ. 0.0) THEN
C
C  FLAT TERRAIN always gives flat sfcs
C
           SLOPER =0.0
        ELSE
C
C  sfc rise at this pt proportional to terrain increment as a fraction 
c  of maximum terrain elevation differences.
C
           SLOPER = (HERE-SFCMIN)/HIRISE
        END IF
C
        RETURN
        END
c
c**********************************************************************
         SUBROUTINE STRAT
C**********************************************************************
C
C  READS SOUNDING INFORMATION,CALCULATES LAPSE RATES & OTHER PARAM-
C  ETERS FOR VARIOUS FLOW LEVELS. MODIFIED BY LUDWIG, DECEMBER 1987.
C	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
c
	logical JGOOD(NSITES)     
c
        COMMON /ANCHOR/ SLAT,SLNG,UTMAPX,UTMAPY,MDATE,
     $                  IMO,IHOUR,NEND,XORIG,YORIG
        COMMON /compnt/ U(NXGRD,NYGRD,NZGRD),V(NXGRD,NYGRD,NZGRD),z0,
     $        USIG(NSITES,NZGRD),VSIG(NSITES,NZGRD),UCOMP(NSITES),
     $        VCOMP(NSITES),IUGRAF(NXGRD,NYGRD,NHORIZ),Z10,zzero,
     $	      IVGRAF(NXGRD,NYGRD,NHORIZ),SPDMET(NXGRD,NYGRD,NZGRD),
     $        DIRMET(NXGRD,NYGRD,NZGRD),SPDCNV
        COMMON/CSFC/ SFCHT(NXGRD,NYGRD),SIGMA(NZGRD),
     $               SFCLOW,SFCHI,ZRISE
        COMMON /CVOS/ RCM,RMF,IV,DSCRS,kgridx,kgridy,D2MIN,
     $	              HT2DIS,ADJMAX,dtwt
        COMMON /FLOWER/ GRDHI,GRDLO,DZMAX(NTSITE,NZGRD),
     $                  RHS(NXGRD,NYGRD,NZGRD),AVTHK,SLFAC,
     $                  RHSLO(NTSITE,NZGRD),CMPRES,DPTMIN,NFLAT,
     $                  ZCHOOZ(NZGRD),levbot(NXGRD,NYGRD)
        COMMON /LIMITS/ NCOL,NROW,NLVL,NCOLM1,NROWM1,
     $                  LOWIX(5,ntsite),LOWIY(5,ntsite)
        COMMON /NUMOBS/ NUMDOP,NUMNWS,NUMTOT,NUMTMP
	COMMON/PARMS/ ZTOP,DS,NLVLM1,TDSI,TYM	 
	COMMON /STALOC/ XG(NSITES),YG(NSITES),ZG(NSITES),
     $                  WTDOP(NXGRD,NYGRD,NWSITE),JJJ,IDOP(NWSITE),
     $                  ITMP(NTSITE),WTEMP(NXGRD,NYGRD,NTSITE),JGOOD
        COMMON /TSONDS/ ZSND(NSNDHT,NTSITE),TSND(NSNDHT,NTSITE),
     $     PSND(NSNDHT,NTSITE),ZMIDS(NSNDHT,NTSITE),NTLEV(NTSITE),
     $     DPTDZS(NSNDHT,NTSITE),POTEMP(NSNDHT,NTSITE),
     $     PTLAPS(NTSITE,NZGRD),T0(NTSITE,NZGRD),
     $     ZSIGL(NTSITE,NZGRD),DPWD(NSNDHT),DPWS(NSNDHT)
c
	save
C
C  STATEMENT FUNCTIONS FOR POTENTIAL TEMPERATURE & LINEAR INTERPOLATION
C
        THETA(PP,TT)=(TT+273.13)*((1000./PP)**0.288)
        XLINTR(X0,X1,ZZZ,Z0,Z1)=X0+(X1-X0)*(ZZZ-Z0)/(Z1-Z0)
c
        IF(numtmp.gt.0) THEN
c
	   DO 200 JTSOND=1,NUMTMP
C
C  FILL ARRAYS WITH MSG DATA IDENTIFIER
C
              DO 8 I=1,NSNDHT
                 ZSND(I,JTSOND)=-999.
                 TSND(I,JTSOND)=-999.
                 PSND(I,JTSOND)=-999.
                 ZMIDS(I,JTSOND)=-999.
                 DPTDZS(I,JTSOND)=-999.
                 POTEMP(I,JTSOND)=-999.
 8            CONTINUE
              II=ITMP(JTSOND)
c
c  read  station no.,no. of heights (nhites) in sounding & data input
c  type (ityp) data are read as follows for ityp= :
c       1--height(m--msl),temperature(c), potential temperature
c          lapse rate(k/m)
c       2--height(m--msl), temp(c) & pressure(mb)
c       3--height (in ft. msl), temp(c), pressure (mb)
c       4--height (in ft. msl), temp (c), dewpt (c), press (mb)
c
              read(12,*)
              read(12,*)  
              read(12,*) utme,utmn 
              read(12,*) ntlev(jtsond),ityp
6001	      format (1x,2i4)
              read(12,*)
              nhites=ntlev(jtsond)
              if (nhites .gt.0) then
c
c  get terrain height at nearest grid pt.
c
                  jx=max(1,nint(xg(ii)))
                  jy=max(1,nint(yg(ii)))
                  jx=min(jx,ncol)
                  jy=min(jy,nrow)
                  zhere=sfcht(jx,jy)
c
                  i=0
                  do 10 j=1,nhites
                      if(ityp.le.3) then
	                 read(12,*,end=186) z,t,p
	              else
	                 read(12,*,end=186) z,t,dp,p
	              end if
                      IF(Z.LE.-999. .OR. T.LE.-999.
     $                         .OR. P.LE.-999.)     GO TO 10
                      I=I+1
                      IF (ITYP.GE.2) THEN
C
C  CONVERT TO METERS & T TO POT TMP, IF REQUIRED (ITYP=3 or 4)
C
                         IF (ITYP .GE. 3) Z=0.3048*Z
                         IF (I .LE. NSNDHT) THEN
	                    ZSND(I,JTSOND)=Z
	                    TSND(I,JTSOND)=THETA(P,T)
	                    PSND(I,JTSOND)=P
                         ELSE
	                    ZSND(NSNDHT,JTSOND)=Z
	                    TSND(NSNDHT,JTSOND)=THETA(P,T)
	                    PSND(NSNDHT,JTSOND)=P
                         END IF
                      ELSE IF (ITYP.EQ.1) THEN
C
C  ESTIMATE POT TEMPS FROM INTEGRATED POT TMP LAPSE RATES
C
                         IF (I.EQ.1) THEN
	                    TSND(I,JTSOND)=T+273.13
	                    DPTDZS(I,JTSOND)=P
	                    ZMIDS(I,JTSOND)=Z
                         ELSE IF(I .LE. NSNDHT) THEN
	                    DPTDZS(I,JTSOND)=P
	                    ZMIDS(I,JTSOND)=Z
	                    TSND(I,JTSOND)=TSND(I-1,JTSOND)+
     $                           0.5*(ZMIDS(I,JTSOND)-
     $                           ZMIDS(I-1,JTSOND))*
     $                           (DPTDZS(I-1,JTSOND)+DPTDZS(I,JTSOND))
                         ELSE
                            DPTDZS(NSNDHT,JTSOND)=P
                            ZMIDS(NSNDHT,JTSOND)=Z
                            TSND(NSNDHT,JTSOND)=TSND(NSNDHT-1,JTSOND)+
     $                           0.5*(Z-ZMIDS(NSNDHT-1,JTSOND))*
     $                           (DPTDZS(NSNDHT-1,JTSOND)+P)
                         END IF
                      END IF
 10                CONTINUE
 186               NHITES=I
                   IF(NHITES .GT. NSNDHT) NHITES=NSNDHT
                   IF(ITYP .GE. 2) THEN
                      LL=0
                      DO 30 I=1,NHITES-1
                         IF (ZSND(I+1,JTSOND).GT.ZSND(I,JTSOND)) THEN
                            LL=LL+1
                            T1=TSND(I,JTSOND)
                            T2=TSND(I+1,JTSOND)
                            DPTDZS(LL,JTSOND)=(T2-T1)/
     $                          (ZSND(I+1,JTSOND)-ZSND(I,JTSOND))
                            ZMIDS(LL,JTSOND)=0.5*
     $                          (ZSND(I,JTSOND)+ZSND(I+1,JTSOND))
                            TSND(LL,JTSOND)=0.5*(T1+T2)
                         END IF
 30                   CONTINUE
                      NHITES=LL
                   END IF
C
C  CONVERT TO HEIGHT ABOVE SFC-- DISCARD ANY BELOW SFC POINTS (ERROR).
C
                   LL=0
                   DO 35 I=1,NHITES
                      ZMIDS(I,JTSOND)=ZMIDS(I,JTSOND)-ZHERE
                      IF (ZMIDS(I,JTSOND) .GT. 0.0) THEN
                         LL=LL+1
                         ZMIDS(LL,JTSOND)=ZMIDS(I,JTSOND)
                         DPTDZS(LL,JTSOND)=DPTDZS(I,JTSOND)
                         TSND(LL,JTSOND)=TSND(I,JTSOND)
                      END IF
 35                CONTINUE
                   NHITES=LL
C
C  INTERPOLATE (OR EXTRPOLATE) TO FLOW SURFACES.
C
	         DO 100 I=1,NLVL
	            ZBAR=0.0
C
C  GET AVERAGE HT. ABOVE LOW SPOTS FOR FLOW SURFACES & INTERPOLATE
C  TEMPERATURES & LAPSE RATES.
C
	            IF (I.GT.1) THEN
	               DO 80 J=1,5
	                  LLLX=LOWIX(J,jtsond)
	                  LLLY=LOWIY(J,jtsond)
	                  ZBAR=ZBAR+RHS(LLLX,LLLY,I)
80	               CONTINUE
                       ZBAR=ZBAR/5.0
                    END IF
C
C  USE TOP LAPSE RATE & EXTRAPOLATE TO LEVELS ABOVE TOP OF SOUNDING.
C
                    IF (ZBAR .GT. ZMIDS(NHITES,JTSOND)) THEN
                       PTLAPS(JTSOND,I)=DPTDZS(NHITES,JTSOND)
                       T0(JTSOND,I)=TSND(NHITES,JTSOND)+
     $                    PTLAPS(JTSOND,I)*(ZBAR-ZMIDS(NHITES,JTSOND))
C
C  USE BOTTOM LAPSE & EXTRPOLATE TO LEVELS BELOW LOWEST SOUNDING HT.
C
                    ELSE IF (ZBAR .LE. ZMIDS(1,JTSOND)) THEN
                       PTLAPS(JTSOND,I)=DPTDZS(1,JTSOND)
                       T0(JTSOND,I)=TSND(1,JTSOND)+
     $                    PTLAPS(JTSOND,I)*(ZBAR-ZMIDS(1,JTSOND))
C
C  LINEARLY INTERPOLATE FOR LEVELS BETWEEN SOUNDING POINTS
C
                    ELSE
	               DO 90 J=2,NHITES
                          Z0=ZMIDS(J-1,JTSOND)
                          DTDZ0=DPTDZS(J-1,JTSOND)
                          T00=TSND(J-1,JTSOND)
                          IF (Z0 .LE. 0.0) THEN
                             Z0=0.0
                             T00=TSND(J,JTSOND)
                             DTDZ0=DPTDZS(J,JTSOND)
                          END IF
                          Z1=ZMIDS(J,JTSOND)
                          IF (ZBAR.GT.Z0 .AND. ZBAR.LE.Z1) THEN
                             PTLAPS(JTSOND,I)=
     $                       XLINTR(DTDZ0,DPTDZS(J,JTSOND),ZBAR,Z0,Z1)
                             T0(JTSOND,I)=
     $                          XLINTR(T00,TSND(J,JTSOND),ZBAR,Z0,Z1)
	                  END IF
90	               CONTINUE
                       PTLAPS(JTSOND,1)=DPTDZS(1,JTSOND)
                       T0(JTSOND,1)=TSND(1,JTSOND)
                    END IF
100	         CONTINUE
              END IF
200	   CONTINUE
c
        END IF
c
C
        RETURN
        END
c
c*********************************************************************
        SUBROUTINE TOPWND
c*********************************************************************
C
C  INTERPOLATES WINDS AT TOPMOST LEVEL BETWEEN OBSERVATION PTS--WHEN
C  ONLY ONE WIND AVAILABLE IT'S USED EVERYWHERE. LUDWIG, NOVEMBER,1987.
C	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
c
	logical JGOOD(NSITES)
	real UU(NWSITE),VV(NWSITE)     
c
        COMMON /compnt/ U(NXGRD,NYGRD,NZGRD),V(NXGRD,NYGRD,NZGRD),z0,
     $        USIG(NSITES,NZGRD),VSIG(NSITES,NZGRD),UCOMP(NSITES),
     $        VCOMP(NSITES),IUGRAF(NXGRD,NYGRD,NHORIZ),Z10,zzero,
     $	      IVGRAF(NXGRD,NYGRD,NHORIZ),SPDMET(NXGRD,NYGRD,NZGRD),
     $        DIRMET(NXGRD,NYGRD,NZGRD),SPDCNV
        COMMON /LIMITS/ NCOL,NROW,NLVL,NCOLM1,NROWM1,
     $                  LOWIX(5,ntsite),LOWIY(5,ntsite)
        COMMON /NUMOBS/ NUMDOP,NUMNWS,NUMTOT,NUMTMP	 
	COMMON /STALOC/ XG(NSITES),YG(NSITES),ZG(NSITES),
     $                  WTDOP(NXGRD,NYGRD,NWSITE),JJJ,IDOP(NWSITE),
     $                  ITMP(NTSITE),WTEMP(NXGRD,NYGRD,NTSITE),JGOOD
C
	save
c
C  USE THE ONE WIND AVAILABLE EVERYWHERE
C
        IF (NUMDOP.EQ.1) THEN
           UU(1)=USIG(1,NLVL)
           VV(1)=VSIG(1,NLVL)
           DO 25 I = 1,NCOL
              DO 25 J = 1,NROW
                 U(I,J,NLVL)=UU(1)
                 V(I,J,NLVL)=VV(1)
25         CONTINUE
        ELSE
           DO 30 I=1,NUMDOP
              UU(I)=USIG(I,NLVL)
              VV(I)=VSIG(I,NLVL)
30         CONTINUE
c
	   DO 65 I = 1,NCOL
	      xx=float(i)
              DO 60 J = 1,NROW
	         yy=float(j)
	         sumdop=0.0
	         U(I,J,NLVL)=0.0
                 V(I,J,NLVL)=0.0
                 DO 50 IK=1,NUMDOP
c
c  following statement changed [ from IT=IDOP(JK) ] by fll 5/24/2000
c  bug found by doug miller of the naval postgraduate school.  it
c  appears to affect only results obtained when more than one
c  sounding is available.
c
                    IT=IDOP(IK)
	            dwate=WNDWT(XX,YY,XG(IT),YG(IT))
	            sumdop=sumdop+dwate
                    U(I,J,NLVL)=U(I,J,NLVL)+dwate*UU(IT)
                    V(I,J,NLVL)=V(I,J,NLVL)+dwate*VV(IT)
50               continue
	         if (sumdop.gt.0.0) then
                    U(I,J,NLVL)=U(I,J,NLVL)/sumdop
                    V(I,J,NLVL)=V(I,J,NLVL)/sumdop
		 end if
60            CONTINUE
65	   CONTINUE
        END IF
C
        RETURN
        END
C
C*********************************************************************
         SUBROUTINE TOPO
C*********************************************************************
C
C   READ  AND  COMPUTE TOPOGRAPHY AT GRID POINTS.  Get
C   FIRST GUESS FLOW SURFACES.  RESIG will BE USED LATER TO DEFINE THEM
C   IN TERMS OF THE CRITICAL STREAMLINE PARAMETERS. 
C	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
c
        COMMON/CSFC/ SFCHT(NXGRD,NYGRD),SIGMA(NZGRD),
     $               SFCLOW,SFCHI,ZRISE
        COMMON /FLOWER/ GRDHI,GRDLO,DZMAX(NTSITE,NZGRD),
     $                  RHS(NXGRD,NYGRD,NZGRD),AVTHK,SLFAC,
     $                  RHSLO(NTSITE,NZGRD),CMPRES,DPTMIN,NFLAT,
     $                  ZCHOOZ(NZGRD),levbot(NXGRD,NYGRD)
        COMMON /LIMITS/ NCOL,NROW,NLVL,NCOLM1,NROWM1,
     $                  LOWIX(5,ntsite),LOWIY(5,ntsite)
        COMMON /NUMOBS/ NUMDOP,NUMNWS,NUMTOT,NUMTMP
	COMMON/PARMS/ ZTOP,DS,NLVLM1,TDSI,TYM
C
c  MAKE SURE THAT  THINGS READ ON 1ST PASS ARE THERE ON LATER CALLS
c
        SAVE
C
C  READ TERRAIN HEIGHT VALUES AT GRID POINTS IN METERS, ALL GRIDS
C
	SFCLOW=1.E6
	SFCHI=-1.E6
C
C  IN DATA FILE, northern ROW IS FIRST.
C  READ HEIGHTS AT GRID POINTS & set initial boundary layer ht.
c  use terrainfollowing 1st; will reset in resig later.
c
7	CONTINUE
c		 
	DO 28 jy=nrow,1,-1
           READ (11,6001) (sfcht(i,jy),i=1,ncol)
           DO 18 ix=1,ncol
              IF(sfcht(ix,jy).LT.SFCLOW) SFCLOW=sfcht(ix,jy)
              IF(sfcht(ix,jy).GT. SFCHI) SFCHI =sfcht(ix,jy)
18	   CONTINUE
28	CONTINUE
c
	ZRISE=SFCHI-SFCLOW
	RELHT=AVTHK
C
C  GET HEIGHT of each 1st guess surface RELATIVE TO TERRAIN FOR EACH 
C  GRID PT. CHANGE IN HT RELATIVE TO THE LOWEST HT (MSL) IS ASSUMED
C  PROPORTIONAL TO SIGMA FOR THAT SFC.
C
	DO 67 jy=1,NROW
           DO 67 ix=1,NCOL
	      relsfc=(sfcht(ix,jy)-sfclow)/zrise
              DO 65 kz=1,NLVL
		 RHS (ix,jy,kz) = sigma(kz)*(avthk-relsfc*(1.0-slfac))
65         CONTINUE
67	CONTINUE
C
6001	Format(f3.0,107f4.0)
6011	format (10f10.1)
c
      	RETURN
      	END
c
C**********************************************************************
        SUBROUTINE LEVWND
C**********************************************************************
C
C  INTERPOLATES MASS ADJUSTED FIELD TO ANEMOMETER HEIGHT 
c  (Z10, FOR Z INDEX=1) & TO NHORIZ-1 FLAT PLANES (FOR Z INDICES 
c  2-NHORIZ) SET AT THE HS ZCHOOZ ABOVE THE LOWEST TERRRAIN GRID POINT
c  IN THE COARSE GRID, AS DEFINEDTHE VARIABLE ARRAY SIGMA.  THIS 
c  SUBROUTINE ALSO CONVERTS INTERPOLATED WINDS BACK TO METEOROLOGICAL 
c  SPEEDS AND ANGLES THAT CAN BE PLOTTED IF DESIRED.  
c
c	LUDWIG--JANUARY 1988
C
	PARAMETER(RAD2D=180./3.14159)
C	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
c
        COMMON /ANCHOR/ SLAT,SLNG,UTMAPX,UTMAPY,MDATE,
     $                  IMO,IHOUR,NEND,XORIG,YORIG
        COMMON /compnt/ U(NXGRD,NYGRD,NZGRD),V(NXGRD,NYGRD,NZGRD),z0,
     $        USIG(NSITES,NZGRD),VSIG(NSITES,NZGRD),UCOMP(NSITES),
     $        VCOMP(NSITES),IUGRAF(NXGRD,NYGRD,NHORIZ),Z10,zzero,
     $	      IVGRAF(NXGRD,NYGRD,NHORIZ),SPDMET(NXGRD,NYGRD,NZGRD),
     $        DIRMET(NXGRD,NYGRD,NZGRD),SPDCNV
        COMMON /CSFC/ SFCHT(NXGRD,NYGRD),SIGMA(NZGRD),
     $                SFCLOW,SFCHI,ZRISE
        COMMON /FLOWER/ GRDHI,GRDLO,DZMAX(NTSITE,NZGRD),
     $                  RHS(NXGRD,NYGRD,NZGRD),AVTHK,SLFAC,
     $                  RHSLO(NTSITE,NZGRD),CMPRES,DPTMIN,NFLAT,
     $                  ZCHOOZ(NZGRD),levbot(NXGRD,NYGRD)
	COMMON /LIMITS/ NCOL,NROW,NLVL,NCOLM1,NROWM1,
     $                  LOWIX(5,ntsite),LOWIY(5,ntsite)
C
C  MAKE SURE EVERYTHING IS STILL HERE ON SUBSEQUENT CALLS
C
	SAVE
c
	write (*,*) 'starting levwnd'
c
	z0=zzero
	z10 = 10.0
	DO 200 IX=1,NCOL
	   DO 200 IY=1,NROW
C
C  1ST GET LOCAL HT RELATIVE TO THE LOWEST PT IN THE DOMAIN.
C
	      ZSFC=SFCHT(IX,IY)
C
C  INTERPOLATE TO HORIZONTAL PLANES (iz>1).
C
	      DO 100 IZ=1,NHORIZ
C
C  GET HEIGHT OF HORIZONTAL PLANE ABOVE LOCAL GROUND LEVEL (ZAGL)
c  for 1st level use a terrain following surface, z10 m above sfc
c  everywhere.
C
	      if (iz .gt. 1) then
                 ZAGL=ZCHOOZ(IZ)-ZSFC
	      else
	         ZAGL=z10
	      end if
C
C  WIND = 0 BELOW GROUND.
C
	      IF (ZAGL .LE. Z0 .or. iz.gt.nflat) THEN
                 iUGRAF(IX,IY,IZ)=0
                 iVGRAF(IX,IY,IZ)=0
C
C  USE TOPMOST WIND (converted to integer cm/s) ABOVE DOMAIN.
C
	      ELSE IF (ZAGL .GE. RHS(IX,IY,NLVL)) THEN
c
                 iUGRAF(IX,IY,IZ)=nint(100.0*U(IX,IY,NLVL))
                 iVGRAF(IX,IY,IZ)=nint(100.0*V(IX,IY,NLVL))
C
C  LOG INTERPOLATE FOR OTHER CASES
C
	      ELSE
                 DO 75 LL=2,NLVL
                    IF (ZAGL.GE.RHS(IX,IY,LL-1) .AND.
     $                           ZAGL.LE.RHS(IX,IY,LL)) THEN
                       BOTTOM=RHS(IX,IY,LL-1)
                       TOP=RHS(IX,IY,LL)
                       IF (BOTTOM .LT. Z0) THEN
                          BOTTOM=Z0
                          U0=ZERO
                          V0=ZERO
                       ELSE
                          U0=U(IX,IY,LL-1)
                          V0=V(IX,IY,LL-1)
                       END IF
                       U1=U(IX,IY,LL)
                       V1=V(IX,IY,LL)
	               CALL LGNTRP(uuuf,BOTTOM,TOP,ZAGL,U0,U1)
	               CALL LGNTRP(vvvf,BOTTOM,TOP,ZAGL,V0,V1)
c
c  winds (m/s) are converted to integer cm/s for wrinting files
c
	               iUGRAF(IX,IY,IZ)=nint(100.0*uuuf)
	               iVGRAF(IX,IY,IZ)=nint(100.0*vvvf)
                    END IF
75	         CONTINUE
              END IF
C
100	   CONTINUE
C
200	CONTINUE
C
C  CONVERT COMPONENTS IN HORIZONTAL PLANES TO METEOROLOGICAL WINDS
c
	DO 300 LEV=1,nflat
	DO 300 IX=1,NCOL
	DO 300 IY=1,NROW
C
c   convert integer cm/s back to real m/s
c
	   UG=0.01*float(iUGRAF(IX,IY,LEV))
	   VG=0.01*float(iVGRAF(IX,IY,LEV))
C
	   IF (UG.EQ.ZERO.AND.VG.EQ.ZERO) THEN
              SPDMET(IX,IY,LEV)=ZERO
              DIRMET(IX,IY,LEV)=ZERO
	   ELSE
	      SPDMET(IX,IY,LEV)=sp(UG,VG)
	      DIRMET(IX,IY,LEV)=dd(UG,VG)
	   END IF
300	CONTINUE
C
	RETURN
	END
C
c**************************************************************
	real function sp(uu,vv)
c**************************************************************
c
c  function to get speed from components
c
	sp=sqrt(uu*uu+vv*vv)
c
	return
	end
c**********************************************************
	real function dd(uu,vv)
c*********************************************************
c
c  function to get direction (degrees) from components
c
	parameter (rad2d=180./3.14159)
c
	if (sp(uu,vv).gt.0.0) then
	   dd=amod(540.+ rad2d*atan2(uu,vv),360.)
	else
	   dd=0.0
	end if
c
	return
	end
c
c**********************************************************
	subroutine cateob(numnws,xg,yg,uu,vv)
c**********************************************************
c
c  writes observed winds to windsuv.dat & windsuv.query for plotting
c  & querying. a 999 fill except for the grid point nearest an obs
c  (windsuv.dat) and a a 5 by 5 array centered on the grid point 
c  nearest an observation (windsuv.query).
c  
c  
c   fludwig 6/2005
c
c	
	PARAMETER (NXGRD=108,NYGRD=123,NZGRD=16)
	PARAMETER (NSITES=200,NWSITE=5,NTSITE=5)
	PARAMETER (NSNDHT=30,NHORIZ=10,NARRAY=NTSITE*NZGRD)
c
	real xg(NSITES),yg(NSITES),uu(NSITES),vv(NSITES)
	integer iu(NXGRD,NYGRD), iv(NXGRD,NYGRD)
c
	common /buggsy/ nskip
c
	save
c
c  fill arrays with 999
c
	nine3=999
	call setint (nine3,iu,NXGRD,NYGRD)
	call setint (nine3,iv,NXGRD,NYGRD)
c
c  first do plotting file (windsuv.dat)
c
	do 10 is=1,numnws
	   ix=nint(xg(is))
	   iy=nint(yg(is))
c
c  check to see that the pt is on the grid.  if so convert
c  to deciknots and put the component values in (and around)
c  the point.  maximum component value for plotting is 58 kts
c

	   if (ix.le.NXGRD .and. iy.le.NYGRD 
     $              .and. ix.ge.1 .and. iy.ge.1) then
	      jju=nint(19.4*uu(is))
	      if (jju.gt.582) jju=582
	      jjv=nint(19.4*vv(is))
	      if (jjv.gt.582) jjv=582
	      iu(ix,iy)=jju
	      iv(ix,iy)=jjv
	   end if
10	continue
c
c  arrays filled, now write results
c
	OPEN (20,FILE='windsuv.dat',STATUS='UNKNOWN',form='formatted')
c
	do 12 ky=NYGRD,1,-nskip
	   WRITE(20,6001)(iu(kx,ky),kx=1,NXGRD,nskip)
12	continue
	do 14 ky=NYGRD,1,-nskip
	   WRITE(20,6001)(iv(kx,ky),kx=1,NXGRD,nskip)
14	continue
	close (20)
c
c  now do file for querying
c
	do 20 is=1,numnws
	   ix=nint(xg(is))
	   iy=nint(yg(is))
c
c  check to see that the pt is on the grid.  if so convert
c  to deciknots and put the component values in (and around)
c  the point.  maximum component value for plotting is 58 kts
c
	   if (ix.le.NXGRD .and. iy.le.NYGRD 
     $              .and. ix.ge.1 .and. iy.ge.1) then
	      jju=nint(19.4*uu(is))
	      if (jju.gt.582) jju=582
	      jjv=nint(19.4*vv(is))
	      if (jjv.gt.582) jjv=582
	      do 18 jx=max(1,ix-1),min(ix+1,NXGRD)
	         do 16 jy=max(1,iy-1),min(iy+1,NYGRD)
	            iu(jx,jy)=jju
	            iv(jx,jy)=jjv
16	         continue
18	      continue
	   end if
20	continue
c
c  arrays filled, now write results
c
 	OPEN (20, FILE='windsuv.query', STATUS='UNKNOWN',
     $                                  form='formatted')
c
	do 22 ky=NYGRD,1,-nskip
	   WRITE(20,6001)(iu(kx,ky),kx=1,NXGRD,nskip)
22	continue
	do 24 ky=NYGRD,1,-nskip
	   WRITE(20,6001)(iv(kx,ky),kx=1,NXGRD,nskip)
24	continue
	close (20)
c
6001	FORMAT(1X,150I5)
	return
c
	end
c