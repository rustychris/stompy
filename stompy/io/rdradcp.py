""" 
A mostly direct translation of rdradcp.m to python.
1/3/2013: Updated with DKR changes to rdradcp.m

"""

from __future__ import division
from __future__ import print_function

#  cruft that 2to3 added - but seems to make less compatible with older code
#  and unclear that it's needed for py3k.
# from builtins import range
# from builtins import object
# from past.utils import old_div

import sys,os,re,math
from numpy import *
import numpy as np
import scipy.stats.stats as sp
import scipy.stats.morestats as ssm
from matplotlib.dates import date2num,num2date
import datetime

from . import nmea

cfac=180.0/(2**31)

def msg_print(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def get_ens_dtype(sourceprog = 'WINRIVER'):
    
    ens_dtype = [('mtime',float64),
                 ('number',int32),
                 ('pitch',float64), ('roll',float64), ('heading',float64),
                 ('pitch_std',float64), ('roll_std',float64), ('heading_std',float64),
                 ('depth',float64),
                 ('temperature',float64),
                 ('salinity',float64),
                 ('pressure',float64),('pressure_std',float64),
                 ('bt_mode',float64),
                 ('bt_range',(float64,4)),
                 ('bt_vel',(float64,4)),
                 ('bt_corr',(float64,4)),
                 ('bt_ampl',(float64,4)),
                 ('bt_perc_good',(float64,4))
                 ]

    if sourceprog  == 'WINRIVER':
    # if cfg.sourceprog  in ['WINRIVER']:
    # if cfg.sourceprog  in ('WINRIVER',):
        ens_dtype += [('nav_mtime',float64),
                      ('nav_longitude',float64),
                      ('nav_latitude',float64)]
    elif sourceprog == 'VMDAS':
        ens_dtype += [('nav_smtime',float64),
                      ('nav_emtime',float64),
                      ('nav_slongitude',float64),
                      ('nav_elongitude',float64),
                      ('nav_slatitude',float64),
                      ('nav_elatitude',float64),
                      ('nav_mtime',float64)]
    else:
        pass  
    return ens_dtype
    
def get_bin_dtype():

    # things of the shape [n_cells,n]
    return      [('east_vel',float64),
                 ('north_vel',float64),
                 ('vert_vel',float64),
                 ('error_vel',float64),
                 ('corr',(float64,4)),
                 ('status',(float64,4)),
                 ('intens',(float64,4)),
                 ('perc_good',(float64,4))
                 ]


class Adcp(object):
    pass

#function [adcp,cfg,ens,hdr]=rdradcp(name,varargin);
# 
def rdradcp(name,
            num_av=5,
            nens=-1, # or [start,stop] as 1-based, inclusive
            baseyear=2000,
            despike='no',
            log_fp=None):
    """
    The original documentation from Rich Pawlowicz's code:

    RDRADCP  Read (raw binary) RDI ADCP files, 
    ADCP=RDRADCP(NAME) reads the raw binary RDI BB/Workhorse ADCP file NAME and
    puts all the relevant configuration and measured data into a data structure 
    ADCP (which is self-explanatory). This program is designed for handling data
    recorded by moored instruments (primarily Workhorse-type but can also read
    Broadband) and then downloaded post-deployment. For vessel-mount data I
    usually make p-files (which integrate nav info and do coordinate transformations)
    and then use RDPADCP. 
    
    This current version does have some handling of VMDAS, WINRIVER, and WINRIVER2 output
    files, but it is still 'beta'. There are (inadequately documented) timestamps
    of various kinds from VMDAS, for example, and caveat emptor on WINRIVER2 NMEA data.
    
    (ADCP,CFG)=RDRADCP(...) returns configuration data in a
    separate data structure.

    Various options can be specified on input:
    (..)=RDRADCP(NAME,NUMAV) averages NUMAV ensembles together in the result.
    (..)=RDRADCP(NAME,NUMAV,NENS) reads only NENS ensembles (-1 for all).
    (..)=RDRADCP(NAME,NUMAV,(NFIRST NEND)) reads only the specified range
    of ensembles. This is useful if you want to get rid of bad data before/after
    the deployment period.
    
    Notes:
    - sometimes the ends of files are filled with garbage. In this case you may
      have to rerun things explicitly specifying how many records to read (or the
      last record to read). I don't handle bad data very well. Also - in Aug/2007
      I discovered that WINRIVER-2 files can have a varying number of bytes per
      ensemble. Thus the estimated number of ensembles in a file (based on the
      length of the first ensemble and file size) can be too high or too low.

    - I don't read in absolutely every parameter stored in the binaries;
      just the ones that are 'most' useful. Look through the code if
      you want to get other things.

    - chaining of files does not occur (i.e. read .000, .001, etc.). Sometimes
      a ping is split between the end of one file and the beginning of another.
      The only way to get this data is to concatentate the files, using
      cat file1.000 file1.001 > file1   (unix)
      copy file1.000/B+file2.001/B file3.000/B     (DOS/Windows)

      (as of Dec 2005 we can probably read a .001 file)

    - velocity fields are always called east/north/vertical/error for all
      coordinate systems even though they should be treated as
      1/2/3/4 in beam coordinates etc.

    String parameter/option pairs can be added after these initial parameters:

    'baseyear': Base century for BB/v8WH firmware (default to 2000).

    'despike': 'no' | 'yes' | 3-element vector 
    
    Controls ensemble averaging. With 'no' a simple mean is used 
    (default). With 'yes' a mean is applied to all values that fall 
    within a window around the median (giving some outlier rejection). 
    This is useful for noisy data. Window sizes are [.3 .3 .3] m/s 
    for [ horiz_vel vert_vel error_vel ] values. If you want to 
    change these values, set 'despike' to the 3-element vector.

    R. Pawlowicz (rich@eos.ubc.ca) - 17/09/99

    R. Pawlowicz - 17/Oct/99 
    5/july/00 - handled byte offsets (and mysterious 'extra" bytes) slightly better, Y2K

    5/Oct/00 - bug fix - size of ens stayed 2 when NUMAV==1 due to initialization,
    hopefully this is now fixed.

    10/Mar/02 - #bytes per record changes mysteriously,
    tried a more robust workaround. Guess that we have an extra
    2 bytes if the record length is even?

    28/Mar/02 - added more firmware-dependent changes to format; hopefully this
    works for everything now (put previous changes on firmer footing?)

    30/Mar/02 - made cfg output more intuitive by decoding things.
    An early version of WAVESMON and PARSE which split out this
    data from a wave recorder inserted an extra two bytes per record.
    I have removed the code to handle this but if you need it see line 509

    29/Nov/02  - A change in the bottom-track block for version 4.05 (very old!).

    29/Jan/03  - Status block in v4.25 150khzBB two bytes short?

    14/Oct/03  - Added code to at least 'ignore' WinRiver GPS blocks.

    11/Nov/03  - VMDAS navigation block, added hooks to output
    navigation data.

    26/Mar/04
    - better decoding of nav blocks
    - better handling of weird bytes at beginning and end of file
    - (code fixes due to Matt Drennan).

    25/Aug/04  - fixes to "junk bytes" handling.

    27/Jan/05  - even more fixed to junk byte handling (move 1 byte at a time rather than
    two for odd lengths.

    29/Sep/2005 - median windowing done slightly incorrectly in a way which biases
    results in a negative way in data is *very* noisy. Now fixed.

    28/Dc/2005  - redid code for recovering from ensembles that mysteriously change length, added
    'checkheader' to make a complete check of ensembles.

    Feb/2006  - handling of firmware version 9 (navigator)

    23/Aug/2006 - more firmware updates (16.27)

    23/Aug2006  - ouput some bt QC stiff

    29/Oct/2006 - winriver bottom track block had errors in it - now fixed.

    30/Oct/2006 - pitch_std, roll_std now uint8 and not int8 (thanks Felipe pimenta)

    13/Aug/2007 - added Rio Grande (firmware v 10), 
    better handling of those cursed winriver ASCII NMEA blocks whose
    lengths change unpredictably.
    skipping the inadequately documented 2022 WINRIVER-2 NMEA block

    13/Mar/2010 - firmware version 50 for WH.
   
    31/Aug/2012 - Rusty Holleman / RMA - ported to python
    
    Python port details:

    log_fp: a file-like object - the message are the same as in the matlab code,
    but this allows them to be redirected elsewhere.
    """
    if log_fp is None:
        log_fp = sys.stdout
    def msg(s):
        log_fp.write(s)
        log_fp.flush()
        
    century=baseyear  # ADCP clock does not have century prior to firmware 16.05.
    vels=despike   # Default to simple averaging


    # Check file information first
    if not os.path.exists(name):
        msg("ERROR******* Can't find file %s\n"%name)
        return None

    msg("\nOpening file %s\n\n"%name)
    fd=open(name,'rb') # NB: no support at the file level for 'ieee-le'

    # Read first ensemble to initialize parameters

    [ens,hdr,cfg,pos]=rd_buffer(fd,-2,msg) # Initialize and read first two records
    if ens is None: #  ~isstruct(ens) & ens==-1,
        msg("No Valid data found\n")
        return None

    fd.seek(pos)              # Rewind

    if (cfg.prog_ver<16.05 and cfg.prog_ver>5.999) or cfg.prog_ver<5.55:
        msg("***** Assuming that the century begins year %d (info not in this firmware version)\n"%century)
    elif cfg.prog_ver>23.18 and cfg.prog_ver<23.20:
        msg("***** Maybe this is an ocean surveyor, and century is 2000\n")
        century=2000
    else:
        century=0  # century included in clock.  

    def ensemble_dates(ensx):
        """ helper routine to extract dates from the given ensemble, return 
        as an array of datenums
        """
        # handle hours, minutes, seconds, 100ths manually, but date with date2num
        dats = [date2num(datetime.date(int(century+ensx.rtc[i,0]),
                                       int(ensx.rtc[i,1]),
                                       int(ensx.rtc[i,2]))) \
                    + sum( ensx.rtc[i,3:7] * [1./24, 1./(24*60), 1./86400, 1./8640000 ])

                for i in range(len(ensx.rtc))]
        dats = array(dats)
        return dats

    dats = ensemble_dates(ens)

    t_int=diff(dats)[0]
    msg( "Record begins at %s\n"%( num2date(dats[0]).strftime('%c') ))
    msg( "Ping interval appears to be %ss\n\n"%( 86400*t_int ))

    # Estimate number of records (since I don't feel like handling EOFs correctly,
    # we just don't read that far!)


    # Now, this is a puzzle - it appears that this is not necessary in
    # a firmware v16.12 sent to me, and I can't find any example for
    # which it *is* necessary so I'm not sure why its there. It could be
    # a leftoever from dealing with the bad WAVESMON/PARSE problem (now
    # fixed) that inserted extra bytes.
    # ...So its out for now.
    #if cfg.prog_ver>=16.05, extrabytes=2 else extrabytes=0 end # Extra bytes
    extrabytes=0

    naminfo = os.stat(name)
    nensinfile=int(naminfo.st_size/(hdr.nbyte+2+extrabytes))
    msg("\nEstimating %d ensembles in this file\n"%nensinfile)

    # [python] nens, if a sequence, is taken to be 1-based, inclusive indices.
    # This is counter to the normal python interpretation, but instead 
    # consistent with the original matlab.
    if isinstance(nens,int) or isinstance(nens,integer):
        if nens==-1:
            nens=nensinfile
        msg("   Reading %d ensembles, reducing by a factor of %d\n"%(nens,num_av) )
    else:
        msg("   Reading ensembles %d-%d, reducing by a factor of %d\n"%(nens[0],nens[1],num_av) )
        fd.seek((hdr.nbyte+2+extrabytes)*(nens[0]-1),os.SEEK_CUR)
        nens=nens[1] - nens[0] + 1

    # Number of records after averaging.

    n=int(nens/num_av)
    msg("Final result %d values\n"%n)

    if num_av>1:
        if type(vels) == str:
           msg("\n Simple mean used for ensemble averaging\n")
        else:
           msg("\n Averaging after outlier rejection with parameters %s\n"%vels)



    # Structure to hold all ADCP data 
    # Note that I am not storing all the data contained in the raw binary file, merely
    # things I think are useful.

    # types for the data arrays - first, the fields common to all sourceprog:

    adcp = Adcp()
    adcp.name = 'adcp'
    adcp.config=cfg

    # things of the shape [1,n]
    ens_dtype = get_ens_dtype(cfg.sourceprog)
    # things of the shape [n_cells,n]
    bin_dtype = get_bin_dtype()

    # NB: we may not actually have n ensembles - don't know until
    # the whole file is actually read - so at the end of this function
    # these arrays may get truncated
    adcp.ensemble_data = zeros(n,dtype=ens_dtype)
    adcp.bin_data = zeros((n,cfg.n_cells), dtype=bin_dtype)

   
        
    # Calibration factors for backscatter data

    # Loop for all records
    ens = None # force it to reinitialize
    
    for k in range(n): # [python] k switched to zero-based
        # Gives display so you know something is going on...
  
        if k%50==0:
            msg("%d\n"%(k*num_av))
        msg(".") 
  
        # Read an ensemble
  
        [ens,hdr,cfg1,pos]=rd_buffer(fd,num_av,msg)
  
        if ens is None: # ~isstruct(ens), # If aborting...
            msg("Only %d records found..suggest re-running RDRADCP using this parameter\n"%( (k-1)*num_av ))
            msg("(If this message preceded by a POSSIBLE PROGRAM PROBLEM message, re-run using %d)\n"%( (k-1)*num_av-1))
            n = k
            break

        dats = ensemble_dates(ens)

        adcp.ensemble_data['mtime'][k]       =median(dats)  
        adcp.ensemble_data['number'][k]      =ens.number[0]
        adcp.ensemble_data['heading'][k]     =ssm.circmean(ens.heading*pi/180.)*180/pi
        adcp.ensemble_data['pitch'][k]       =mean(ens.pitch)
        adcp.ensemble_data['roll'][k]        =mean(ens.roll)
        adcp.ensemble_data['heading_std'][k] =mean(ens.heading_std)
        adcp.ensemble_data['pitch_std'][k]   =mean(ens.pitch_std)
        adcp.ensemble_data['roll_std'][k]    =mean(ens.roll_std)
        adcp.ensemble_data['depth'][k]       =mean(ens.depth)
        adcp.ensemble_data['temperature'][k] =mean(ens.temperature)
        adcp.ensemble_data['salinity'][k]    =mean(ens.salinity)
        adcp.ensemble_data['pressure'][k]    =mean(ens.pressure)
        adcp.ensemble_data['pressure_std'][k]=mean(ens.pressure_std)

        # [python] - order of indices for bin data is opposite matlab -
        #   adcp.east_vel[ ensemble index, bin_index ] 
        if type(vels) == str:
            adcp.bin_data['east_vel'][k,:]    =nmean(ens.east_vel ,0) # [python] axis changed to 0-based, and switched!
            adcp.bin_data['north_vel'][k,:]   =nmean(ens.north_vel,0) # assume ens.east_vel[sample_index,bin_index]
            adcp.bin_data['vert_vel'][k,:]    =nmean(ens.vert_vel ,0)
            adcp.bin_data['error_vel'][k,:]   =nmean(ens.error_vel,0)
        else:
            adcp.bin_data['east_vel'][k,:]    =nmedian(ens.east_vel  ,vels[0],0)
            adcp.bin_data['north_vel'][k,:]   =nmedian(ens.north_vel ,vels[0],0)
            adcp.bin_data['vert_vel'][k,:]    =nmedian(ens.vert_vel  ,vels[1],0)
            adcp.bin_data['error_vel'][k,:]   =nmedian(ens.error_vel ,vels[2],0)
          
        # per-beam, per bin data - 
        # adcp.corr[ensemble index, bin_index, beam_index ]
        adcp.bin_data['corr'][k,:,:]        =nmean(ens.corr,0)        # added correlation RKD 9/00
        adcp.bin_data['status'][k,:,:]	=nmean(ens.status,0)   
  
        adcp.bin_data['intens'][k,:,:]     =nmean(ens.intens,0)
        adcp.bin_data['perc_good'][k,:,:]  =nmean(ens.percent,0)  # felipe pimenta aug. 2006
  
        adcp.ensemble_data['bt_range'][k,:]   =nmean(ens.bt_range,0)
        adcp.ensemble_data['bt_mode'][k]   = nmedian(ens.bt_mode)
        adcp.ensemble_data['bt_vel'][k,:]     =nmean(ens.bt_vel,0)

        adcp.ensemble_data['bt_corr'][k,:]=nmean(ens.bt_corr,0)          # felipe pimenta aug. 2006
        adcp.ensemble_data['bt_ampl'][k,:]=nmean(ens.bt_ampl,0)          #  "
        adcp.ensemble_data['bt_perc_good'][k,:]=nmean(ens.bt_perc_good,0)#  " 
  
        if cfg.sourceprog == 'WINRIVER':
            #if cfg.sourceprog in ('instrument','WINRIVER'):
            adcp.ensemble_data['nav_mtime'][k]=nmean(ens.smtime)
            # these are sometimes nan - and note that nmean
            # modifies the input, so it looks like it should
            adcp.ensemble_data['nav_longitude'][k]=nmean(ens.slongitude)
            adcp.ensemble_data['nav_latitude'][k]=nmean(ens.slatitude)  
            # DBG
            #print "nmean(%s) => %s"%(ens.slongitude,adcp.ensemble_data['nav_longitude'][k])            
            #print "nmean(%s) => %s"%(ens.slatitude,adcp.ensemble_data['nav_latitude'][k])            
            
            # out of curiosity, does this ever happen??
            #if cfg.sourceprog=='instrument' and isfinite(adcp.nav_latitude[k]) and adcp.nav_latitude[k]!=0:
            #    print "##################### instrument has some data ###################"
        elif cfg.sourceprog == 'VMDAS':
            adcp.ensemble_data['nav_smtime'][k]   =ens.smtime[0]
            adcp.ensemble_data['nav_emtime'][k]   =ens.emtime[0]
            adcp.ensemble_data['nav_slatitude'][k]=ens.slatitude[0]
            adcp.ensemble_data['nav_elatitude'][k]=ens.elatitude[0]
            adcp.ensemble_data['nav_slongitude'][k]=ens.slongitude[0]
            adcp.ensemble_data['nav_elongitude'][k]=ens.elongitude[0]
            adcp.ensemble_data['nav_mtime'][k]=nmean(ens.nmtime)
        ##

  
    msg("\nRead to byte %d in a file of size %d bytes\n"%( fd.tell(),naminfo.st_size ) )
    if fd.tell()+hdr.nbyte<naminfo.st_size:
        msg("-->There may be another %d ensembles unread\n" % int((naminfo.st_size-fd.tell())/(hdr.nbyte+2)) )
        
    fd.close()
    
    if n < len(adcp.ensemble_data):
        msg("Truncating data to the valid set of records\n")
        adcp.ensemble_data = adcp.ensemble_data[:n]
        adcp.bin_data = adcp.bin_data[:n]

    # RH: invalidate bad bottom track
    bt_invalid = adcp.ensemble_data['bt_vel'][:,0]==-32768
    adcp.ensemble_data['bt_vel'][bt_invalid]=np.nan

    # and make the fields show up more like the matlab code:
    for name,typ in ens_dtype:
        setattr(adcp,name,adcp.ensemble_data[name])
    for name,typ in bin_dtype:
        setattr(adcp,name,adcp.bin_data[name])

    # and normalize the latitude/longitude naming:
    adcp.latitude = None
    adcp.longitude = None

    if cfg:
        if cfg.sourceprog == 'VMDAS':
            # punting on which lat/lon fields to reference
            msg("VMDAS input - assuming nav_s* fields are better than nav_e*\n")
            adcp.latitude = adcp.nav_slatitude
            adcp.longitude = adcp.nav_slongitude        
            # the arrays are there, but the values aren't set yet
            #print("adcp lat/lon %f %f\n"%(adcp.latitude,adcp.longitude))
        elif cfg.sourceprog in ('WINRIVER'):
            adcp.latitude = adcp.nav_latitude
            adcp.longitude = adcp.nav_longitude
            # too early to print 
            #print("adcp lat/lon %f %f\n"%(adcp.latitude[0],adcp.longitude[0]))
        

    return adcp

    


#----------------------------------------
#function valid=checkheader(fd)
def checkheader(fd):
    """ Given an open file object, read the ensemble size, skip
    ahead, make sure we can read the cfg bytes of the *next* 
    ensemble, come back to the starting place, and report success.
    """
    valid=0
    starting_pos=fd.tell()
    try:
        # have to use the file descriptor version, since we're just getting
        # the file object, not a filename
        # info = os.fstat(fd.fileno())
        numbytes=fromfile(fd,int16,1)          # Following the header bytes is numbytes               
        if len(numbytes) and numbytes[0]>0:                                         # and we move forward numbytes>0
            fd.seek(numbytes[0]-2,os.SEEK_CUR) 
            cfgid=fromfile(fd,uint8,2) # while return [] if hit EOF
            if len(cfgid)==2:                # will Skip the last ensemble (sloppy code)
                # fprintf([dec2hex(cfgid(1)) ' ' dec2hex(cfgid(2)) '\n'])          
                if cfgid[0]==0x7F and cfgid[1]==0x7F:          # and we have *another* 7F7F
                    valid=1                                                    # ...ONLY THEN it is valid.   
    finally:
        fd.seek(starting_pos)
    return valid
     

#-------------------------------------
# function [hdr,pos]=rd_hdr(fd)
def rd_hdr(fd,msg=msg_print):
    # Read config data
    # Changed by Matt Brennan to skip weird stuff at BOF (apparently
    # can happen when switching from one flash card to another
    # in moored ADCPs).
    # on failure, return hdr=None

    cfgid=fromfile(fd,uint8,2)
    nread=0
    # departure from matlab code - check to see if cfgid itself was
    # truncated at EOF
    while len(cfgid)<2 or (cfgid[0] != 0x7F or cfgid[1]!=0x7F) or not checkheader(fd):
        nextbyte=fromfile(fd,uint8,1)
        pos=fd.tell()
        nread+=1
        if len(nextbyte)==0:  # End of file
            msg('EOF reached before finding valid cfgid\n')
            hdr=None
            return hdr,pos
        # seems backwards, but they're the same value - 0x7F
        cfgid[1],cfgid[0] = cfgid[0],nextbyte[0]

        if pos % 1000==0:
            msg("Still looking for valid cfgid at file position %d...\n"%pos)
        #end
    #end 

    pos=fd.tell()-2
    if nread>0:
        msg("Junk found at BOF...skipping %d bytes until\n"%nread )
        msg("cfgid=%x %x  at file pos %d\n"%(cfgid[0],cfgid[1],pos))
    #end

    hdr=rd_hdrseg(fd)
    return hdr,pos

#-------------------------------------
#function cfg=rd_fix(fd)
def rd_fix(fd,msg=msg_print):
    # Read config data
    cfgid=fromfile(fd,uint16,1)
    if len(cfgid) == 0:
        msg("WARNING: ran into end of file reading Fixed header ID\n")
    elif cfgid[0] != 0: # 0x0000
        msg("WARNING: Fixed header ID %x incorrect - data corrupted or not a BB/WH raw file?\n"%cfgid[0])
    #end 

    cfg,nbytes=rd_fixseg(fd)
    return cfg


#--------------------------------------
#function [hdr,nbyte]=rd_hdrseg(fd)
class Header(object):
    pass

def rd_hdrseg(fd):
    # Reads a Header
    hdr = Header()
    hdr.nbyte          =fromfile(fd,int16,1)[0]
    fd.seek(1,os.SEEK_CUR)
    ndat=fromfile(fd,int8,1)[0]
    hdr.dat_offsets    =fromfile(fd,int16,ndat)
    nbyte=4+ndat*2
    return hdr,nbyte

#-------------------------------------
#function opt=getopt(val,varargin)
def getopt(val,*args):
    # Returns one of a list (0=first in varargin, etc.)
    val = int(val) # in case it's a boolean
    if val>=len(args):
        return 'unknown'
    else:
        return args[val]
   			
#
#-------------------------------------
# function [cfg,nbyte]=rd_fixseg(fd)
class Config(object):
    pass

def rd_fixseg(fd):
    """ returns Config, nbyte
    Reads the configuration data from the fixed leader
    """

    ##disp(fread(fd,10,'uint8'))
    ##fseek(fd,-10,os.SEEK_CUR)
    cfg = Config()
    cfg.name='wh-adcp'
    cfg.sourceprog='instrument'  # default - depending on what data blocks are
                                  # around we can modify this later in rd_buffer.
    cfg.prog_ver       =fromfile(fd,uint8,1)[0]+fromfile(fd,uint8,1)/100.0

    # 8,9,16 - WH navigator
    # 10 -rio grande
    # 15, 17 - NB
    # 19 - REMUS, or customer specific
    # 11- H-ADCP
    # 31 - Streampro
    # 34 - NEMO
    # 50 - WH, no bottom track (built on 16.31)
    # 51 - WH, w/ bottom track
    # 52 - WH, mariner

    if int(cfg.prog_ver) in (4,5):
        cfg.name='bb-adcp'
    elif int(cfg.prog_ver) in (8,9,10,16,50,51,52):
        cfg.name='wh-adcp'
    elif int(cfg.prog_ver) in (14,23):  # phase 1 and phase 2
        cfg.name='os-adcp'
    else:
        cfg.name='unrecognized firmware version'       
    #end    

    config         =fromfile(fd,uint8,2)  # Coded stuff
    cfg.config          ="%2o-%2o"%(config[1],config[0])
    cfg.beam_angle     =getopt(config[1]&3,15,20,30)
    # RCH: int is needed here because the equality evaluates to a boolean, but
    # getopt expects an integer which can be used to index the list of arguments.
    # in the expression above, config[1]&3 evaluates to an integer
    cfg.numbeams       =getopt( config[1]&16==16,4,5)
    cfg.beam_freq      =getopt(config[0]&7,75,150,300,600,1200,2400,38)
    cfg.beam_pattern   =getopt(config[0]&8==8,'concave','convex') # 1=convex,0=concave
    cfg.orientation    =getopt(config[0]&128==128,'down','up')    # 1=up,0=down

    ## HERE - 
    # 8/31/12: code above here has been translated to python 
    #  code below is still matlab.
    # a few notes on the translation:
    #   fread(fd,count,'type') => fromfile(fd,type,count)
    #     note that fromfile always returns an array - so when count==1, you 
    #     may want fromfile(...)[0] to get back a scalar value
    #   returns are sneaky - since matlab defines the return values at the beginning
    #     but python must explicitly specify return values at each return statement
    #   to get something like a matlab struct, declare an empty class (see Config above)
    #   then you can do things like cfg.simflag = 123

    # RCH: fromfile returns a list, so index it with [0] to get an int
    cfg.simflag        =getopt(fromfile(fd,uint8,1)[0],'real','simulated') # Flag for simulated data

    fd.seek(1,os.SEEK_CUR) # fseek(fd,1,'cof') 

    cfg.n_beams        =fromfile(fd,uint8,1)[0]
    cfg.n_cells        =fromfile(fd,uint8,1)[0]
    cfg.pings_per_ensemble=fromfile(fd,uint16,1)[0]
    cfg.cell_size      =fromfile(fd,uint16,1)[0]*.01	 # meters
    cfg.blank          =fromfile(fd,uint16,1)[0]*.01	 # meters
    cfg.prof_mode      =fromfile(fd,uint8,1)[0]
    cfg.corr_threshold =fromfile(fd,uint8,1)[0]
    cfg.n_codereps     =fromfile(fd,uint8,1)[0]
    cfg.min_pgood      =fromfile(fd,uint8,1)[0]
    cfg.evel_threshold =fromfile(fd,uint16,1)[0]

    cfg.time_between_ping_groups = sum( fromfile(fd,uint8,3) * array([60, 1, .01]) ) # seconds

    coord_sys      =fromfile(fd,uint8,1)[0]                                # Lots of bit-mapped info
    cfg.coord="%2o"%coord_sys
    # just like C...
    cfg.coord_sys      =getopt( (coord_sys >> 3)&3,'beam','instrument','ship','earth')
    # RCH: need into since it's an equality comparison which gives a boolean
    cfg.use_pitchroll  =getopt(coord_sys&4==4,'no','yes')  
    cfg.use_3beam      =getopt(coord_sys&2==2,'no','yes')
    cfg.bin_mapping    =getopt(coord_sys&1==1,'no','yes')

    cfg.xducer_misalign=fromfile(fd,int16,1)[0]*.01    # degrees
    cfg.magnetic_var   =fromfile(fd,int16,1)[0]*.01	# degrees
    cfg.sensors_src    ="%2o"%(fromfile(fd,uint8,1)[0])
    cfg.sensors_avail  ="%2o"%(fromfile(fd,uint8,1)[0])
    cfg.bin1_dist      =fromfile(fd,uint16,1)[0]*.01	# meters
    cfg.xmit_pulse     =fromfile(fd,uint16,1)[0]*.01	# meters
    cfg.water_ref_cells=fromfile(fd,uint8,2)
    cfg.fls_target_threshold =fromfile(fd,uint8,1)[0]
    fd.seek(1,os.SEEK_CUR) # fseek(fd,1,'cof')
    cfg.xmit_lag       =fromfile(fd,uint16,1)[0]*.01 # meters
    nbyte=40

    if int(cfg.prog_ver) in (8,10,16,50,51,52):

        if cfg.prog_ver>=8.14:  # Added CPU serial number with v8.14
            cfg.serialnum      =fromfile(fd,uint8,8)
            nbyte+=8 
        #end

        if cfg.prog_ver>=8.24:  # Added 2 more :w  bytes with v8.24 firmware
            cfg.sysbandwidth  =fromfile(fd,uint8,2)
            nbyte+=2
        #end

        if cfg.prog_ver>=16.05:                      # Added 1 more bytes with v16.05 firmware
            cfg.syspower      =fromfile(fd,uint8,1)[0]
            nbyte+=1
        #end

        if cfg.prog_ver>=16.27:   # Added bytes for REMUS, navigators, and HADCP
            cfg.navigator_basefreqindex=fromfile(fd,uint8,1)[0]
            nbyte+=1
            cfg.remus_serialnum=fromfile(fd,uint8,4)
            nbyte+=4
            cfg.h_adcp_beam_angle=fromfile(fd,uint8,1)[0]
            nbyte+=1
        #end
    elif int(cfg.prog_ver)==9:

        if cfg.prog_ver>=9.10:  # Added CPU serial number with v8.14
            cfg.serialnum      =fromfile(fd,uint8,8)
            nbyte+=8 
            cfg.sysbandwidth  =fromfile(fd,uint8,2)
            nbyte+=2
        end

    elif int(cfg.prog_ver) in (14,16):

        cfg.serialnum      =fromfile(fd,uint8,8)  # 8 bytes 'reserved'
        nbyte+=8

    # It is useful to have this precomputed.
    cfg.ranges=cfg.bin1_dist+arange(cfg.n_cells)*cfg.cell_size
    if cfg.orientation==1:
        cfg.ranges *= -1

    return cfg,nbyte
	
#-----------------------------
#function [ens,hdr,cfg,pos]=rd_buffer(fd,num_av)
ens_alloc = None 
ens_alloc_num_av = None

hdr = None
FIXOFFSET = None
SOURCE = None
def rd_buffer(fd,num_av,msg=msg_print):
    """ RH: return ens=None, hdr=None if there's a problem

    returns (ens,hdr,cfg,pos)
    """
    # To save it being re-initialized every time.
    # [python] cache the preallocated array in ens_alloc, and remember 
    # what num_av was, so we can reallocate when called with a different num_av.
    # otherwise global/local is too confusing, as other parts of the code use
    # ens both for a local variable and a global variable, or kind of appear to do
    # so.
    global ens_alloc,ens_alloc_num_av, hdr
    pos = None 
    # A fudge to try and read files not handled quite right.
    global FIXOFFSET, SOURCE

    # If num_av<0 we are reading only 1 element and initializing
    if num_av<0:
        SOURCE=0

    class Ensemble(object):
        pass
    
    cfg=None
    ens=None

    if num_av == ens_alloc_num_av:
        ens = ens_alloc

    # This reinitializes to whatever length of ens we want to average.
    if num_av<0 or ens is None:
        FIXOFFSET=0   
        n=abs(num_av)
        [hdr,pos]=rd_hdr(fd,msg)
        if hdr is None:
            return ens,hdr,cfg,pos
        cfg=rd_fix(fd,msg)
        fd.seek(pos,os.SEEK_SET)

        ens_dtype = [('number',float64),
                     ('rtc',(float64,7)),
                     ('BIT',float64),
                     ('ssp',float64),
                     ('depth',float64),
                     ('pitch',float64),
                     ('roll',float64),
                     ('heading',float64),
                     ('temperature',float64),
                     ('salinity',float64),
                     ('mpt',float64),
                     ('heading_std',float64),
                     ('pitch_std',float64),
                     ('roll_std',float64),
                     ('adc',(float64,8)),
                     ('error_status_wd',float64),
                     ('pressure',float64),
                     ('pressure_std',float64),
                     ('east_vel',(float64,cfg.n_cells)),
                     ('north_vel',(float64,cfg.n_cells)),
                     ('vert_vel',(float64,cfg.n_cells)),
                     ('error_vel',(float64,cfg.n_cells)),
                     ('intens',(float64,(cfg.n_cells,4))),
                     ('percent',(float64,(cfg.n_cells,4))),
                     ('corr',(float64,(cfg.n_cells,4))),
                     ('status',(float64,(cfg.n_cells,4))),
                     ('bt_mode',float64),
                     ('bt_range',(float64,4)),
                     ('bt_vel',(float64,4)),
                     ('bt_corr',(float64,4)),
                     ('bt_ampl',(float64,4)),
                     ('bt_perc_good',(float64,4)),
                     ('smtime',float64),
                     ('emtime',float64),
                     ('slatitude',float64),
                     ('slongitude',float64),
                     ('elatitude',float64),
                     ('elongitude',float64),
                     ('nmtime',float64),
                     ('flags',float64)  ]
        ens = Ensemble()
        ens.ensemble_data = zeros( n, dtype=ens_dtype)

        for name,typ in ens_dtype:
            setattr(ens,name,ens.ensemble_data[name])

        ens_alloc = ens
        ens_alloc_num_av = num_av

        num_av=abs(num_av)

    k=-1  # a bit tricky - it gets incremented at the beginning of an ensemble
    while k+1<num_av:
        # This is in case junk appears in the middle of a file.
        num_search=6000
        id1=fromfile(fd,uint8,2)

        search_cnt=0
        while search_cnt<num_search and \
                   ((id1[0]!=0x7F or id1[1]!=0x7F ) or  not checkheader(fd)):
            search_cnt+=1
            nextbyte=fromfile(fd,uint8,1)
            if len(nextbyte)==0:  # End of file
                msg("EOF reached after %d bytes searched for next valid ensemble start\n"%search_cnt)
                ens=None
                return ens,hdr,cfg,pos
            id1[1]=id1[0]
            id1[0]=nextbyte[0]
        # fprintf([dec2hex(id1(1)) '--' dec2hex(id1(2)) '\n'])

        if search_cnt==num_search:
            print("ERROR: Searched %d entries..."%search_cnt)
            print("Not a workhorse/broadband file or bad data encountered: -> %x%x"%(id1[0],id1[1]))
            ens = None
            return ens,hdr,cfg,pos
        elif search_cnt>0:
            msg("Searched %d bytes to find next valid ensemble start\n"%search_cnt)

        startpos=fd.tell()-2  # Starting position.

        # Read the # data types.
        [hdr,nbyte]=rd_hdrseg(fd)     
        byte_offset=nbyte+2
        ## fprintf('# data types = %d\n  ',(length(hdr.dat_offsets)))
        ## fprintf('Blocklen = %d\n  ',hdr.nbyte)

        # Read all the data types.

        for n in range(len(hdr.dat_offsets)): # n: 1 => 0 based
            id_="%04X"%fromfile(fd,uint16,1)[0]

            # handle all the various segments of data. Note that since I read the IDs as a two
            # byte number in little-endian order the high and low bytes are exchanged compared to
            # the values given in the manual.
            #
            winrivprob=0

            #print("n,id = %d %s\n"%(n,id_))
            if id_ == '0000':
                # case '0000',   # Fixed leader
                [cfg,nbyte]=rd_fixseg(fd)
                nbyte+=2
            elif id_ == '0080':   # Variable Leader
                # So I think that we need to increment k here, as this marks the
                # beginning of a record, but we want k to remain 0-based, so above
                # it was initialized to -1 (just as in the matlab code it is initialized
                # to 0).
                k+=1 
                
                ens.number[k]         =fromfile(fd,uint16,1)[0]
                ens.rtc[k,:]          =fromfile(fd,uint8,7)
                ens.number[k]         =ens.number[k]+65536*fromfile(fd,uint8,1)[0]
                ens.BIT[k]            =fromfile(fd,uint16,1)[0]
                ens.ssp[k]            =fromfile(fd,uint16,1)[0]
                ens.depth[k]          =fromfile(fd,uint16,1)[0]*.1   # meters
                ens.heading[k]        =fromfile(fd,uint16,1)[0]*.01  # degrees
                ens.pitch[k]          =fromfile(fd,int16,1)[0]*.01   # degrees
                ens.roll[k]           =fromfile(fd,int16,1)[0]*.01   # degrees
                ens.salinity[k]       =fromfile(fd,int16,1)[0]       # PSU
                ens.temperature[k]    =fromfile(fd,int16,1)[0]*.01   # Deg C
                ens.mpt[k]            =sum( fromfile(fd,uint8,3) * array([60,1,.01])) # seconds
                ens.heading_std[k]    =fromfile(fd,uint8,1)[0]     # degrees
                ens.pitch_std[k]      =fromfile(fd,uint8,1)[0]*.1   # degrees
                ens.roll_std[k]       =fromfile(fd,uint8,1)[0]*.1   # degrees
                ens.adc[k,:]          =fromfile(fd,uint8,8)
                nbyte=2+40

                if cfg.name =='bb-adcp': 
                    if cfg.prog_ver>=5.55:
                        fd.seek(15,os.SEEK_CUR) # 14 zeros and one byte for number WM4 bytes
                        cent=fromfile(fd,uint8,1)[0]            # possibly also for 5.55-5.58 but
                        ens.rtc[k,:] = fromfile(fd,uint8,7)    # I have no data to test.
                        ens.rtc[k,0] += cent*100
                        nbyte+=15+8
                    # end
                elif cfg.name == 'wh-adcp': # for WH versions.		
                    ens.error_status_wd[k]=fromfile(fd,uint32,1)[0]
                    nbyte+=4
 
                    if int(cfg.prog_ver) in (8,10,16,50,51,52):
                        if cfg.prog_ver>=8.13:  # Added pressure sensor stuff in 8.13
                            fd.seek(2,os.SEEK_CUR)   
                            ens.pressure[k]       =fromfile(fd,uint32,1)[0]
                            ens.pressure_std[k]   =fromfile(fd,uint32,1)[0]
                            nbyte+=10  
                        # end
 
                        if cfg.prog_ver>=8.24:  # Spare byte added 8.24
                            fd.seek(1,os.SEEK_CUR)
                            nbyte+=1
                        # end
 
                        if ( cfg.prog_ver>=10.01 and cfg.prog_ver<=10.99 ) or \
                                cfg.prog_ver>=16.05:   # Added more fields with century in clock 16.05
                            cent=fromfile(fd,uint8,1)[0] 
                            ens.rtc[k,:]=fromfile(fd,uint8,7)   
                            ens.rtc[k,0]+=cent*100
                            nbyte+=8
                        # end
                    elif int(cfg.prog_ver)==9:
                        fd.seek(2,os.SEEK_CUR)   
                        ens.pressure[k]       =fromfile(fd,uint32,1)[0]  
                        ens.pressure_std[k]   =fromfile(fd,uint32,1)[0]
                        nbyte+=10  
  
                        if cfg.prog_ver>=9.10:  # Spare byte added 8.24
                            fd.seek(1,os.SEEK_CUR)
                            nbyte+=1
                        # end
                    # end
                

                elif cfg.name=='os-adcp':
                    fd.seek(16,os.SEEK_CUR) # 30 bytes all set to zero, 14 read above
                    nbyte+=16
 
                    if cfg.prog_ver>23:
                        fd.seek(2,os.SEEK_CUR)
                        nbyte+=2
                    #end    
                #end
            elif id_ == '0100':  # Velocities
                # RCH: will need to check array ordering on these - may have rows/cols
                # switched!
                vels=fromfile(fd,int16,4*cfg.n_cells).reshape([cfg.n_cells,4]) * 0.001     # m/s
                ens.east_vel[k,:]  =vels[:,0]
                ens.north_vel[k,:] =vels[:,1]
                ens.vert_vel[k,:]  =vels[:,2]
                ens.error_vel[k,:] =vels[:,3]
                nbyte=2+4*cfg.n_cells*2
            elif id_ == '0200':  # Correlations
                # RCH check array ordering:
                ens.corr[k,:,:]   =fromfile(fd,uint8,4*cfg.n_cells).reshape([cfg.n_cells,4])
                nbyte=2+4*cfg.n_cells
            elif id_ == '0300':  # Echo Intensities  
                # RCH check array ordering:
                ens.intens[k,:,:]   =fromfile(fd,uint8,4*cfg.n_cells).reshape([cfg.n_cells,4])
                nbyte=2+4*cfg.n_cells
            elif id_ == '0400':  # Percent good
                ens.percent[k,:,:]  =fromfile(fd,uint8,4*cfg.n_cells).reshape([cfg.n_cells,4])
                nbyte=2+4*cfg.n_cells
            elif id_ == '0500':  # Status
                # RESUME TRANSLATION HERE
                # Rusty, I was not consistent about retaining "end" statements
                # I noticed after deleting several that you had been keeping 
                # commented out versions.
                if cfg.name=='os-adcp':
                    # fd.seek(00,os.SEEK_CUR) # zero seek is in the original matlab...
                    nbyte=2 # +00
                else:
                    # Note in one case with a 4.25 firmware SC-BB, it seems like
                    # this block was actually two bytes short!
                    ens.status[k,:,:]   =fromfile(fd,uint8,4*cfg.n_cells).reshape([cfg.n_cells,4])
                    nbyte=2+4*cfg.n_cells
            elif id_ == '0600': # Bottom track
                # In WINRIVER GPS data is tucked into here in odd ways, as long
                # as GPS is enabled.
                if SOURCE==2:
                    fd.seek(2,os.SEEK_CUR)
                    # Rusty, I added the [0] below and in several other places
                    long1=fromfile(fd,uint16,1)[0]
                    
                    # added bt mode extraction  - ben
                    fd.seek(3,os.SEEK_CUR)                    
                    ens.bt_mode[k] = float64(fromfile(fd,uint8,1)[0]) # fromfile(fd,uint8,1)[0]
                    
                    fd.seek(2,os.SEEK_CUR)
                    #fd.seek(6,os.SEEK_CUR)           
                    ens.slatitude[k]  =fromfile(fd,int32,1)[0]*cfac
                    if ens.slatitude[k]==0:
                        ens.slatitude[k]=nan
                else:
                    #fd.seek(14,os.SEEK_CUR) # Skip over a bunch of stuff
                    fd.seek(7,os.SEEK_CUR) # Skip over a bunch of stuff
                    ens.bt_mode[k] = float64(fromfile(fd,uint8,1)[0])
                    fd.seek(6,os.SEEK_CUR) # Skip over a bunch of stuff
                    
                # end
                ens.bt_range[k,:]=fromfile(fd,uint16,4)*.01 #
                ens.bt_vel[k,:]  =fromfile(fd,int16,4)
                ens.bt_corr[k,:] =fromfile(fd,uint8,4)      # felipe pimenta aug. 2006
                ens.bt_ampl[k,:]=fromfile(fd,uint8,4)      # "
                ens.bt_perc_good[k,:]=fromfile(fd,uint8,4) # "
                if SOURCE==2:
                    fd.seek(2,os.SEEK_CUR)
                    # The original rdradcp code:
                    # ens.slongitude[k]=(long1+65536*fromfile(fd,uint16,1)[0])*cfac
                    # Fix from DKR:
                    tmp=(long1+65536*fromfile(fd,uint16,1)[0])*cfac
                    if long1==0:
                        ens.slongitude[k]=nan #dkr --> else longitudes bad
                    else:
                        ens.slongitude[k]=tmp
                    #end

                    #fprintf('\n k %d %8.3f %f ',long1,ens.slongitude(k),(ens.slongitude(k)/cfac-long1)/65536)
                    if ens.slongitude[k]>180:
                        ens.slongitude[k]=ens.slongitude[k]-360
                    if ens.slongitude[k]==0:
                        ens.slongitude[k]=nan
                    fd.seek(16,os.SEEK_CUR)
                    qual=fromfile(fd,uint8,1)
                    if qual==0: 
                        ## fprintf('qual==%d,%f %f',qual,ens.slatitude(k),ens.slongitude(k))
                        ens.slatitude[k]=nan
                        ens.slongitude[k]=nan
                    fd.seek(71-45-21,os.SEEK_CUR)
                else:   
                    fd.seek(71-45,os.SEEK_CUR)
                # end
                nbyte=2+68
                if cfg.prog_ver>=5.3:    # Version 4.05 firmware seems to be missing these last 11 bytes.
                    fd.seek(78-71,os.SEEK_CUR)  
                    ens.bt_range[k,:]=ens.bt_range[k,:]+fromfile(fd,uint8,4)*655.36
                    nbyte+=11

                    if cfg.name == 'wh-adcp':
                        if cfg.prog_ver>=16.20:   # RDI documentation claims these extra bytes were added in v 8.17
                            fd.seek(4,os.SEEK_CUR)  # but they don't appear in my 8.33 data - conversation with
                            nbyte+=4       # Egil suggests they were added in 16.20
                        #end
                    #end
                #end
                                
            # end # id_==0600 # bottom track
            elif id_ == '2000':  # Something from VMDAS.
                # The raw files produced by VMDAS contain a binary navigation data
                # block. 
                cfg.sourceprog='VMDAS'
                if SOURCE != 1:
                    msg("\n***** Apparently a VMDAS file \n")
                #end    
                SOURCE=1
                utim  =fromfile(fd,uint8,4)
                mtime =datenum(utim[3]+utim[4]*256,utim[2],utim[1])
                ens.smtime[k]     =mtime+fromfile(fd,uint32,1)[0]/8640000.
                fd.seek(4,os.SEEK_CUR)  # PC clock offset from UTC
                ens.slatitude[k]  =fromfile(fd,int32,1)[0]*cfac
                ens.slongitude[k] =fromfile(fd,int32,1)[0]*cfac
                ens.emtime[k]     =mtime+fromfile(fd,uint32,1)[0]/8640000.
                ens.elatitude[k]  =fromfile(fd,int32,1)[0]*cfac
                ens.elongitude[k] =fromfile(fd,int32,1)[0]*cfac
                fd.seek(12,os.SEEK_CUR)   
                ens.flags[k]      =fromfile(fd,uint16,1)[0]	
                fd.seek(6,os.SEEK_CUR)
                utim  =fromfile(fd,uint8,4)
                mtime =datenum(utim(1)+utim(2)*256,utim(4),utim(3))
                ens.nmtime[k]     =mtime+fromfile(fd,uint32,1)[0]/8640000.
                # in here we have 'ADCP clock' (not sure how this
                # differs from RTC (in header) and UTC (earlier in this block).
                fd.seek(16,os.SEEK_CUR)
                nbyte=2+76
            elif id_ == '2022':  # New NMEA data block from WInRiverII
                cfg.sourceprog='WINRIVER2'
                if SOURCE != 2:
                    msg("\n***** Apparently a WINRIVER file - Raw NMEA data handler not yet implemented\n")
                #end 
                SOURCE=2
 
                specID=fromfile(fd,uint16,1)[0]
                msgsiz=fromfile(fd,int16,1)[0]
                deltaT=fromfile(fd,uint8,8)
                nbyte=2+12
 
                fd.seek(msgsiz,os.SEEK_CUR)
                nbyte+=msgsiz
                # print "post msgsiz, nbyte=%d"%nbyte
 
                ## do nothing code on specID
                #    fprintf(' %d ',specID)
                #              switch specID,
                #                  case 100,
                #                  case 101,
                #                  case 102,
                #                  case 103,
                #              end
 
 
            # The following blocks come from WINRIVER files, they aparently contain
            # the raw NMEA data received from a serial port.
            #
            # Note that for WINRIVER files somewhat decoded data is also available
            # tucked into the bottom track block.
            #
            # I've put these all into their own block because RDI's software apparently completely ignores the
            # stated lengths of these blocks and they very often have to be changed. Rather than relying on the
            # error coding at the end of the main block to do this (and to produce an error message) I will
            # do it here, without an error message to emphasize that I am kludging the WINRIVER blocks only!
            elif id_ in ('2100','2101','2102','2103','2104'):
                winrivprob=1
 
                if id_ == '2100': # $xxDBT  (Winriver addition) 38
                    cfg.sourceprog='WINRIVER'
                    if SOURCE != 2:
                        msg("\n***** Apparently a WINRIVER file - Raw NMEA data handler not yet implemented\n")
                    SOURCE=2
                    str_=fd.read(38) # fromfile(fd,uchar,38)
                    nbyte=2+38
 
                elif id_ == '2101': # $xxGGA  (Winriver addition) 94 in manual but 97 seems to work
                    # Except for a winriver2 file which seems to use 77.
                    cfg.sourceprog='WINRIVER'
                    if SOURCE != 2:
                        msg("\n***** Apparently a WINRIVER file - Raw NMEA data handler not yet implemented\n")
                    SOURCE=2
                    str_=fd.read(97) # setstr(fromfile(fd,uchar,97))
                    nbyte=2+97

                    l = str_.find('$GPGGA')
                    if l >= 0:
                        # original indices: str(l+7:l+8) str(l+9:l+10) str(l+11:l+12)
                        # but we are both zero-based, and ending index is exclusive...
                        # but since l is already zero-based instead of 1 based, we only have to change
                        # the ending indexes in each case.
                        try:
                            # occasionally the GPS will have logged an incomplete reading -
                            # and this may fail.
                            hh,mm,ss = int(str_[l+7:l+9]), int(str_[l+9:l+11]), float(str_[l+11:l+13])
                            ens.smtime[k]=(hh+(mm+ss/60.)/60.)/24.
                        except ValueError:
                            msg('Corrupt GPS string - skipping')
                        # disp(['->' setstr_(str_(1:50)) '<-'])
                elif id_ == '2102': # $xxVTG  (Winriver addition) 45 (but sometimes 46 and 48)
                    cfg.sourceprog='WINRIVER'
                    if SOURCE != 2:
                        msg("\n***** Apparently a WINRIVER file - Raw NMEA data handler not yet implemented\n")
                    #end
                    SOURCE=2
                    str_=fd.read(45)
                    nbyte=2+45
                    #disp(setstr(str_))
 
                elif id_ == '2103': # $xxGSA  (Winriver addition) 60
                    cfg.sourceprog='WINRIVER'
                    if SOURCE != 2:
                        msg("\n***** Apparently a WINRIVER file - Raw NMEA data handler not yet implemented\n")
                    #end
                    SOURCE=2
                    str_=fd.read(60)
                    nbyte=2+60
 
                elif id_ == '2104':  #xxHDT or HDG (Winriver addition) 38
                    cfg.sourceprog='WINRIVER'
                    if SOURCE != 2:
                        msg("\n***** Apparently a WINRIVER file - Raw NMEA data handler not yet implemented\n")
                    #end
                    SOURCE=2
                    str_=fd.read(38)
                    nbyte=2+38
 
            elif id_ == '0701': # Number of good pings
                fd.seek(4*cfg.n_cells,os.SEEK_CUR)
                nbyte=2+4*cfg.n_cells
 
            elif id_ == '0702': # Sum of squared velocities
                fd.seek(4*cfg.n_cells,os.SEEK_CUR)
                nbyte=2+4*cfg.n_cells
 
            elif id_ == '0703': # Sum of velocities      
                fd.seek(4*cfg.n_cells,os.SEEK_CUR)
                nbyte=2+4*cfg.n_cells
 
            # These blocks were implemented for 5-beam systems
 
            elif id_ == '0A00': # Beam 5 velocity (not implemented)
                fd.seek(cfg.n_cells,os.SEEK_CUR)
                nbyte=2+cfg.n_cells
 
            elif id_ == '0301': # Beam 5 Number of good pings (not implemented)
                fd.seek(cfg.n_cells,os.SEEK_CUR)
                nbyte=2+cfg.n_cells
 
            elif id_ == '0302': # Beam 5 Sum of squared velocities (not implemented)
                fd.seek(cfg.n_cells,os.SEEK_CUR)
                nbyte=2+cfg.n_cells
 
            elif id_ == '0303': # Beam 5 Sum of velocities (not implemented)
                fd.seek(cfg.n_cells,os.SEEK_CUR)
                nbyte=2+cfg.n_cells
 
            elif id_ == '020C': # Ambient sound profile (not implemented)
                fd.seek(4,os.SEEK_CUR)
                nbyte=2+4
 
            elif id_ == '3000':  # Fixed attitude data format for OS-ADCPs (not implemented)	     
                fd.seek(32,os.SEEK_CUR)
                nbyte=2+32
 
            else:
                # This is pretty idiotic - for OS-ADCPs (phase 2) they suddenly decided to code
                # the number of bytes into the header ID word. And then they don't really
                # document what they did! So, this is cruft of a high order, and although
                # it works on the one example I have - caveat emptor....
                #
                # Anyway, there appear to be codes 0340-03FC to deal with. I am not going to
                # decode them but I am going to try to figure out how many bytes to
                # skip.
                #if strcmp(id(1:2),'30'),
                if id_[:2] == '30':
                    # I want to count the number of 1s in the middle 4 bits of the
                    # 2nd two bytes.
                    
                    nflds= bin( int(id_[2:4],16) & 0x3C ).count('1')
                    # I want to count the number of 1s in the highest 2 bits of byte 3
                    dfac =  bin(int(id_[2],16)&0x0C).count('1')
                    fd.seek(12*nflds*dfac,os.SEEK_CUR)
                    nbyte=2+12*nflds*dfac
 
                else:
                    msg( "Unrecognized ID code: %s"%id_ )
                    # DBG: 
                    #raise Exception,"STOP"
                    nbyte=2
                    ens = None
                    return ens,hdr,cfg,pos

                ## ens=-1
                ## 
            #end
 
            # here I adjust the number of bytes so I am sure to begin
            # reading at the next valid offset. If everything is working right I shouldn't have
            # to do this but every so often firware changes result in some differences.

            # print '#bytes is %d, original offset is %d'%(nbyte,byte_offset)
            byte_offset=byte_offset+nbyte   
 
            # both n and hdr.dat_offsets are now 0-based, but len() is unchanged - so be
            # careful on comparisons to len(hdr.dat_offsets)
            if n+1<len(hdr.dat_offsets):
                if hdr.dat_offsets[n+1] != byte_offset:
                    if not winrivprob: 
                        msg("%s: Adjust location by %d\n"%(id_,hdr.dat_offsets[n+1]-byte_offset) )
                    fd.seek(hdr.dat_offsets[n+1]-byte_offset,os.SEEK_CUR)
                #end	
                byte_offset=hdr.dat_offsets[n+1]
            else:
                if hdr.nbyte-2 != byte_offset:
                    if not winrivprob:
                        msg("%s: Adjust location by %d\n"%(id_,hdr.nbyte-2-byte_offset))
                    fd.seek(hdr.nbyte-2-byte_offset,os.SEEK_CUR)
                #end
                byte_offset=hdr.nbyte-2
            #end
        #end
 
        # Now at the end of the record we have two reserved bytes, followed
        # by a two-byte checksum = 4 bytes to skip over.
 
        readbytes=fd.tell()-startpos
        offset=(hdr.nbyte+2)-byte_offset # The 2 is for the checksum
 
        if offset !=4 and FIXOFFSET==0: 
            # in python, no direct test for eof (annoying), so step back one byte,
            # and try to read it.  not sure that this will do the right thing if the
            # last thing we did was a failed read - it definitely works if the last thing
            # was a bad seek
            fd.seek(-1,os.SEEK_CUR)
            feof = len(fd.read(1)) == 0
 
            msg("\n*****************************************************\n")
            if feof:
                msg("EOF reached unexpectedly - discarding this last ensemble\n")
                ens=-1
            else:
                msg("Adjust location by %d (readbytes=%d, hdr.nbyte=%d)\n"%(offset,readbytes,hdr.nbyte))
                msg(" NOTE - If this appears at the beginning of the read, it is\n")
                msg("        is a program problem, possibly fixed by a fudge\n")
                msg("        PLEASE REPORT TO rich@eos.ubc.ca WITH DETAILS!!\n\n")
                msg("      -If this appears at the end of the file it means\n")
                msg("       The file is corrupted and only a partial record has \n ")
                msg("       has been read\n")
            #end
            msg("******************************************************\n")
            FIXOFFSET=offset-4
        #end  
        fd.seek(4+FIXOFFSET,os.SEEK_CUR) 
 
        # An early version of WAVESMON and PARSE contained a bug which stuck an additional two
        # bytes in these files, but they really shouldn't be there 
        #if cfg.prog_ver>=16.05,    
        #	  fd.seek(2,os.SEEK_CUR)
        #end
 
    #end
 
    # Blank out stuff bigger than error velocity
    # big_err=abs(ens.error_vel)>.2
    # big_err=0
 
    # Blank out invalid data 
    # RCH: removed big_err references    
    ens.east_vel[ens.east_vel==-32.768]=nan
    ens.north_vel[ens.north_vel==-32.768]=nan 
    ens.vert_vel[ens.vert_vel==-32.768]=nan
    ens.error_vel[ens.error_vel==-32.768]=nan

    return ens,hdr,cfg,pos
 

#--------------------------------------
#function y=nmedian(x,window,dim)
def nmedian(x,window=inf,dim=None):
    # python: dim is 0-based now!
    # Copied from median but with handling of NaN different.
    # RH: assume that this means calculate median of x, ignoring
    # nans, along the axis given by dim
    # window means only consider values which are within window
    # of the median in the median (circular, yes)

    x = array(x) # probably not necessary

    if dim is None: 
        # choose dim to be the first non-unity dimension of x
        long_dims = [d for d in range(x.ndim) if x.shape[d]>1]
        # and if none are long, revert to summing over 0
        dim = (long_dims + [0])[0]
    #end

    # Depart slightly from the original matlab for dealing with 
    # the case when dim>=x.ndim.  Make x one dimension bigger, 
    # and set dim to be that.  Then all the computations are simpler,
    # and if necessary the dimensions can be remangled at the end
    orig_dim = dim
    if dim>=x.ndim:
        dim = x.ndim
        x = x[...,None]

    # The original shape of x, but with _one_ extra dimension if
    # dim was beyond the original size of x
    shape_with_dimpad = x.shape

    # siz = x.shape # no longer need to explicitly do this.
    n = x.shape[dim]

    # Permute and reshape so that DIM becomes the row dimension of a 2-D array
    # basically rotate the dimensions so that dim becomes the first:
    perm = (arange(x.ndim) - dim) % x.ndim
    unperm = (arange(x.ndim) + dim) % x.ndim

    # x = reshape(permute(x,perm), n,prod(siz)/n)
    x = x.transpose(perm)
    dims_permuted = x.shape
    x = x.reshape( (n,-1) )

    # Sort along first dimension
    x.sort(axis=0) # in place sort, and puts nan at the end, while -inf,inf are in order.
    [n1,n2]=x.shape

    if n1==1: # each column has only one row - no stats to be taken, just copy.
        y=x 
    else:
        if n2==1:
            # summing booleans is safe - 
            kk=sum(isfinite(x))
            if kk > 0:
                # x1,x2:  if kk is even, the two middle elements
                #         if kk is odd, both are set to the middle element
                x1=x[ int((kk-1)/2) ] 
                x2=x[ int(kk/2)     ] 
                deviations = abs(x-(x1+x2)/2.)
                x[deviations>window]=nan
            #end
            x.sort(axis=0)
            # repeat once since we may have nan'd some values.
            kk=sum(isfinite(x))
            x[isnan(x)]=0
            y=NaN
            if kk>0:
                y=sum(x)/kk
            #end
        else:
            # count finite values in each column
            kk=sum(isfinite(x),axis=0)
            ll = kk<n1-2 # ll is true for rows with at least 2 nans
            kk[ll]=0 ; x[:,ll]=nan # in those cases, the answer is nan.  seems harsh.

            # whoa - presumably want to pull the middle two valid values from 
            # each row
            low_median = ((kk-1)/2).clip(0,inf).astype(int32)
            high_median = (kk/2).clip(0,inf).astype(int32)
            x1=x[ low_median, list(range(n2))]
            x2=x[ high_median, list(range(n2))]
     
            # x1,x2 have to get the extra dimension for the broadcast to work
            deviations = abs(x - (x1+x2)[None,...]/2.) 
            x[deviations>window]=nan
            x.sort(axis=0)
            kk=sum(isfinite(x),axis=0)
            x[ isnan(x) ]=0
            y=nan+ones(n2)
            if any(kk):
                valid = kk>0
                y[valid] = sum(x[:,valid],axis=0) / kk[valid]
            # end
        #end
    #end 
                
    # Now we have y, which has shape x.shape[1:]
    # make that back into the shape of x, first by undoing
    # the reshape (recalling that we squished the first dimension of x)
    y_dims_permuted = list(dims_permuted)
    y_dims_permuted[0] = 1
    y = y.reshape(y_dims_permuted)
    # and then undoing the permute:
    y = y.transpose(unperm)

    # and finally, pad out some one-entry dimensions in case the user requested
    # dim>x.ndim
    while x.ndim <= orig_dim:
        x = x[...,None]

    # note that this will leave a 1 entry dimension along the axis
    # of the median - unlike normal functions which lose that dimension

    return y

#--------------------------------------
#function y=nmean(x,dim)
def nmean(x,dim=None):
    # R_NMEAN Computes the mean of matrix ignoring NaN
    #         values
    #   R_NMEAN(X,DIM) takes the mean along the dimension DIM of X. 
    #
    xorig = x
    x=x.copy() # to get matlab semantics
    kk=isfinite(x)
    x[~kk]=0

    if dim is None: 
        # choose dim to be the first non-unity dimension of x
        long_dims = [d for d in range(x.ndim) if x.shape[d]>1]
        # and if none are long, revert to summing over 0
        dim = (long_dims + [0])[0]
    #end

    if dim >= x.ndim:
        y=x # For matlab 5.0 only!!! Later versions have a fixed 'sum'
    else:
        # it's possible that x is just a vector - in which case
        # this sum will return a scalar
        ndat=atleast_1d( sum(kk,axis=dim) )
        indat=(ndat==0)
        # If there are no good data then it doesn't matter what
        # we average by - and this avoid div-by-zero warnings.
        ndat[indat]=1

        # division is always elementwise in numpy
        y = atleast_1d(sum(x,axis=dim))/ndat.astype(float64)
        y[indat]=nan
    #end
    
    return y


# related functions 
def adcp_merge_nmea(r,gps_fn,adjust_to_utc=False):
    """
    parse a NMEA file from WinRiver (i.e. with RDENS sentences),
    and add lat/lon to r.
    adjust_to_utc: use GPS time to modify the hours place of r.mtime
    """
    sents=nmea.parse_nmea(gps_fn)

    fixes=[] # [(ens, lat, lon, dn), ... ]

    last_rdens=None
    last_rmc=None
    for sent in sents:
        if sent['sentence']=='$RDENS':
            last_rdens=sent
        elif sent['sentence']=='$GPRMC':
            last_rmc=sent
        elif sent['sentence']=='$GPGGA':
            lat=sent['lat']
            lon=sent['lon']
            if last_rdens:
                ens=int(last_rdens['ensemble'])
            else:
                ens=-1
            if last_rmc:
                dn=last_rmc['dn']
            else:
                dn=np.nan
            fixes.append( [ens,lat,lon,dn] )
            last_rmc=last_rdens=None

    fixes=np.array(fixes)
    if len(fixes):
        valid=fixes[:,0]>=0
        fixes=fixes[valid]
    
    if len(fixes):
        lats=np.interp(r.ensemble_data['number'],
                       fixes[:,0],fixes[:,1],left=np.nan,right=np.nan)
        lons=np.interp(r.ensemble_data['number'],
                       fixes[:,0],fixes[:,2],left=np.nan,right=np.nan)
        r.gps_dn=np.interp(r.ensemble_data['number'],
                           fixes[:,0],fixes[:,3],left=np.nan,right=np.nan)
        if adjust_to_utc:
            deltas=24*(r.gps_dn - r.mtime)
            hour_correction= np.round( np.median(deltas[np.isfinite(deltas)]))
            if np.isnan(hour_correction):
                print("%s: WARNING: cannot determine time shift - no valid GPS data"%(gps_fn))
            else:
                if 1: # hour_correction!=0.0:
                    print("%s: shifting by %d hours, based on GPS time"%(gps_fn,hour_correction))
                shift_min=np.round( np.nanmin(deltas) )
                shift_max=np.round( np.nanmax(deltas) )
                if shift_min != shift_max:
                    print("%s: WARNING: time zone shift is ambiguous (%.0f-%.0fh)"%(gps_fn,
                                                                                    shift_min,shift_max))
                r.mtime += hour_correction/24.
    else:
        lats=np.nan * r.ensemble_data['number']
        lons=lats

    ll=np.array([lons,lats]).T
    r.ll=ll
    r.latitude=lats
    r.longitude=lons
    return r


def ship_to_earth(r):
    """ 
    rotate r.east_vel, r.north_vel, r.bt_vel
    by compass heading.
    ship values moved to r.ship_**var**
    """
    if r.config.coord_sys!='ship':
        raise Exception("applying ship_to_earth but coord sys is %s"%r.config.coord_sys)

    r.ship_east_vel=r.east_vel
    r.ship_north_vel=r.north_vel

    theta=np.exp(-1j*r.heading*np.pi/180)
    vel=theta[:,None] * (r.east_vel + 1j*r.north_vel)
    r.east_vel=np.real(vel)
    r.north_vel=np.imag(vel)

    r.ship_bt_vel=r.bt_vel.copy()
    bt_vel=theta * (r.bt_vel[:,0] + 1j*r.bt_vel[:,1])
    r.bt_vel[:,0] = np.real(bt_vel)
    r.bt_vel[:,1] = np.imag(bt_vel)
    r.config.coord_sys='earth'

def to_earth(r):
    if r.config.coord_sys=='ship':
        ship_to_earth(r)

def add_depth(r):
    r.depth=np.median(r.bt_range,axis=1)

def invalidate_from_bed(r):
    """
    where bottom track is good, nan out data in bottom 5%
    """
    shape=r.east_vel.shape
    bt_valid=np.isfinite(r.bt_vel[:,0]) 

    Depth=-r.depth[:,None]*np.ones(shape)
    Range=(-r.config.ranges[None,:])*np.ones(shape)
    Depth[~bt_valid,:]=-np.inf
    r.east_vel[ Range<0.9*Depth ] = np.nan
    r.north_vel[ Range<0.9*Depth ] = np.nan
