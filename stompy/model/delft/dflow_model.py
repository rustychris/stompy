import ..generic.model as gen_model
import .io as dio

class DFlowModel(gen_model.Model):
    dfm_bin_dir=None # .../bin  giving directory containing dflowfm
    num_procs=1 
    run_dir="." # working directory when running dflowfm
    
    mdu_basename='flowfm.mdu'

    mdu=None
    
    def __init__(self):
        pass
    
    def load_mdu(self,fn):
        self.mdu=dio.MDUFile(fn)
        
    
# class Scalar(gen_bc.Scalar):
#     def write_(self,model,feature,grid):
#         print("Writing feature: %s"%(feature['name']))
# 
#         name=feature['name']
#         old_bc_fn=mdu.filepath( ['external forcing','ExtForceFile'] )
# 
#         assert feature['geom'].type=='LineString'
#         pli_data=[ (name, np.array(feature['geom'].coords)) ]
#         base_fn=os.path.join(mdu.base_path,"%s_%s"%(name,self.var_name))
#         pli_fn=base_fn+'.pli'
#         dio.write_pli(pli_fn,pli_data)
# 
#         if self.var_name=='salinity':
#             quant='salinitybnd'
#         elif self.var_name=='temperature':
#             quant='temperaturebnd'
#         else:
#             assert False
# 
#         with open(old_bc_fn,'at') as fp:
#             lines=["QUANTITY=%s"%quant,
#                    "FILENAME=%s_%s.pli"%(name,self.var_name),
#                    "FILETYPE=9",
#                    "METHOD=3",
#                    "OPERAND=O",
#                    ""]
#             fp.write("\n".join(lines))
# 
#         self.write_data(mdu,feature,self.var_name,base_fn)
# 
#     def write_data(self,mdu,feature,var_name,base_fn):
#         ref_date,start_date,end_date = mdu.time_range()
#         period=np.array([start_date,end_date])
#         elapsed_minutes=(period - ref_date)/np.timedelta64(60,'s')
# 
#         # just write a single node
#         tim_fn=base_fn + "_0001.tim"
#         with open(tim_fn,'wt') as fp:
#             for t in elapsed_minutes:
#                 fp.write("%g %g\n"%(t,self.value))
#     
