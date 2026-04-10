import argparse
import sys
import os
import filecmp
import shutil

from . import result_reader, rasmap


def list_plans(project_path):
    print(f"Opening project in '{project_path}'")
    prj = result_reader.RasProject(project_path)
    for plan_file in prj.getPlanFiles():
        print(plan_file)

def map_add_result(result_path, project_path, rasmap_path=None):
    if rasmap_path is None:
        assert ".prj" in project_path,"Expected project path with .prj extension"
        rasmap_path = project_path.replace('.prj','.rasmap')
    rm = rasmap.RasMap(rasmap_path)
    rm.add_result_layer(result_path)
    out_fn=rasmap_path.replace('.rasmap','-new.rasmap')
    print(f"Writing new rasmap file to {out_fn}")
    rm.write(out_fn)
    
def copy_result(project_path, src_result, dry_run=False):
    # Check for source files and compare to existing files
    assert os.path.isfile(project_path)

    project_dir=os.path.dirname(project_path)
    src_base,ext = os.path.splitext(src_result)
    plan_number = int(ext.replace('.p',''))
    print(f"Plan number is {plan_number}")
    plan_text_fn = src_result
    plan_hdf_fn  = src_result + ".hdf"
    ic_fn = src_base + f".ic.o{plan_number:02}"
    b_fn  = src_base + f".b{plan_number:02}"
    bco_fn= src_base + f".bco{plan_number:02}"

    # preflight checks
    all_fns = [plan_text_fn,plan_hdf_fn,ic_fn,b_fn,bco_fn]
    to_copy=[] # (src_fn,dst_fn)
    for src_fn in all_fns:
        src_exists=os.path.exists(src_fn)
        
        dst_fn=os.path.join(project_dir,os.path.basename(src_fn))
        dst_exists=os.path.exists(dst_fn)

        assert src_exists

        if dst_exists:
            # Compare files
            src_size=os.path.getsize(src_fn)
            dst_size=os.path.getsize(dst_fn)
            if dst_size>src_size:
                # Not specifically an error but suggests possible bad input
                raise Exception(f"Source file is smaller than destination: src: {src_size}  dst: {dst_size}")
            if filecmp.cmp(src_fn,dst_fn,shallow=False):
                #print(f"  Files are the same")
                print(f" Skipping identical {os.path.basename(src_fn)}")
            else:
                if src_fn!=plan_hdf_fn:
                    # Not sure when this would happen - maybe plan has never been run at all in dst?
                    raise Exception("Expected only the HDF file to differ")
                else:
                    to_copy.append( (src_fn,dst_fn) )
        else:
            to_copy.append( (src_fn,dst_fn) )
            
    for src_fn,dst_fn in to_copy:
        if dry_run:
            print(f"  [dry run] copy {src_fn} => {dst_fn}")
        else:
            print(f"  copy {src_fn} => {dst_fn}")
            shutil.copyfile(src_fn,dst_fn)

            
def main(args):
    parser = argparse.ArgumentParser(description='Manipulate RAS 6.x projects')

    parser.add_argument("-p", "--project", help="Path to RAS .prj project file", type=str)
    parser.add_argument("-m", "--map-path", help="Path to .rasmap file",type=str)
    parser.add_argument("--copy-result",
                        help=("Copy output from other folder into this project, assumes plan exists\n"
                              "e.g. path/to/other_project/name.p01"),
                        type=str)
    parser.add_argument("--list-plans",help="List plans",action='store_true')
    parser.add_argument("--map-add-result",help="Add a result to rasmap, does not copy\nSpecify as path to .pNN file")
    parser.add_argument("--sync-name-to-hdf",help="Update names in the given plan/result HDF from the corresponding text plan file",
                        type=str)

    args=parser.parse_args(args=args)

    if args.project:
        print(f"Will read project from {args.project}")
    
    if args.copy_result is not None:
        copy_result(project_path=args.project, src_result = args.copy_result)

    if args.list_plans:
        list_plans(args.project)

    if args.map_add_result:
        map_add_result(args.map_add_result,args.project,args.map_path)

    if args.sync_name_to_hdf:
        result_reader.RasReader.sync_name_to_hdf(args.sync_name_to_hdf)


if __name__=='__main__':
    main(sys.argv[1:])

