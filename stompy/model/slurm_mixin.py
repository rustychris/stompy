"""
First cut at refactoring slurm interaction
"""

# For farm -- assumes that for any mpi work we're already in a job.
class SlurmMixin:
    @classmethod
    def slurm_jobid(cls):
        return os.environ.get('SLURM_JOBID',None)
    
    @classmethod
    def assert_slurm(cls):
        # Could use srun outside of an existing job to synchronously
        # schedule and run.  But then we'd have to get the partition and task
        # details down to here.
        assert cls.slurm_jobid() is not None,"mpi tasks need to be run within a job!"

    @classmethod
    def slurm_num_procs(cls):
        return int(os.environ['SLURM_NTASKS'])

    @classmethod
    def slurm_pack_options(cls,n):
        """
        Return options to pass to srun to invoke an mpi task
        with n cpus.
        """
        # is there a single component that works?
        n_het_tasks=0 # running count of tasks across components
        packs=[] # list of components needed to get to n

        # Scan up to 10 components:
        for group in range(10):
            group=str(group)
            group_size=int(os.environ.get( f'SLURM_NTASKS_PACK_GROUP_{group}', 0))
            if group_size==0: break
            if group_size>=n:
                packs=[group]
                break
            if n_het_tasks<n:
                n_het_tasks+=group_size
                packs.append(group)

        if len(packs):
            options=[f"--pack-group={','.join(packs)}"]
            print(f"Scanned PACK_GROUPS: {n} requested can be satisfied with {options[0]}",
                  flush=True)
            return options

        if n_het_tasks>0:
            print(f"Scanned PACK_GROUPS and found {n_het_tasks} < {n}. No can do.",
                  flush=True)
            raise Exception("Insufficient cpus for MPI request")
        else:
            # Not a pack job.
            n_tasks=int(os.environ.get('SLURM_NTASKS',0))
            if n_tasks==n:
                print(f"Homogeneous job, and n==NTASKS")
                return []
            elif n_tasks<n:
                raise Exception(f"MPI job size {n} > SLURM ntasks {n_tasks}")
            else:
                options=['-n',str(n)]
                print(f"Homogeneous oversized job.  Add {' '.join(options)}",
                      flush=True)
                return options
    
    def run_mpi(self,sun_args):
        # This needs to be better abstracted, not suntans specific.
        self.assert_slurm()

        sun="sun"
        if self.sun_bin_dir is not None:
            sun=os.path.join(self.sun_bin_dir,sun)

        pack_opts=self.slurm_pack_options(self.num_procs)
        cmd=["srun"]
        cmd+=pack_opts
        cmd+=[sun] + sun_args
        print("About to invoke MPI cmd:")
        print(cmd,flush=True)
        subprocess.call(cmd)

