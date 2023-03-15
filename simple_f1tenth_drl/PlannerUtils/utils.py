import os, shutil, yaml
from argparse import Namespace
  
import cProfile
import pstats
import io
from pstats import SortKey


def init_file_struct(path):
    if os.path.exists(path):
        try:
            os.rmdir(path)
        except:
            shutil.rmtree(path)
    os.mkdir(path)

    

       
def load_conf(fname):
    full_path =  "experiments/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf 



    
def profile_and_save(function):
    with cProfile.Profile(builtins=False) as pr:
        function()
        
        with open("Data/Profiling/main.prof", "w") as f:
            ps = pstats.Stats(pr, stream=f)
            ps.strip_dirs()
            ps.sort_stats('cumtime')
            ps.print_stats()
            
        with open("Data/Profiling/main_total.prof", "w") as f:
            ps = pstats.Stats(pr, stream=f)
            ps.strip_dirs()
            ps.sort_stats('tottime')
            ps.print_stats()