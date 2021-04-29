import os
import sys
sys.path.append(os.getcwd())
print(os.getcwd())
# os.chdir("..")
import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import glob
import sys


if __name__ == "__main__":

    if len(sys.argv)>1:
        CDU.newEntryToDataDict(sys.argv[1:])
    else:
        pkl_files = [fn.replace("\\", "/") for fn in glob.glob('RegistrosDP_PP/*.pklspikes')]
        CDU.newEntryToDataDict(pkl_files)