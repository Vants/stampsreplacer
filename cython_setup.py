from distutils.core import setup
from pathlib import Path

from Cython.Build import cythonize

compileable_files_path = [Path("scripts", "utils", "ArrayUtils.py"),
                          Path("scripts", "utils", "MatlabUtils.py"),
                          Path("scripts", "utils", "MatrixUtils.py"),
                          Path("scripts", "processes", "CreateLonLat.py"),
                          Path("scripts", "processes", "PsEstGamma.py"),
                          Path("scripts", "processes", "PsFiles.py"),
                          Path("scripts", "processes", "PhaseCorrection.py"),
                          Path("scripts", "processes", "PsWeed.py"),
                          Path("scripts", "processes", "PsSelect.py"),
                          Path("scripts", "funs", "PsTopofit.py")]
files_str = []

for file_name_with_path in compileable_files_path:
    if not file_name_with_path.exists():
        print("No file '{0}'".format(str(file_name_with_path)))
    else:
        files_str.append(str(file_name_with_path))


setup(ext_modules=cythonize(files_str))
