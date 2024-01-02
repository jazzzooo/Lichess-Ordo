# Jazzo
Like Ordo, but made by me, jazzzooo.

## Description
Program for quickly calculating ~optimal~ good chess ratings

## Getting Started

### Dependencies
* Python 3
* tqdm
* Numpy
* Cython
* OpenMP

### Building
```
CC="clang" CFLAGS="-march=native -mtune=native -Ofast -fopenmp -g0" python3 setup.py build_ext --inplace
```
Try experimenting with different flags and compilers. AMD users might benefit from AOCC.

### Running
Extract the ratings file in the `data` folder with `pzstd -d data/packed-blitz.zst`

```
./jazzo
```

## License
This project is licensed under the Unlicense - see the LICENSE.md file for details

## Acknowledgments
* [Ordo](https://github.com/michiguel/Ordo)
