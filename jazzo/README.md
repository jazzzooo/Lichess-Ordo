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
python3 setup.py build_ext --inplace
```
Try experimenting with different flags and compilers in `setup.py`. AMD users might benefit from AOCC.

### Running
Extract the ratings file in the `data` folder with `pzstd -d data/blitz.packed.zst`

```
./jazzo.py
```

## License
This project is licensed under the Unlicense - see the LICENSE.md file for details

## Acknowledgments
* [Ordo](https://github.com/michiguel/Ordo)
