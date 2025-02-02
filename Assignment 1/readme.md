# For Assignment1.cpp

To compile the file

```bash
mpic++ Assignment1.cpp -o out
```

To run the file, an argument is needed (i.e. 2,3,4, 1 is filename itself) to choose which part is to be run,like so

```bash
mpirun -n <num_processors> --oversubscibe ./out <arg>
```

Example

```bash
mpirun -n 16 --oversubscribe ./out 2
```

# For randomWalk.cpp

To compile the file

```bash
mpic++ randomWalk.cpp -o out
```

To run the file

```bash
mpic++ -n 16 --oversubscribe ./out
```