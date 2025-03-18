# Execution and Speedup Analysis

## Compilation and Execution Instructions

### 1. Odd-Even Sort

To compile:

```bash
mpic++ oddEvenSort.cpp -o oddEvenSort
```

To run:

```bash
mpirun -np <num_processors> --oversubscribe ./oddEvenSort <num_elements>
```

### 2. Monte Carlo Method

To compile:

```bash
mpic++ MonteCarlo.cpp -o MonteCarlo
```

To run:

```bash
mpirun -n <num_processors> --oversubscribe ./MonteCarlo <num_points>
```

### 3. Matrix Multiplication

To compile:

```bash
mpic++ matMul.cpp -o matMul
```

To run:

```bash
mpirun -n <num_processors> --oversubscribe ./matMul <mat_dim>
```

### 4. Dot Product

To compile:

```bash
mpic++ dotProduct.cpp -o dotProduct
```

To run:

```bash
mpirun -n <num_processors> --oversubscribe ./dotProduct <num_elements>
```

### Graphs

#### Execution Time vs Processors
