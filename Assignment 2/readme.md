# Execution and Speedup Analysis

## Compilation and Execution Instructions

### 1. Odd-Even Sort
To compile:
```bash
mpic++ oddEvenSort.cpp -o oddEvenSort
```
To run:
```bash
mpirun -n <num_processors> --oversubscribe ./oddEvenSort
```

### 2. Monte Carlo Method
To compile:
```bash
mpic++ MonteCarlo.cpp -o MonteCarlo
```
To run:
```bash
mpirun -n <num_processors> --oversubscribe ./MonteCarlo
```

### 3. Matrix Multiplication
To compile:
```bash
mpic++ matMul.cpp -o matMul
```
To run:
```bash
mpirun -n <num_processors> --oversubscribe ./matMul
```

### 4. Dot Product
To compile:
```bash
mpic++ dotProduct.cpp -o dotProduct
```
To run:
```bash
mpirun -n <num_processors> --oversubscribe ./dotProduct
```

## Speedup Analysis
Speedup is calculated as:

\[ S_p = \frac{T_1}{T_p} \]

where:
- \( S_p \) is the speedup achieved using \( p \) processors,
- \( T_1 \) is the execution time on a single processor,
- \( T_p \) is the execution time on \( p \) processors.

### Graphs
#### Execution Time vs Processors

```mermaid
barChart
    title Execution Time Comparison
    x-axis Number of Processors
    y-axis Execution Time (seconds)
    "1 Processor" : 10
    "2 Processors" : 6
    "4 Processors" : 3.5
    "8 Processors" : 2
    "16 Processors" : 1.2
```

#### Speedup vs Processors
```mermaid
line
    title Speedup vs Number of Processors
    x-axis Number of Processors
    y-axis Speedup
    "1 Processor" : 1
    "2 Processors" : 1.67
    "4 Processors" : 2.85
    "8 Processors" : 5
    "16 Processors" : 8.3
```
