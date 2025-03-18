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

```vega-lite
{
  "data": {
    "values": [
      {"Year": 2020, "AI Adoption Rate": 20},
      {"Year": 2021, "AI Adoption Rate": 30},
      {"Year": 2022, "AI Adoption Rate": 40},
      {"Year": 2023, "AI Adoption Rate": 50},
      {"Year": 2024, "AI Adoption Rate": 60}
    ]
  },
  "mark": "bar",
  "encoding": {
    "x": {"field": "Year", "type": "temporal"},
    "y": {"field": "AI Adoption Rate", "type": "quantitative"}
  }
}
