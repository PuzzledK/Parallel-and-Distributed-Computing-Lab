
# Assignment Codes for Parallel and Distributed Lab

I am compiling and running the programs locally on linux, so I advice to do the same, or use WSL to use a full fledged linux distro in windows itself.

# To install openMPI

```bash
#For Ubuntu
sudo apt install -y openmpi-bin openmpi-common openmpi-doc libopenmpi-dev

#For Arch Linux with Pacman
sudo pacman -S openmpi
```

# Verify openMPI Installation

```bash
mpirun --version
```

# To Compile Programs

```bash
mpic++ <filename>.cpp -o <output_filename>
```

# To run the compiled programs

```bash
mpirun -n <num_processes> --oversubscribe ./<output_filename>
#Use --oversubscribe if you are using more processes than Physical cores on your CPU
```

# Timer.hpp header file

I have created a simple timer class to help with measuring time of execution of tasks efficiently,make sure to include this file before compiling the programs.

```cpp
#include<Timer.hpp>
```
Replace this with

```cpp
#include "../helpers/Timer.hpp"
```

The Chronos library has to be installed in your c++ compiler for this header file to work.