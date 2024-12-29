# Running Cadence Circuit Simulations in a Docker Container

This document provides instructions for setting up and running Cadence IC6 (Spectre) simulations in a Docker container. It covers basic requirements, container setup, and how to configure your simulation environment. The goal is to enable any user—regardless of Linux familiarity—to quickly get up and running with Cadence tools.

If you are already familiar with Docker containers and have everything installed, you can skip directly to [Section 4.1](#41-run-container-with-a-shared-directory) to set up the container. Then, update directories as described in [Part 1 of Section 7](#7-configuring-cadence-simulations) and proceed to [Section 8](#8-run-aicircuit-simulation) to run the simulation code.

## Table of Contents
1. [Introduction](#1-introduction)
2. [System Requirements](#2-system-requirements)
3. [Docker Image Preparation](#3-docker-image-preparation)
4. [Running the Container](#4-running-the-container)
   1. [Run Container with a Shared Directory](#41-run-container-with-a-shared-directory)  
   2. [Run Container with GPU Support (Optional)](#42-run-container-with-gpu-support-optional) 
5. [Accessing the Container](#5-accessing-the-container)
6. [Folder Structure & Simulation Files](#6-folder-structure--simulation-files)
7. [Configuring Cadence Simulations](#7-configuring-cadence-simulations)
8. [Run AICircuit Simulation](#8-run-aicircuit-simulation)
9. [Notes & Recommendations](#9-notes--recommendations)
10. [References](#10-references)

## 1. Introduction
Cadence tools (Spectre, Virtuoso, etc.) typically require specific Linux distributions (e.g., Red Hat or RockLinux). If you are using another distribution such as Ubuntu, you can run Cadence inside a Docker container that emulates a supported environment. This document will help you:

- Understand the container-based setup for Cadence IC6.
- Configure the environment to run simulations (Spectre, netlists, etc.).
- Share files between your host machine and the Docker container.

## 2. System Requirements
- **Host Operating System:** Any recent Linux distribution (e.g., Ubuntu) with Docker installed.
- **Docker:** Ensure you have Docker Engine installed.  
  See [Docker’s official documentation](https://docs.docker.com/engine/install/) for installation steps.
- **License & Access to Cadence Tools:** You must legally own or have permission to use Cadence software.

## 3. Docker Image Preparation
1. **Base Image:** You need a Docker image that contains a supported Linux environment (e.g., Red Hat or RockLinux) with Cadence tools installed.  
   - Example image name: `rlinux8-cadence`
2. **Tools in the Image:**  
   - Cadence IC6.1.8 (or your desired version)  
   - NCSU Design Kit (PDK) version 1.6  
   - Optional: Python / PyTorch or other ML libraries if you need to run machine learning experiments inside the container.

> **Note:** If you do not have a prebuilt image, you will need to create one by installing the required OS, Cadence tools, and any additional dependencies in a Dockerfile.

## 4. Running the Container
Below are common `docker run` commands you might use. Adjust them as needed.

### 4.1 Run Container with a Shared Directory
```bash
docker run -dit \
  --privileged \
  --volume /path/on/host:/path/in/docker \
  --net=host \
  --name my_cadence_container \
  rlinux8-cadence
```

- `--volume /path/on/host:/path/in/docker`: Mounts a shared folder for exchanging files between the host and the container.  
- `--net=host`: Allows the container to use the host’s network stack.  
- `--privileged`: Some Cadence operations require privileged access.  

### 4.2 Run Container with GPU Support (Optional)
If you plan to run GPU-accelerated computations (not typically needed for analog circuit simulation but maybe used for ML workloads):
```bash
docker run -dit \
  --privileged \
  --gpus=all \
  --net=host \
  --name my_cadence_container \
  rlinux8-cadence
```

> **Note:** GPU access requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) if you are using NVIDIA GPUs.

## 5. Accessing the Container

To open an interactive shell session inside the running container:
```bash
docker exec -it --user <your_username> my_cadence_container bash
```

Replace `<your_username>` with a valid user in the container (or omit `--user` if not needed).

Once inside the container, you can run Cadence commands (e.g., `virtuoso`, `spectre`) or any additional software installed in the image.

## 6. Folder Structure & Simulation Files

A typical repository or project structure might look like this on the **host** system:

```bash
my_project/
├── shared_dir/            # This folder is mounted inside the Docker
│   ├── Netlists/          # Netlist files (Spectre format, etc.)
│   ├── Ocean/             # Ocean scripts for simulation automation
│   ├── Model/             # Device model files
│   └── ...                # Other relevant simulation artifacts
├── Config/
│   └── sim_config         # Configuration for simulation environment
└── README.md
```
- **Netlists**: Contains your circuit netlists.  
- **Ocean**: Contains Ocean scripts to automate Spectre/Analog Artist runs.  
- **Model**: Holds device model (BSIM, etc.) files for transistor-level simulations.  
- **Config/sim_config**: A configuration file to set environment variables, directory paths, or other simulation parameters.  

Before running simulations, ensure these folders and files are populated with your custom content.

## 7. Configuring Cadence Simulations

1. **Update Directory Paths:**
   - In your Ocean scripts (`.ocn` or `.ocean` files), specify the correct paths to the netlists and model files. Additionally, define the directory where the simulation results should be saved.
   - In `Config/sim_config`, set the required environment variables. For example, configure `ocean` to point to `/path/on/host` and `oceandocker` to point to `/path/in/docker`.

2. **Run Simulation Commands:**
   - From within the container, navigate to the directory containing your Ocean scripts or netlists.
   - Invoke Spectre or an Ocean script, for example:

   ```bash
   cd /path/in/docker/Ocean
   ocean -nograph -replay run_sim.ocn
   ```

3. **Post-Simulation Checks:**
   - Review simulation logs (usually `.log` or `.out` files).
   - Check results in the designated results directory.

## 8. Run AICircuit Simulation

To run the simulation, first ensure that the desired Docker command is set at the top of the `Simulation/simulator.py` file:
```bash
docker exec --user <your_username> my_cadence_container /bin/tcsh -c
```

This script orchestrates the simulation environment and calls the required components.

Once the setup is complete, execute the simulation using the following command:

```bash
python simulation.py --circuit=CSVA --model=MultiLayerPerceptron --npoints=100
```

## 9. Notes & Recommendations

- **Licensing:** Ensure that your Cadence license server is accessible to the container (often by using `--net=host` or appropriate firewall rules).

- **Performance Considerations:**
  - Using Docker on top of virtualized environments can add overhead.
  - GPU passthrough is primarily useful for ML workloads, not analog Spectre runs.

- **Security:** Running a container with `--privileged` can be a security risk. If possible, consider a non-privileged setup.

- **Documentation:** Refer to Cadence official docs for tool-specific commands, environment variables, or advanced features like ADE (Analog Design Environment).

## 10. References

- [Cadence Support & Manuals](https://www.cadence.com/en_US/home.html): Official documentation and resources for Cadence tools.
- [Docker Documentation](https://docs.docker.com/): Comprehensive guide to Docker installation and usage.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html): Documentation for enabling GPU support in Docker containers.
- [Red Hat Linux](https://www.redhat.com/en): Information about Red Hat Enterprise Linux, a supported OS for Cadence tools.
- [Ocean Scripting Documentation](https://support.cadence.com): Official Cadence documentation for writing and using Ocean scripts (requires valid account).



