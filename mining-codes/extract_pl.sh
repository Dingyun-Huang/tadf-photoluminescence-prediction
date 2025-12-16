#!/bin/bash -l
#PBS -N cde_extract
#PBS -q debug
#PBS -A SolarWindowsADSP
#PBS -l filesystems=home:grand:eagle
#PBS -l nodes=2
#PBS -l walltime=00:60:00

module use /soft/modulefiles
module load conda
module load nvhpc-mixed
module load craype-accel-nvidia80
conda activate
source /grand/SolarWindowsADSP/dingyun/venvs/cde-tadf/bin/activate

cd $PBS_O_WORKDIR

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job started at: {$TSTAMP}"

NNODES=$(wc -l <$PBS_NODEFILE)
PPN=16
NPROCS=$((PPN * NNODES))

echo "*************************************************************"
echo "STARTING A NEW RUN ON ${NPROCS} processes across ${NNODES} NODES"
echo "DATE: ${TSTAMP}"
echo "NPROCS = PPN * NNODES = ${NPROCS}"
echo "*************************************************************"

export MPICH_GPU_SUPPORT_ENABLED=0
export LD_LIBRARY_PATH=/grand/SolarWindowsADSP/dingyun/envs/cde-tadf/lib64/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH


PYTHON_PROGRAMME=/grand/SolarWindowsADSP/dingyun/tadf_reextract/polaris_extraction_managed_workers.py
PAPER_ROOT=/grand/projects/SolarWindowsADSP/dingyun/tadf/papers/wiley_md/
SAVE_ROOT=/grand/SolarWindowsADSP/dingyun/tadf_reextract/extracted_results/debug_20250411/
START=0
END=3000

echo "Using Python: $(which python)"
echo "Using MPI: $(which mpiexec)"

echo ""
module l
echo ""

OUTPUT_DIR="wiley_table"

mpiexec -n ${NPROCS} -ppn ${PPN} \
    --hostfile ${PBS_NODEFILE} \
    --depth 1 \
    --cpu-bind depth \
    ./set_affinity_gpu_polaris.sh \
    python ${PYTHON_PROGRAMME} \
    ${PAPER_ROOT} \
    ${SAVE_ROOT} \
    ${START} \
    ${END} \
    ${OUTPUT_DIR}
