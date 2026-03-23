#!/bin/bash -x 
#SBATCH --account=xxx
#SBATCH --partition=xxx
#SBATCH --job-name=xxx
#SBATCH --output=slurm-%x.%N.%j.out 
#SBATCH --error=slurm-%x.%N.%j.err 
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=0-01:00:00

# Load the Python module.
module load python/3.6/anaconda

# ***** USER-DEFINED VARIABLES *****
# Set simulation frame range (modify these numbers before submission)
START_FRAME=15000
FINISH_FRAME=49999
# ***********************************

# Create temporary directories for geometry and FFT outputs.
mkdir -p geo_tempo
mkdir -p out_tempo

#Based on the command-line argument (1,2,3,4), it outputs:
#  1) Mode 1 (only OH): Bonds: H–O and O–Si, Angle: H–O–Si.
#  2) Mode 2 (only CH3): Bonds: C–H (for each H in CH3) and C–Si;
#       Angles: all H–C–H combinations and H–C–Si angles.
#  3) Mode 3 (only surface angles): Angles at the Si center between any two functional groups.
#  4) Mode 4 (all): Both groups’ bonds/angles plus the surface angles.
#  5) Mode 5 (only water): Bonds: H–O (only once per bond) and angle: H–O–H.
echo "Running find_indices.py to extract H atom indices..."
python find_indices_water_2.py 5

# Step 2: Run the second script (process_simulation.py) in parallel.
# Each non-empty line in config-out.dat is treated as one configuration.
num_configs=$(wc -l < config-out.dat)
echo "Found $num_configs configuration lines in config-out.dat."

export NSCM=1

echo "Launching process_simulation.py in parallel for each configuration..."
for (( i=1; i<=num_configs; i++ )); do
    echo "Launching configuration $i with frames ${START_FRAME} to ${FINISH_FRAME}..."
    srun --exclusive -N1 -n1 bash -c "python process_simulation_2.py $i $START_FRAME $FINISH_FRAME && \
        mv geo_${i}.dat geo_tempo/ && mv output_${i}.dat out_tempo/" &
done
wait
echo "All process_simulation.py instances have completed."

# Step 3: Run the averaging script in out_tempo.
echo "Copying average_output.py into out_tempo folder..."
cp average_output.py out_tempo/

echo "Changing to out_tempo folder and running average_output.py..."
cd out_tempo
python average_output.py

# After averaging, copy the final output file back to the main folder.
cp output_average.dat ..
cd ..
echo "Final averaged spectra (output_average.dat) copied to main folder."
