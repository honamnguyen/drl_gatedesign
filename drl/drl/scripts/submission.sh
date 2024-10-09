#!/bin/bash -x
#SBATCH --job-name=hardware
#SBATCH --account=netenesyquma
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=/p/project/netenesyquma/nam/drl_gatedesign/data/out.%j
#SBATCH --error=/p/project/netenesyquma/nam/drl_gatedesign/data/err.%j
#SBATCH --time=24:00:00
#SBATCH --partition=batch
eval export CONDA_EXE='/p/project/netenesyquma/nam/miniconda3/bin/conda'
export _CE_M=''
export _CE_CONDA=''
export CONDA_PYTHON_EXE='/p/project/netenesyquma/nam/miniconda3/bin/python'

# Copyright (C) 2012 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause

__add_sys_prefix_to_path() {
    # In dev-mode CONDA_EXE is python.exe and on Windows
    # it is in a different relative location to condabin.
    if [ -n "${_CE_CONDA}" ] && [ -n "${WINDIR+x}" ]; then
        SYSP=$(\dirname "${CONDA_EXE}")
    else
        SYSP=$(\dirname "${CONDA_EXE}")
        SYSP=$(\dirname "${SYSP}")
    fi

    if [ -n "${WINDIR+x}" ]; then
        PATH="${SYSP}/bin:${PATH}"
        PATH="${SYSP}/Scripts:${PATH}"
        PATH="${SYSP}/Library/bin:${PATH}"
        PATH="${SYSP}/Library/usr/bin:${PATH}"
        PATH="${SYSP}/Library/mingw-w64/bin:${PATH}"
        PATH="${SYSP}:${PATH}"
    else
        PATH="${SYSP}/bin:${PATH}"
    fi
    \export PATH
}

__conda_exe() (
    __add_sys_prefix_to_path
    "$CONDA_EXE" $_CE_M $_CE_CONDA "$@"
)

__conda_hashr() {
    if [ -n "${ZSH_VERSION:+x}" ]; then
        \rehash
    elif [ -n "${POSH_VERSION:+x}" ]; then
        :  # pass
    else
        \hash -r
    fi
}

__conda_activate() {
    if [ -n "${CONDA_PS1_BACKUP:+x}" ]; then
        # Handle transition from shell activated with conda <= 4.3 to a subsequent activation
        # after conda updated to >= 4.4. See issue #6173.
        PS1="$CONDA_PS1_BACKUP"
        \unset CONDA_PS1_BACKUP
    fi
    \local ask_conda
    ask_conda="$(PS1="${PS1:-}" __conda_exe shell.posix "$@")" || \return
    \eval "$ask_conda"
    __conda_hashr
}

__conda_reactivate() {
    \local ask_conda
    ask_conda="$(PS1="${PS1:-}" __conda_exe shell.posix reactivate)" || \return
    \eval "$ask_conda"
    __conda_hashr
}

conda() {
    \local cmd="${1-__missing__}"
    case "$cmd" in
        activate|deactivate)
            __conda_activate "$@"
            ;;
        install|update|upgrade|remove|uninstall)
            __conda_exe "$@" || \return
            __conda_reactivate
            ;;
        *)
            __conda_exe "$@"
            ;;
    esac
}

if [ -z "${CONDA_SHLVL+x}" ]; then
    \export CONDA_SHLVL=0
    # In dev-mode CONDA_EXE is python.exe and on Windows
    # it is in a different relative location to condabin.
    if [ -n "${_CE_CONDA:+x}" ] && [ -n "${WINDIR+x}" ]; then
        PATH="$(\dirname "$CONDA_EXE")/condabin${PATH:+":${PATH}"}"
    else
        PATH="$(\dirname "$(\dirname "$CONDA_EXE")")/condabin${PATH:+":${PATH}"}"
    fi
    \export PATH

    # We're not allowing PS1 to be unbound. It must at least be set.
    # However, we're not exporting it, which can cause problems when starting a second shell
    # via a first shell (i.e. starting zsh from bash).
    if [ -z "${PS1+x}" ]; then
        PS1=
    fi
fi

conda activate base
module load OpenGL
conda activate julius
srun --exclusive --ntasks=1 --cpus-per-task=5 python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_4w_IBMvalencia_q0_ratio3_chan0_TISE_dur45dt_seg9_sub0.4_hid100,200,100_average__seed1 -numtransmon=1 -targetgate=X90 -numworkers=4 -IBMbackend=valencia -IBMqubits=0 -IBMUDratio=3 -seed=1 -channels=0 -evolvemethod=TISE -duration=45 -numseg=9 -subactionscale=0.4 -numiter=5000 -rewardtype=average -hidsizes=100,200,100 -worstfidmethod= &

sleep 60

srun --exclusive --ntasks=1 --cpus-per-task=5 python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_4w_IBMvalencia_q0_ratio3_chan0_TISE_dur45dt_seg9_sub0.4_hid100,200,100_average__seed2 -numtransmon=1 -targetgate=X90 -numworkers=4 -IBMbackend=valencia -IBMqubits=0 -IBMUDratio=3 -seed=2 -channels=0 -evolvemethod=TISE -duration=45 -numseg=9 -subactionscale=0.4 -numiter=5000 -rewardtype=average -hidsizes=100,200,100 -worstfidmethod= &

sleep 60

srun --exclusive --ntasks=1 --cpus-per-task=5 python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_4w_IBMvalencia_q0_ratio3_chan0_TISE_dur45dt_seg9_sub0.4_hid100,200,100_average__seed3 -numtransmon=1 -targetgate=X90 -numworkers=4 -IBMbackend=valencia -IBMqubits=0 -IBMUDratio=3 -seed=3 -channels=0 -evolvemethod=TISE -duration=45 -numseg=9 -subactionscale=0.4 -numiter=5000 -rewardtype=average -hidsizes=100,200,100 -worstfidmethod= &

sleep 60

srun --exclusive --ntasks=1 --cpus-per-task=5 python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_4w_IBMvalencia_q0_ratio3_chan0_TISE_dur45dt_seg9_sub0.4_hid100,200,100_average__seed4 -numtransmon=1 -targetgate=X90 -numworkers=4 -IBMbackend=valencia -IBMqubits=0 -IBMUDratio=3 -seed=4 -channels=0 -evolvemethod=TISE -duration=45 -numseg=9 -subactionscale=0.4 -numiter=5000 -rewardtype=average -hidsizes=100,200,100 -worstfidmethod= &

sleep 60

srun --exclusive --ntasks=1 --cpus-per-task=5 python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_4w_IBMvalencia_q0_ratio3_chan0_TISE_dur45dt_seg9_sub0.4_hid50,100,50_average__seed1 -numtransmon=1 -targetgate=X90 -numworkers=4 -IBMbackend=valencia -IBMqubits=0 -IBMUDratio=3 -seed=1 -channels=0 -evolvemethod=TISE -duration=45 -numseg=9 -subactionscale=0.4 -numiter=5000 -rewardtype=average -hidsizes=50,100,50 -worstfidmethod= &

sleep 60

srun --exclusive --ntasks=1 --cpus-per-task=5 python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_4w_IBMvalencia_q0_ratio3_chan0_TISE_dur45dt_seg9_sub0.4_hid50,100,50_average__seed2 -numtransmon=1 -targetgate=X90 -numworkers=4 -IBMbackend=valencia -IBMqubits=0 -IBMUDratio=3 -seed=2 -channels=0 -evolvemethod=TISE -duration=45 -numseg=9 -subactionscale=0.4 -numiter=5000 -rewardtype=average -hidsizes=50,100,50 -worstfidmethod= &

sleep 60

srun --exclusive --ntasks=1 --cpus-per-task=5 python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_4w_IBMvalencia_q0_ratio3_chan0_TISE_dur45dt_seg9_sub0.4_hid50,100,50_average__seed3 -numtransmon=1 -targetgate=X90 -numworkers=4 -IBMbackend=valencia -IBMqubits=0 -IBMUDratio=3 -seed=3 -channels=0 -evolvemethod=TISE -duration=45 -numseg=9 -subactionscale=0.4 -numiter=5000 -rewardtype=average -hidsizes=50,100,50 -worstfidmethod= &

sleep 60

srun --exclusive --ntasks=1 --cpus-per-task=5 python /p/project/netenesyquma/nam/drl_gatedesign/drl/drl/scripts/rl_train.py -study juwels_4w_IBMvalencia_q0_ratio3_chan0_TISE_dur45dt_seg9_sub0.4_hid50,100,50_average__seed4 -numtransmon=1 -targetgate=X90 -numworkers=4 -IBMbackend=valencia -IBMqubits=0 -IBMUDratio=3 -seed=4 -channels=0 -evolvemethod=TISE -duration=45 -numseg=9 -subactionscale=0.4 -numiter=5000 -rewardtype=average -hidsizes=50,100,50 -worstfidmethod= &

sleep 60


wait
