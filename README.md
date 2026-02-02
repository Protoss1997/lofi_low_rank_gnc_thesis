# lofi_low_rank_gnc_thesis


## Activar conda
source ~/anaconda3/etc/profile.d/conda.sh

## Crear ambiente
conda create -n ambiente python=3.12 -y

## Ingresar ambiente
conda activate ambiente

## Instalaciones
python -m pip install -U pip
python -m pip install jupyterlab
pip install -U ipywidgets

pip install -U jax
pip install -U chex jaxtyping jax-tqdm matplotlib
pip install -U "tfp-nightly[jax]"


## Salir de conda
conda deactivate