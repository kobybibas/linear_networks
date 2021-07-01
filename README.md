# Linear networks
Experimneting with linear networks.

# Install requirements
```bash
# Add channels. Last added is with the highest priorety
conda config --add channels pytorch
conda config --add channels conda-forge
conda config --add channels anaconda

# Install pip for fallback
conda install --yes pip

# Install with conda. If package installation fails, install with pip.
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt
```

# Execute
```bash
python main_real_data_set.py  --multirun set_names="1028_SWD","1030_ERA","1196_BNG_pharynx","1199_BNG_echoMonths", "1201_BNG_breastTumor","215_2dplanes","218_house_8L", "225_puma8NH", "229_pwLinear","344_mv","522_pm10", "537_houses","542_pollution" hydra/launcher=joblib
```
