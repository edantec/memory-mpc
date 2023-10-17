# HPP dataset generation

To build the container:

```bash
sudo apptainer build pfc-hpp.sif pfc-hpp.def
scp pfc-hpp.sif $USER@pfcalcul.laas.fr:/pfcalcul/work/$USER/hpp-dataset/
```

Then start a manager on the pfc frontend:
```bash
ssh $USER@pfcalcul.laas.fr
cd /pfcalcul/work/$USER/pfc-hpp
apptainer run --app manager pfc-hpp.sif
```

And run a boss wherever you want:
```
apptainer run --app boss pfc-hpp.sif
```

Then, you should be good for `sbatch ./schedule.sh`
