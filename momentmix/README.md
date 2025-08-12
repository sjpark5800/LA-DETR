<!-- ```
# for dataset with temporal dependencies
PYTHONPATH=$PYTHONPATH:. python momentmix/temporal/mmix.py hl 5 0

# for dataset without temporal dependencies
PYTHONPATH=$PYTHONPATH:. python momentmix/non_temporal/mmix.py tacos 10 1
PYTHONPATH=$PYTHONPATH:. python momentmix/non_temporal/mmix.py charades 10 2
PYTHONPATH=$PYTHONPATH:. python momentmix/non_temporal/mmix.py charades_vgg 10 5
``` -->




## MomentMix Dataset Generation

The following scripts generate MomentMix-augmented datasets.

### Usage

```bash
# For datasets with temporal dependencies
PYTHONPATH=$PYTHONPATH:. python momentmix/temporal/mmix.py <dataset_name> <epsilon_cut> <seed>

# For datasets without temporal dependencies
PYTHONPATH=$PYTHONPATH:. python momentmix/non_temporal/mmix.py <dataset_name> <epsilon_cut> <seed>
```

### Arguments

* **`dataset_name`** (*str*): Name of the dataset to process (e.g., `hl`, `tacos`, `charades`, `charades_vgg`).
* **`epsilon_cut`** (*float*): The hyperparameter $\varepsilon_{\text{cut}}$ controlling the extent to which each sub-foreground is shortened relative to its original long-foreground duration.
* **`seed`** (*int*): Random seed for reproducibility.

### Examples

```bash
# Temporal dependencies
PYTHONPATH=$PYTHONPATH:. python momentmix/temporal/mmix.py hl 5 0

# Non-temporal dependencies
PYTHONPATH=$PYTHONPATH:. python momentmix/non_temporal/mmix.py tacos 10 1
PYTHONPATH=$PYTHONPATH:. python momentmix/non_temporal/mmix.py charades 10 2
PYTHONPATH=$PYTHONPATH:. python momentmix/non_temporal/mmix.py charades_vgg 10 5
```

