## Running ```run.py```

### Train and test
```
python3 run.py
```

### Skip train and only run test on a checkpoint
```
python3 run.py --evaluate --checkpoint CHECKPOINT_DIRECTORY
```

### Output visualization of test data prediction
```
python3 run.py --evaluate --checkpoint CHECKPOINT_DIRECTORY --visualization
```

### Use a checkpoint to enable live webcam feature
```
python3 run.py --evaluate --checkpoint CHECKPOINT_DIRECTORY --live
```