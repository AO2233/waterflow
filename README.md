# (6th) Solution for [2024 IEEE GRSS Data Fusion Contest Track 1](https://codalab.lisn.upsaclay.fr/competitions/16702#learn_the_details) 

## train data structure
```log
├── data
│   ├── dev
│   │   ├── p1
│   │   └── p2
│   └── Track1
│       ├── train
│       │   ├── images
│       │   └── labels
│       └── val
│           └── images
|
|
└── waterflow
      |....
```

## Run the code

```bash
pip install -r requirements.txt

# infer
python run.py
```

## Method

**Model** = Unetpp + ResNeSt 269e

**Loss** = Diceloss + BCEloss -> rate=3:1

**Pre-process** = Normalize(img_mean, img_std)

**Output** = sigmoid + FindBest(threshold)

**Trick** = TTA

## Result

> lr=3e-5, wd=4e-3, warm-up=0.2, ep=400

| model\threshold | 0.5    | 0.3     | 0.1    | 
|-----------------|--------|---------|--------|
| upp 269e        | 0.9210 |         |        |
| upp 269e + TTA  | 0.9240 | 0.92560 | 0.9260 |

## TODO
- [ ] K-Fold
- [ ] Visualize
- [ ] pre-process upgrade
- [ ] ...