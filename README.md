Para entrenar modelo YOLO:

en terminal:

```bash
yolo task=segment mode=train epochs=30 data=dataset.yaml model=yolo11s-seg.pt
```

Para armar dataset, ver repo: https://github.com/rooneysh/Labelme2YOLO