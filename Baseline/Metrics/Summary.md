The baseline uses those metrics:
1.  Nel file Common Code/evaluation.py vengono calcolate (sulle pipeline, non sui singoli modelli):

    - Accuracy
    - F1 macro
    - Precision
    - Recall
    - Confusion Matrix
    - F1 weighted (commentato)
    - Balanced Accuracy (commentato)

    Lo stesso schema è usato in Complete_eval.py per la pipeline con depth threshold.

2. Nel Carlo/test_end_to_end_pipeline.py vengono inoltre contate le
    - Wrong pairing (tool not in GT): cioè quante volte la pipeline ha associato un tool a un tissue/interazione che non corrisponde al ground truth.
    - Missed positives: cioè quante interazioni TTI positive del ground truth non sono state recuperate dalla pipeline. 

3. Per il modello YOLO il repository ha test_yolo.py che conta:
    - Wrong classes
    - No preds
    - Good classes
    - Preds
    Mi è stato suggerito di usare le "metriche standard" di Ultralytics (Precision, Recall, mAP@50, mAP@50–95, F1, IoU) sia per la Box che la Mask.

4. Per il modello U-Net mi è stato suggerito di aggiungere 
    - Dice
    - IoU
    - Precision
    - Recall
    Se vuoi una valutazione più completa: Dice per classe, IoU per classe, mean Dice, mean IoU.

5. Esiste inoltre una misura fatta sul tempo FPS/latency nel caso dei video (considerata nella pipeline della cartella di Carlo)