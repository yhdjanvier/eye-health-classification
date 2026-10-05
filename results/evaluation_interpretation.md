# Evaluation interpretation (auto-generated)

* **Overall accuracy on the unseen test set: 88.47%** - judged acceptable. Macro precision 88.86%, macro recall 88.29%, macro F1 88.43%.
* **Best class:** Cataract (F1 95.39%). **Weakest class:** Glaucoma (F1 81.25%, recall 77.48%).
* **Most frequent confusion:** Glaucoma images predicted as Healthy (Normal) (31 cases) - these classes look most similar to the network.
* **Generalisation check:** validation accuracy 88.13% vs test accuracy 88.47% (gap -0.33%). The small gap shows the model is not over-fitted to the validation set.
* **Healthy-vs-Diseased screening view:** sensitivity (diseased eyes detected) 91.31%, specificity (healthy eyes confirmed) 87.58%. 41 diseased images were missed (called healthy) - in screening, missed disease is the costliest error.
* **Confidence behaviour:** mean confidence when correct 90.22% vs when wrong 69.05%. A threshold of 0.80 (chosen on validation data) marks low-confidence results as 'uncertain' in the app.

## Suggested improvements if results are below target

1. Weakest-class recall low -> train longer, raise `--unfreeze` (e.g. 60), or keep class weights on.
2. Large train/validation gap -> raise `--dropout` (0.4-0.5) or reduce `--lr-finetune`.
3. Persistent confusion between two classes -> collect more/cleaner images for them or use a larger backbone.
4. Re-run `python src/train.py` then `python src/evaluate.py` and compare rows in `results/experiments.csv`.