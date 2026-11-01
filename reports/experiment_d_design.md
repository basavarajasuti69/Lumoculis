# Experiment D design

Change: reduced class-weighted loss strength from 1.0 (Experiment C) to 0.5 while retaining weighted sampling and all other Experiment C settings.

Rationale: Experiment C improved minority-grade recall but overpredicted grade 4 and reduced overall QWK/accuracy. Halving loss-weight strength interpolates halfway between uniform and full dampened inverse-frequency weights, expected to reduce overemphasis on rare classes while preserving some minority benefit.

Run command:
python -m lumoculis.train --config configs/experiment_d_sampling_reduced_loss_weights.yaml

Evaluate checkpoint after training using val_preds.npz and existing metrics utilities.
