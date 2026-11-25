# Grafici dei 30 specialisti

Il pacchetto con le schede e le matrici lo scrive il job CPU sul cluster, in `<dest>/graphics`, dopo l’aggregazione. Questo indice descrive il codice e le figure dei 12 modelli già pubblicate. Non contiene numeri della campagna dei 30.

Nota di procedura: [docs/nonideal/04_specialist_graphics.md](../../../../docs/nonideal/04_specialist_graphics.md).
Contratto: [configs/nonideal/specialist_graphics_contract_v1.json](../../../../configs/nonideal/specialist_graphics_contract_v1.json).
Script: [scripts/70_nonideal_specialist_graphics.py](../../../../scripts/70_nonideal_specialist_graphics.py).
Launcher: [scripts/engaging/nonideal_graphics.sbatch](../../../../scripts/engaging/nonideal_graphics.sbatch).
Test sintetici (matrice completa e input mancanti o incoerenti): [tests/test_nonideal_specialist_graphics.py](../../../../tests/test_nonideal_specialist_graphics.py).

## Figure dei 12 modelli originali, già pubblicate

Il job non le ridisegna. I pannelli di campo pubblicati sono quattro, tutti a seed 0, e mostrano il modulo dell’errore:

- [fig5 U-Net p_on seed 0](../package/figures/fig5_fields_O-C149-on-s0.png)
- [fig5 U-Net p_off seed 0](../package/figures/fig5_fields_O-C149-off-s0.png)
- [fig5 GLU p_on seed 0](../package/figures/fig5_fields_S7-C149-glu-on-s0.png)
- [fig5 GLU p_off seed 0](../package/figures/fig5_fields_S7-C149-glu-off-s0.png)

RMSE assoluta e peggioramento rispetto al caso pulito, sui dodici modelli: [fig1](../package/figures/fig1_macro_rmse_by_cell.png), [fig2](../package/figures/fig2_deterioration_vs_R0.png). Lettura successiva: [../reading/INDEX.md](../reading/INDEX.md).

## Schede che il job aggiunge, a campagna completa

Una PDF per run, sotto `cards/`, e una pagina PNG per condizione VAL. L’indice del job elenca ogni file con la sorgente in `figure_manifest.json`.
