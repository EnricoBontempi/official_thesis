# Step 2 missione 04 — PASSAGGIO DI CONSEGNE (storico, superato)

**Nota successiva:** dopo il ripristino della quota il lavoro è stato completato in `../step2_modelli_di_misura.md` e `../step2_modelli_fonti_e_ricerche.md`, che integrano anche i report dei subagenti. Questo file resta come traccia dell'interruzione.

Interrotto per limite d'uso. **Non è la consegna dello step 2**: `step2_modelli_di_misura.md` e `step2_modelli_fonti_e_ricerche.md` NON sono stati scritti.

## Fatto e verificato dal coordinatore (testo integrale letto, sezioni pertinenti) — vedi `appunti_verificati_coordinatore.md`
- Virginia Tech, rapporto DOE OSTI 1463254 (https://www.osti.gov/servlets/purl/1463254): Cap. 4 pp. 81–82, 134–138; Cap. 5 pp. 165–166; Cap. 6 pp. 188–199 (sonda tipo B, 2 Hz fino a std<0,5 %, liner rimosso, Tab. 6-1 errori e formule, PIV R1 stessa condizione nominale, TiO2 1–2 µm). Incoerenze interne annotate.
- TU Delft JSC: Sampat et al., Fuel 395 (2025) 135059 (https://repository.tudelft.nl/file/File_1f7eab10-17ca-49e4-8a3d-10a7b58f6c14) §2.1–2.2, §3.2, Eq. (1); tesi Sampat 2025 (https://repository.tudelft.nl/file/File_bd189a65-5831-4433-8cbf-4f73c2857607) §3.2.5, §6.2.2, App. C.2.1 (Nu Hilpert, ε=0,175 da Bradley & Entwistle 1961, ε_g WSGG), §7.3 limiti. Diametro sonda, f campionamento, τ: [M].
- TU Delft JHC: van den Berg et al., AECS 20 (2024) 100298 (https://repository.tudelft.nl/file/File_3e5ff27b-6bbc-43d9-8e1e-0c40a3c3a845) §2.2, Eq. (1)–(3) (Churchill–Bernstein; ε_S(T) da Shaddix 1999), guaina 3 mm, 0,62 mm/s, 10 Hz, 71 campioni, PIV solo non reagente.
- EM2C: Mercier et al. 2016, HAL hal-01405801 §2.3–2.5, §3.2–3.3, Fig. 3–6 (K 1,5 mm in ORZ, <1200 K, non corrette, ≤5 %).
- Altri testi scaricati ma NON letti: tesi Link (TU Delft, File_b0ecc49d-bba9-478f-9050-0c421c9539f8), paper Link (File_53187369-...). GT2022-81808: file chiuso ("campus only").

## Stima illustrativa (NON parametro qualificato)
`stima_tau_correlazioni.py` / `stima_tau_output.txt`: τ e errore radiativo con correlazioni (Churchill–Bernstein, Ranz–Marshall), proprietà Cantera gri30 di prodotti CH4/aria in equilibrio, ρc di materiali **assunte** (3,0–4,2 MJ/m³K), ε=0,2, T_w=600 K **assunte**. Ordini: fili 50–76 µm 10–80 ms; bead 0,3 mm 70–190 ms; bead 0,762 mm 0,5–1,4 s. Rilevante: finestra LES RealPDEBench = 0,5 s.

## Subagenti — NESSUN risultato ricevuto
Tre subagenti (SA1 dinamica/compensazione; SA2 PoliMi/UMD/LDV/rig NH3; SA3 sensore virtuale LES/FDS e alternative) terminati per limite API **senza consegna**. Nessun REPORT.md prodotto. Esistono solo download parziali nella scratchpad (elenchi in `scratch_sa*_elenco.txt`; es. Jiménez 2022, Petit 1982, Hung 2005, Shaddix 1999/2017, NIST TN 2083, Kato 2007, Yilmaz 2008, ASICI 42, CORE): **non letti né verificati**. SA3 ha segnalato solo "un paper LES che applica un modello di TC a dati simulati", non identificato.

## File temporanei (effimeri, persi a fine sessione)
Scratchpad: `/tmp/claude-0/-home-user-official-thesis/ae261236-95b1-589c-9117-9ab4edd4e2e2/scratchpad/` → `me/vt/vt.pdf|vt.txt`, `me/delft/*.pdf|*.txt`, `me/em2c/m.pdf`, `sa1/ sa2/ sa3/`. Per recuperarli: riscaricare dagli URL sopra (nessun PDF in repository).

## Aperto
PoliMi (Olivani 2007, Solero 2017), UMD Archer 2005, LDV+TC compensate (Heitor 1985, Ferrão & Heitor 1998) ancora solo AB; modelli dinamici e compensazione; catalisi Pt in fiamme NH3; sensore virtuale in LES; alternative superficie/ottiche; schede, tabella parametri, requisiti LES, domande per agente dataset, opzioni step 3: tutti da scrivere.
