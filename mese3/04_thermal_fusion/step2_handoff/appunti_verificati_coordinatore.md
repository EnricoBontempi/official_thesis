# Bozza coordinatore — schede verificate personalmente (da integrare)

## VT (OSTI 1463254) — verificato sul PDF OSTI, testo estratto
- Cap. 6 pp. 188–199: sonda casalinga, fili tipo B (PtRh/Pt), canna ceramica ~45 cm, su tubo d'acciaio 1 m, traslatore 2D; inserita dallo scarico; box isolante con piastrelle ceramiche e pasta di zirconia; connettore raffreddato ad aria; giunto di riferimento alla connessione con estensione in rame; T_ref max < 200 °C (testo) / 430 K (Tab. 6-1).
- NI TC01 USB; 3 campioni a 2 Hz per punto; salvataggio quando std < soglia (tipicamente 0,5 %); poi spostamento radiale.
- Condizione: Re 50 000, φ = 0,65, pilota 6–7 %; "liner and transition piece were disconnected during thermocouple scanning in the primary zone".
- Exit: 5 posizioni assiali A (x/Dc=1,64)–E (3,34); profili radiali fino a r/DT=0,45; media bulk ~1300 K; PF da 0,1 a 0,04.
- Primary zone: scansioni assiali a r/DN = 0, 0,71, near wall; 400–1600 K; mappa 2D 320 nodi (16×20), area dichiarata "0 < r/DN < 1.36, 0.14 < x/DN < 0.29" (probabile refuso sull'intervallo assiale [I]).
- Errori: isteresi per direzione di scansione; "Transient error is within 10%"; ref junction 200 °C → ~3 % ≈ −30 K a 1000 K (testo) vs Tab. 6-1 "1.2% ~ −19.2 K (at 1600K)" con max ref 430 K e "Voltage error 0.1 mV / 8.2 mV"; conduzione 0,15 K; radiazione 11,1 K; transiente 0,0016 K (Tab.) — incoerente con "within 10%" [I: la tabella valuta l'errore a regime quasi stazionario, l'isteresi misura la non-equilibrio].
- Formule Tab. 6-1: E_C = (T_r − T_M)/cosh[L (4h_c/(d k_s))^{1/2}]; E_R = K_R σ ε A_R (T_J⁴ − T_W⁴)/(h_c A_c); E_tr = τ dT_J/dt = (ρ c d/(4 h_c)) × √6 (STD). Diametro del filo tipo B: [M]. h_c, ε, L: [M] (valori non riportati).
- CFD: zona di ricircolo d'angolo misurata (testo "100K-1200 K", refuso probabile) vs CFD 1500–1700 K; attribuito a condizioni termiche ideali nella CFD.
- "it might not possible to repeat temperature measurement with different combustor downstream geometry" (limite dell'intrusività).
- Cap. 4 pp. 81–82 (non reagente): TC tipo T 40 AWG (0,076 mm), "accuracy of 0.5 K and response time constants of ∼ 0.05 seconds (for air at a speed of 20 m s-1) per the manufacturer"; due all'uscita dell'ugello, uno ~1 cm dalla parete; "the combustor flow was sufficient to shift the thermocouples out of position"; stratificazione 12 K.
- Cap. 4 pp. 134–138 (reagente, transitorio di accensione, Re ~12 000/24 000): TC K bead 0,762 mm in ricircolo e uscita, "approximate time response of 1.5 seconds" (costruttore); guaine ceramiche → possibile stagnazione e risposta più lenta; "Thinner thermocouples could not be used during the flame experiments due to breakage"; HTC sovrastimato nei transitori; riferimento da TC abbandonato.
- PIV (Cap. 5 p. 165–166): Nd:YAG 140 mJ, FlowSense 4M 2048², TiO2 1–2 µm, seminati nella settling chamber; 7,4 Hz; casi R1 φ=0,65, pilota 6 %, Re 50k → stessa condizione nominale della scansione TC; PIV con liner in quarzo [I: implicito dalla misura ottica attraverso il liner; Fig. 5.1 non letta in immagine].
- Fouling del liner con semina (Cap. 3, glicole nel caso isotermo).

## TU Delft JSC (Sampat 2025 Fuel; tesi Sampat 2025) — verificato
- Camera quarzo ID 208 mm, sp. 3 mm; camera acciaio 304 stesse dimensioni con 6 porte P1–P6; 12 ugelli D=6,67 mm; T_air 350 °C; velocità bulk ugello ≈100 m/s; 30–80 kW; φ 0,3–1,0; H2 0–100 %.
- PIV solo H2=0 %, 60 kW, φ 0,6 e 0,8, camera in quarzo; TiO2 0,5 µm via un ugello; 15 Hz, 2000 immagini; finestra 24×24 px 75 % → spaziatura 305,6 µm, finestra 1222,4 µm.
- TC tipo S con guaina Pt 10 %; boccola con foro 0,5 mm più largo della sonda; sovrapressione → eventuale fuga verso l'esterno, non diluizione; traversata radiale a velocità costante: 2 diametri di sonda percorsi mentre si acquisiscono 200 campioni; media mobile su un diametro di sonda; Zaber.
- Eq. (1): ε_tc(σ ε_g T_g⁴ + σ ε_wall T_wall⁴) + h(T_g − T_tc) − σ ε_tc T_tc⁴ = 0; T_wall = lettura TC a r=0,0 (alla parete) [D].
- Tesi App. C.2.1: h = Nu k/D, D diametro sonda, k dei gas combusti; Nu da Hilpert (via Çengel & Ghajar): 5 intervalli di Re; ε_tc = 0,175 fisso, da Bradley & Entwistle 1961 (fili Pt-10Rh: 0,1287–0,2243 tra 600 e 1450 °C); ε_g da WSGG (HITEMP2010, 4 gas grigi, polinomio 4° ordine) [rif. 172]. "corrected for radiation and convection based on the velocity field".
- Diametro sonda [M]; frequenza di campionamento [M]; velocità di traversata [M]; τ [M]; lunghezza di percorso per WSGG [M].
- Limiti dichiarati (tesi §7.3): "probe measurements are intrusive and, hence, may affect the flow field"; "the temperature distribution on the thermocouple sheath would also vary with radial location"; rimedio proposto CARS.
- Fig. C.4 riporta i dati grezzi non corretti (non estratti numericamente).
- Tabella condizioni: temperature = media dei valori non corretti di TC K inguainate Inconel (servizio).

## TU Delft JHC chiuso (van den Berg 2024) — verificato
- Camera ottagonale ID 120 mm, 8 finestre 40×300 mm; una finestra sostituita da piastra acciaio con 10 porte a 30 mm; 1 atm; Δp ~80 mbar verso lo scarico; coflow riferimento 1000 °C, getto 300 °C; Tab. 2: jet 97–119 m/s, coflow bulk 1,32 m/s (11 kW).
- SPIV solo non reagente; DEHS ~1 µm nel coflow; 800 immagini; FOV da ~58 a 92 mm dal getto; condizioni "different from those of the reacting condition measurements. However, they are similar enough to represent the flow physics qualitatively".
- TC tipo S, guaina Pt 10 %, diametro esterno 3 mm; traversata radiale alla porta 1 (Y/D=19,5) con Zaber a 0,62 mm/s; 10 Hz; media su un diametro (71 punti); IC 95 % da media mobile.
- Eq. (1) (rif. Jiménez 2022): (Nu k_g/d)(T_g − T_th) = ε σ (T_th⁴ − T_w⁴); Eq. (2) Churchill–Bernstein; Eq. (3) ε_S-type = −0,1 + 3,24e-4 T − 1,25e-7 T² + 2,18e-11 T³ (rif. Shaddix 1999), T = temperatura della TC. T_w: definizione [M] nel paper.
- Conclusioni: "The non-reacting flow field measurements were used to correct the thermocouple measurements".

## EM2C (Mercier 2016, HAL) — verificato §2.3–2.5, §3.2–3.3, Fig. 3–6
- TC K da 1,5 mm introdotte attraverso una barra d'acciaio verticale; ORZ; T < 1200 K; errore radiativo ≤ 5 %; non corrette; misure dopo equilibrio termico (~45 min, monitorato da una TC in una barra); dettagli nel rif. [6] (Guiberti 2015, non letto).
- Confronto con RANS con pareti misurate imposte (Fig. 6): "good agreement" — usato per validare le condizioni al contorno termiche.
- PIV 10 Hz con ZrO2 (reagente) e olio (non reagente).
- τ_ORZ ≈ 40 ms (rif. [6]).
