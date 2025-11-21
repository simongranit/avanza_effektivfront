# Avanza Portföljsimulator

Streamlit-app för att ladda Avanza-fonder, klassificera dem och simulera portföljer med Monte Carlo och effektiva fronter.

## Plan och antaganden
- Fokus ligger på att göra fondhanteringen enklare via en dedikerad flik för att lägga till, ta bort och klassificera fonder.
- Ingen databas används; all data hämtas direkt från Avanza-API:t baserat på angivna fond-ID:n.
- Konfigurationer sparas lokalt som `fondkonfiguration.json` och kan laddas upp via sidpanelen.

## Kom igång
1. Installera beroenden:
   ```bash
   pip install -r requirements.txt
   ```
2. Starta appen:
   ```bash
   streamlit run app.py
   ```
3. Öppna länken som Streamlit visar (oftast `http://localhost:8501`).

## Användning
- **Fondhantering**: Lägg till Avanza-ID:n, inkludera fonder från en uppladdad konfiguration och justera klassificeringar via datatabellen. Du kan även ta bort fonder och spara den aktuella uppsättningen till disk.
- **Analys**: Välj vilka fonder som ska ingå i simuleringen och kör sedan Monte Carlo och effektiva fronter. Resultat visas i diagram, tabeller och pajer för fördelning mellan fondtyper.

## Miljövariabler
- Inga specifika miljövariabler krävs. All extern data hämtas via Avanza utan autentisering.

## Testning
- Snabb syntaxkontroll kan göras med:
  ```bash
  python -m py_compile app.py config_io.py data_loader.py optimization.py portfolio_engine.py
  ```

## Föreslagna beroendeändringar
- Inga ändringar föreslagna; befintliga paket räcker för nuvarande funktioner.

## Kända begränsningar
- Avanza-API:t kan begränsa antalet anrop; stora mängder fond-ID:n kan därför ta tid eller misslyckas.
- Monte Carlo-simuleringar kan vara beräkningsintensiva vid mycket höga portföljantal.
