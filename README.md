Use this branch for live experiments where data is being streamed from either a robot's ESP32 or from a .csv file.

Don't use this branch for processing that data to get SNR values. For that, use the snr-tools branch!

There are two primary scripts in this branch:
(1) LDC SCANNER V6 streams data from the robot or a .csv file, visualizes the data, and supports logging that data to a .csv file and taking .pdf screenshots.

(2) SNR_Interactive_Analysis processes SNR data and visualizes it in various ways. Usually the input to this script will come from Scanimator V4, which is in the snr-tools branch.


Getting started: create a Python environment and install the dependencies with `pip install -r requirements.txt` (the live GUI needs PyQt5, pyqtgraph, PyOpenGL, pyserial, numpy, and matplotlib). Launch the live tool by running `python "LDC SCANNER V6.py"` from the repository root — start it from here so it can import the `scanner/` package that holds its backend logic. The window opens even without a robot attached. Use the connection panel at the lower right to choose "Live Serial" (pick your ESP32's port and baud rate, then press Connect), or switch to "Simulate CSV", browse to a recorded file, and press Start Replay to play it back at an adjustable speed.

Once data is streaming you'll see three live views: a 3D R_p/L surface on the left, a phase-space / time-trace plot on the upper right (click its x-axis to toggle between L-vs-R_p and L-vs-time), and a crack-event plot below. Turn on "Write to File" to log every incoming sample to a CSV for later analysis, and press Snapshot to export all four plot views as publication-quality PDFs into a timestamped `Snapshots/` folder. A few keyboard shortcuts help while the window is focused: Space clears the plots, P pauses/resumes the stream, F toggles CSV logging, 1/2/3 switch 3D camera presets, and — in replay mode — the Left/Right arrow keys scrub back and forth through the recording.

If you want to read or modify the code, start with `ARCHITECTURE.md`, which has flowcharts and a "where do I change X?" guide; the live app is one entry file plus the small, documented `scanner/` package (parsing, data sources, state, surface geometry, and PDF export). For the second tool, open `SNR_Interactive_Analysis.ipynb` in Jupyter and run the cells top to bottom — it uses a file picker to load the CSVs you recorded, computes and plots SNR, and writes figures and reports under `outputs/`. Keep in mind that the heavier SNR processing pipeline (Scanimator V4) lives on the snr-tools branch, so switch branches if that is what you need.


LDC SCANNER flowchart:
<img width="1536" height="1024" alt="ldc_scanner_v6_flowchart" src="https://github.com/user-attachments/assets/80c43625-3ac3-4b4d-bdd3-797f50faa72a" />
