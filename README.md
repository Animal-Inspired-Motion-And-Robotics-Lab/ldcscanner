Use this branch for live experiments where data is being streamed from either a robot's ESP32 or from a .csv file.

Don't use this branch for processing that data to get SNR values. For that, use the snr-tools branch!

There are two primary scripts in this branch:
(1) LDC SCANNER V6 streams data from the robot or a .csv file, visualizes the data, and supports logging that data to a .csv file and taking .pdf screenshots.

(2) SNR_Interactive_Analysis processes SNR data and visualizes it in various ways. Usually the input to this script will come from Scanimator V4, which is in the snr-tools branch.


LDC SCANNER flowchart:
<img width="1536" height="1024" alt="ldc_scanner_v6_flowchart" src="https://github.com/user-attachments/assets/80c43625-3ac3-4b4d-bdd3-797f50faa72a" />
