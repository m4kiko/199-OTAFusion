# Meta-Transformer Setup & Signal Format Guide

This guide covers two main parts:

1. **Part 1: Setup on a New Computer** - How to install the project from scratch.
2. **Part 2: Input Signal Format** - The exact data format required for the `meta_blackbox.py` classifier.

---

## Part 1: Installation on a New Computer

Follow these steps to get the project and classifier running.

### Step 1.1: Environment & Dependencies

**1. Clone the Repository:**

```bash
git clone https://github.com/your-username/meta-transformer-amc.git
cd meta-transformer-amc
```

**2. Create Python Environment:**

This project was built with **Python 3.8**. It is highly recommended to use this version.

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install Dependencies:**

Install all required packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

---

### Step 1.2: Download Data & Models

The classifier requires two large components that are not in the Git repository.

**1. Dataset: Download the RadioML2018.01A dataset.**

* Link: https://www.deepsig.ai/datasets
* File needed: `GOLD_XYZ_OSC.0001_1024.hdf5`

**2. Pre-trained Models: Download the `.tar` model files from the Google Drive link in the original `README`.**

* Link: https://drive.google.com/file/d/1x_amkYkb3m3bdpPeRztMWhJUWxmBSzmk/view

---

### Step 1.3: Place Files in Correct Folders

⚠️ **This is the most critical step.** The code will fail if this structure is not exact.

Inside your `meta-transformer-amc` folder, create the directories and place the files as shown:

```
meta-transformer-amc/
│
├── amc_dataset/
│   └── RML2018/
│       └── GOLD_XYZ_OSC.0001_1024.hdf5  <-- Place the HDF5 dataset here
│
├── checkpoint/
│   └── learning/
│       ├── 0.tar                       <-- Place all .tar model files here
│       ├── 1.tar
│       ├── 49.tar
│       └── ... (and so on)
│
├── config/
├── data/
├── models/
│
├── main.py
├── meta_blackbox.py
├── signal_generator.py
└── requirements.txt
```

---

### Step 1.4: Critical Configuration (CPU vs. GPU)

The code defaults to using an NVIDIA GPU (CUDA). If your new computer does not have one, you must change this setting.

**1. Open the file:** `config/config.yaml`

**2. Find the `cuda:` setting.**

**A) If your new computer does NOT have an NVIDIA GPU:**

Change the setting to `False`:

```yaml
cuda: False
```

This will force the models to run on your CPU. If you forget this, you will get a `RuntimeError: Cannot access accelerator device`.

**B) If your new computer HAS an NVIDIA GPU:**

Leave the setting as `True`:

```yaml
cuda: True
```

---

### Step 1.5: Verify Installation

You can now run a test to confirm everything is working. This command will load the `vit_main` model, build its 24-way reference prototypes from the HDF5 file, and classify a demo signal.

```bash
python meta_blackbox.py
```

If the setup is correct, you will see `Classifier ready.` followed by a classification result.

---

## Part 2: Input Signal Format (`.npy`)

To use the `classify_from_file("your_signal.npy")` function in `meta_blackbox.py`, your signal file must match the following format.

### Requirements

* **File Type:** `.npy` (a NumPy binary file)
* **Data Type:** `numpy.float32` (or any float, as it will be cast)
* **Shape:** `(2, 1024)`

### Signal Structure

The signal must be in **I/Q (In-phase and Quadrature)** format.

* `signal[0, :]` (Row 1): The **In-phase (I)** component. An array of 1024 float samples.
* `signal[1, :]` (Row 2): The **Quadrature (Q)** component. An array of 1024 float samples.

This `(2, 1024)` shape represents 1024 complex samples, which is the required input length for the `vit_main` model.

---

### How to Generate a Valid Signal File

You can use the `signal_generator.py` script to extract test signals from the HDF5 dataset in this exact format.

**Example command:**

```bash
# This will create a 'qpsk_test.npy' file with the correct format
python signal_generator.py -m QPSK -s 20
```

(You will then be prompted to enter the output path: `qpsk_test.npy`)

---

### How to Use the Classifier

**1. Classify a single signal file:**

```bash
python meta_blackbox.py /path/to/your/signal.npy
```

**2. Or modify the default file in the script:**

Edit `meta_blackbox.py` and change the `YOUR_SIGNAL_FILE` variable to point to your signal file.

```python
YOUR_SIGNAL_FILE = 'path/to/your/signal.npy'
```

Then run:

```bash
python meta_blackbox.py
```

---

## Troubleshooting

### Common Issues

**1. `FileNotFoundError: Model file not found`**
* Ensure the `.tar` model files are placed in `checkpoint/learning/`

**2. `FileNotFoundError: Could not load HDF5 dataset`**
* Ensure `GOLD_XYZ_OSC.0001_1024.hdf5` is in `amc_dataset/RML2018/`

**3. `RuntimeError: Cannot access accelerator device`**
* Set `cuda: False` in `config/config.yaml` if you don't have an NVIDIA GPU

**4. `ValueError: Invalid signal shape. Expected (2, 1024)`**
* Your `.npy` file has the wrong shape. Use `signal_generator.py` to create valid test signals.

---

## Additional Resources

* Original Meta-Transformer Paper: [Insert link if available]
* RadioML Dataset Documentation: https://www.deepsig.ai/datasets
* For questions or issues, please open an issue on the GitHub repository.

---

## License

[Include your license information here]


---

Here is a new `README.md` file tailored as a personal guide for you to set up this project on new devices, including the CUDA fix we just worked through.

-----

# `meta-transformer-amc` - Personal Setup Guide

This guide is for setting up the Meta-Transformer AMC project on a new device.

## Step 1: Python Environment

1.  **Clone the Repository:**

    ```bash
    git clone [your-repo-url]
    cd meta-transformer-amc
    ```

2.  **Create Python Environment:**

      * This project was built with **Python 3.8**. It's highly recommended to use this version.

    <!-- end list -->

    ```bash
    python -m venv .venv
    .\.venv\Scripts\Activate
    ```

3.  **Install Dependencies:**

      * Install all required packages from the `requirements.txt` file.

    <!-- end list -->

    ```bash
    pip install -r requirements.txt
    ```

## Step 2: Data & Model Files

The code requires the dataset and pre-trained models to be placed in a specific folder structure.

1.  **Download Files:**

      * **Dataset:** Download `GOLD_XYZ_OSC.0001_1024.hdf5` (from the original `README`'s DeepSig link).
      * **Models:** Download the pre-trained `.tar` files (from the original `README`'s Google Drive link).

2.  **Create Folder Structure:**

      * Place the downloaded files into the directories as shown below. You will need to create the `amc_dataset/RML2018` and `checkpoint/learning` folders if they don't exist.

    <!-- end list -->

    ```
    meta-transformer-amc/
    │
    ├── amc_dataset/
    │   └── RML2018/
    │       └── GOLD_XYZ_OSC.0001_1024.hdf5  <-- Place dataset here
    │
    ├── checkpoint/
    │   └── learning/
    │       ├── 0.tar                       <-- Place pre-trained models here
    │       ├── 1.tar
    │       └── ... (and so on)
    │
    ├── config/
    ├── data/
    ├── models/
    │
    ├── main.py
    └── requirements.txt
    ```

## Step 3: Hardware Configuration (Important\!)

This step is crucial to avoid runtime errors. The code is set to use a GPU (CUDA) by default.

1.  Open the file: `config/config.yaml`.
2.  Find the `cuda:` setting.

<!-- end list -->

  * **To run on a GPU:**

      * Make sure you have an NVIDIA GPU with the correct CUDA drivers installed.
      * Leave the setting as:
        ```yaml
        cuda: True
        ```

  * **To run on a CPU (No GPU):**

      * If you don't have an NVIDIA GPU or if CUDA is not installed, you **must** change this setting.
      * Change the setting to:
        ```yaml
        cuda: False
        ```
      * **Note:** If you leave this as `True` without a GPU, you will get this error: `RuntimeError: Cannot access accelerator device when none is available.`

## Step 4: Run

Once setup is complete, you can run the default test:

```bash
python main.py test
```