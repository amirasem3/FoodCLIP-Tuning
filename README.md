## Windows Installation & Execution Guide

This project requires **Python 3.9 or newer (64-bit)**.
Please follow the steps below carefully.

---

### 1️⃣ Install Python (Required)

Download and install **Python 3.14.2 (64-bit)** from the official Python website:

🔗 **Python installer (Windows, 64-bit)**
[https://www.python.org/ftp/python/3.14.2/python-3.14.2-amd64.exe](https://www.python.org/ftp/python/3.14.2/python-3.14.2-amd64.exe)

During installation:

* ✅ **Check** “Add Python to PATH”
* ✅ Choose **Install for all users** (recommended)

After installation, **close and reopen** your terminal.

Verify installation:

```powershell
python --version
```

You should see something like:

```
Python 3.14.2
```

---

### 2️⃣ Disable Microsoft Store Python Alias (Important)

To avoid conflicts, disable Windows’ built-in Python alias:

1. Open **Settings**
2. Go to **Apps → Advanced app settings → App execution aliases**
3. Turn **OFF**:

   * `python.exe`
   * `python3.exe`

---

### 3️⃣ Run the Project (Automatic Setup)

From the project root directory, run:

```cmd
windows_runner.bat
```

Or double-click **`windows_runner.bat`**.

This script will automatically:

* Create a virtual environment
* Install PyTorch (CPU)
* Install all required dependencies
* Run:

  1. Zero-shot CLIP baseline
  2. Prompt-tuning training
  3. Final evaluation & plots

---

### 4️⃣ Output Files

After successful execution, results will be saved in:

```
results/
checkpoints/
```

These include:

* Accuracy metrics
* Per-class improvement plots
* Saved prompt-tuning model

---

### ⚠️ Troubleshooting

If you see errors related to Python:

* Ensure Python was added to PATH
* Ensure App Execution Aliases are disabled
* Reopen your terminal after installation

---

### ✅ Tested On

* Windows 10 / 11 (64-bit)
* Python 3.11 – 3.14
* CPU-only environments

---

If you want, I can also:

* Add **macOS/Linux instructions**
* Generate a **full README.md** (abstract, method, results, citation)
* Write a **“Reproducibility”** section like real IEEE papers
