# romgeo-spg-management

A management system for Romgeo SPG, designed to streamline operations.

![Work in Progress](https://img.shields.io/badge/status-WIP-yellow.svg)

## Features

- Load and save SPG files
- View and edit SPG file structure
- Import and export grid data (CSV, Surfer GRT, legacy TXT)
- Preview geodetic shifts and geoid heights as heatmaps
- Compare SPG files
- Add or reset metadata
- Create new SPG files (template)
- Tabbed interface for viewing/editing X, Y, and Z grids
- Tree view for SPG structure navigation

## Install Instructions

Follow the steps for your operating system.

### Windows

```powershell
git clone https://github.com/mihail-cretu/romgeo-spg-management.git
cd romgeo-spg-management
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main_gui.py
```

### Linux

```bash
git clone https://github.com/mihail-cretu/romgeo-spg-management.git
cd romgeo-spg-management
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python main_gui.py
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.
