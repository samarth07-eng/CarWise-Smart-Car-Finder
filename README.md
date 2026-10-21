
# CarWise — Smart Car Finder

CarWise is a Python-based car recommendation web application that helps users explore and compare cars according to their budget and preferences.

## Features

- Budget-based car filtering
- Fuel type and transmission preferences
- Minimum seating, mileage and safety requirements
- Maintenance-level filtering
- Content-based similarity recommendations using scikit-learn
- Top car recommendations
- Detailed car explorer
- Side-by-side car comparison
- Downloadable PDF recommendation report
- Downloadable CSV recommendations and catalogue

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- ReportLab

## Project Structure

```text
Car_Recommendation_System/
├── app.py
├── cars.csv
├── requirements.txt
└── README.md
```

## Run Locally

1. Install Python.
2. Open the project folder in VS Code.
3. Open the terminal.
4. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

5. Start the application:

```bash
python -m streamlit run app.py
```

## Dataset

The application reads car specifications from `cars.csv`.

Required columns:

- name
- price
- fuel
- transmission
- seats
- type
- mileage
- safety

The `maintenance` and `image_url` columns are optional.

## Recommendation Method

CarWise transforms categorical features using one-hot encoding and numerical features using MinMaxScaler. It then uses cosine similarity to rank cars according to the selected preferences.

The similarity score is not supervised-model prediction accuracy. Recommendation quality depends on the dataset's accuracy and coverage.

## Limitations

- Recommendations are limited to the available dataset.
- Listed prices and specifications may become outdated.
- The app does not guarantee the best purchase.
- Users should verify specifications, prices, safety information and maintenance costs before buying.

## Future Improvements

- Expand and verify the car dataset.
- Add estimated ownership costs.
- Add user accounts and saved favourites.
- Improve recommendation explanations.
- Deploy the application online.

## Author

Developed as a student software project.
