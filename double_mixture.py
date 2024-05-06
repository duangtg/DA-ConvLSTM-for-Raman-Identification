import pandas as pd
import os

# Pure mineral Raman spectra
fo_file = 'data/mixed_spectra/sd_fo.csv'
aug_file = 'data/mixed_spectra/sd_aug.csv'
an_file = 'data/mixed_spectra/sd_an.csv'
en_file = 'data/mixed_spectra/sd_en.csv'

# Read data
fo_spectra = pd.read_csv(fo_file)
aug_spectra = pd.read_csv(aug_file)
an_spectra = pd.read_csv(an_file)
en_spectra = pd.read_csv(en_file)

# Get the wavelength column
wavelength_columns = fo_spectra.columns[:-4]

# Read mix ratio file
mixing_ratios_file = 'double_mixture.csv'
mixing_ratios = pd.read_csv(mixing_ratios_file)

# Create blend spectra for each blend ratio and save them separately
output_folder = './mixed_spectra'
os.makedirs(output_folder, exist_ok=True)

for index, row in mixing_ratios.iterrows():
    mixed_spectra_list = []
    for _ in range(400):
        # Extract spectra randomly
        fo_spectrum = fo_spectra.sample(1, ignore_index=True)
        aug_spectrum = aug_spectra.sample(1, ignore_index=True)
        an_spectrum = an_spectra.sample(1, ignore_index=True)
        en_spectrum = en_spectra.sample(1, ignore_index=True)

        # Linear mix
        mixed_spectrum_data = {
            **{col: row['fo'] / 100 * fo_spectrum[col].iloc[0] +
                    row['aug'] / 100 * aug_spectrum[col].iloc[0] +
                    row['an'] / 100 * an_spectrum[col].iloc[0] +
                    row['en'] / 100 * en_spectrum[col].iloc[0]
               for col in wavelength_columns},
            'fo': row['fo'],
            'aug': row['aug'],
            'an': row['an'],
            'en': row['en']
        }

        mixed_spectra_list.append(mixed_spectrum_data)

    # Save result
    mixed_spectra = pd.DataFrame(mixed_spectra_list)
    output_file_path = os.path.join(output_folder, f'mixed_spectra_ratio_{index}.csv')
    mixed_spectra.to_csv(output_file_path, index=False)

print(f"all results have been saved：{output_folder}")
