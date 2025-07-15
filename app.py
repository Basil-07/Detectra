import os
import secrets
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import joblib
from fpdf import FPDF

# Import functions from model_1.py and run_1.py
from model_1 import predict_sample, prepare_training_data, train_models, DRUG_INFO
import run_1

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['REPORT_FOLDER'] = 'reports'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Ensure upload and report directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

# Global variables for models and label encoder (load them once on app startup)
binary_model = None
multiclass_model = None
label_encoder = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_models():
    """Load pre-trained models or train them if not found."""
    global binary_model, multiclass_model, label_encoder
    try:
        binary_model = joblib.load('drug_binary_xgb.pkl')
        multiclass_model = joblib.load('drug_multiclass_xgb.pkl')
        label_encoder = joblib.load('drug_label_encoder.pkl')
        print("✅ Models loaded successfully!")
    except FileNotFoundError:
        print("⚠️ Models not found. Training new models...")
        # Define input files for training (ensure these CSVs are in the same directory as app.py)
        drug_files = {
            'cocaine': "cocaine.csv",
            'heroin': "heroin.csv",
            'methadone': "methadone.csv",
            'morphine': "morphine.csv",
            'meth' : "meth.csv"
        }
        non_drug_files = [
            "lactic.csv",
            "citric.csv",
            "ethanol.csv",
            "glucose.csv"
        ]
        try:
            # Ensure the CSV files are present for training
            for f in list(drug_files.values()) + non_drug_files:
                if not os.path.exists(f):
                    print(f"❌ Required training file not found: {f}. Please ensure all .csv files (cocaine.csv, heroin.csv, etc.) are in the same directory as app.py.")
                    raise FileNotFoundError(f"Missing training data file: {f}")

            features_df = prepare_training_data(drug_files, non_drug_files)
            binary_model, multiclass_model, label_encoder = train_models(features_df)
            print("✅ Models trained and saved successfully!")
        except Exception as e:
            print(f"❌ Error during model training: {e}")
            import traceback
            traceback.print_exc()
            # If model training fails, set models to None to indicate they are not ready
            binary_model, multiclass_model, label_encoder = None, None, None


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/pure_compound', methods=['GET', 'POST'])
def pure_compound():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if binary_model is None or multiclass_model is None or label_encoder is None:
                flash('Model not loaded or trained. Please check server logs for errors.', 'danger')
                return redirect(url_for('home'))

            try:
                prediction_result = predict_sample(filepath, binary_model, multiclass_model, label_encoder)

                if prediction_result.get('error'):
                    flash(f"Prediction error: {prediction_result['error']}", 'danger')
                    return redirect(url_for('pure_compound'))

                # Generate spectrum graph for output
                df = pd.read_csv(filepath)
                plt.figure(figsize=(10, 6))
                plt.plot(df['wavenumber'], df['absorbance'], label='Sample Spectrum')
                
                # Mark detected peaks
                detected_peaks_wn = prediction_result.get('detected_peaks', [])
                
                if detected_peaks_wn:
                    max_abs = df['absorbance'].max()
                    height_threshold = max_abs * 0.1
                    
                    for detected_wn in detected_peaks_wn:
                        # Find the corresponding absorbance in the original dataframe
                        idx = (df['wavenumber'] - detected_wn).abs().idxmin()
                        actual_intensity_at_peak = df['absorbance'].iloc[idx]

                        if actual_intensity_at_peak > height_threshold:
                            plt.plot(detected_wn, actual_intensity_at_peak, 'rx', markersize=10, label='_nolegend_')
                            plt.text(detected_wn, actual_intensity_at_peak * 1.05,
                                     f'{detected_wn:.0f}', rotation=45, ha='center')


                # Prepare data for peak matching table in template
                peak_matching_data = []
                if prediction_result['is_drug']:
                    predicted_drug = prediction_result['drug_type']
                    plt.title(f'Detected Drug: {predicted_drug} Spectrum')
                    
                    # Highlight expected peaks if a drug is detected
                    expected_peaks = prediction_result.get('expected_peaks', [])
                    for peak in expected_peaks:
                        # Find closest point on the actual spectrum
                        idx = (df['wavenumber'] - peak).abs().idxmin()
                        plt.plot(df['wavenumber'].iloc[idx], df['absorbance'].iloc[idx], 'go', markersize=8, fillstyle='none', label='_nolegend_') # Green circle for expected
                        plt.text(df['wavenumber'].iloc[idx], df['absorbance'].iloc[idx] * 0.95,
                                 f'Exp:{peak:.0f}', rotation=45, ha='center', color='green')

                        # Logic for peak matching data for template
                        closest_detected_for_table = None
                        is_match = False
                        if detected_peaks_wn:
                            # Find the closest detected peak to the current expected peak
                            closest_idx_detected = np.argmin(np.abs(np.array(detected_peaks_wn) - peak))
                            candidate_detected_peak = detected_peaks_wn[closest_idx_detected]
                            # Changed tolerance from 10 to 20 for matching
                            if abs(candidate_detected_peak - peak) <= 20: 
                                closest_detected_for_table = candidate_detected_peak
                                is_match = True
                        
                        peak_matching_data.append({
                            'expected_peak': peak,
                            'detected_peak': closest_detected_for_table,
                            'is_match': is_match
                        })


                else:
                    plt.title('Sample Spectrum (No Drug Detected)')
                    # If no drug is detected, populate peak_matching_data with top detected peaks
                    for wn, intensity in zip(prediction_result['detected_peaks'], prediction_result['peak_intensities']):
                        peak_matching_data.append({
                            'wavenumber': wn,
                            'absorbance': intensity
                        })


                plt.xlabel('Wavenumber (cm^-1)')
                plt.ylabel('Absorbance')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                graph_url = base64.b64encode(img.getvalue()).decode()
                plt.close() # Close plot to free memory

                # Generate PDF Report
                report_filename = f"report_pure_{os.path.splitext(filename)[0]}.pdf"
                report_filepath = os.path.join(app.config['REPORT_FOLDER'], report_filename)
                
                generate_pure_compound_report(report_filepath, prediction_result, filepath, graph_url, peak_matching_data)

                return render_template('output.html',
                                       prediction=prediction_result,
                                       graph_url=graph_url,
                                       report_filename=report_filename,
                                       peak_matching_data=peak_matching_data) # Pass the new data

            except Exception as e:
                flash(f"An error occurred during processing: {e}", 'danger')
                print(f"Error details: {e}")
                import traceback
                traceback.print_exc()
                return redirect(url_for('pure_compound'))
        else:
            flash('Invalid file type. Please upload a CSV file.', 'danger')
            return redirect(request.url)
    return render_template('upload.html')


@app.route('/mixture_compound', methods=['GET', 'POST'])
def mixture_compound():
    drugs_list = sorted(list(run_1.DRUGS.keys()))
    non_drugs_list = sorted(list(run_1.NON_DRUGS.keys()))

    if request.method == 'POST':
        try:
            drug = request.form['drug']
            drug_weight = float(request.form['drug_weight'])
            non_drug = request.form['non_drug']
            non_drug_weight = float(request.form['non_drug_weight'])

            if not (1 <= drug_weight <= 100) or not (0 <= non_drug_weight <= (100 - drug_weight)):
                flash("Invalid percentages. Drug percentage must be 1-100, and cutting agent percentage must be 0- (100 - drug_percentage).", 'danger')
                return redirect(url_for('mixture_compound'))

            # Generate mixture spectrum
            mixture_spectrum = run_1.generate_mixture(drug, drug_weight, non_drug, non_drug_weight)

            # Find characteristic peaks in the mixture for the chosen drug
            found_peaks_mixture = run_1.find_characteristic_peaks(mixture_spectrum, drug)

            # Generate Mixture Spectrum Graph
            plt.figure(figsize=(12, 6))
            plt.plot(run_1.wavenumbers, mixture_spectrum, label='Mixture Spectrum', color='blue')
            plt.title(f'Simulated Mixture: {drug} ({drug_weight}%) + {non_drug.replace("_", " ")} ({non_drug_weight}%)')
            plt.xlabel('Wavenumber (cm^-1)')
            plt.ylabel('Absorbance')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            # Mark found characteristic peaks on the mixture graph
            for target, detected, intensity in found_peaks_mixture:
                plt.plot(detected, intensity, 'rx', markersize=10, label='_nolegend_') # Red X for detected peaks
                plt.text(detected, intensity * 1.05, f'{detected:.0f}', rotation=45, ha='center')

            mixture_img = io.BytesIO()
            plt.savefig(mixture_img, format='png')
            mixture_img.seek(0)
            mixture_graph_url = base64.b64encode(mixture_img.getvalue()).decode()
            plt.close()

            # Generate Pure Drug Spectrum Graph (for the drug selected)
            pure_drug_spectrum = run_1.generate_compound_spectrum(run_1.DRUGS[drug], 1.0) # Pure drug with 100% weight
            
            plt.figure(figsize=(12, 6))
            plt.plot(run_1.wavenumbers, pure_drug_spectrum, label=f'Pure {drug} Spectrum', color='green')
            plt.title(f'Pure {drug} Spectrum with Key Peaks')
            plt.xlabel('Wavenumber (cm^-1)')
            plt.ylabel('Absorbance')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            # Mark characteristic peaks for the pure drug from DRUGS dictionary
            for peak in run_1.DRUGS[drug]:
                # Find the absorbance at the characteristic peak in the pure spectrum
                idx = np.argmin(np.abs(run_1.wavenumbers - peak))
                intensity = pure_drug_spectrum[idx]
                plt.plot(peak, intensity, 'ro', markersize=8, fillstyle='none', label='_nolegend_') # Red circle for characteristic peaks
                plt.text(peak, intensity * 1.05, f'{peak:.0f}', rotation=45, ha='center', color='red')

            pure_drug_img = io.BytesIO()
            plt.savefig(pure_drug_img, format='png')
            pure_drug_img.seek(0)
            pure_drug_graph_url = base64.b64encode(pure_drug_img.getvalue()).decode()
            plt.close()

            match_percentage = len(found_peaks_mixture) / len(run_1.DRUGS[drug]) * 100 if run_1.DRUGS[drug] else 0

            # Generate PDF Report for Mixture
            report_filename = f"report_mixture_{drug}_{non_drug}.pdf"
            report_filepath = os.path.join(app.config['REPORT_FOLDER'], report_filename)
            generate_mixture_compound_report(report_filepath, drug, drug_weight, non_drug, non_drug_weight, found_peaks_mixture, match_percentage, mixture_graph_url, pure_drug_graph_url)


            return render_template('mixture_output.html',
                                   drug=drug,
                                   drug_weight=drug_weight,
                                   non_drug=non_drug,
                                   non_drug_weight=non_drug_weight,
                                   found_peaks=found_peaks_mixture,
                                   match_percentage=f"{match_percentage:.1f}",
                                   mixture_graph_url=mixture_graph_url,
                                   pure_drug_graph_url=pure_drug_graph_url,
                                   report_filename=report_filename)
        except Exception as e:
            flash(f"An error occurred during mixture simulation: {e}", 'danger')
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()
            return redirect(url_for('mixture_compound'))

    return render_template('mixture.html', drugs=drugs_list, non_drugs=non_drugs_list)

@app.route('/download_report/<filename>')
def download_report(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)


def generate_pure_compound_report(filepath, prediction_result, original_filepath, graph_url_base64, peak_matching_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 20, "Detectraa - Pure Compound Analysis Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Analysis of: {os.path.basename(original_filepath)}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Prediction Results:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Is Drug Detected: {'Yes' if prediction_result['is_drug'] else 'No'}", ln=True)
    pdf.cell(0, 10, f"Drug Probability: {prediction_result['probability']:.1%}", ln=True)

    if prediction_result['is_drug']:
        pdf.cell(0, 10, f"Predicted Drug Type: {prediction_result['drug_type']}", ln=True)
        pdf.cell(0, 10, f"Confidence: {prediction_result['confidence']:.1%}", ln=True)
        pdf.cell(0, 10, f"Matched Peaks: {prediction_result['matched_peaks']}", ln=True)
        
        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Drug Information:", ln=True)
        pdf.set_font("Arial", "", 12)
        for key, value in prediction_result['drug_info'].items():
            pdf.cell(0, 7, f"- {key.replace('_', ' ').title()}: {value}", ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Peak Matching Analysis:", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.cell(50, 7, "Expected Peak (cm^-1)", 1, 0, 'C')
        pdf.cell(50, 7, "Detected Peak (cm^-1)", 1, 0, 'C')
        pdf.cell(30, 7, "Match", 1, 1, 'C')

        # Use the pre-processed peak_matching_data
        for item in peak_matching_data:
            pdf.cell(50, 7, f"{item['expected_peak']:.0f}", 1, 0, 'C')
            if item['detected_peak'] is not None:
                pdf.cell(50, 7, f"{item['detected_peak']:.0f}", 1, 0, 'C')
                pdf.cell(30, 7, "Yes" if item['is_match'] else "No", 1, 1, 'C') # Use is_match from data
            else:
                pdf.cell(50, 7, "N/A", 1, 0, 'C')
                pdf.cell(30, 7, "No", 1, 1, 'C')
    else:
        pdf.cell(0, 10, f"Reason: {prediction_result.get('reason', 'N/A')}", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Top Detected Peaks:", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.cell(50, 7, "Wavenumber (cm^-1)", 1, 0, 'C')
        pdf.cell(50, 7, "Absorbance", 1, 1, 'C')
        # Use the pre-processed peak_matching_data for non-drug case
        for item in peak_matching_data:
            pdf.cell(50, 7, f"{item['wavenumber']:.1f}", 1, 0, 'C')
            pdf.cell(50, 7, f"{item['absorbance']:.3f}", 1, 1, 'C')


    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Spectrum Graph:", ln=True)
    pdf.ln(5)
    
    # Save the base64 image to a temporary file
    img_data = base64.b64decode(graph_url_base64)
    temp_img_path = "temp_spectrum_graph.png"
    with open(temp_img_path, "wb") as f:
        f.write(img_data)
    
    # Add image to PDF
    pdf.image(temp_img_path, x=10, w=pdf.w - 20)
    
    # Remove temporary image
    os.remove(temp_img_path)

    pdf.output(filepath)


def generate_mixture_compound_report(filepath, drug, drug_weight, non_drug, non_drug_weight, found_peaks, match_percentage, mixture_graph_url_base64, pure_drug_graph_url_base64):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 20, "Detectraa - Mixture Compound Analysis Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Simulated Mixture Composition:", ln=True)
    pdf.cell(0, 10, f"- Drug: {drug} ({drug_weight:.1f}%)", ln=True)
    pdf.cell(0, 10, f"- Cutting Agent: {non_drug.replace('_', ' ')} ({non_drug_weight:.1f}%)", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Identified Characteristics:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Predicted Dominant Drug: {drug}", ln=True)
    pdf.cell(0, 10, f"Percentage of Characteristic Peaks Matched: {match_percentage:.1f}%", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Characteristic Peaks Found in Mixture:", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(40, 7, "Target (cm^-1)", 1, 0, 'C')
    pdf.cell(40, 7, "Detected (cm^-1)", 1, 0, 'C')
    pdf.cell(40, 7, "Intensity", 1, 1, 'C')
    if found_peaks:
        for target, detected, intensity in found_peaks:
            pdf.cell(40, 7, f"{target:.0f}", 1, 0, 'C')
            pdf.cell(40, 7, f"{detected:.0f}", 1, 0, 'C')
            pdf.cell(40, 7, f"{intensity:.3f}", 1, 1, 'C')
    else:
        pdf.cell(120, 7, "No characteristic peaks detected for the chosen drug.", 1, 1, 'C')

    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Mixture Spectrum Graph:", ln=True)
    pdf.ln(5)
    
    # Save mixture graph image to a temporary file
    img_data_mixture = base64.b64decode(mixture_graph_url_base64)
    temp_mixture_img_path = "temp_mixture_spectrum_graph.png"
    with open(temp_mixture_img_path, "wb") as f:
        f.write(img_data_mixture)
    pdf.image(temp_mixture_img_path, x=10, w=pdf.w - 20)
    os.remove(temp_mixture_img_path)

    pdf.add_page() # New page for the pure drug spectrum for better layout
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Pure {drug} Spectrum Graph with Key Peaks:", ln=True)
    pdf.ln(5)

    # Save pure drug graph image to a temporary file
    img_data_pure = base64.b64decode(pure_drug_graph_url_base64)
    temp_pure_img_path = f"temp_pure_{drug}_spectrum_graph.png"
    with open(temp_pure_img_path, "wb") as f:
        f.write(img_data_pure)
    pdf.image(temp_pure_img_path, x=10, w=pdf.w - 20)
    os.remove(temp_pure_img_path)

    pdf.output(filepath)


if __name__ == '__main__':
    # Call load_models() when the application starts
    load_models()
    app.run(debug=True)