from fpdf import FPDF
import os

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Detectraa Analysis Report', ln=True, align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def generate_pdf_report(result, graph_path, output_path, mixture=False):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)

    if mixture:
        pdf.cell(0, 10, f"Mixture Analysis", ln=True)
        pdf.cell(0, 10, f"Drug: {result['drug']} ({result['drug_weight']}%)", ln=True)
        pdf.cell(0, 10, f"Cutting Agent: {result['agent']} ({result['agent_weight']}%)", ln=True)
        pdf.ln(5)

        pdf.cell(0, 10, "Matched Peaks:", ln=True)
        if result['matched_peaks']:
            for t, d, i in result['matched_peaks']:
                pdf.cell(0, 10, f"- Target: {t} | Detected: {d:.1f} | Intensity: {i:.3f}", ln=True)
        else:
            pdf.cell(0, 10, "No characteristic peaks found.", ln=True)

    else:
        if 'error' in result:
            pdf.set_text_color(255, 0, 0)
            pdf.cell(0, 10, f"Error: {result['error']}", ln=True)
            pdf.set_text_color(0, 0, 0)
        else:
            pdf.cell(0, 10, f"Drug Detected: {'Yes' if result['is_drug'] else 'No'}", ln=True)
            pdf.cell(0, 10, f"Probability: {result['probability']:.2f}", ln=True)

            if result.get('drug_type'):
                pdf.cell(0, 10, f"Identified Drug: {result['drug_type']}", ln=True)
                pdf.cell(0, 10, f"Confidence: {result['confidence']:.2f}", ln=True)

                drug_info = result.get('drug_info', {})
                for key, value in drug_info.items():
                    pdf.multi_cell(0, 10, f"{key.capitalize()}: {value}")

            if 'detected_peaks' in result:
                peaks = result['detected_peaks']
                intensities = result['peak_intensities']
                pdf.ln(5)
                pdf.cell(0, 10, "Top Detected Peaks:", ln=True)
                for p, i in zip(peaks, intensities):
                    pdf.cell(0, 10, f"{p:.1f} cm-1 (Intensity: {i:.3f})", ln=True)

    # Add graph image
    if os.path.exists(graph_path):
        pdf.ln(10)
        pdf.image(graph_path, x=10, w=180)

    pdf.output(output_path)
    return output_path
