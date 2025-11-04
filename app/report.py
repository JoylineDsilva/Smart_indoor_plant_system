from fpdf import FPDF

def generate_pdf_report(status, confidence, recommendations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Plant Health Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Status: {status}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence}", ln=True)
    pdf.cell(200, 10, txt=f"Recommendations: {recommendations}", ln=True)

    pdf.output("report.pdf")
