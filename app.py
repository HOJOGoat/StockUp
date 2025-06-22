from flask import Flask, render_template, request, send_file
from ppa import StockAnalyzer
from weasyprint import HTML
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    report = ""
    chart_path = None
    pdf_ready = False
    ticker = ""

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        analyzer = StockAnalyzer(ticker)
        report = analyzer.generate_report()

        # Save report as HTML temporarily
        with open(f"templates/{ticker}_report.html", "w") as f:
            f.write(f"<pre>{report}</pre>")

        # Convert to PDF
        HTML(f"templates/{ticker}_report.html").write_pdf(f"{ticker}_report.pdf")
        pdf_ready = True

        chart_path = f"/static/{ticker}_technicals.png"

    return render_template('index.html', report=report, chart_path=chart_path, pdf_ready=pdf_ready, ticker=ticker)

@app.route('/download/<ticker>')
def download_pdf(ticker):
    path = f"{ticker}_report.pdf"
    return send_file(path, as_attachment=True)
