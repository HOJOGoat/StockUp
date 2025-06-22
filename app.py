from flask import Flask, render_template, request
from ppa import StockAnalyzer  # your existing class

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def index():
    report = ""
    chart_path = None
    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        analyzer = StockAnalyzer(ticker)
        report = analyzer.generate_report()
        chart_path = f"/static/{ticker}_technicals.png"
    return render_template('index.html', report=report, chart_path=chart_path)

if __name__ == '__main__':
    app.run(debug=True)
