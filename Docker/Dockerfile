FROM python:3.8
RUN pip install numpy pandas matplotlib
ADD Appscript.py .
ADD data_daily.csv .
CMD ["python", "./Appscript.py"]