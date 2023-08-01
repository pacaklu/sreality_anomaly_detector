from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
import smtplib, ssl


MESSAGE_BODY = 'anomaly flat detection'
EMAIL_SUBJECT = 'anomaly flat detection'
EMAIL_FROM = 'LPTestingEmailPython@gmail.com'
EMAIL_TO = 'pacaklu@seznam.cz'
SMTP_SERVER ='smtp.gmail.com'
SMTP_PORT = 587
SMTP_PASSWORD = 'yqzjbynvkfcqnuur'
FILE_NAME = 'testing_csv'

def send_mail(PATH_TO_CSV_FILE):
    # Create a multipart message
    msg = MIMEMultipart()
    body_part = MIMEText(MESSAGE_BODY, 'plain')
    msg['Subject'] = EMAIL_SUBJECT
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO
    # Add body to email
    msg.attach(body_part)
    # Open and read the CSV file in binary
    with open(PATH_TO_CSV_FILE,'rb') as file:
    # Attach the file with filename to the email
        msg.attach(MIMEApplication(file.read(), Name=FILE_NAME))

    # Create context
    context = ssl.create_default_context()

    # Login to the server and send message
    with smtplib.SMTP(SMTP_SERVER, port=SMTP_PORT) as smtp:
        smtp.starttls(context=context)
        smtp.login(EMAIL_FROM, SMTP_PASSWORD)
        smtp.send_message(msg)
