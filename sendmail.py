import smtplib, ssl
port = 587  # For starttls
smtp_server = "smtp.gmail.com"
sender_email = "sdanime1999@gmail.com"
receiver_email = "souvikdutta7012@gmail.com"
password = input("Enter you password:")
message = """\
Subject: Congratulations

Your model has been trained with accuracy more than 80%."""

context = ssl.create_default_context()
with smtplib.SMTP(smtp_server, port) as server:
    server.ehlo()  # Can be omitted
    server.starttls(context=context)
    server.ehlo()  # Can be omitted
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message)